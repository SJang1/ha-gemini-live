"""Frontend module for Gemini Live integration."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from homeassistant.components.frontend import add_extra_js_url, remove_extra_js_url
from homeassistant.components.http import StaticPathConfig
from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

FRONTEND_PATH = Path(__file__).parent / "frontend"
CARD_FILENAME = "gemini-live-card.js"
CARD_URL = f"/{DOMAIN}/{CARD_FILENAME}"

# Track if frontend is already registered
DATA_FRONTEND_REGISTERED = f"{DOMAIN}_frontend_registered"


def _get_card_version() -> int:
    """Get the current version from the JS file's modification time."""
    try:
        return int((FRONTEND_PATH / CARD_FILENAME).stat().st_mtime)
    except Exception:
        return 1


async def async_register_frontend(hass: HomeAssistant) -> None:
    """Register the frontend files."""
    # Check if already registered to avoid duplicate registration on reload
    if hass.data.get(DATA_FRONTEND_REGISTERED):
        _LOGGER.debug("Gemini Live frontend already registered")
        return

    # Register static path for the card
    try:
        await hass.http.async_register_static_paths(
            [
                StaticPathConfig(
                    CARD_URL,
                    str(FRONTEND_PATH / "gemini-live-card.js"),
                    cache_headers=False,
                )
            ]
        )
    except RuntimeError as err:
        # aiohttp raises RuntimeError if the same route/method is already registered.
        # This can happen when the integration is reloaded or configuration changes
        # while the frontend static path is already present. Log and continue.
        _LOGGER.debug("Static path registration skipped (already registered): %s", err)

    # Get versioned URL for cache busting
    card_version = _get_card_version()
    card_url_versioned = f"{CARD_URL}?v={card_version}"
    
    # Remove any previously registered URL to prevent duplicates on reload/version change
    old_url = hass.data.get(f"{DOMAIN}_card_url_versioned")
    if old_url and old_url != card_url_versioned:
        try:
            remove_extra_js_url(hass, old_url)
        except Exception:
            pass  # Ignore if already removed
    
    # Add the card to the frontend (versioned URL for cache busting)
    try:
        add_extra_js_url(hass, card_url_versioned)
    except Exception as e:
        _LOGGER.debug("Failed to add extra JS URL for frontend: %s", e)

    # Log file existence to help debug 'custom element not found' issues
    try:
        card_path = FRONTEND_PATH / CARD_FILENAME
        if card_path.exists():
            _LOGGER.debug("Frontend card file exists: %s (size=%d)", card_path, card_path.stat().st_size)
        else:
            _LOGGER.warning("Frontend card file missing at %s", card_path)
    except Exception as e:
        _LOGGER.debug("Error checking frontend card file: %s", e)

    # Store the versioned URL for later unregistration
    hass.data[f"{DOMAIN}_card_url_versioned"] = card_url_versioned

    # Add to Lovelace resources
    await async_add_lovelace_resource(hass)

    # Diagnostic: check Lovelace resources for the card and log presence
    try:
        resources = await _get_lovelace_resources(hass)
        if resources is None:
            _LOGGER.debug("Lovelace resources unavailable when registering frontend")
        else:
            found = any((r.get("url", "").startswith(CARD_URL) for r in resources))
            _LOGGER.debug("Lovelace resource for Gemini card present: %s", found)
    except Exception as e:
        _LOGGER.debug("Error checking Lovelace resources: %s", e)

    # Mark as registered
    hass.data[DATA_FRONTEND_REGISTERED] = True

    _LOGGER.debug("Registered Gemini Live frontend at %s", CARD_URL)


async def async_unregister_frontend(hass: HomeAssistant) -> None:
    """Unregister the frontend files."""
    if not hass.data.get(DATA_FRONTEND_REGISTERED):
        return

    # Remove extra JS URL
    try:
        card_url_versioned = hass.data.get(f"{DOMAIN}_card_url_versioned")
        if card_url_versioned:
            remove_extra_js_url(hass, card_url_versioned)
    except Exception as e:
        _LOGGER.debug("Error removing extra JS URL: %s", e)

    # Do not remove Lovelace resources here to avoid removing the user's
    # dashboard resource when the integration is reloaded. Removing resources
    # during unload has caused the card to disappear unexpectedly for users
    # when restarting or reloading the integration. Preserve the Lovelace
    # resource so dashboards remain stable; administrators can remove the
    # resource manually if desired.
    _LOGGER.debug("Preserving Lovelace resource for Gemini Live card on unregister")

    # Mark as unregistered
    hass.data[DATA_FRONTEND_REGISTERED] = False

    _LOGGER.debug("Unregistered Gemini Live frontend")


async def async_add_lovelace_resource(hass: HomeAssistant) -> None:
    """Add the card to Lovelace resources."""
    try:
        # Get the Lovelace resources
        resources = await _get_lovelace_resources(hass)
        if resources is None:
            _LOGGER.debug("Lovelace resources not available (YAML mode?)")
            return

        # Use the versioned URL stored during registration for consistency
        card_url_versioned = hass.data.get(f"{DOMAIN}_card_url_versioned")
        if not card_url_versioned:
            # Fallback if not stored yet (shouldn't happen in normal flow)
            card_version = _get_card_version()
            card_url_versioned = f"{CARD_URL}?v={card_version}"

        # Check if already exists with the SAME version (check both versioned and unversioned)
        for resource in resources:
            resource_url = resource.get("url", "")
            if resource_url.startswith(CARD_URL):
                if resource_url == card_url_versioned:
                    _LOGGER.debug("Lovelace resource already exists with current version")
                    return
                # Resource exists but with old version - update it
                _LOGGER.debug("Lovelace resource exists with old version, updating")
                await _update_lovelace_resource(hass, resource.get("id"), card_url_versioned)
                return

        # Add the resource with version
        await _add_lovelace_resource(hass, card_url_versioned, "module")
        _LOGGER.info("Added Gemini Live card to Lovelace resources")

    except Exception as e:
        _LOGGER.warning("Failed to add Lovelace resource: %s", e)


async def async_remove_lovelace_resource(hass: HomeAssistant) -> None:
    """Remove the card from Lovelace resources."""
    try:
        resources = await _get_lovelace_resources(hass)
        if resources is None:
            return

        # Find and remove the resource (check both versioned and unversioned)
        for resource in resources:
            resource_url = resource.get("url", "")
            if resource_url.startswith(CARD_URL):
                await _remove_lovelace_resource(hass, resource.get("id"))
                _LOGGER.info("Removed Gemini Live card from Lovelace resources")
                return

    except Exception as e:
        _LOGGER.warning("Failed to remove Lovelace resource: %s", e)


async def _get_lovelace_resources(hass: HomeAssistant) -> list[dict[str, Any]] | None:
    """Get Lovelace resources from storage."""
    try:
        # Try to get the resources collection from lovelace component
        lovelace_data = hass.data.get("lovelace")
        if not lovelace_data:
            return None

        # Get resources using attribute access (not dict access) per HA 2026.2 deprecation
        resources_collection = getattr(lovelace_data, "resources", None)
        if resources_collection is None:
            return None

        # Ensure the collection is loaded before accessing items
        if hasattr(resources_collection, "loaded") and not resources_collection.loaded:
            if hasattr(resources_collection, "async_load"):
                await resources_collection.async_load()
                resources_collection.loaded = True

        if hasattr(resources_collection, "async_items"):
            # async_items() is synchronous in HA despite the name
            return list(resources_collection.async_items() or [])
        elif hasattr(resources_collection, "data"):
            return list(resources_collection.data.values())

        return None

    except Exception as e:
        _LOGGER.debug("Error getting Lovelace resources: %s", e)
        return None


async def _add_lovelace_resource(hass: HomeAssistant, url: str, resource_type: str) -> None:
    """Add a resource to Lovelace."""
    lovelace_data = hass.data.get("lovelace")
    if not lovelace_data:
        return

    resources_collection = getattr(lovelace_data, "resources", None)
    if resources_collection and hasattr(resources_collection, "async_create_item"):
        # CRITICAL: Ensure the resources collection is loaded before creating items.
        # If we call async_create_item before the collection is loaded, the collection's
        # internal data dict will be empty, and when it saves, it will OVERWRITE all
        # existing resources with just our new item, deleting all other custom cards.
        if hasattr(resources_collection, "loaded") and not resources_collection.loaded:
            if hasattr(resources_collection, "async_load"):
                await resources_collection.async_load()
                resources_collection.loaded = True
        
        # Double-check: If we think it's loaded but data is empty, something went wrong.
        # In this case, bail out to avoid potentially overwriting existing resources.
        if hasattr(resources_collection, "data") and not resources_collection.data:
            # Check if there really are no resources by trying to get items
            existing_items = resources_collection.async_items() if hasattr(resources_collection, "async_items") else None
            if existing_items is None or len(list(existing_items)) == 0:
                # Could be empty OR could be a load failure - check if storage exists
                _LOGGER.debug("Resources collection appears empty, proceeding cautiously")
        
        # Different HA versions/implementations accept slightly different
        # payload shapes when creating a resource. Try payloads in order of
        # likelihood to succeed (res_type is the correct field name in modern HA).
        payloads = [
            {"url": url, "res_type": resource_type},
            {"url": url, "res_type": "module"},
            {"url": url, "type": resource_type},
            {"url": url, "resource_type": resource_type},
            {"url": url},
        ]

        for payload in payloads:
            try:
                await resources_collection.async_create_item(payload)
                _LOGGER.debug("Created Lovelace resource with payload: %s", payload)
                return
            except Exception as exc:  # pylint: disable=broad-except
                _LOGGER.debug(
                    "Failed to create Lovelace resource with payload %s: %s",
                    payload,
                    exc,
                )

        _LOGGER.warning("All attempts to add Lovelace resource failed for URL: %s", url)


async def _update_lovelace_resource(hass: HomeAssistant, resource_id: str | None, url: str) -> None:
    """Update an existing Lovelace resource URL."""
    if not resource_id:
        return

    lovelace_data = hass.data.get("lovelace")
    if not lovelace_data:
        return

    resources_collection = getattr(lovelace_data, "resources", None)
    if resources_collection and hasattr(resources_collection, "async_update_item"):
        # Ensure the collection is loaded before updating items to avoid data corruption
        if hasattr(resources_collection, "loaded") and not resources_collection.loaded:
            if hasattr(resources_collection, "async_load"):
                await resources_collection.async_load()
                resources_collection.loaded = True
        
        # Verify we can find the item before updating (ensures data is loaded)
        if hasattr(resources_collection, "data") and resource_id not in resources_collection.data:
            _LOGGER.debug("Resource %s not found in collection, skipping update", resource_id)
            return
        
        try:
            await resources_collection.async_update_item(resource_id, {"url": url})
            _LOGGER.info("Updated Gemini Live card resource to version: %s", url)
        except Exception as exc:
            _LOGGER.debug("Failed to update Lovelace resource: %s", exc)


async def _remove_lovelace_resource(hass: HomeAssistant, resource_id: str | None) -> None:
    """Remove a resource from Lovelace."""
    if not resource_id:
        return

    lovelace_data = hass.data.get("lovelace")
    if not lovelace_data:
        return

    resources_collection = getattr(lovelace_data, "resources", None)
    if resources_collection and hasattr(resources_collection, "async_delete_item"):
        # Ensure the collection is loaded before deleting items to avoid data corruption
        if hasattr(resources_collection, "loaded") and not resources_collection.loaded:
            if hasattr(resources_collection, "async_load"):
                await resources_collection.async_load()
                resources_collection.loaded = True
        
        # Verify we can find the item before deleting (ensures data is loaded)
        if hasattr(resources_collection, "data") and resource_id not in resources_collection.data:
            _LOGGER.debug("Resource %s not found in collection, skipping delete", resource_id)
            return
        
        try:
            await resources_collection.async_delete_item(resource_id)
        except Exception as exc:
            _LOGGER.debug("Failed to delete Lovelace resource %s: %s", resource_id, exc)
