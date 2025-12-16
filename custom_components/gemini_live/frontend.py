"""Frontend module for Gemini Live integration."""
from __future__ import annotations

import logging
from pathlib import Path

from homeassistant.components.frontend import add_extra_js_url, remove_extra_js_url
from homeassistant.components.http import StaticPathConfig
from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

FRONTEND_PATH = Path(__file__).parent / "frontend"
CARD_FILENAME = "gemini-live-card.js"
CARD_URL = f"/{DOMAIN}/{CARD_FILENAME}"


def _get_card_version() -> int:
    """Get the current version from the JS file's modification time."""
    try:
        return int((FRONTEND_PATH / CARD_FILENAME).stat().st_mtime)
    except Exception:
        return 1


async def async_register_frontend(hass: HomeAssistant) -> None:
    """Register the frontend files using add_extra_js_url with versioning."""
    # Register static path for the card with NO caching
    try:
        await hass.http.async_register_static_paths(
            [
                StaticPathConfig(
                    CARD_URL,
                    str(FRONTEND_PATH / CARD_FILENAME),
                    cache_headers=False,
                )
            ]
        )
        _LOGGER.debug("Static path registered for %s", CARD_URL)
    except RuntimeError as err:
        _LOGGER.debug("Static path already registered: %s", err)

    # Get versioned URL for cache busting
    card_version = _get_card_version()
    card_url_versioned = f"{CARD_URL}?v={card_version}"
    
    # Remove any previously registered URL to prevent duplicates
    old_url = hass.data.get(f"{DOMAIN}_card_url_versioned")
    if old_url and old_url != card_url_versioned:
        try:
            remove_extra_js_url(hass, old_url)
            _LOGGER.debug("Removed old extra JS URL: %s", old_url)
        except Exception:
            pass

    # Add the card to the frontend with versioning
    try:
        add_extra_js_url(hass, card_url_versioned)
        _LOGGER.info("Registered Gemini Live card at %s", card_url_versioned)
    except Exception as e:
        _LOGGER.error("Failed to add extra JS URL: %s", e)

    # Store the versioned URL for later
    hass.data[f"{DOMAIN}_card_url_versioned"] = card_url_versioned


async def async_unregister_frontend(hass: HomeAssistant) -> None:
    """Unregister the frontend files."""
    try:
        card_url_versioned = hass.data.get(f"{DOMAIN}_card_url_versioned")
        if card_url_versioned:
            remove_extra_js_url(hass, card_url_versioned)
            _LOGGER.info("Unregistered Gemini Live card")
    except Exception as e:
        _LOGGER.debug("Error removing extra JS URL: %s", e)
