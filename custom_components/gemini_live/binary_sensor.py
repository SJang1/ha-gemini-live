"""Binary sensor for Gemini Live integration.

Provides sensors for listening/speaking state.
"""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the binary sensor platform."""
    data = hass.data[DOMAIN][entry.entry_id]

    async_add_entities(
        [
            GeminiLiveListeningSensor(hass, entry, data),
            GeminiLiveSpeakingSensor(hass, entry, data),
            GeminiLiveConnectedSensor(hass, entry, data),
        ],
        True,
    )


class GeminiLiveBaseSensor(BinarySensorEntity):
    """Base class for Gemini Live binary sensors."""

    _attr_has_entity_name = True

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        data: dict[str, Any],
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self._entry = entry
        self._data = data
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
                "name": "Gemini Live",
            "manufacturer": "Google",
            "model": "Gemini Live API",
        }


class GeminiLiveListeningSensor(GeminiLiveBaseSensor):
    """Binary sensor for listening state."""

    _attr_name = "Listening"
    _attr_device_class = BinarySensorDeviceClass.SOUND

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        data: dict[str, Any],
    ) -> None:
        """Initialize the sensor."""
        super().__init__(hass, entry, data)
        self._attr_unique_id = f"{entry.entry_id}_listening"
        self._remove_listener = None

    @property
    def is_on(self) -> bool:
        """Return true if listening."""
        return self._data.get("is_listening", False)

    async def async_added_to_hass(self) -> None:
        """Handle entity added to hass."""
        @callback
        def handle_state_change(event) -> None:
            """Handle listening state change event."""
            if event.data.get("entry_id") == self._entry.entry_id:
                self.async_write_ha_state()

        self._remove_listener = self.hass.bus.async_listen(
            f"{DOMAIN}_listening_state_changed",
            handle_state_change,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity removal."""
        if self._remove_listener:
            self._remove_listener()


class GeminiLiveSpeakingSensor(GeminiLiveBaseSensor):
    """Binary sensor for speaking state."""

    _attr_name = "Speaking"
    _attr_device_class = BinarySensorDeviceClass.SOUND

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        data: dict[str, Any],
    ) -> None:
        """Initialize the sensor."""
        super().__init__(hass, entry, data)
        self._attr_unique_id = f"{entry.entry_id}_speaking"
        self._remove_listener = None

    @property
    def is_on(self) -> bool:
        """Return true if speaking."""
        return self._data.get("is_speaking", False)

    async def async_added_to_hass(self) -> None:
        """Handle entity added to hass."""
        @callback
        def handle_state_change(event) -> None:
            """Handle speaking state change event."""
            if event.data.get("entry_id") == self._entry.entry_id:
                self.async_write_ha_state()

        self._remove_listener = self.hass.bus.async_listen(
            f"{DOMAIN}_speaking_state_changed",
            handle_state_change,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity removal."""
        if self._remove_listener:
            self._remove_listener()


class GeminiLiveConnectedSensor(GeminiLiveBaseSensor):
    """Binary sensor for connection state."""

    _attr_name = "Connected"
    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        data: dict[str, Any],
    ) -> None:
        """Initialize the sensor."""
        super().__init__(hass, entry, data)
        self._attr_unique_id = f"{entry.entry_id}_connected"
        self._remove_listener = None

    @property
    def is_on(self) -> bool:
        """Return true if connected."""
        client = self._data.get("client")
        return client.connected if client else False

    async def async_added_to_hass(self) -> None:
        """Handle entity added to hass."""
        @callback
        def handle_state_change(event) -> None:
            """Handle connected state change event."""
            if event.data.get("entry_id") == self._entry.entry_id:
                self.async_write_ha_state()

        self._remove_listener = self.hass.bus.async_listen(
            f"{DOMAIN}_connected_state_changed",
            handle_state_change,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity removal."""
        if self._remove_listener:
            self._remove_listener()
