"""Media player for Gemini Live integration.

This provides a virtual media player for audio output from the Gemini API.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.components.media_player import (
    MediaPlayerDeviceClass,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
    MediaType,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .audio_processor import AudioBuffer, AudioProcessor
from .const import AUDIO_OUTPUT_SAMPLE_RATE, DOMAIN, EVENT_AUDIO_DELTA, EVENT_TURN_COMPLETE
from .live_client import GeminiLiveClient

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the media player platform."""
    data = hass.data[DOMAIN][entry.entry_id]

    async_add_entities(
        [GeminiLiveMediaPlayer(hass, entry, data)],
        True,
    )


class GeminiLiveMediaPlayer(MediaPlayerEntity):
    """Media player for Gemini Live audio output."""

    _attr_has_entity_name = True
    _attr_name = "Audio Output"
    _attr_device_class = MediaPlayerDeviceClass.SPEAKER
    _attr_supported_features = (
        MediaPlayerEntityFeature.PLAY
        | MediaPlayerEntityFeature.STOP
        | MediaPlayerEntityFeature.VOLUME_SET
        | MediaPlayerEntityFeature.VOLUME_MUTE
    )

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        data: dict[str, Any],
    ) -> None:
        """Initialize the media player."""
        self.hass = hass
        self._entry = entry
        self._data = data
        self._attr_unique_id = f"{entry.entry_id}_media_player"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Gemini Live",
            "manufacturer": "Google",
            "model": "Gemini Live API",
        }

        self._state = MediaPlayerState.IDLE
        self._volume = 1.0
        self._muted = False
        self._audio_buffer = AudioBuffer(AUDIO_OUTPUT_SAMPLE_RATE)
        self._playback_task: asyncio.Task | None = None

    @property
    def state(self) -> MediaPlayerState:
        """Return the state of the media player."""
        return self._state

    @property
    def volume_level(self) -> float:
        """Return the volume level."""
        return self._volume

    @property
    def is_volume_muted(self) -> bool:
        """Return if volume is muted."""
        return self._muted

    async def async_added_to_hass(self) -> None:
        """Handle entity added to hass."""
        client: GeminiLiveClient = self._data.get("client")

        if client:
            # Register for audio events
            client.on(EVENT_AUDIO_DELTA, self._handle_audio_delta)
            client.on(EVENT_TURN_COMPLETE, self._handle_turn_complete)

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity removal."""
        client: GeminiLiveClient = self._data.get("client")

        if client:
            client.off(EVENT_AUDIO_DELTA, self._handle_audio_delta)
            client.off(EVENT_TURN_COMPLETE, self._handle_turn_complete)

        if self._playback_task:
            self._playback_task.cancel()

    async def _handle_audio_delta(self, data: dict[str, Any]) -> None:
        """Handle incoming audio data."""
        audio_b64 = data.get("audio", "")
        if audio_b64:
            audio_bytes = AudioProcessor.decode_base64(audio_b64)
            self._audio_buffer.write(audio_bytes)

            if self._state != MediaPlayerState.PLAYING:
                self._state = MediaPlayerState.PLAYING
                self._data["is_speaking"] = True
                self.async_write_ha_state()

                # Fire event
                self.hass.bus.async_fire(
                    f"{DOMAIN}_speaking_state_changed",
                    {"entry_id": self._entry.entry_id, "is_speaking": True},
                )

    async def _handle_turn_complete(self, data: dict[str, Any]) -> None:
        """Handle turn complete event."""
        if self._state == MediaPlayerState.PLAYING:
            self._state = MediaPlayerState.IDLE
            self._data["is_speaking"] = False
            self.async_write_ha_state()

            # Fire event
            self.hass.bus.async_fire(
                f"{DOMAIN}_speaking_state_changed",
                {"entry_id": self._entry.entry_id, "is_speaking": False},
            )

    async def async_media_play(self) -> None:
        """Start playback."""
        self._state = MediaPlayerState.PLAYING
        self.async_write_ha_state()

    async def async_media_stop(self) -> None:
        """Stop playback."""
        self._audio_buffer.clear()
        self._state = MediaPlayerState.IDLE
        self._data["is_speaking"] = False
        self.async_write_ha_state()

    async def async_set_volume_level(self, volume: float) -> None:
        """Set volume level."""
        self._volume = volume
        self.async_write_ha_state()

    async def async_mute_volume(self, mute: bool) -> None:
        """Mute volume."""
        self._muted = mute
        self.async_write_ha_state()

    def get_audio_buffer(self) -> AudioBuffer:
        """Get the audio buffer for external playback."""
        return self._audio_buffer

    def get_audio_as_wav(self) -> bytes | None:
        """Get buffered audio as WAV file."""
        if self._audio_buffer.size > 0:
            return self._audio_buffer.get_wav()
        return None
