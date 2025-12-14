"""WebSocket API for Gemini Live Audio integration.

This provides WebSocket endpoints for frontend clients to interact
with the Gemini Live API.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback

from .const import (
    DOMAIN,
    EVENT_AUDIO_DELTA,
    EVENT_ERROR,
    EVENT_FUNCTION_CALL,
    EVENT_INPUT_TRANSCRIPTION,
    EVENT_INTERRUPTED,
    EVENT_OUTPUT_TRANSCRIPTION,
    EVENT_SESSION_STARTED,
    EVENT_SESSION_RESUMED,
    EVENT_SESSION_RESUMPTION_UPDATE,
    EVENT_GO_AWAY,
    EVENT_GENERATION_COMPLETE,
    EVENT_TURN_COMPLETE,
)
from .live_client import GeminiLiveClient
from .mcp_handler import HomeAssistantMCPTools

_LOGGER = logging.getLogger(__name__)


def _get_client(hass: HomeAssistant, entry_id: str | None = None) -> tuple[GeminiLiveClient | None, str | None]:
    """Get the Gemini Live Audio client.
    
    If entry_id is provided, returns that specific client.
    Otherwise, auto-detects the first available client.
    
    Returns tuple of (client, entry_id).
    """
    if DOMAIN not in hass.data:
        return None, None

    # If entry_id provided, use it directly
    if entry_id and entry_id in hass.data[DOMAIN]:
        data = hass.data[DOMAIN][entry_id]
        if isinstance(data, dict):
            return data.get("client"), entry_id
        return None, None

    # Auto-detect first available client
    for eid, data in hass.data[DOMAIN].items():
        # Skip special keys and non-dict entries
        if eid.startswith("_") or not isinstance(data, dict):
            continue
        client = data.get("client")
        if client:
            return client, eid

    return None, None


def _get_data(hass: HomeAssistant, entry_id: str | None = None) -> tuple[dict | None, str | None]:
    """Get the integration data for an entry.
    
    If entry_id is provided, returns that specific data.
    Otherwise, auto-detects the first available entry.
    
    Returns tuple of (data dict, entry_id).
    """
    if DOMAIN not in hass.data:
        return None, None

    # If entry_id provided, use it directly
    if entry_id and entry_id in hass.data[DOMAIN]:
        data = hass.data[DOMAIN][entry_id]
        if isinstance(data, dict):
            return data, entry_id
        return None, None

    # Auto-detect first available entry
    for eid, data in hass.data[DOMAIN].items():
        # Skip special keys and non-dict entries
        if eid.startswith("_") or not isinstance(data, dict):
            continue
        if "client" in data:
            return data, eid

    return None, None


def async_register_websocket_api(hass: HomeAssistant) -> None:
    """Register WebSocket API commands."""
    websocket_api.async_register_command(hass, websocket_connect)
    websocket_api.async_register_command(hass, websocket_disconnect)
    websocket_api.async_register_command(hass, websocket_send_text)
    websocket_api.async_register_command(hass, websocket_send_audio)
    websocket_api.async_register_command(hass, websocket_send_image)
    websocket_api.async_register_command(hass, websocket_start_listening)
    websocket_api.async_register_command(hass, websocket_stop_listening)
    websocket_api.async_register_command(hass, websocket_get_status)
    websocket_api.async_register_command(hass, websocket_subscribe)


@websocket_api.websocket_command(
    {
        "type": "gemini_live/connect",
        vol.Optional("entry_id"): str,
        vol.Optional("resumption_handle"): str,
    }
)
@websocket_api.async_response
async def websocket_connect(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Connect to the Gemini Live Audio API."""
    data, entry_id = _get_data(hass, msg.get("entry_id"))

    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    client: GeminiLiveClient = data.get("client")
    ha_tools: HomeAssistantMCPTools = data.get("ha_tools")
    mcp_handler = data.get("mcp_handler")

    if not client:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio client not found")
        return

    try:
        # Force disconnect any stale connection first
        if client.connected:
            _LOGGER.info("Disconnecting stale connection before new connect")
            await client.disconnect()
            # Fire disconnected event
            if entry_id:
                hass.bus.async_fire(
                    f"{DOMAIN}_connected_state_changed",
                    {"entry_id": entry_id, "is_connected": False},
                )
        
        # Set resumption handle if provided (for session resumption)
        resumption_handle = msg.get("resumption_handle")
        if resumption_handle:
            _LOGGER.info("Using resumption handle for session resume")
            client.set_resumption_handle(resumption_handle)
        
        # Register function call handler BEFORE connecting
        async def on_function_call(event_data: dict[str, Any]) -> None:
            """Handle function calls from Gemini and send results back."""
            call_id = event_data.get("call_id", "")
            function_name = event_data.get("name", "")
            arguments = event_data.get("arguments", {})
            
            _LOGGER.info("Handling function call: %s (call_id=%s)", function_name, call_id)
            
            result = None
            
            # Check for Home Assistant built-in tools
            if ha_tools and function_name in ["get_entity_state", "call_service", "get_entities_by_domain", "get_area_entities"]:
                result = await ha_tools.execute_tool(function_name, arguments)
            # Check if this is an MCP server function (format: server_name__tool_name)
            elif mcp_handler and "__" in function_name:
                parsed = mcp_handler.parse_function_name(function_name)
                if parsed:
                    server_name, tool_name = parsed
                    _LOGGER.info("Calling MCP tool: %s/%s", server_name, tool_name)
                    result = await mcp_handler.call_tool(server_name, tool_name, arguments)
            
            if result is None:
                result = {"error": f"Unknown function: {function_name}"}
            
            # Send result back to Gemini
            _LOGGER.info("Sending function result for %s: %s", function_name, result)
            await client.send_function_result(call_id, result)
        
        # Register the handler
        client.on(EVENT_FUNCTION_CALL, on_function_call)
        # Store for cleanup
        data["_function_call_handler"] = on_function_call
        
        success = await client.connect()
        
        # Fire connected event for binary sensor
        if success and entry_id:
            hass.bus.async_fire(
                f"{DOMAIN}_connected_state_changed",
                {"entry_id": entry_id, "is_connected": True},
            )
        
        connection.send_result(msg["id"], {"connected": success})
    except Exception as e:
        connection.send_error(msg["id"], "connection_failed", str(e))


@websocket_api.websocket_command(
    {
        "type": "gemini_live/disconnect",
        vol.Optional("entry_id"): str,
    }
)
@websocket_api.async_response
async def websocket_disconnect(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Disconnect from the Gemini Live Audio API."""
    data, entry_id = _get_data(hass, msg.get("entry_id"))

    if data:
        client = data.get("client")
        
        # Remove function call handler if registered
        handler = data.pop("_function_call_handler", None)
        if handler and client:
            client.off(EVENT_FUNCTION_CALL, handler)
        
        if client:
            await client.disconnect()
        
        # Fire disconnected event for binary sensor
        if entry_id:
            hass.bus.async_fire(
                f"{DOMAIN}_connected_state_changed",
                {"entry_id": entry_id, "is_connected": False},
            )

    connection.send_result(msg["id"], {"connected": False})


@websocket_api.websocket_command(
    {
        "type": "gemini_live/send_text",
        vol.Optional("entry_id"): str,
        vol.Required("text"): str,
    }
)
@websocket_api.async_response
async def websocket_send_text(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Send text to the Gemini Live Audio API."""
    text = msg["text"]
    client, entry_id = _get_client(hass, msg.get("entry_id"))

    if not client or not client.connected:
        connection.send_error(msg["id"], "not_connected", "Gemini Live Audio not connected")
        return

    try:
        response = await client.send_text(text)
        connection.send_result(
            msg["id"],
            {
                "success": True,
                "text": response.text,
                "audio_transcript": response.audio_transcript,
            },
        )
    except Exception as e:
        connection.send_error(msg["id"], "send_failed", str(e))


@websocket_api.websocket_command(
    {
        "type": "gemini_live/send_audio",
        vol.Optional("entry_id"): str,
        vol.Required("audio"): str,  # Base64 encoded audio
    }
)
@websocket_api.async_response
async def websocket_send_audio(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Send audio to the Gemini Live Audio API."""
    audio_b64 = msg["audio"]
    client, entry_id = _get_client(hass, msg.get("entry_id"))

    if not client or not client.connected:
        connection.send_error(msg["id"], "not_connected", "Gemini Live Audio not connected")
        return

    try:
        await client.send_audio_base64(audio_b64)
        connection.send_result(msg["id"], {"success": True})
    except Exception as e:
        connection.send_error(msg["id"], "send_failed", str(e))


@websocket_api.websocket_command(
    {
        "type": "gemini_live/send_image",
        vol.Optional("entry_id"): str,
        vol.Required("image"): str,  # Base64 encoded image
        vol.Optional("mime_type"): str,  # MIME type (default: image/jpeg)
    }
)
@websocket_api.async_response
async def websocket_send_image(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Send image to the Gemini Live Audio API."""
    image_b64 = msg["image"]
    mime_type = msg.get("mime_type", "image/jpeg")
    client, entry_id = _get_client(hass, msg.get("entry_id"))

    if not client or not client.connected:
        connection.send_error(msg["id"], "not_connected", "Gemini Live Audio not connected")
        return

    try:
        await client.send_image_base64(image_b64, mime_type)
        connection.send_result(msg["id"], {"success": True})
    except Exception as e:
        connection.send_error(msg["id"], "send_failed", str(e))


@websocket_api.websocket_command(
    {
        "type": "gemini_live/start_listening",
        vol.Optional("entry_id"): str,
    }
)
@websocket_api.async_response
async def websocket_start_listening(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Start listening for audio input."""
    data, entry_id = _get_data(hass, msg.get("entry_id"))

    if not data or not entry_id:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    client: GeminiLiveClient = data.get("client")

    if not client:
        connection.send_error(msg["id"], "not_found", "Client not found")
        return

    # Connect if not already connected - wait for actual connection
    if not client.connected:
        _LOGGER.info("Connecting to Gemini API before starting listening...")
        success = await client.connect()
        if not success:
            connection.send_error(msg["id"], "connection_failed", "Failed to connect to Gemini API")
            return
        
        # Fire connected event for binary sensor
        hass.bus.async_fire(
            f"{DOMAIN}_connected_state_changed",
            {"entry_id": entry_id, "is_connected": True},
        )

    # Update listening state
    data["is_listening"] = True

    # Fire event for binary sensor
    hass.bus.async_fire(
        f"{DOMAIN}_listening_state_changed",
        {"entry_id": entry_id, "is_listening": True},
    )

    connection.send_result(msg["id"], {"listening": True, "connected": True})


@websocket_api.websocket_command(
    {
        "type": "gemini_live/stop_listening",
        vol.Optional("entry_id"): str,
    }
)
@websocket_api.async_response
async def websocket_stop_listening(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Stop listening for audio input."""
    data, entry_id = _get_data(hass, msg.get("entry_id"))

    if not data or not entry_id:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    client: GeminiLiveClient = data.get("client")

    # Signal end of audio stream
    if client and client.connected:
        await client.send_audio_stream_end()

    # Update listening state
    data["is_listening"] = False

    # Fire event for binary sensor
    hass.bus.async_fire(
        f"{DOMAIN}_listening_state_changed",
        {"entry_id": entry_id, "is_listening": False},
    )

    connection.send_result(msg["id"], {"listening": False})


@websocket_api.websocket_command(
    {
        "type": "gemini_live/get_status",
        vol.Optional("entry_id"): str,
    }
)
@websocket_api.async_response
async def websocket_get_status(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Get the current status of the Gemini Live Audio connection."""
    data, entry_id = _get_data(hass, msg.get("entry_id"))

    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    client: GeminiLiveClient = data.get("client")

    connection.send_result(
        msg["id"],
        {
            "connected": client.connected if client else False,
            "is_listening": data.get("is_listening", False),
            "is_speaking": data.get("is_speaking", False),
        },
    )


@websocket_api.websocket_command(
    {
        "type": "gemini_live/subscribe",
        vol.Optional("entry_id"): str,
    }
)
@callback
def websocket_subscribe(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Subscribe to Gemini Live Audio events."""
    # Note: We can't use async _get_client here since this is a @callback
    # So we need to do manual detection synchronously
    entry_id = msg.get("entry_id")
    data = None
    client = None
    
    if DOMAIN in hass.data:
        if entry_id and entry_id in hass.data[DOMAIN]:
            data = hass.data[DOMAIN][entry_id]
            if isinstance(data, dict):
                client = data.get("client")
        else:
            # Auto-detect
            for eid, d in hass.data[DOMAIN].items():
                if eid.startswith("_") or not isinstance(d, dict):
                    continue
                if "client" in d:
                    data = d
                    client = d.get("client")
                    entry_id = eid
                    break

    if not data or not client:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    # Track subscription for cleanup
    subscriptions = data.setdefault("subscriptions", {})
    connection_id = id(connection)

    # Clear any existing subscriptions for this client to prevent duplicate events
    # This happens when user reconnects without proper disconnect
    if connection_id in subscriptions:
        old_handlers = subscriptions.pop(connection_id, {})
        for event_type, handler in old_handlers.items():
            client.off(event_type, handler)
        _LOGGER.debug("Cleared stale subscription for connection %s", connection_id)

    # Define event handlers
    async def on_audio_delta(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "audio_delta", "data": event_data},
            )
        )

    async def on_transcript(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "transcript", "data": event_data},
            )
        )

    async def on_output_transcript(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "output_transcript", "data": event_data},
            )
        )

    async def on_turn_complete(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "turn_complete", "data": event_data},
            )
        )

    async def on_error(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "error", "data": event_data},
            )
        )

    async def on_interrupted(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "interrupted", "data": event_data},
            )
        )

    async def on_function_call_event(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "function_call", "data": event_data},
            )
        )

    async def on_session_resumed(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "session_resumed", "data": event_data},
            )
        )

    async def on_session_resumption_update(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "session_resumption_update", "data": event_data},
            )
        )

    async def on_go_away(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "go_away", "data": event_data},
            )
        )

    async def on_generation_complete(event_data: dict[str, Any]) -> None:
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"type": "generation_complete", "data": event_data},
            )
        )

    # Register handlers
    client.on(EVENT_AUDIO_DELTA, on_audio_delta)
    client.on(EVENT_INPUT_TRANSCRIPTION, on_transcript)
    client.on(EVENT_OUTPUT_TRANSCRIPTION, on_output_transcript)
    client.on(EVENT_TURN_COMPLETE, on_turn_complete)
    client.on(EVENT_ERROR, on_error)
    client.on(EVENT_INTERRUPTED, on_interrupted)
    client.on(EVENT_FUNCTION_CALL, on_function_call_event)
    client.on(EVENT_SESSION_RESUMED, on_session_resumed)
    client.on(EVENT_SESSION_RESUMPTION_UPDATE, on_session_resumption_update)
    client.on(EVENT_GO_AWAY, on_go_away)
    client.on(EVENT_GENERATION_COMPLETE, on_generation_complete)

    # Store handlers for cleanup
    subscriptions[connection_id] = {
        EVENT_AUDIO_DELTA: on_audio_delta,
        EVENT_INPUT_TRANSCRIPTION: on_transcript,
        EVENT_OUTPUT_TRANSCRIPTION: on_output_transcript,
        EVENT_TURN_COMPLETE: on_turn_complete,
        EVENT_ERROR: on_error,
        EVENT_INTERRUPTED: on_interrupted,
        EVENT_FUNCTION_CALL: on_function_call_event,
        EVENT_SESSION_RESUMED: on_session_resumed,
        EVENT_SESSION_RESUMPTION_UPDATE: on_session_resumption_update,
        EVENT_GO_AWAY: on_go_away,
        EVENT_GENERATION_COMPLETE: on_generation_complete,
    }

    def on_close() -> None:
        """Handle connection close."""
        handlers = subscriptions.pop(connection_id, {})
        for event_type, handler in handlers.items():
            client.off(event_type, handler)

    connection.subscriptions[msg["id"]] = on_close
    connection.send_result(msg["id"])
