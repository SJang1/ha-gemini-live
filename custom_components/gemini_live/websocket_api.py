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
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class ConnectionRoute:
    """Mapping between a user websocket and backend (hassio/google) websocket.

    Helps decide where events for a given websocket should be routed.
    """

    # frontend websocket id (id(connection))
    user_ws: Optional[int] = None
    # backend client websocket id or client object id (if known)
    hassio_google_ws: Optional[int] = None
    # server-side client uuid mapping (e.g. "ws_<id>" or legacy id)
    client_uuid: Optional[str] = None
    # creation timestamp (asyncio loop time)
    created_at: Optional[float] = None
    # whether the backend session/client is connected
    server_connected: bool = False
    # last seen timestamp for server events
    last_seen: Optional[float] = None
    # stored subscription handlers keyed by connection id
    subscriptions: dict = field(default_factory=dict)
    # per-connection client state (listening/speaking)
    client_states: dict = field(default_factory=dict)
    # arbitrary metadata
    meta: dict = field(default_factory=dict)

    def owns(self, connection_id: int) -> bool:
        """Return True if this route owns the given connection id."""
        return connection_id == self.user_ws or connection_id == self.hassio_google_ws

    def to_debug(self) -> dict:
        """Return a small serializable debug summary."""
        return {
            "user_ws": self.user_ws,
            "hassio_google_ws": self.hassio_google_ws,
            "client_uuid": self.client_uuid,
            "created_at": self.created_at,
            "subscriptions_count": len(self.subscriptions),
        }


async def _disconnect_and_cleanup(data: dict, client_uuid: str, hass: HomeAssistant | None = None, entry_id: str | None = None) -> None:
    """Disconnect and destroy a client session."""
    clients = data.get("clients", {})
    routes = data.get("routes", {})
    
    if client_uuid and client_uuid in clients:
        client_entry = clients[client_uuid]
        client = client_entry.get("client")
        
        # Unregister handlers
        meta = client_entry.get("meta", {})
        handler = meta.pop("_function_call_handler", None)
        if handler and client:
            client.off(EVENT_FUNCTION_CALL, handler)
        
        session_handler = meta.pop("_session_handler", None)
        if session_handler and client:
            client.off(EVENT_SESSION_STARTED, session_handler)
            client.off(EVENT_SESSION_RESUMED, session_handler)

        # IMPORTANT: Preserve subscriptions before destroying client entry.
        # If we are about to destroy the client, we must move its stored subscriptions
        # back to the global map so they can be migrated to the NEW client that
        # will likely be created immediately after (e.g. in start_listening).
        try:
            stored_subs = client_entry.get("subscriptions", {})
            _LOGGER.info(
                "Cleanup: client %s has stored_subs keys=%s existing_global_subs_keys=%s",
                client_uuid,
                list(stored_subs.keys()) if stored_subs else None,
                list(data.get("subscriptions", {}).keys()),
            )
            if stored_subs:
                global_subs = data.setdefault("subscriptions", {})
                for conn_id, handlers in stored_subs.items():
                    # Restore to global map keyed by connection_id
                    global_subs[conn_id] = handlers
                    _LOGGER.info("Cleanup: preserved handlers for conn_id=%s types=%s to global", conn_id, list(handlers.keys()))
                _LOGGER.debug("Preserved subscriptions for %d connections from destroyed client %s", len(stored_subs), client_uuid)
        except Exception as e:
            _LOGGER.warning("Failed to preserve subscriptions during cleanup: %s", e)

        # Clear listening/speaking state for this client's owner connection
        try:
            owner_ws = client_entry.get("owner_ws")
            if owner_ws:
                client_states = data.setdefault("client_states", {})
                if owner_ws in client_states:
                    client_states[owner_ws]["listening"] = False
                    client_states[owner_ws]["speaking"] = False
                
                # Update aggregate state and fire events to ensure sensors reset
                data["is_listening"] = any(s.get("listening") for s in client_states.values())
                data["is_speaking"] = any(s.get("speaking") for s in client_states.values())
                
                if hass and entry_id:
                    hass.bus.async_fire(
                        f"{DOMAIN}_listening_state_changed",
                        {"entry_id": entry_id, "is_listening": data.get("is_listening", False)},
                    )
                    hass.bus.async_fire(
                        f"{DOMAIN}_speaking_state_changed",
                        {"entry_id": entry_id, "is_speaking": data.get("is_speaking", False)},
                    )
        except Exception:
            pass

        if client:
            _LOGGER.info("Destroying client %s", client_uuid)
            try:
                await client.disconnect()
            except Exception:
                pass
            
        # Remove from clients
        clients.pop(client_uuid, None)
        
        # Remove from routes
        routes.pop(client_uuid, None)

async def _create_and_connect_client(
    hass: HomeAssistant,
    data: dict,
    client_uuid: str,
    connection: websocket_api.ActiveConnection,
    entry_id: str,
    resumption_handle: str | None = None
) -> GeminiLiveClient:
    """Create and connect a new GeminiLiveClient."""
    clients = data.setdefault("clients", {})
    connections_by_ws = data.setdefault("connections_by_ws", {})
    
    api_key = data.get("api_key")
    session_config = data.get("session_config")
    if not api_key or not session_config:
        raise ValueError("Integration not configured properly")

    # Create a new GeminiLiveClient instance
    new_client = GeminiLiveClient(api_key=api_key, session_config=session_config)
    
    # Tag the client with the owning frontend websocket id
    try:
        new_client.set_owner_ws(id(connection))
    except Exception:
        pass
        
    clients[client_uuid] = {
        "client": new_client,
        "created_at": asyncio.get_event_loop().time(),
        "subscriptions": {},
        "client_states": {},
        "owner_ws": id(connection),
        "meta": {"conversation_enabled": True, "is_placeholder": False},
    }
    
    # Ensure route entry exists
    try:
        routes = data.setdefault("routes", {})
        routes.setdefault(client_uuid, ConnectionRoute(client_uuid=client_uuid))
        routes[client_uuid].user_ws = id(connection)
        routes[client_uuid].hassio_google_ws = id(new_client)
        routes[client_uuid].created_at = asyncio.get_event_loop().time()
        routes[client_uuid].meta.setdefault("is_placeholder", False)
    except Exception:
        pass
        
    client = new_client
    # Map this websocket connection to the client_uuid
    connections_by_ws[id(connection)] = client_uuid
    
    # Migrate any stored subscriptions
    try:
        connection_id = id(connection)
        legacy_client = data.get("client")
        _migrate_subscriptions_to_client(data, client_uuid, connection_id, new_client, legacy_client)
    except Exception:
        _LOGGER.debug("Failed to migrate subscription handlers to new client %s", client_uuid)

    # Set resumption handle if provided
    if resumption_handle:
        _LOGGER.info("Using resumption handle for session resume")
        client.set_resumption_handle(resumption_handle)
    
    # Register handlers
    ha_tools = data.get("ha_tools")
    mcp_handler = data.get("mcp_handler")
    
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
        # Check if this is an MCP server function
        elif mcp_handler and "__" in function_name:
            parsed = mcp_handler.parse_function_name(function_name)
            if parsed:
                server_name, tool_name = parsed
                _LOGGER.info("Calling MCP tool: %s/%s", server_name, tool_name)
                result = await mcp_handler.call_tool(server_name, tool_name, arguments)

        # Agent fallback
        if result is None:
            try:
                agent = data.get("agent")
                if agent and hasattr(agent, "_execute_function"):
                    try:
                        result = await agent._execute_function(function_name, call_id, arguments)
                    except Exception:
                        result = None
            except Exception:
                result = None

        if result is None:
            result = {"error": f"Unknown function: {function_name}"}
        
        _LOGGER.info("Sending function result for %s: %s", function_name, result)
        await client.send_function_result(call_id, result)
    
    async def on_session_started_or_resumed(event_data: dict[str, Any]) -> None:
        """Update routes with new Google WS ID when session starts or resumes."""
        try:
            routes = data.setdefault("routes", {})
            if client_uuid and client_uuid in routes:
                google_ws_id = None
                try:
                    google_ws_id = client.get_google_ws_id()
                except Exception:
                    google_ws_id = id(client)
                
                routes[client_uuid].hassio_google_ws = google_ws_id
                routes[client_uuid].server_connected = True
                routes[client_uuid].last_seen = asyncio.get_event_loop().time()
        except Exception as e:
            _LOGGER.warning("Failed to update route after session event: %s", e)
    
    client.on(EVENT_FUNCTION_CALL, on_function_call)
    client.on(EVENT_SESSION_STARTED, on_session_started_or_resumed)
    client.on(EVENT_SESSION_RESUMED, on_session_started_or_resumed)
    
    # Store for cleanup
    clients[client_uuid].setdefault("meta", {})
    clients[client_uuid]["meta"]["_function_call_handler"] = on_function_call
    clients[client_uuid]["meta"]["_session_handler"] = on_session_started_or_resumed
    
    success = await client.connect()
    
    # Update route/server status
    try:
        routes = data.setdefault("routes", {})
        if client_uuid and client_uuid in routes:
            google_ws_id = None
            try:
                google_ws_id = client.get_google_ws_id()
            except Exception:
                google_ws_id = id(client)
            
            routes[client_uuid].hassio_google_ws = google_ws_id
            routes[client_uuid].server_connected = bool(success)
            routes[client_uuid].last_seen = asyncio.get_event_loop().time()
    except Exception:
        pass
        
    if success and entry_id:
        hass.bus.async_fire(
            f"{DOMAIN}_connected_state_changed",
            {"entry_id": entry_id, "is_connected": True},
        )
        hass.bus.async_fire(f"{DOMAIN}_client_added", {"entry_id": entry_id, "client_uuid": client_uuid})

    return client


def _ensure_ws_client_uuid(data: dict, connection: websocket_api.ActiveConnection) -> str:
    """Ensure there is a server-side client UUID mapped to this websocket.

    Uses the websocket id as the stable identifier. If no mapping exists,
    creates a lightweight placeholder client entry pointing at the legacy
    shared client until `connect()` creates a per-connection `GeminiLiveClient`.
    Returns the client_uuid string.
    """
    clients = data.setdefault("clients", {})
    connections_by_ws = data.setdefault("connections_by_ws", {})
    conn_id = id(connection)
    if conn_id in connections_by_ws:
        return connections_by_ws[conn_id]

    # Create a server-generated id based on websocket id
    client_uuid = f"ws_{conn_id}"
    # Placeholder entry: do NOT reference the shared legacy client object
    # (store None) so later replacement with a per-connection client does
    # not accidentally mutate other placeholders that pointed to the
    # same legacy reference.
    clients[client_uuid] = {
        "client": None,
        "created_at": asyncio.get_event_loop().time(),
        "subscriptions": {},
        "client_states": {},
        "owner_ws": conn_id,
        "meta": {"conversation_enabled": True, "is_placeholder": True},
    }
    # Track routing information for this server-side client uuid
    routes = data.setdefault("routes", {})
    try:
        routes[client_uuid] = ConnectionRoute(
            user_ws=conn_id,
            hassio_google_ws=None,
            client_uuid=client_uuid,
            created_at=asyncio.get_event_loop().time(),
            subscriptions={},
            client_states={},
            meta={"is_placeholder": True},
        )
    except Exception:
        pass
    connections_by_ws[conn_id] = client_uuid
    _LOGGER.debug("Created server-side client_uuid %s for ws=%s", client_uuid, conn_id)
    return client_uuid


def _route_allows_send(data: dict, connection_id: int, mapped_uuid: str | None, client_obj: GeminiLiveClient | None) -> bool:
    """Decide whether a message should be sent to the websocket `connection_id`.

    Checks the `routes` mapping and per-client ownership to ensure events
    are delivered only to intended websockets (user_ws or hassio_google_ws).
    """
    try:
        routes = data.setdefault("routes", {})
        if mapped_uuid and mapped_uuid in routes:
            route = routes[mapped_uuid]
            if route.user_ws == connection_id or route.hassio_google_ws == connection_id:
                return True
            return False

        # Fallback: check clients owner_ws if present
        clients = data.setdefault("clients", {})
        if mapped_uuid and mapped_uuid in clients:
            owner = clients.get(mapped_uuid, {}).get("owner_ws")
            if owner is None:
                return True
            return owner == connection_id

        # If no mapping info, be permissive but rely on existing checks elsewhere
        return True
    except Exception:
        return True


def _get_active_connection(data: dict, connection_id: int):
    """Return ActiveConnection object for given connection id if known."""
    try:
        return data.setdefault("active_connections", {}).get(connection_id)
    except Exception:
        return None


def _send_via_route(data: dict, mapped_uuid: str | None, primary_conn: websocket_api.ActiveConnection | None, event_msg: dict) -> None:
    """Send `event_msg` to all ActiveConnection targets for `mapped_uuid`.

    Targets include `user_ws` and `hassio_google_ws` recorded in routes.
    Falls back to sending to `primary_conn` when no route targets exist.
    """
    try:
        routes = data.setdefault("routes", {})
        sent_targets = set()
        if mapped_uuid and mapped_uuid in routes:
            route = routes.get(mapped_uuid)
            active = data.setdefault("active_connections", {})
            # Collect user_ws and hassio_google_ws targets
            for cid in (route.user_ws, route.hassio_google_ws):
                try:
                    if cid:
                        conn = active.get(cid)
                        if conn:
                            sent_targets.add(conn)
                except Exception:
                    pass

        # Always include primary_conn if present (ensures local send)
        if primary_conn:
            sent_targets.add(primary_conn)

        # Send to all unique targets
        for conn in list(sent_targets):
            try:
                conn.send_message(event_msg)
            except Exception:
                _LOGGER.debug("Failed to send routed event to connection %s", getattr(conn, "id", None))
        return
    except Exception:
        # Best-effort fallback
        try:
            if primary_conn:
                primary_conn.send_message(event_msg)
        except Exception:
            pass


def _deliver_event(data: dict, mapped_uuid: str | None, primary_conn: websocket_api.ActiveConnection, msg_id: int, ev_type: str, ev_data: Any) -> None:
    try:
        event_msg = websocket_api.event_message(msg_id, {"type": ev_type, "data": ev_data})
        _send_via_route(data, mapped_uuid, primary_conn, event_msg)
    except Exception:
        try:
            primary_conn.send_message(websocket_api.event_message(msg_id, {"type": ev_type, "data": ev_data}))
        except Exception:
            pass


def _migrate_subscriptions_to_client(
    data: dict,
    client_uuid: str,
    connection_id: int,
    new_client: GeminiLiveClient,
    legacy_client: GeminiLiveClient | None = None,
) -> None:
    """Attach any stored subscription handlers for `connection_id` to `new_client`.

    This will look in both the global `subscriptions` map and any per-client
    `clients[client_uuid]["subscriptions"]` entries. After attaching, stored
    references are removed so handlers are not duplicated.
    """
    try:
        subscriptions = data.setdefault("subscriptions", {})
        migrated_global = 0
        migrated_perclient = 0

        _LOGGER.info(
            "Migration starting for client_uuid=%s connection_id=%s: global_subs_keys=%s new_client=%s",
            client_uuid,
            connection_id,
            list(subscriptions.keys()),
            id(new_client),
        )

        # First migrate global subscriptions for this connection id
        old_handlers = subscriptions.pop(connection_id, None)
        _LOGGER.info(
            "Migration: global old_handlers found=%s handler_types=%s",
            old_handlers is not None,
            list(old_handlers.keys()) if old_handlers else None,
        )
        if old_handlers:
            for ev_type, handler in list(old_handlers.items()):
                try:
                    if legacy_client:
                        try:
                            legacy_client.off(ev_type, handler)
                        except Exception:
                            pass
                    new_client.on(ev_type, handler)
                    migrated_global += 1
                    _LOGGER.info("Migration: attached global handler %s to client %s", ev_type, id(new_client))
                except Exception as e:
                    _LOGGER.warning("Failed attaching migrated handler %s to new client %s: %s", ev_type, client_uuid, e)
            # Record under new client's stored subscriptions for cleanup
            clients = data.setdefault("clients", {})
            clients.setdefault(client_uuid, {}).setdefault("subscriptions", {})[connection_id] = old_handlers
            try:
                routes = data.setdefault("routes", {})
                if client_uuid in routes:
                    routes[client_uuid].subscriptions[connection_id] = old_handlers
            except Exception:
                pass

        # Then migrate any handlers stored under the per-client entry
        try:
            clients = data.setdefault("clients", {})
            client_entry = clients.get(client_uuid, {})
            per_client_subs = client_entry.get("subscriptions", {})
            per_handlers = per_client_subs.pop(connection_id, None)
            if per_handlers:
                for ev_type, handler in list(per_handlers.items()):
                    try:
                        if legacy_client:
                            try:
                                legacy_client.off(ev_type, handler)
                            except Exception:
                                pass
                        new_client.on(ev_type, handler)
                        migrated_perclient += 1
                    except Exception:
                        _LOGGER.debug("Failed attaching per-client migrated handler %s to new client %s", ev_type, client_uuid)
                clients.setdefault(client_uuid, {}).setdefault("subscriptions", {})[connection_id] = per_handlers
                try:
                    routes = data.setdefault("routes", {})
                    if client_uuid in routes:
                        routes[client_uuid].subscriptions[connection_id] = per_handlers
                except Exception:
                    pass
        except Exception:
            pass

        # Log summary of migration
        try:
            route_debug = None
            try:
                route_debug = data.setdefault("routes", {}).get(client_uuid).to_debug() if client_uuid in data.setdefault("routes", {}) else None
            except Exception:
                route_debug = None
            _LOGGER.debug(
                "Migrated subscriptions to client %s (global=%d perclient=%d) for connection %s route=%s",
                client_uuid,
                migrated_global,
                migrated_perclient,
                connection_id,
                route_debug,
            )
        except Exception:
            pass

    except Exception:
        _LOGGER.debug("Error while migrating subscriptions to new client %s", client_uuid)



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
    websocket_api.async_register_command(hass, websocket_clear_conversation)
    websocket_api.async_register_command(hass, websocket_set_conversation_enabled)
    websocket_api.async_register_command(hass, websocket_send_text)
    websocket_api.async_register_command(hass, websocket_send_audio)
    websocket_api.async_register_command(hass, websocket_send_image)
    websocket_api.async_register_command(hass, websocket_start_listening)
    websocket_api.async_register_command(hass, websocket_stop_listening)
    websocket_api.async_register_command(hass, websocket_get_status)
    websocket_api.async_register_command(hass, websocket_debug_state)
    websocket_api.async_register_command(hass, websocket_set_resumption)
    websocket_api.async_register_command(hass, websocket_subscribe)

    # Start a background debug reporter to log routes/clients periodically
    try:
        async def _periodic_debug_reporter() -> None:
            # Run as a background task to aid debugging of routing/race conditions
            while True:
                try:
                    if not _LOGGER.isEnabledFor(logging.DEBUG):
                        await asyncio.sleep(10.0)
                        continue

                    data_root = hass.data.get(DOMAIN, {})
                    for eid, data in list(data_root.items()):
                        try:
                            if not isinstance(data, dict):
                                continue
                            clients = data.get("clients", {})
                            routes = data.get("routes", {})
                            active = data.get("active_connections", {})
                            
                            # Build client summary with actual handler counts from client object
                            clients_summary = {}
                            for k, v in clients.items():
                                client_obj = v.get("client")
                                handler_count = 0
                                if client_obj and hasattr(client_obj, "_event_handlers"):
                                    try:
                                        handler_count = sum(len(h) for h in client_obj._event_handlers.values())
                                    except Exception:
                                        pass
                                clients_summary[k] = {
                                    "id": id(client_obj) if client_obj else None,
                                    "owner_ws": v.get("owner_ws"),
                                    "subs_stored": len(v.get("subscriptions", {})),
                                    "handlers_attached": handler_count,
                                }
                            
                            summary = {
                                "entry_id": eid,
                                "clients": clients_summary,
                                "routes": {k: v.to_debug() if hasattr(v, "to_debug") else str(v) for k, v in routes.items()},
                                "active_connections": list(active.keys()),
                            }
                            _LOGGER.debug("Periodic routing debug: %s", summary)
                        except Exception:
                            _LOGGER.debug("Error building periodic debug summary for entry %s", eid)
                    await asyncio.sleep(10.0)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(10.0)

        # Schedule the reporter as a background task
        try:
            hass.loop.create_task(_periodic_debug_reporter())
        except Exception:
            # Fallback if loop attribute not present
            asyncio.create_task(_periodic_debug_reporter())
    except Exception:
        _LOGGER.debug("Failed to start periodic debug reporter")


@websocket_api.websocket_command(
    {
        "type": "gemini_live/connect",
        vol.Optional("entry_id"): str,
        vol.Optional("client_uuid"): str,
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

    # Use server-side ws-based client uuid
    client_uuid = _ensure_ws_client_uuid(data, connection)
    clients = data.setdefault("clients", {})

    # Check if client already exists and is not a placeholder
    client_entry = clients.get(client_uuid)
    client = client_entry.get("client") if client_entry else None
    is_placeholder = client_entry.get("meta", {}).get("is_placeholder", True) if client_entry else True

    # If we need a new client (doesn't exist or is placeholder), create it
    if not client or is_placeholder:
        try:
            client = await _create_and_connect_client(
                hass, data, client_uuid, connection, entry_id, msg.get("resumption_handle")
            )
        except Exception as e:
            connection.send_error(msg["id"], "connection_failed", str(e))
            return
    else:
        # Reuse existing client
        if not client.connected:
            # Set resumption handle if provided
            resumption_handle = msg.get("resumption_handle")
            if resumption_handle:
                client.set_resumption_handle(resumption_handle)
            await client.connect()

    connection.send_result(msg["id"], {"connected": client.connected, "client_uuid": client_uuid})


@websocket_api.websocket_command(
    {
        "type": "gemini_live/disconnect",
        vol.Optional("entry_id"): str,
        vol.Optional("client_uuid"): str,
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
        clients = data.setdefault("clients", {})
        connections_by_ws = data.setdefault("connections_by_ws", {})
        client_uuid = _ensure_ws_client_uuid(data, connection)

        req_uuid = msg.get("client_uuid")
        if req_uuid and req_uuid in clients:
            client_uuid = req_uuid
        elif not req_uuid:
            mapped_uuid = connections_by_ws.get(id(connection))
            if mapped_uuid and mapped_uuid in clients:
                client_uuid = mapped_uuid

        current_mapped = connections_by_ws.get(id(connection))
        if client_uuid and client_uuid in clients and current_mapped and current_mapped != client_uuid:
            _LOGGER.debug(
                "Disconnect request for client %s from ws=%s refers to a different owner",
                client_uuid,
                id(connection),
            )
            connection.send_result(msg["id"], {"connected": False})
            return

        _LOGGER.debug("Disconnect requested for client %s (ws=%s)", client_uuid, id(connection))
        await _disconnect_and_cleanup(data, client_uuid, hass, entry_id)
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
        vol.Optional("client_uuid"): str,
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
    data, entry_id = _get_data(hass, msg.get("entry_id"))
    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    # Resolve per-connection client first, fall back to legacy
    clients = data.setdefault("clients", {})
    connections_by_ws = data.setdefault("connections_by_ws", {})
    client_uuid = connections_by_ws.get(id(connection)) or _ensure_ws_client_uuid(data, connection)
    client: GeminiLiveClient | None = None
    if client_uuid and client_uuid in clients:
        client = clients[client_uuid].get("client")
    else:
        client = data.get("client")

    if not client or not client.connected:
        connection.send_error(msg["id"], "not_connected", "Gemini Live Audio not connected")
        return

    try:
        _LOGGER.debug("send_text from ws=%s client_uuid=%s text_len=%s", id(connection), client_uuid, len(text))
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
        vol.Optional("client_uuid"): str,
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
    data, entry_id = _get_data(hass, msg.get("entry_id"))
    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    clients = data.setdefault("clients", {})
    connections_by_ws = data.setdefault("connections_by_ws", {})
    client_uuid = connections_by_ws.get(id(connection)) or _ensure_ws_client_uuid(data, connection)
    client: GeminiLiveClient | None = None
    if client_uuid and client_uuid in clients:
        client = clients[client_uuid].get("client")
    else:
        client = data.get("client")

    if not client or not client.connected:
        connection.send_error(msg["id"], "not_connected", "Gemini Live Audio not connected")
        return

    try:
        _LOGGER.debug("send_audio from ws=%s client_uuid=%s size=%d", id(connection), client_uuid, len(audio_b64))
        # Record send timestamp for this websocket connection to help measure
        # delivery latency / detect race conditions between sends and incoming events.
        now = asyncio.get_event_loop().time()
        try:
            last_send_times = data.setdefault("last_send_time", {})
            last_send_times[id(connection)] = now
        except Exception:
            _LOGGER.debug("Failed to record last_send_time for connection %s", id(connection))

        await client.send_audio_base64(audio_b64)
        connection.send_result(msg["id"], {"success": True})
    except Exception as e:
        connection.send_error(msg["id"], "send_failed", str(e))


@websocket_api.websocket_command(
    {
        "type": "gemini_live/send_image",
        vol.Optional("entry_id"): str,
        vol.Optional("client_uuid"): str,
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
    data, entry_id = _get_data(hass, msg.get("entry_id"))
    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    clients = data.setdefault("clients", {})
    connections_by_ws = data.setdefault("connections_by_ws", {})
    client_uuid = connections_by_ws.get(id(connection)) or _ensure_ws_client_uuid(data, connection)
    client: GeminiLiveClient | None = None
    if client_uuid and client_uuid in clients:
        client = clients[client_uuid].get("client")
    else:
        client = data.get("client")

    if not client or not client.connected:
        connection.send_error(msg["id"], "not_connected", "Gemini Live Audio not connected")
        return

    try:
        # If frontend provided an empty mime_type (e.g., iOS HEIC often lacks file.type),
        # fall back to a reasonable default and log for debugging.
        if not mime_type:
            _LOGGER.debug("send_image: empty mime_type from frontend, defaulting to image/jpeg for ws=%s client_uuid=%s", id(connection), client_uuid)
            mime_type = "image/jpeg"

        _LOGGER.debug("send_image from ws=%s client_uuid=%s mime=%s size=%d", id(connection), client_uuid, mime_type, len(image_b64))
        await client.send_image_base64(image_b64, mime_type)
        connection.send_result(msg["id"], {"success": True})
    except Exception as e:
        _LOGGER.error("Failed to send media from ws=%s client_uuid=%s mime=%s error=%s", id(connection), client_uuid, mime_type, e)
        connection.send_error(msg["id"], "send_failed", str(e))


@websocket_api.websocket_command(
    {
        "type": "gemini_live/start_listening",
        vol.Optional("entry_id"): str,
        vol.Optional("client_uuid"): str,
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

    # Resolve per-connection client if available
    connections_by_ws = data.setdefault("connections_by_ws", {})
    client_uuid = connections_by_ws.get(id(connection)) or _ensure_ws_client_uuid(data, connection)
    clients = data.setdefault("clients", {})

    # Check if client_uuid is existing on class, if exist, destroy
    if client_uuid and client_uuid in clients:
        await _disconnect_and_cleanup(data, client_uuid, hass, entry_id)
    
    # Make a new one to connect, connect it
    try:
        client = await _create_and_connect_client(hass, data, client_uuid, connection, entry_id)
    except Exception as e:
        connection.send_error(msg["id"], "connection_failed", str(e))
        return

    # Update listening state for this websocket connection
    try:
        connection_id = id(connection)
        client_states = data.setdefault("client_states", {})
        client_states.setdefault(connection_id, {"listening": False, "speaking": False})
        client_states[connection_id]["listening"] = True

        if client_uuid and client_uuid in clients:
            clients.setdefault(client_uuid, {}).setdefault("client_states", {})
            clients[client_uuid]["client_states"][connection_id] = client_states[connection_id]

        data["is_listening"] = any(s.get("listening") for s in client_states.values())
        hass.bus.async_fire(
            f"{DOMAIN}_listening_state_changed",
            {"entry_id": entry_id, "is_listening": data.get("is_listening", False)},
        )
    except Exception:
        _LOGGER.debug("Error setting listening state for connection")

    connection.send_result(msg["id"], {"listening": True, "connected": True})


@websocket_api.websocket_command(
    {
        "type": "gemini_live/stop_listening",
        vol.Optional("entry_id"): str,
        vol.Optional("client_uuid"): str,
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

    # Resolve per-connection client for stop_listening as well
    connections_by_ws = data.setdefault("connections_by_ws", {})
    client_uuid = connections_by_ws.get(id(connection)) or _ensure_ws_client_uuid(data, connection)
    
    # Destroy client session on stop listening per user request
    await _disconnect_and_cleanup(data, client_uuid, hass, entry_id)

    # Update listening state for this websocket connection
    try:
        connection_id = id(connection)
        client_states = data.setdefault("client_states", {})
        if connection_id in client_states:
            client_states[connection_id]["listening"] = False

        data["is_listening"] = any(s.get("listening") for s in client_states.values())
        hass.bus.async_fire(
            f"{DOMAIN}_listening_state_changed",
            {"entry_id": entry_id, "is_listening": data.get("is_listening", False)},
        )
    except Exception:
        _LOGGER.debug("Error clearing listening state for connection")

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

    # Prefer per-connection / per-client status when possible so each browser
    # sees its own listening/speaking state rather than the aggregated one.
    clients = data.setdefault("clients", {})
    connections_by_ws = data.setdefault("connections_by_ws", {})
    connection_id = id(connection)
    client_uuid = connections_by_ws.get(connection_id) or _ensure_ws_client_uuid(data, connection)

    connected = False
    is_listening = False
    is_speaking = False

    if client_uuid and client_uuid in clients:
        client_entry = clients[client_uuid]
        client_obj = client_entry.get("client")
        connected = bool(getattr(client_obj, "connected", False))
        # per-client client_states keyed by connection id
        cs = client_entry.get("client_states", {})
        state = cs.get(connection_id) or {}
        is_listening = bool(state.get("listening", False))
        is_speaking = bool(state.get("speaking", False))
    else:
        # Fallback to aggregated view for legacy clients
        client: GeminiLiveClient = data.get("client")
        connected = (
            any(c.get("client") and getattr(c.get("client"), "connected", False) for c in data.get("clients", {}).values())
            or (client.connected if client else False)
        )
        is_listening = data.get("is_listening", False)
        is_speaking = data.get("is_speaking", False)

    connection.send_result(msg["id"], {"connected": connected, "is_listening": is_listening, "is_speaking": is_speaking})


@websocket_api.websocket_command(
    {
        "type": "gemini_live/set_resumption",
        vol.Optional("entry_id"): str,
        vol.Optional("client_uuid"): str,
        vol.Optional("enable"): bool,
        vol.Optional("clear_handle"): bool,
    }
)
@websocket_api.async_response
async def websocket_set_resumption(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Enable/disable session resumption and optionally clear stored handle."""
    data, entry_id = _get_data(hass, msg.get("entry_id"))

    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    session_config = data.get("session_config")
    if not session_config:
        connection.send_error(msg["id"], "not_found", "Session config not available")
        return

    enable = msg.get("enable")
    clear = bool(msg.get("clear_handle", False))

    try:
        if enable is not None:
            try:
                session_config.enable_session_resumption = bool(enable)
            except Exception:
                pass

        if clear:
            try:
                session_config.session_resumption_handle = None
            except Exception:
                pass

        connection.send_result(msg["id"], {"ok": True, "enable": getattr(session_config, "enable_session_resumption", None), "handle_cleared": clear})
    except Exception as e:
        connection.send_error(msg["id"], "set_failed", str(e))


@websocket_api.websocket_command(
    {
        "type": "gemini_live/run_bisect",
        vol.Optional("entry_id"): str,
        vol.Optional("timeout"): int,
    }
)
@websocket_api.async_response
async def websocket_run_bisect(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Run bisect on configured tools and per-MCP groups and return results.

    Returns a dict with `flat` (results for flat tools) and `groups`
    mapping server_name -> per-tool results.
    """
    data, entry_id = _get_data(hass, msg.get("entry_id"))

    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    # Resolve a client (prefer per-connection/legacy)
    client_obj, _ = _get_client(hass, entry_id)
    if not client_obj:
        connection.send_error(msg["id"], "not_connected", "No Gemini client available")
        return

    timeout = float(msg.get("timeout") or 20.0)

    session_config = data.get("session_config")
    if not session_config:
        connection.send_error(msg["id"], "not_found", "Session config not available")
        return

    results: dict[str, Any] = {"flat": {}, "groups": {}}

    try:
        # First, run bisect on current flat tools (if any)
        try:
            _LOGGER.info("Running bisect on flat tools (timeout=%s)", timeout)
            flat = await client_obj.bisect_tools_and_report(timeout=timeout)
            results["flat"] = flat
        except Exception as e:
            _LOGGER.debug("Flat bisect failed: %s", e, exc_info=True)
            results["flat"] = {"error": str(e)}

        # Now run bisect per MCP group (if present)
        mcp_groups = getattr(session_config, "mcp_tool_groups", {}) or {}
        orig_tools = list(getattr(session_config, "tools", []) or [])

        for server_name, funcs in list(mcp_groups.items()):
            try:
                if not funcs:
                    results["groups"][server_name] = {"error": "no_functions"}
                    continue

                # Build simple per-function tool list so bisect_tools tests each function
                simple_tools: list[dict[str, Any]] = []
                for f in funcs:
                    try:
                        if isinstance(f, dict):
                            name = f.get("name") or f.get("id") or "<unnamed>"
                            desc = f.get("description") or (f.get("annotations") or {}).get("title") or ""
                            params = f.get("parameters") or f.get("inputSchema") or f.get("input_schema") or {}
                            simple_tools.append({"name": name, "description": desc, "parameters": params})
                        else:
                            simple_tools.append({"name": str(f)})
                    except Exception:
                        simple_tools.append({"name": "<error>"})

                # Temporarily replace session tools with this group's simple tools
                session_config.tools = simple_tools
                _LOGGER.info("Running bisect for MCP group %s (%d funcs)", server_name, len(simple_tools))
                try:
                    grp_res = await client_obj.bisect_tools(timeout=timeout)
                    results["groups"][server_name] = grp_res
                except Exception as e:
                    _LOGGER.debug("Bisect for group %s failed: %s", server_name, e, exc_info=True)
                    results["groups"][server_name] = {"error": str(e)}
                finally:
                    # Restore original tools after each group run
                    session_config.tools = orig_tools
            except Exception as e:
                _LOGGER.debug("Error preparing bisect for group %s: %s", server_name, e, exc_info=True)
                results["groups"][server_name] = {"error": str(e)}

        connection.send_result(msg["id"], {"ok": True, "results": results})
    except Exception as e:
        _LOGGER.error("websocket_run_bisect failed: %s", e, exc_info=True)
        connection.send_error(msg["id"], "bisect_failed", str(e))


@websocket_api.websocket_command(
    {
        "type": "gemini_live/subscribe",
        vol.Optional("entry_id"): str,
        vol.Optional("client_uuid"): str,
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
                # Prefer per-connection client if mapping exists
                clients = data.setdefault("clients", {})
                connections_by_ws = data.setdefault("connections_by_ws", {})
                # Determine server-side client uuid for this websocket
                client_uuid = connections_by_ws.get(id(connection)) or _ensure_ws_client_uuid(data, connection)
                # If a client_uuid is provided by the frontend but the server
                # hasn't created a per-connection client yet (connect may
                # come later), create a placeholder mapping so subscriptions
                # attach to the intended client_uuid instead of the legacy
                # shared client. This prevents subscribe->connect races where
                # the handlers end up attached to the wrong client object.
                if client_uuid and client_uuid in clients:
                    # If the existing entry is a placeholder referencing the
                    # legacy shared client, replace it with a real
                    # per-connection GeminiLiveClient when credentials are
                    # available so this websocket gets its own connected
                    # client instead of remaining a placeholder.
                    client_entry = clients[client_uuid]
                    client = client_entry.get("client")
                    legacy_client_obj = data.get("client")
                    _LOGGER.info(
                        "Subscribe: found existing client_uuid=%s client=%s legacy=%s is_same=%s is_placeholder=%s",
                        client_uuid,
                        id(client) if client else None,
                        id(legacy_client_obj) if legacy_client_obj else None,
                        client is legacy_client_obj,
                        client_entry.get("meta", {}).get("is_placeholder", False),
                    )
                    if client is legacy_client_obj:
                        api_key = data.get("api_key")
                        session_config = data.get("session_config")
                        if api_key and session_config:
                            new_client = GeminiLiveClient(api_key=api_key, session_config=session_config)
                            client_entry.update({
                                "client": new_client,
                                "created_at": asyncio.get_event_loop().time(),
                                "owner_ws": id(connection),
                                "meta": client_entry.get("meta", {"conversation_enabled": True}),
                            })
                            # Tag replaced client with owner websocket id
                            try:
                                new_client.set_owner_ws(id(connection))
                            except Exception:
                                pass
                            client = new_client
                            connections_by_ws[id(connection)] = client_uuid
                            # attempt to connect the new client
                            try:
                                hass.async_create_task(client.connect())
                            except Exception:
                                _LOGGER.debug("Failed to schedule connect() for replaced per-connection client %s", client_uuid)
                            # migrate any existing subscription handlers for this conn
                            try:
                                subscriptions = data.setdefault("subscriptions", {})
                                conn_id = id(connection)
                                if conn_id in subscriptions:
                                    old_handlers = subscriptions.get(conn_id, {})
                                    for ev_type, handler in list(old_handlers.items()):
                                        try:
                                            if legacy_client_obj and legacy_client_obj is not client:
                                                legacy_client_obj.off(ev_type, handler)
                                        except Exception:
                                            pass
                                        try:
                                            client.on(ev_type, handler)
                                        except Exception:
                                            pass
                                    client_entry.setdefault("subscriptions", {})
                                    client_entry["subscriptions"][conn_id] = old_handlers
                                    _LOGGER.debug("Reattached %d existing subscription handlers to replaced client %s for connection %s", len(old_handlers), client_uuid, conn_id)
                            except Exception:
                                _LOGGER.debug("Error migrating subscriptions to replaced client %s", client_uuid)
                elif client_uuid and client_uuid not in clients:
                    # Frontend supplied a client_uuid but the server doesn't have
                    # a client entry yet. Create a real per-connection client
                    # now so events are routed only to this websocket. If we
                    # don't have credentials/config we fall back to a
                    # lightweight placeholder referencing the legacy client.
                    api_key = data.get("api_key")
                    session_config = data.get("session_config")
                    if api_key and session_config:
                        new_client = GeminiLiveClient(api_key=api_key, session_config=session_config)
                        # Tag the new per-connection client with the owning ws id
                        try:
                            new_client.set_owner_ws(id(connection))
                        except Exception:
                            pass
                        clients[client_uuid] = {
                            "client": new_client,
                            "created_at": asyncio.get_event_loop().time(),
                            "subscriptions": {},
                            "client_states": {},
                            "owner_ws": id(connection),
                            "meta": {"conversation_enabled": True},
                        }
                        connections_by_ws[id(connection)] = client_uuid
                        client = new_client
                        # Ensure the new per-connection client actually connects
                        try:
                            hass.async_create_task(client.connect())
                        except Exception:
                            _LOGGER.debug("Failed to schedule connect() for per-connection client %s", client_uuid)

                        # If there are any existing subscription handlers previously
                        # registered for this websocket (likely attached to the
                        # legacy shared client), move them to the newly created
                        # per-connection client to ensure handlers receive events
                        # from the correct client object.
                        try:
                            subscriptions = data.setdefault("subscriptions", {})
                            conn_id = id(connection)
                            if conn_id in subscriptions:
                                old_handlers = subscriptions.get(conn_id, {})
                                legacy_client = data.get("client")
                                for ev_type, handler in list(old_handlers.items()):
                                    try:
                                        if legacy_client and legacy_client is not new_client:
                                            legacy_client.off(ev_type, handler)
                                    except Exception:
                                        pass
                                    try:
                                        new_client.on(ev_type, handler)
                                    except Exception:
                                        pass
                                clients.setdefault(client_uuid, {}).setdefault("subscriptions", {})
                                clients[client_uuid]["subscriptions"][conn_id] = old_handlers
                                _LOGGER.debug("Reattached %d existing subscription handlers to new client %s for connection %s", len(old_handlers), client_uuid, conn_id)
                        except Exception:
                            _LOGGER.debug("Error reattaching existing subscriptions to new per-connection client %s", client_uuid)

                        _LOGGER.info("Created per-connection client %s via subscribe for entry %s (ws=%s)", client_uuid, entry_id, id(connection))
                        try:
                            hass.bus.async_fire(f"{DOMAIN}_client_added", {"entry_id": entry_id, "client_uuid": client_uuid})
                        except Exception:
                            _LOGGER.debug("Failed to fire client_added event for new per-connection client")
                    else:
                        # Could not create a per-connection client because credentials/config
                        # are not available. Map this websocket connection to a synthetic
                        # legacy uuid so `mapped_uuid` is set (avoids `None` in logs)
                        connections_by_ws = data.setdefault("connections_by_ws", {})
                        clients = data.setdefault("clients", {})
                        legacy_uuid = f"legacy_{entry_id or 'unknown'}"
                        if legacy_uuid not in clients:
                            clients[legacy_uuid] = {
                                "client": data.get("client"),
                                "created_at": asyncio.get_event_loop().time(),
                                "subscriptions": {},
                                "client_states": {},
                            }
                        connections_by_ws[id(connection)] = legacy_uuid
                
                else:
                    # If there's no per-connection mapping, avoid returning the legacy shared client
                    # Create a new per-websocket client so each browser gets an isolated session
                    client = data.get("client")
                    if client and (not client_uuid):
                        # No mapping for this websocket; create an isolated client instance
                        api_key = data.get("api_key")
                        session_config = data.get("session_config")
                        if api_key and session_config:
                            import uuid as _uuid
                            new_uuid = _uuid.uuid4().hex
                            new_client = GeminiLiveClient(api_key=api_key, session_config=session_config)
                            # Tag isolated per-websocket client with owner ws id
                            try:
                                new_client.set_owner_ws(id(connection))
                            except Exception:
                                pass
                            clients[new_uuid] = {
                                "client": new_client,
                                "created_at": asyncio.get_event_loop().time(),
                                "subscriptions": {},
                                "client_states": {},
                                "owner_ws": id(connection),
                            }
                            connections_by_ws[id(connection)] = new_uuid
                            client = new_client
                            # Ensure the new per-connection client actually connects so
                            # it starts its receive loop and can emit audio events
                            try:
                                hass.async_create_task(client.connect())
                            except Exception:
                                _LOGGER.debug("Failed to schedule connect() for per-connection client %s", new_uuid)
                            # If there are any existing subscription handlers previously
                            # registered for this websocket (likely attached to the
                            # legacy shared client), move them to the newly created
                            # per-connection client to ensure handlers receive events
                            # from the correct client object.
                            try:
                                subscriptions = data.setdefault("subscriptions", {})
                                conn_id = id(connection)
                                if conn_id in subscriptions:
                                    old_handlers = subscriptions.get(conn_id, {})
                                    legacy_client = data.get("client")
                                    # Detach from legacy client and attach to new client
                                    for ev_type, handler in list(old_handlers.items()):
                                        try:
                                            if legacy_client and legacy_client is not new_client:
                                                legacy_client.off(ev_type, handler)
                                        except Exception:
                                            pass
                                        try:
                                            new_client.on(ev_type, handler)
                                        except Exception:
                                            pass
                                    # Also record these subscriptions under the new client's info
                                    clients.setdefault(new_uuid, {}).setdefault("subscriptions", {})
                                    clients[new_uuid]["subscriptions"][conn_id] = old_handlers
                                    _LOGGER.debug("Reattached %d existing subscription handlers to new client %s for connection %s", len(old_handlers), new_uuid, conn_id)
                                    try:
                                        _migrate_subscriptions_to_client(data, new_uuid, conn_id, new_client, data.get("client"))
                                    except Exception:
                                        pass
                                    try:
                                        _migrate_subscriptions_to_client(data, new_uuid, conn_id, new_client, data.get("client"))
                                    except Exception:
                                        pass
                            except Exception:
                                _LOGGER.debug("Error reattaching existing subscriptions to new per-connection client %s", new_uuid)

                            _LOGGER.info("Created per-connection client %s via subscribe for entry %s (ws=%s)", new_uuid, entry_id, id(connection))
                            try:
                                hass.bus.async_fire(f"{DOMAIN}_client_added", {"entry_id": entry_id, "client_uuid": new_uuid})
                            except Exception:
                                _LOGGER.debug("Failed to fire client_added event for new per-connection client")
                        else:
                            # Could not create a per-connection client because credentials/config
                            # are not available. Map this websocket connection to a synthetic
                            # legacy uuid so `mapped_uuid` is set (avoids `None` in logs)
                            connections_by_ws = data.setdefault("connections_by_ws", {})
                            clients = data.setdefault("clients", {})
                            legacy_uuid = f"legacy_{entry_id or 'unknown'}"
                            if legacy_uuid not in clients:
                                clients[legacy_uuid] = {
                                    "client": client,
                                    "created_at": asyncio.get_event_loop().time(),
                                    "subscriptions": {},
                                    "client_states": {},
                                }
                            connections_by_ws[id(connection)] = legacy_uuid
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
    # Track per-connection client state (listening/speaking)
    client_states = data.setdefault("client_states", {})
    connection_id = id(connection)
    # Register this active connection so routed sends can reach other endpoints
    try:
        data.setdefault("active_connections", {})[connection_id] = connection
    except Exception:
        pass
    # Determine mapped client_uuid for this websocket (if any)
    connections_by_ws = data.setdefault("connections_by_ws", {})
    # Ensure a server-side client uuid exists for this websocket so mapped_uuid
    # is never None (prevents subscribe->connect race showing null in logs)
    mapped_uuid = connections_by_ws.get(connection_id) or _ensure_ws_client_uuid(data, connection)
    _LOGGER.debug("Subscribe requested for connection %s entry=%s mapped_uuid=%s", connection_id, entry_id, mapped_uuid)

    # Clear any existing subscriptions for this client to prevent duplicate events
    # This happens when user reconnects without proper disconnect. Use the
    # actual client object currently mapped for this websocket (fallback to
    # legacy shared client) so we detach handlers from the correct object.
    if connection_id in subscriptions:
        old_handlers = subscriptions.pop(connection_id, {})
        # Determine which client object these handlers were originally
        # intended for (prefer mapped_uuid entry, fall back to legacy).
        # If the mapping is a placeholder referencing the legacy client,
        # stored handlers may live under clients[mapped_uuid]["subscriptions"].
        try:
            clients = data.setdefault("clients", {})
            target_client = None
            client_entry = clients.get(mapped_uuid, {}) if mapped_uuid else {}
            entry_client_obj = client_entry.get("client") if client_entry else None
            is_placeholder = bool(client_entry.get("meta", {}).get("is_placeholder", False))
            # If placeholder referencing legacy, remove from per-client storage
            if is_placeholder and entry_client_obj is data.get("client"):
                try:
                    client_entry.get("subscriptions", {}).pop(connection_id, None)
                except Exception:
                    pass
                target_client = None
            else:
                target_client = entry_client_obj or data.get("client")
        except Exception:
            target_client = data.get("client")

        for event_type, handler in old_handlers.items():
            try:
                if target_client:
                    target_client.off(event_type, handler)
            except Exception:
                pass
        _LOGGER.debug("Cleared stale subscription for connection %s (mapped_uuid=%s)", connection_id, mapped_uuid)

    # Define event handlers
    async def on_audio_delta(event_data: dict[str, Any]) -> None:
        # Instrument timing: measure delay from last audio send to this handler invocation
        try:
            now = asyncio.get_event_loop().time()
            last_send_times = data.setdefault("last_send_time", {})
            last_send = last_send_times.get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            now = None
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_AUDIO_DELTA for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            # Allow a placeholder that's explicitly owned by this websocket
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring audio_delta for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            # Ensure this websocket is the owner of the mapped client when owner is set
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring audio_delta for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here — rely on
            # placeholder/owner checks and routed delivery. Comparing against
            # the `client` closure can be stale due to migration timing and
            # causes valid events to be ignored.
        except Exception:
            pass

        # Mark this connection as receiving speaking audio
        try:
            client_states.setdefault(connection_id, {"listening": False, "speaking": False})
            client_states[connection_id]["speaking"] = True
            # Aggregate speaking state across all connections
            data["is_speaking"] = any(s.get("speaking") for s in client_states.values())
            # Fire speaking state change if needed
            hass.bus.async_fire(
                f"{DOMAIN}_speaking_state_changed",
                {"entry_id": entry_id, "is_speaking": data.get("is_speaking", False)},
            )
        except Exception:
            _LOGGER.debug("Error updating client speaking state for connection %s", connection_id)

        _deliver_event(data, mapped_uuid, connection, msg["id"], "audio_delta", event_data)
        _LOGGER.debug("Emitted audio_delta to connection %s client_uuid=%s", connection_id, mapped_uuid)

    # Tag handlers with websocket mapping for debug/tracing
    try:
        for _h in (on_audio_delta, on_transcript, on_output_transcript, on_turn_complete, on_error, on_interrupted, on_function_call_event, on_session_resumed, on_session_resumption_update, on_go_away, on_generation_complete):
            try:
                setattr(_h, "_ws_mapping", (connection_id, mapped_uuid))
            except Exception:
                pass
    except Exception:
        pass

    async def on_transcript(event_data: dict[str, Any]) -> None:
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring output_transcript for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring output_transcript for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Avoid strict client-object comparison here; rely on ownership checks.
        except Exception:
            pass
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring turn_complete for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring turn_complete for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Avoid strict client-object comparison here; rely on ownership checks.
        except Exception:
            pass
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring error for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring error for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Avoid strict client-object comparison here; rely on ownership checks.
        except Exception:
            pass
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring interrupted for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring interrupted for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Avoid strict client-object comparison here; rely on ownership checks.
        except Exception:
            pass
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring function_call event for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring function_call event for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Avoid strict client-object comparison here; rely on ownership checks.
        except Exception:
            pass
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring session_resumed for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring session_resumed for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Avoid strict client-object comparison here; rely on ownership checks.
        except Exception:
            pass
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring session_resumption_update for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring session_resumption_update for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Avoid strict client-object comparison here; rely on ownership checks.
        except Exception:
            pass
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring go_away for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring go_away for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Avoid strict client-object comparison here; rely on ownership checks.
        except Exception:
            pass
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring generation_complete for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring generation_complete for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Avoid strict client-object comparison here; rely on ownership checks.
        except Exception:
            pass
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_INPUT_TRANSCRIPTION for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring transcript for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring transcript for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "transcript", event_data)
        _LOGGER.debug("Emitted transcript to connection %s client_uuid=%s", connection_id, mapped_uuid)

    async def on_text_delta(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_TEXT_DELTA for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring text_delta for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring text_delta for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "text_delta", event_data)
        _LOGGER.debug("Emitted text_delta to connection %s client_uuid=%s", connection_id, mapped_uuid)

    async def on_output_transcript(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_OUTPUT_TRANSCRIPTION for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring output_transcript for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring output_transcript for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "output_transcript", event_data)
        _LOGGER.debug("Emitted output_transcript to connection %s client_uuid=%s", connection_id, mapped_uuid)

    async def on_turn_complete(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_TURN_COMPLETE for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring turn_complete for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring turn_complete for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        # Mark this connection as no longer speaking
        try:
            if connection_id in client_states:
                client_states[connection_id]["speaking"] = False
                data["is_speaking"] = any(s.get("speaking") for s in client_states.values())
                hass.bus.async_fire(
                    f"{DOMAIN}_speaking_state_changed",
                    {"entry_id": entry_id, "is_speaking": data.get("is_speaking", False)},
                )
        except Exception:
            _LOGGER.debug("Error updating client speaking state on turn_complete for %s", connection_id)

        _deliver_event(data, mapped_uuid, connection, msg["id"], "turn_complete", event_data)
        _LOGGER.debug("Emitted turn_complete to connection %s client_uuid=%s", connection_id, mapped_uuid)

    async def on_error(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_ERROR for connection %s mapped_uuid=%s client_obj=%s data=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            event_data,
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring error for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring error for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "error", event_data)
        _LOGGER.debug("Emitted error to connection %s client_uuid=%s: %s", connection_id, mapped_uuid, event_data)

        # Perform server-side cleanup for fatal client errors so we don't leave
        # stale per-connection clients, routes or subscriptions behind.
        try:
            clients = data.setdefault("clients", {})
            connections_by_ws = data.setdefault("connections_by_ws", {})
            routes = data.setdefault("routes", {})

            # Prefer the explicit mapped uuid for this websocket, fall back to
            # stored mapping or provided mapped_uuid variable
            mapped = connections_by_ws.get(connection_id) or mapped_uuid
            if mapped and mapped in clients:
                client_entry = clients.get(mapped, {})
                client_obj = client_entry.get("client")

                # Mark route as disconnected
                try:
                    if mapped in routes:
                        routes[mapped].server_connected = False
                        routes[mapped].last_seen = asyncio.get_event_loop().time()
                except Exception:
                    _LOGGER.debug("Failed to update route status for %s", mapped)

                # Attempt to disconnect/cleanup the client object if present
                try:
                    if client_obj:
                        # Schedule a graceful disconnect; await so session tears down
                        try:
                            await client_obj.disconnect()
                        except Exception:
                            # Best-effort: log and continue
                            _LOGGER.debug("Error while disconnecting client object for %s", mapped)
                except Exception:
                    _LOGGER.debug("Error attempting client_obj.disconnect() for %s", mapped)

                # Clear listening/speaking state for this connection and fire events
                try:
                    # Clear for this specific connection
                    client_states.setdefault(connection_id, {})
                    client_states[connection_id]["listening"] = False
                    client_states[connection_id]["speaking"] = False
                    
                    # Update global aggregate state
                    data["is_listening"] = any(s.get("listening") for s in client_states.values())
                    data["is_speaking"] = any(s.get("speaking") for s in client_states.values())
                    
                    if entry_id:
                        hass.bus.async_fire(
                            f"{DOMAIN}_listening_state_changed",
                            {"entry_id": entry_id, "is_listening": data.get("is_listening", False)},
                        )
                        hass.bus.async_fire(
                            f"{DOMAIN}_speaking_state_changed",
                            {"entry_id": entry_id, "is_speaking": data.get("is_speaking", False)},
                        )
                except Exception as e:
                    _LOGGER.debug("Error clearing sensor state in on_error: %s", e)

                # Clear stored client reference and per-connection state/subscriptions
                try:
                    client_entry["client"] = None
                    client_entry.pop("client_states", None)
                    # Remove any stored per-client subscriptions for this connection
                    try:
                        subs = client_entry.get("subscriptions", {})
                        subs.pop(connection_id, None)
                    except Exception:
                        pass
                except Exception:
                    _LOGGER.debug("Failed to clear client entry for %s", mapped)

                # Remove mapping from connections_by_ws
                try:
                    connections_by_ws.pop(connection_id, None)
                except Exception:
                    pass

                _LOGGER.info("Cleaned up server-side client state for ws=%s client_uuid=%s", connection_id, mapped)

                # Fire connected state change for sensors / automation
                try:
                    if entry_id:
                        hass.bus.async_fire(f"{DOMAIN}_connected_state_changed", {"entry_id": entry_id, "is_connected": False})
                except Exception:
                    pass
        except Exception as e:
            _LOGGER.debug("Error during on_error cleanup: %s", e)

    async def on_interrupted(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_INTERRUPTED for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring interrupted for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring interrupted for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "interrupted", event_data)
        _LOGGER.debug("Emitted interrupted to connection %s client_uuid=%s", connection_id, mapped_uuid)

    async def on_function_call_event(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_FUNCTION_CALL for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring function_call event for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring function_call event for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "function_call", event_data)
        _LOGGER.debug("Emitted function_call event to connection %s client_uuid=%s", connection_id, mapped_uuid)

    async def on_session_resumed(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_SESSION_RESUMED for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring session_resumed for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring session_resumed for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "session_resumed", event_data)
        _LOGGER.debug("Emitted session_resumed to connection %s client_uuid=%s", connection_id, mapped_uuid)

    async def on_session_resumption_update(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_SESSION_RESUMPTION_UPDATE for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring session_resumption_update for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring session_resumption_update for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "session_resumption_update", event_data)
        _LOGGER.debug("Emitted session_resumption_update to connection %s client_uuid=%s", connection_id, mapped_uuid)

    async def on_go_away(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_GO_AWAY for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring go_away for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring go_away for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "go_away", event_data)
        _LOGGER.debug("Emitted go_away to connection %s client_uuid=%s", connection_id, mapped_uuid)

    async def on_generation_complete(event_data: dict[str, Any]) -> None:
        try:
            now = asyncio.get_event_loop().time()
            last_send = data.setdefault("last_send_time", {}).get(connection_id)
            delta = (now - last_send) if last_send else None
        except Exception:
            delta = None

        _LOGGER.debug(
            "Incoming EVENT_GENERATION_COMPLETE for connection %s mapped_uuid=%s client_obj=%s data_keys=%s delay_since_last_send=%s",
            connection_id,
            mapped_uuid,
            id(client) if client is not None else None,
            list(event_data.keys()) if isinstance(event_data, dict) else type(event_data),
            f"{delta:.3f}s" if delta is not None else "-",
        )

        # Ensure this event is intended for this websocket's current mapped client
        try:
            _clients = data.setdefault("clients", {})
            _conns = data.setdefault("connections_by_ws", {})
            _mapped = _conns.get(connection_id)
            _client_entry = _clients.get(_mapped, {}) if _mapped else {}
            _current_client = _client_entry.get("client") if _client_entry else None
            is_placeholder = _client_entry.get("meta", {}).get("is_placeholder", False)
            owner = _client_entry.get("owner_ws")
            if is_placeholder and owner != connection_id:
                _LOGGER.debug("Ignoring generation_complete for connection %s mapped_uuid=%s: placeholder client (owner=%s)", connection_id, _mapped, owner)
                return
            if owner is not None and owner != connection_id:
                _LOGGER.debug("Ignoring generation_complete for connection %s mapped_uuid=%s: owner mismatch (owner=%s)", connection_id, _mapped, owner)
                return
            # Do not strictly compare the live client object here; rely on
            # placeholder/owner checks and routed delivery as in `on_audio_delta`.
        except Exception:
            pass

        _deliver_event(data, mapped_uuid, connection, msg["id"], "generation_complete", event_data)
        _LOGGER.debug("Emitted generation_complete to connection %s client_uuid=%s", connection_id, mapped_uuid)

    # Register handlers on the actual client object mapped to this websocket.
    # This prevents subscription handlers from being attached to the wrong
    # client object when placeholders or legacy/shared clients are involved.
    try:
        clients = data.setdefault("clients", {})
        target_client = None
        
        # Get the client entry for this websocket's mapped uuid
        client_entry = clients.get(mapped_uuid, {}) if mapped_uuid else {}
        entry_client_obj = client_entry.get("client") if client_entry else None
        is_placeholder = client_entry.get("meta", {}).get("is_placeholder", False) if client_entry else True
        
        # Determine if we should attach handlers directly
        # Attach if: we have a real client object AND it's not a placeholder
        should_attach = bool(entry_client_obj and not is_placeholder)
        target_client = entry_client_obj if should_attach else None

        # Log subscription attempt details for debugging
        _LOGGER.info(
            "Subscribe handler attachment decision: connection=%s mapped_uuid=%s should_attach=%s entry_client_obj=%s is_placeholder=%s target_client=%s",
            connection_id,
            mapped_uuid,
            should_attach,
            id(entry_client_obj) if entry_client_obj else None,
            is_placeholder,
            id(target_client) if target_client else None,
        )

        if should_attach and target_client:
            target_client.on(EVENT_AUDIO_DELTA, on_audio_delta)
            target_client.on(EVENT_INPUT_TRANSCRIPTION, on_transcript)
            target_client.on(EVENT_OUTPUT_TRANSCRIPTION, on_output_transcript)
            target_client.on(EVENT_TURN_COMPLETE, on_turn_complete)
            target_client.on(EVENT_ERROR, on_error)
            target_client.on(EVENT_INTERRUPTED, on_interrupted)
            target_client.on(EVENT_FUNCTION_CALL, on_function_call_event)
            target_client.on(EVENT_SESSION_RESUMED, on_session_resumed)
            target_client.on(EVENT_SESSION_RESUMPTION_UPDATE, on_session_resumption_update)
            target_client.on(EVENT_GO_AWAY, on_go_away)
            target_client.on(EVENT_GENERATION_COMPLETE, on_generation_complete)
            attach_to_client = target_client
            _LOGGER.info(
                "Attached subscription handlers to client %s for connection %s mapped_uuid=%s",
                id(target_client),
                connection_id,
                mapped_uuid,
            )
        else:
            # Store handlers for later migration when a per-connection client is created
            stored_handlers = {
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
            clients.setdefault(mapped_uuid or f"legacy_{entry_id}", {}).setdefault("subscriptions", {})[connection_id] = stored_handlers
            _LOGGER.info(
                "Stored subscription handlers for later migration: connection=%s mapped_uuid=%s handler_types=%s",
                connection_id,
                mapped_uuid,
                list(stored_handlers.keys()),
            )
            attach_to_client = None
    except Exception as e:
        _LOGGER.warning("Failed to attach subscription handlers to target client for connection %s mapped_uuid=%s: %s", connection_id, mapped_uuid, e)

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
    _LOGGER.info(
        "Stored global subscriptions for cleanup: connection=%s keys_after=%s",
        connection_id,
        list(subscriptions.keys()),
    )

    # Log registration details including which client object the handlers were attached to
    try:
        handler_count = len(subscriptions[connection_id])
        # Prefer to show the client object id/class we actually attached to, if any
        _client_id = None
        _client_class = None
        _client_owner = None
        try:
            if attach_to_client is not None:
                _client_id = id(attach_to_client)
                _client_class = attach_to_client.__class__.__name__
                _client_owner = getattr(attach_to_client, "_owner_ws", None)
        except Exception:
            _client_id = None
            _client_class = None
            _client_owner = None

        # Route debug summary when available
        _route_debug = None
        try:
            routes = data.setdefault("routes", {})
            if mapped_uuid and mapped_uuid in routes:
                _route_debug = routes[mapped_uuid].to_debug()
        except Exception:
            _route_debug = None

        _LOGGER.debug(
            "Registered %d subscription handlers on client obj id=%s class=%s owner=%s for connection %s mapped_uuid=%s route=%s",
            handler_count,
            _client_id,
            _client_class,
            _client_owner,
            connection_id,
            mapped_uuid,
            _route_debug,
        )
    except Exception:
        _LOGGER.debug("Failed to log subscription registration details for connection %s", connection_id)

    # Send a one-time debug event to confirm subscription/handler registration
    try:
        _deliver_event(
            data,
            mapped_uuid,
            connection,
            msg["id"],
            "debug",
            {"message": "subscription_registered", "client_uuid": mapped_uuid},
        )
        _LOGGER.debug("Sent subscription_registered debug event to connection %s client_uuid=%s", connection_id, mapped_uuid)
    except Exception:
        _LOGGER.debug("Failed to send subscription_registered debug event for connection %s", connection_id)

    # Send one-time per-connection state so new clients don't see the aggregated
    # `is_listening`/`is_speaking` value (which may reflect other browsers).
    try:
        per_conn_listening = False
        per_conn_speaking = False
        if mapped_uuid and mapped_uuid in clients:
            per_client_states = clients.setdefault(mapped_uuid, {}).get("client_states", {})
            state = per_client_states.get(connection_id) or {}
            per_conn_listening = bool(state.get("listening", False))
            per_conn_speaking = bool(state.get("speaking", False))

        _deliver_event(
            data,
            mapped_uuid,
            connection,
            msg["id"],
            "connection_state",
            {
                "client_uuid": mapped_uuid,
                "is_listening": per_conn_listening,
                "is_speaking": per_conn_speaking,
            },
        )
        _LOGGER.debug("Sent initial connection_state to %s mapped_uuid=%s listening=%s speaking=%s", connection_id, mapped_uuid, per_conn_listening, per_conn_speaking)
    except Exception:
        _LOGGER.debug("Failed to send initial connection_state for connection %s", connection_id)

    def on_close() -> None:
        """Handle connection close."""
        handlers = subscriptions.pop(connection_id, {})
        for event_type, handler in handlers.items():
            try:
                # Look up the most-recent client object for this mapped uuid
                cb_client = None
                try:
                    clients = data.setdefault("clients", {})
                    cb_client = clients.get(mapped_uuid, {}).get("client") if mapped_uuid else None
                except Exception:
                    cb_client = None
                if not cb_client:
                    cb_client = data.get("client")

                if cb_client:
                    cb_client.off(event_type, handler)
                _LOGGER.debug("Unregistered handler %s for connection %s client_uuid=%s", event_type, connection_id, mapped_uuid)
            except Exception:
                pass

        # Also remove any stored per-client subscriptions for this connection
        try:
            clients = data.setdefault("clients", {})
            if mapped_uuid and mapped_uuid in clients:
                clients[mapped_uuid].setdefault("subscriptions", {}).pop(connection_id, None)
        except Exception:
            pass

        # Remove per-connection state and update aggregated sensors
        try:
            client_states.pop(connection_id, None)
            data["is_listening"] = any(s.get("listening") for s in client_states.values())
            data["is_speaking"] = any(s.get("speaking") for s in client_states.values())
            hass.bus.async_fire(
                f"{DOMAIN}_listening_state_changed",
                {"entry_id": entry_id, "is_listening": data.get("is_listening", False)},
            )
            hass.bus.async_fire(
                f"{DOMAIN}_speaking_state_changed",
                {"entry_id": entry_id, "is_speaking": data.get("is_speaking", False)},
            )
        except Exception:
            _LOGGER.debug("Error cleaning up client state for connection %s", connection_id)
        
        # NOTE: Do NOT disconnect the per-websocket client here!
        # The on_close callback fires when a SUBSCRIPTION is cancelled, not when
        # the actual websocket connection closes. The frontend may re-subscribe
        # after start_listening/connect and we don't want to destroy the client.
        # The client will be cleaned up by _disconnect_and_cleanup when stop_listening
        # is called or when the actual websocket connection is closed.
        _LOGGER.debug("Subscription closed for connection %s mapped_uuid=%s (client preserved)", connection_id, mapped_uuid)

        # Remove active connection registration
        try:
            data.setdefault("active_connections", {}).pop(connection_id, None)
        except Exception:
            pass

    connection.subscriptions[msg["id"]] = on_close
    connection.send_result(msg["id"])


@websocket_api.websocket_command(
    {
        "type": "gemini_live/debug_state",
        vol.Optional("entry_id"): str,
    }
)
@websocket_api.async_response
async def websocket_debug_state(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return internal integration state for debugging."""
    data, entry_id = _get_data(hass, msg.get("entry_id"))

    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    try:
        clients = data.get("clients", {})
        connections_by_ws = data.get("connections_by_ws", {})
        subscriptions = data.get("subscriptions", {})
        client_states = data.get("client_states", {})

        clients_summary = {
            k: {
                "connected": bool(getattr(v.get("client"), "connected", False)),
                "created_at": v.get("created_at"),
                "subscriptions_count": len(v.get("subscriptions", {})),
            }
            for k, v in clients.items()
        }

        # Sanitize connections_by_ws keys to readable strings
        conn_map = {str(k): v for k, v in connections_by_ws.items()}

        subscription_summary = {str(k): list(v.keys()) for k, v in subscriptions.items()}

        result = {
            "entry_id": entry_id,
            "clients": clients_summary,
            "connections_by_ws": conn_map,
            "subscriptions": subscription_summary,
            "client_states": {str(k): v for k, v in client_states.items()},
            "is_listening": data.get("is_listening", False),
            "is_speaking": data.get("is_speaking", False),
        }

        connection.send_result(msg["id"], result)
    except Exception as e:
        connection.send_error(msg["id"], "error", str(e))


@websocket_api.websocket_command(
    {
        "type": "gemini_live/clear_conversation",
        vol.Optional("entry_id"): str,
    }
)
@websocket_api.async_response
async def websocket_clear_conversation(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Clear conversation for the requesting websocket client."""
    data, entry_id = _get_data(hass, msg.get("entry_id"))
    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    clients = data.setdefault("clients", {})
    connections_by_ws = data.setdefault("connections_by_ws", {})
    client_uuid = connections_by_ws.get(id(connection)) or _ensure_ws_client_uuid(data, connection)
    client_entry = clients.get(client_uuid)
    if client_entry:
        client_obj = client_entry.get("client")
        if client_obj:
            try:
                client_obj.clear_conversation()
            except Exception:
                pass
        connection.send_result(msg["id"], {"cleared": True, "client_uuid": client_uuid})
        return

    connection.send_result(msg["id"], {"cleared": False})


@websocket_api.websocket_command(
    {
        "type": "gemini_live/set_conversation_enabled",
        vol.Optional("entry_id"): str,
        vol.Required("enabled"): bool,
    }
)
@websocket_api.async_response
async def websocket_set_conversation_enabled(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Enable or disable conversation storage for this websocket client."""
    data, entry_id = _get_data(hass, msg.get("entry_id"))
    if not data:
        connection.send_error(msg["id"], "not_found", "Gemini Live Audio not configured")
        return

    enabled = bool(msg.get("enabled", False))
    clients = data.setdefault("clients", {})
    connections_by_ws = data.setdefault("connections_by_ws", {})
    client_uuid = connections_by_ws.get(id(connection)) or _ensure_ws_client_uuid(data, connection)
    client_entry = clients.setdefault(client_uuid, {})
    client_entry.setdefault("meta", {})["conversation_enabled"] = enabled
    # Also set attribute on client object if present
    client_obj = client_entry.get("client")
    if client_obj is not None:
        try:
            setattr(client_obj, "conversation_enabled", enabled)
        except Exception:
            pass

    connection.send_result(msg["id"], {"client_uuid": client_uuid, "conversation_enabled": enabled})
