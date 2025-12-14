"""The Gemini Live Audio integration."""
from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_API_KEY,
    CONF_ENABLE_AFFECTIVE_DIALOG,
    CONF_ENABLE_GOOGLE_SEARCH,
    CONF_ENABLE_PROACTIVE_AUDIO,
    CONF_INSTRUCTIONS,
    CONF_MCP_SERVER_ENABLED,
    CONF_MCP_SERVER_NAME,
    CONF_MCP_SERVER_TYPE,
    CONF_MCP_SERVERS,
    CONF_MODEL,
    CONF_TEMPERATURE,
    CONF_VOICE,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_VOICE,
    DOMAIN,
    MCP_SERVER_TYPE_SSE,
    MCP_SERVER_TYPE_STDIO,
)
from .conversation import (
    GeminiLiveConversationAgent,
    async_setup_conversation,
    async_unload_conversation,
)
from .mcp_handler import MCPServerHandler
from .live_client import GeminiLiveClient, SessionConfig

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.CONVERSATION,
    Platform.MEDIA_PLAYER,
    Platform.BINARY_SENSOR,
]

# Service schemas
SERVICE_SEND_MESSAGE = "send_message"
SERVICE_ADD_MCP_SERVER = "add_mcp_server"
SERVICE_REMOVE_MCP_SERVER = "remove_mcp_server"
SERVICE_LIST_MCP_SERVERS = "list_mcp_servers"
SERVICE_CONNECT_MCP_SERVERS = "connect_mcp_servers"
SERVICE_CLEAR_CONVERSATION = "clear_conversation"
SERVICE_SEND_AUDIO = "send_audio"
SERVICE_START_LISTENING = "start_listening"
SERVICE_STOP_LISTENING = "stop_listening"

SEND_MESSAGE_SCHEMA = vol.Schema({
    vol.Required("message"): cv.string,
})

SEND_AUDIO_SCHEMA = vol.Schema({
    vol.Required("audio_data"): cv.string,
})

ADD_MCP_SERVER_SCHEMA = vol.Schema({
    vol.Required("name"): cv.string,
    vol.Required("server_type"): vol.In([MCP_SERVER_TYPE_SSE, MCP_SERVER_TYPE_STDIO]),
    vol.Optional("url"): cv.url,
    vol.Optional("token"): cv.string,
    vol.Optional("command"): cv.string,
    vol.Optional("args"): vol.All(cv.ensure_list, [cv.string]),
    vol.Optional("env"): dict,
})

REMOVE_MCP_SERVER_SCHEMA = vol.Schema({
    vol.Required("name"): cv.string,
})


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Gemini Live from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    config = {**entry.data, **entry.options}

    # Get aiohttp session
    session = async_get_clientsession(hass)

    # Create MCP handler
    mcp_handler = MCPServerHandler(session)

    # Add configured MCP servers
    mcp_servers = config.get(CONF_MCP_SERVERS, [])
    _LOGGER.info("Loading %d MCP servers from config", len(mcp_servers))
    for i, server_config in enumerate(mcp_servers):
        _LOGGER.info(
            "MCP server %d: %s (type=%s, enabled=%s)",
            i,
            server_config.get(CONF_MCP_SERVER_NAME),
            server_config.get(CONF_MCP_SERVER_TYPE),
            server_config.get(CONF_MCP_SERVER_ENABLED, True),
        )
        mcp_handler.add_server_from_config(server_config)

    # Connect to stdio MCP servers and get their tools
    stdio_servers = mcp_handler.get_stdio_servers()
    if stdio_servers:
        _LOGGER.info("Connecting to %d stdio MCP servers...", len(stdio_servers))
        for server in stdio_servers:
            try:
                connected = await mcp_handler.connect_server(server.name)
                if connected:
                    _LOGGER.info(
                        "Connected to stdio MCP server %s, found %d tools",
                        server.name,
                        len(server.tools),
                    )
                else:
                    _LOGGER.warning("Failed to connect to stdio MCP server %s", server.name)
            except Exception as e:
                _LOGGER.error("Error connecting to stdio MCP server %s: %s", server.name, e)

    # Get built-in Home Assistant tools
    from .mcp_handler import HomeAssistantMCPTools
    ha_tools = HomeAssistantMCPTools(hass)
    builtin_tools = ha_tools.get_builtin_tools()
    
    # Add stdio MCP server tools as function tools
    stdio_tools = mcp_handler.get_tools_as_functions()
    all_tools = builtin_tools + stdio_tools
    _LOGGER.info(
        "Total tools: %d builtin + %d stdio MCP = %d",
        len(builtin_tools),
        len(stdio_tools),
        len(all_tools),
    )

    # Create session configuration with built-in tools
    session_config = SessionConfig(
        model=config.get(CONF_MODEL, DEFAULT_MODEL),
        voice=config.get(CONF_VOICE, DEFAULT_VOICE),
        instructions=config.get(CONF_INSTRUCTIONS, DEFAULT_INSTRUCTIONS),
        temperature=config.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
        mcp_servers=mcp_servers,
        tools=all_tools,
        enable_google_search=config.get(CONF_ENABLE_GOOGLE_SEARCH, True),
        enable_affective_dialog=config.get(CONF_ENABLE_AFFECTIVE_DIALOG, False),
        enable_proactive_audio=config.get(CONF_ENABLE_PROACTIVE_AUDIO, False),
    )

    # Create the live client (no session needed - uses google-genai)
    client = GeminiLiveClient(
        api_key=config[CONF_API_KEY],
        session_config=session_config,
    )

    # Store references
    hass.data[DOMAIN][entry.entry_id] = {
        "client": client,
        "mcp_handler": mcp_handler,
        "session_config": session_config,
        "ha_tools": ha_tools,
    }

    # Set up the conversation agent
    agent = await async_setup_conversation(hass, entry)
    hass.data[DOMAIN][entry.entry_id]["agent"] = agent

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register update listener
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    # Register services
    await async_register_services(hass)

    # Register WebSocket API
    from .websocket_api import async_register_websocket_api
    async_register_websocket_api(hass)

    # Register frontend
    from .frontend import async_register_frontend
    await async_register_frontend(hass)

    _LOGGER.info("Gemini Live Audio integration set up successfully")

    return True


async def async_register_services(hass: HomeAssistant) -> None:
    """Register integration services."""

    async def handle_send_message(call: ServiceCall) -> None:
        """Handle send_message service call."""
        message = call.data["message"]

        for entry_id, data in hass.data.get(DOMAIN, {}).items():
            if entry_id.startswith("_"):
                continue
            agent: GeminiLiveConversationAgent | None = data.get("agent")
            if agent:
                from homeassistant.components.conversation import ConversationInput
                from homeassistant.util import ulid

                user_input = ConversationInput(
                    text=message,
                    context=call.context,
                    conversation_id=ulid.ulid_now(),
                    language="en",
                    device_id=None,
                    satellite_id=None,
                    agent_id=entry_id,
                )
                await agent.async_process(user_input)
                break

    async def handle_add_mcp_server(call: ServiceCall) -> None:
        """Handle add_mcp_server service call."""
        name = call.data["name"]
        server_type = call.data.get("server_type", MCP_SERVER_TYPE_SSE)
        url = call.data.get("url", "")
        token = call.data.get("token")
        command = call.data.get("command", "")
        args = call.data.get("args", [])
        env = call.data.get("env", {})

        for entry_id, data in hass.data.get(DOMAIN, {}).items():
            if entry_id.startswith("_"):
                continue
            mcp_handler: MCPServerHandler | None = data.get("mcp_handler")
            if mcp_handler:
                mcp_handler.add_server(
                    name=name,
                    server_type=server_type,
                    url=url,
                    token=token,
                    command=command,
                    args=args,
                    env=env,
                )
                await mcp_handler.connect_server(name)
                _LOGGER.info("Added MCP server: %s (type: %s)", name, server_type)
                break

    async def handle_remove_mcp_server(call: ServiceCall) -> None:
        """Handle remove_mcp_server service call."""
        name = call.data["name"]

        for entry_id, data in hass.data.get(DOMAIN, {}).items():
            if entry_id.startswith("_"):
                continue
            mcp_handler: MCPServerHandler | None = data.get("mcp_handler")
            if mcp_handler:
                mcp_handler.remove_server(name)
                _LOGGER.info("Removed MCP server: %s", name)
                break

    async def handle_list_mcp_servers(call: ServiceCall) -> dict:
        """Handle list_mcp_servers service call."""
        servers = []
        for entry_id, data in hass.data.get(DOMAIN, {}).items():
            if entry_id.startswith("_"):
                continue
            mcp_handler: MCPServerHandler | None = data.get("mcp_handler")
            if mcp_handler:
                servers = mcp_handler.get_server_configs()
                break
        return {"servers": servers}

    async def handle_connect_mcp_servers(call: ServiceCall) -> dict:
        """Handle connect_mcp_servers service call."""
        results = {}
        for entry_id, data in hass.data.get(DOMAIN, {}).items():
            if entry_id.startswith("_"):
                continue
            mcp_handler: MCPServerHandler | None = data.get("mcp_handler")
            if mcp_handler:
                results = await mcp_handler.connect_all_servers()
                _LOGGER.info("Connected MCP servers: %s", results)
                break
        return {"results": results}

    async def handle_clear_conversation(call: ServiceCall) -> None:
        """Handle clear_conversation service call."""
        for entry_id, data in hass.data.get(DOMAIN, {}).items():
            client: GeminiLiveClient | None = data.get("client")
            if client:
                client.clear_conversation()
                _LOGGER.info("Cleared conversation history")
                break

    async def handle_send_audio(call: ServiceCall) -> None:
        """Handle send_audio service call."""
        import base64
        audio_data_b64 = call.data["audio_data"]

        for entry_id, data in hass.data.get(DOMAIN, {}).items():
            client: GeminiLiveClient | None = data.get("client")
            if client:
                if not client.connected:
                    await client.connect()
                audio_bytes = base64.b64decode(audio_data_b64)
                await client.send_audio(audio_bytes)
                _LOGGER.debug("Sent audio data: %d bytes", len(audio_bytes))
                break

    async def handle_start_listening(call: ServiceCall) -> None:
        """Handle start_listening service call."""
        for entry_id, data in hass.data.get(DOMAIN, {}).items():
            client: GeminiLiveClient | None = data.get("client")
            if client:
                if not client.connected:
                    await client.connect()
                _LOGGER.info("Started listening session")
                break

    async def handle_stop_listening(call: ServiceCall) -> None:
        """Handle stop_listening service call."""
        for entry_id, data in hass.data.get(DOMAIN, {}).items():
            client: GeminiLiveClient | None = data.get("client")
            if client:
                # Signal end of audio stream
                await client.send_audio_stream_end()
                _LOGGER.info("Stopped listening session")
                break

    # Register services if not already registered
    if not hass.services.has_service(DOMAIN, SERVICE_SEND_MESSAGE):
        hass.services.async_register(
            DOMAIN, SERVICE_SEND_MESSAGE, handle_send_message, schema=SEND_MESSAGE_SCHEMA
        )
    if not hass.services.has_service(DOMAIN, SERVICE_SEND_AUDIO):
        hass.services.async_register(
            DOMAIN, SERVICE_SEND_AUDIO, handle_send_audio, schema=SEND_AUDIO_SCHEMA
        )
    if not hass.services.has_service(DOMAIN, SERVICE_START_LISTENING):
        hass.services.async_register(
            DOMAIN, SERVICE_START_LISTENING, handle_start_listening
        )
    if not hass.services.has_service(DOMAIN, SERVICE_STOP_LISTENING):
        hass.services.async_register(
            DOMAIN, SERVICE_STOP_LISTENING, handle_stop_listening
        )
    if not hass.services.has_service(DOMAIN, SERVICE_ADD_MCP_SERVER):
        hass.services.async_register(
            DOMAIN, SERVICE_ADD_MCP_SERVER, handle_add_mcp_server, schema=ADD_MCP_SERVER_SCHEMA
        )
    if not hass.services.has_service(DOMAIN, SERVICE_REMOVE_MCP_SERVER):
        hass.services.async_register(
            DOMAIN, SERVICE_REMOVE_MCP_SERVER, handle_remove_mcp_server, schema=REMOVE_MCP_SERVER_SCHEMA
        )
    if not hass.services.has_service(DOMAIN, SERVICE_LIST_MCP_SERVERS):
        hass.services.async_register(
            DOMAIN, SERVICE_LIST_MCP_SERVERS, handle_list_mcp_servers
        )
    if not hass.services.has_service(DOMAIN, SERVICE_CONNECT_MCP_SERVERS):
        hass.services.async_register(
            DOMAIN, SERVICE_CONNECT_MCP_SERVERS, handle_connect_mcp_servers
        )
    if not hass.services.has_service(DOMAIN, SERVICE_CLEAR_CONVERSATION):
        hass.services.async_register(
            DOMAIN, SERVICE_CLEAR_CONVERSATION, handle_clear_conversation
        )


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        # Unload conversation agent
        await async_unload_conversation(hass, entry)

        # Disconnect client
        data = hass.data[DOMAIN].pop(entry.entry_id, {})
        client: GeminiLiveClient | None = data.get("client")
        if client:
            await client.disconnect()

        # Disconnect agent
        agent: GeminiLiveConversationAgent | None = data.get("agent")
        if agent:
            await agent.async_disconnect()

        # If this is the last entry, unregister frontend
        remaining_entries = [
            key for key in hass.data.get(DOMAIN, {}).keys()
            if not key.startswith("_")
        ]
        if not remaining_entries:
            from .frontend import async_unregister_frontend
            await async_unregister_frontend(hass)
            _LOGGER.info("Unregistered Gemini Live Audio frontend (last entry removed)")

    return unload_ok


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug("Migrating from version %s", entry.version)

    if entry.version == 1:
        # Future migration logic
        pass

    return True


def get_client(hass: HomeAssistant, entry_id: str) -> GeminiLiveClient | None:
    """Get the Gemini Live Audio client for an entry."""
    if DOMAIN not in hass.data:
        return None
    entry_data = hass.data[DOMAIN].get(entry_id, {})
    return entry_data.get("client")


def get_mcp_handler(hass: HomeAssistant, entry_id: str) -> MCPServerHandler | None:
    """Get the MCP handler for an entry."""
    if DOMAIN not in hass.data:
        return None
    entry_data = hass.data[DOMAIN].get(entry_id, {})
    return entry_data.get("mcp_handler")
