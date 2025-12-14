"""Conversation agent for Gemini Live Audio integration."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import trace
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import ulid

from .const import (
    CONF_API_KEY,
    CONF_ENABLE_AFFECTIVE_DIALOG,
    CONF_ENABLE_GOOGLE_SEARCH,
    CONF_ENABLE_PROACTIVE_AUDIO,
    CONF_INSTRUCTIONS,
    CONF_MCP_SERVERS,
    CONF_MODEL,
    CONF_TEMPERATURE,
    CONF_VOICE,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_VOICE,
)
from .mcp_handler import HomeAssistantMCPTools, MCPServerHandler
from .live_client import GeminiLiveClient, LiveResponse, SessionConfig

_LOGGER = logging.getLogger(__name__)


class GeminiLiveConversationAgent(conversation.AbstractConversationAgent):
    """Gemini Live Audio conversation agent."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self._client: GeminiLiveClient | None = None
        self._mcp_handler: MCPServerHandler | None = None
        self._ha_tools: HomeAssistantMCPTools | None = None
        self._connected = False
        self._conversation_id: str | None = None

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return "*"

    @property
    def attribution(self) -> conversation.Attribution:
        """Return attribution for the agent."""
        return {
            "name": "Gemini Live Audio",
            "url": "https://ai.google.dev/gemini-api/docs/live",
        }

    async def async_initialize(self) -> None:
        """Initialize the conversation agent."""
        config = {**self.entry.data, **self.entry.options}

        session = async_get_clientsession(self.hass)

        # Initialize MCP handler
        self._mcp_handler = MCPServerHandler(session)

        # Add configured MCP servers
        mcp_servers = config.get(CONF_MCP_SERVERS, [])
        for server_config in mcp_servers:
            self._mcp_handler.add_server_from_config(server_config)

        # Connect to stdio MCP servers to get their tools
        stdio_servers = self._mcp_handler.get_stdio_servers()
        for server in stdio_servers:
            try:
                await self._mcp_handler.connect_server(server.name)
                _LOGGER.info(
                    "Connected to stdio MCP server %s with %d tools",
                    server.name,
                    len(server.tools),
                )
            except Exception as e:
                _LOGGER.error("Error connecting to stdio MCP server %s: %s", server.name, e)

        # Initialize Home Assistant tools
        self._ha_tools = HomeAssistantMCPTools(self.hass)
        
        # Build tools list: HA builtin + stdio MCP tools
        builtin_tools = self._ha_tools.get_builtin_tools()
        stdio_tools = self._mcp_handler.get_tools_as_functions()
        all_tools = builtin_tools + stdio_tools
        
        _LOGGER.info(
            "Agent tools: %d builtin + %d stdio MCP = %d total",
            len(builtin_tools),
            len(stdio_tools),
            len(all_tools),
        )

        # Create session configuration
        session_config = SessionConfig(
            model=config.get(CONF_MODEL, DEFAULT_MODEL),
            voice=config.get(CONF_VOICE, DEFAULT_VOICE),
            instructions=self._build_instructions(config),
            temperature=config.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            tools=all_tools,
            mcp_servers=mcp_servers,
            enable_google_search=config.get(CONF_ENABLE_GOOGLE_SEARCH, True),
            enable_affective_dialog=config.get(CONF_ENABLE_AFFECTIVE_DIALOG, False),
            enable_proactive_audio=config.get(CONF_ENABLE_PROACTIVE_AUDIO, False),
        )

        # Create client
        self._client = GeminiLiveClient(
            api_key=config[CONF_API_KEY],
            session_config=session_config,
        )

        # Register event handlers
        self._client.on("function_call", self._handle_function_call)

    def _build_instructions(self, config: dict[str, Any]) -> str:
        """Build the system instructions."""
        base_instructions = config.get(CONF_INSTRUCTIONS, DEFAULT_INSTRUCTIONS)

        ha_context = """
You are integrated with Home Assistant, a home automation platform.
You can control smart home devices and retrieve their states.

Available capabilities:
- Get the state of any entity (lights, sensors, switches, etc.)
- Call services to control devices (turn on/off, set temperature, etc.)
- List entities by domain or area
- Execute MCP server tools if configured

When the user asks to control a device or check its status, use the appropriate function.
Always confirm actions you take and provide helpful feedback.
"""
        return f"{base_instructions}\n\n{ha_context}"

    async def _ensure_connected(self) -> bool:
        """Ensure the client is connected."""
        if not self._client:
            await self.async_initialize()

        if self._client and not self._client.connected:
            self._connected = await self._client.connect()

            if self._connected and self._mcp_handler:
                # Connect to MCP servers
                await self._mcp_handler.connect_all_servers()

        return self._connected

    async def _handle_function_call(self, data: dict[str, Any]) -> None:
        """Handle function calls from the model."""
        call_id = data.get("call_id", "")
        function_name = data.get("name", "")
        arguments = data.get("arguments", {})

        _LOGGER.info("Function call received: name=%s call_id=%s", function_name, call_id)

        # Execute the function
        result = await self._execute_function(function_name, call_id, arguments)

        if self._client and result:
            await self._client.send_function_result(call_id, result)

    async def _execute_function(
        self,
        function_name: str,
        call_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute a function call by name."""
        _LOGGER.info("Executing function: %s with args: %s", function_name, arguments)
        
        # Check for Home Assistant built-in tools
        if function_name in ["get_entity_state", "call_service", "get_entities_by_domain", "get_area_entities"]:
            return await self._ha_tools.execute_tool(function_name, arguments)
        
        # Check if this is an MCP server function (format: server_name__tool_name)
        if self._mcp_handler and "__" in function_name:
            parsed = self._mcp_handler.parse_function_name(function_name)
            if parsed:
                server_name, tool_name = parsed
                _LOGGER.info("Calling MCP tool: %s/%s", server_name, tool_name)
                return await self._mcp_handler.call_tool(server_name, tool_name, arguments)

        return {"error": f"Unknown function: {function_name}"}

    async def async_process(
        self,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Process a user input."""
        if not await self._ensure_connected():
            return conversation.ConversationResult(
                response=intent.IntentResponse(language=user_input.language),
                conversation_id=user_input.conversation_id or ulid.ulid_now(),
            )

        # Set or continue conversation
        conversation_id = user_input.conversation_id or ulid.ulid_now()
        self._conversation_id = conversation_id

        # Trace the conversation
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"text": user_input.text},
        )

        try:
            # Send text to the Live API
            response = await self._client.send_text(user_input.text)

            # Extract the response text
            response_text = self._extract_response_text(response)

            # Create intent response
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(response_text)

            trace.async_conversation_trace_append(
                trace.ConversationTraceEventType.AGENT_DETAIL,
                {"response": response_text},
            )

            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )

        except asyncio.TimeoutError:
            _LOGGER.error("Timeout waiting for Gemini Live Audio response")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Request timed out",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )
        except Exception as e:
            _LOGGER.error("Error processing conversation: %s", e)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Error: {str(e)}",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )

    def _extract_response_text(self, response: LiveResponse) -> str:
        """Extract text from the response."""
        # First try the direct text/transcript
        if response.text:
            return response.text
        if response.audio_transcript:
            return response.audio_transcript

        # Try to extract from output items
        for item in response.output:
            for part in item.parts:
                if isinstance(part, dict):
                    if part.get("text"):
                        return part.get("text", "")

        return "I'm sorry, I couldn't generate a response."

    async def async_disconnect(self) -> None:
        """Disconnect from the API."""
        if self._client:
            await self._client.disconnect()
            self._connected = False


async def async_setup_conversation(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> GeminiLiveConversationAgent:
    """Set up the conversation agent."""
    agent = GeminiLiveConversationAgent(hass, entry)
    await agent.async_initialize()

    conversation.async_set_agent(hass, entry, agent)

    return agent


async def async_unload_conversation(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> None:
    """Unload the conversation agent."""
    conversation.async_unset_agent(hass, entry)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities,
) -> None:
    """Set up conversation platform."""
    pass


async def async_unload_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> bool:
    """Unload conversation platform."""
    return True
