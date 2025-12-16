"""Gemini Live API client using google-genai library.

This module provides a client for Google's Gemini Live API,
supporting real-time audio, video (camera/screen), function calling,
and Google Search.

Based on official documentation:
"""
from __future__ import annotations

import asyncio
import json
import base64
import logging
import re
import unicodedata
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Awaitable, cast

from google import genai
from google.genai import types

from .const import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    DEFAULT_TEMPERATURE,
    EVENT_AUDIO_DELTA,
    EVENT_ERROR,
    EVENT_FUNCTION_CALL,
    EVENT_GENERATION_COMPLETE,
    EVENT_GO_AWAY,
    EVENT_INPUT_TRANSCRIPTION,
    EVENT_INTERRUPTED,
    EVENT_OUTPUT_TRANSCRIPTION,
    EVENT_SESSION_RESUMED,
    EVENT_SESSION_RESUMPTION_UPDATE,
    EVENT_SESSION_STARTED,
    EVENT_TEXT_DELTA,
    EVENT_TURN_COMPLETE,
)

_LOGGER = logging.getLogger(__name__)

# Set google_genai loggers to DEBUG to avoid warning spam about non-data parts
logging.getLogger("google_genai.types").setLevel(logging.DEBUG)
logging.getLogger("google_genai").setLevel(logging.DEBUG)

# Audio constants
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
@dataclass
class SessionConfig:
    """Configuration for a Gemini Live session."""

    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    instructions: str = DEFAULT_INSTRUCTIONS
    temperature: float = DEFAULT_TEMPERATURE
    tools: list[dict[str, Any]] = field(default_factory=list)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    # Mapping of MCP server name -> list of function dicts (grouped by origin)
    mcp_tool_groups: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Allow disabling sending function/tool declarations (useful for debugging)
    enable_tools: bool = True
    enable_google_search: bool = False
    enable_affective_dialog: bool = False
    enable_proactive_audio: bool = False
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM"
    context_window_compression: bool = True
    trigger_tokens: int = 32000
    target_tokens: int = 12800
    input_audio_transcription: bool = True
    output_audio_transcription: bool = True
    # Session resumption - handle from previous session to resume
    session_resumption_handle: str | None = None
    # Enable session resumption for long-running sessions
    enable_session_resumption: bool = True
    # Ephemeral token (use instead of API key for client-side auth)
    ephemeral_token: str | None = None
    # Tools mode: 'full' sends full parameter schemas; 'minimal' sends only name/description
    tools_mode: str = "full"


@dataclass
class ConversationItem:
    """Represents a conversation item."""

    id: str
    role: str
    parts: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"


@dataclass
class LiveResponse:
    """Represents a response from the API."""

    id: str
    status: str
    output: list[ConversationItem] = field(default_factory=list)
    text: str = ""
    audio_transcript: str = ""
    audio_data: bytes = b""


class GeminiLiveClient:
    """Client for Gemini Live API using google-genai library."""

    def __init__(
        self,
        api_key: str,
        session_config: SessionConfig | None = None,
    ) -> None:
        """Initialize the client."""
        self._api_key = api_key
        self._session_config = session_config or SessionConfig()
        self._client: genai.Client | None = None
        self._session = None
        self._session_context = None  # Keep context manager reference for session
        self._connected = False

        # Event handlers
        self._event_handlers: dict[str, list[Callable]] = {}

        # Queue for received audio
        self._audio_in_queue: asyncio.Queue | None = None

        # Background receive task
        self._receive_task: asyncio.Task | None = None

        # Function call handling
        self._pending_function_calls: dict[str, dict[str, Any]] = {}
        # Mapping of function name -> behavior (e.g., 'NON_BLOCKING') from session config
        self._function_behaviors: dict[str, str] = {}
        # Preserve original/raw declarations (name -> original dict)
        self._raw_function_declarations: dict[str, dict[str, Any]] = {}
        # Counter for function responses sent in this session (for debugging)
        self._function_response_count: int = 0
        # Track token usage for context limit monitoring
        self._last_prompt_token_count: int = 0
        self._token_warning_threshold: int = 50000  # Warn when approaching limit

        # Current response tracking
        self._current_response: LiveResponse | None = None
        self._response_futures: dict[str, asyncio.Future] = {}
        # Conversation history (stores completed LiveResponse objects)
        self._conversation_history: list[LiveResponse] = []

        # Session resumption tracking
        self._session_resumption_handle: str | None = None
        self._session_resumable: bool = False

        # Function call processing state - block audio/input during function calls
        self._processing_function_call: bool = False
        # Timestamp of last turn complete - used to add brief delay before new input
        self._last_turn_complete_time: float = 0.0
        # Minimum delay (seconds) after turn complete before allowing new input
        self._turn_complete_delay: float = 0.05  # 50ms delay

        # GoAway tracking
        self._go_away_time_left: int | None = None
        # Background task that periodically re-emits GoAway events
        self._go_away_task: asyncio.Task | None = None
        # Number of frontend/clients currently using this live session
        self._connection_count: int = 0
        # Optional owner frontend websocket id (id(connection)) when this
        # client was created for a specific websocket. May be None for
        # shared/legacy clients.
        self._owner_ws: int | None = None

    @property
    def connected(self) -> bool:
        """Return connection status."""
        return self._connected

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        # Avoid registering the same handler more than once
        try:
            if handler in self._event_handlers[event_type]:
                return
        except Exception:
            pass
        self._event_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event to registered handlers."""
        handlers = self._event_handlers.get(event_type, [])
        _LOGGER.debug("Client %s emitting event '%s' to %d handlers (owner_ws=%s)", id(self), event_type, len(handlers), getattr(self, "_owner_ws", None))
        # Debug: list handler identities and any websocket mapping attached
        try:
            for h in handlers:
                try:
                    mapping = getattr(h, "_ws_mapping", None)
                except Exception:
                    mapping = None
                # If handler is a bound method, include the bound instance class/id
                bound_info = None
                try:
                    bound_self = getattr(h, "__self__", None)
                    if bound_self is not None:
                        bound_info = f"{bound_self.__class__.__name__}@{id(bound_self)}"
                except Exception:
                    bound_info = None
                _LOGGER.debug(
                    "  handler id=%s qual=%s mapping=%s bound=%s",
                    id(h),
                    getattr(h, "__qualname__", str(h)),
                    mapping,
                    bound_info,
                )
        except Exception:
            pass
        if handlers:
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    _LOGGER.error("Error in event handler for %s: %s", event_type, e)

    def _build_tools(self) -> list[types.Tool]:
        """Build tools list for the session."""
        tools_list = []

        # If tools are disabled in the config, return empty tools list immediately
        if not getattr(self._session_config, "enable_tools", True):
            _LOGGER.debug("_build_tools: tools disabled by session_config.enable_tools")
            return tools_list

        # Reset behavior mapping each time we build tools
        try:
            self._function_behaviors = {}
        except Exception:
            self._function_behaviors = {}

        # Add Google Search if enabled
        if self._session_config.enable_google_search:
            tools_list.append(types.Tool(google_search=types.GoogleSearch()))

        # Add function declarations from config
        # First, add any MCP-provided tools grouped by their server of origin.
        # Use the original dicts (function definitions) as-is so we don't
        # mutate schemas coming from MCP servers. This preserves the
        # `function_declarations` objects exactly as discovered.
        try:
            mcp_groups = getattr(self._session_config, "mcp_tool_groups", {}) or {}
            for server_name, funcs in mcp_groups.items():
                if not funcs:
                    continue
                _LOGGER.debug("Building tools for MCP server %s with %d functions", server_name, len(funcs))
                # Helper to sanitize raw function dicts into the minimal allowed
                # shape accepted by the Live API SDK (name, description, parameters).
                def _sanitize_fn_dict(f: dict[str, Any]) -> dict[str, Any]:
                    if not isinstance(f, dict):
                        return {"name": str(f)}
                    name = f.get("name") or f.get("id") or ""
                    desc = f.get("description") or (f.get("annotations") or {}).get("title") or ""
                    params = f.get("parameters") or f.get("inputSchema") or f.get("input_schema") or {}
                    
                    # Add a CONCISE parameter summary to help the model understand how to call the function.
                    # This replaces the old approach of appending str(f) which caused 65k+ token bloat.
                    try:
                        props = params.get("properties", {}) if isinstance(params, dict) else {}
                        required = params.get("required", []) if isinstance(params, dict) else []
                        if props and isinstance(props, dict):
                            # Build concise summary: "Parameters: param1 (type, required), param2 (type)"
                            param_parts = []
                            for pname, pdef in props.items():
                                ptype = pdef.get("type", "any") if isinstance(pdef, dict) else "any"
                                pdesc = pdef.get("description", "") if isinstance(pdef, dict) else ""
                                is_req = pname in required
                                # Keep param description short (first 50 chars)
                                if pdesc and len(pdesc) > 50:
                                    pdesc = pdesc[:47] + "..."
                                if is_req:
                                    param_parts.append(f"{pname}: {ptype} (required){' - ' + pdesc if pdesc else ''}")
                                else:
                                    param_parts.append(f"{pname}: {ptype}{' - ' + pdesc if pdesc else ''}")
                            if param_parts:
                                # Limit to first 10 params to avoid bloat
                                if len(param_parts) > 10:
                                    param_parts = param_parts[:10] + [f"... and {len(param_parts) - 10} more"]
                                desc = desc.rstrip() + " | Params: " + "; ".join(param_parts)
                    except Exception:
                        pass
                    
                    # Ensure parameters is a dict with properties we can augment

                    _LOGGER.debug("Sanitizing function dict for %s: name=%s desc_len=%d", server_name, name, len(desc))
                    
                    try:
                        if not isinstance(params, dict):
                            params = {}
                    except Exception:
                        params = {}

                    if not isinstance(params.get("properties"), dict):
                        params.setdefault("properties", {})

                    # Only inject a `name` property/required for Home Assistant-style
                    # tools. Many third-party tools (e.g. music, generic helpers)
                    # should not be forced to accept a `name` parameter.
                    try:
                        fn_name = (f.get("name") or f.get("id") or "") if isinstance(f, dict) else ""
                    except Exception:
                        fn_name = ""

                    try:
                        props = params.get("properties") if isinstance(params, dict) else None
                        is_ha_like = False
                        if isinstance(fn_name, str) and fn_name.startswith(("Hass", "hass")):
                            is_ha_like = True
                        if isinstance(props, dict):
                            ha_keys = ("domain", "device_class", "area", "floor", "is_volume_muted")
                            if any(k in props for k in ha_keys):
                                is_ha_like = True

                        if is_ha_like:
                            # Ensure a `name` property exists so callers can target entities
                            try:
                                if not isinstance(props, dict):
                                    params["properties"] = {"name": {"type": "string"}}
                                else:
                                    if "name" not in params["properties"]:
                                        params["properties"]["name"] = {"type": "string"}
                            except Exception:
                                try:
                                    params["properties"] = {"name": {"type": "string"}}
                                except Exception:
                                    pass

                            # Ensure `required` includes `name` so callers supply an entity id
                            try:
                                req = params.get("required")
                                if not isinstance(req, list):
                                    params["required"] = []
                                if "name" not in params["required"]:
                                    params["required"].append("name")
                            except Exception:
                                try:
                                    params["required"] = ["name"]
                                except Exception:
                                    pass
                    
                    except Exception:
                        # Safe fallback if HA-detection or schema mutation fails
                        pass

                    return {"name": name, "description": desc, "parameters": params}

                # Preserve the full function list for each MCP server as a
                # single tool entry so the Live API sees all available
                # functions instead of splitting them into multiple tool
                # declarations. Some MCP servers expect the whole set to
                # be present together. Do NOT sanitize or alter the raw
                # dicts received from MCP servers; append them as-is.
                try:
                    # Sanitize MCP-provided function dicts to the minimal
                    # shape accepted by the SDK so pydantic validation does
                    # not reject extra fields like `annotations` or
                    # vendor-specific keys.
                    sanitized = [_sanitize_fn_dict(f) for f in funcs]
                    tools_list.append({"function_declarations": sanitized})
                    # Populate function_behaviors mapping if any functions
                    # declare a non-default behavior (e.g. NON_BLOCKING).
                    for orig in funcs:
                        try:
                            if isinstance(orig, dict):
                                name = orig.get("name") or orig.get("id") or ""
                                behavior = orig.get("behavior") or (orig.get("annotations") or {}).get("behavior")
                                if name and behavior:
                                    try:
                                        self._function_behaviors[name] = str(behavior)
                                    except Exception:
                                        pass
                                # Store the original raw declaration for later use (logging, inspection)
                                try:
                                    if name:
                                        self._raw_function_declarations[name] = orig
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    _LOGGER.debug("Added sanitized Tool dict for server %s with %d functions", server_name, len(sanitized))
                except Exception:
                    _LOGGER.debug("Failed to add MCP tool group for %s; skipping", server_name, exc_info=True)
        except Exception:
            _LOGGER.debug("Error while building MCP tool groups", exc_info=True)

        # Next, process any non-MCP tools that may be present in the flat tools list.
        # If callers provided dict-shaped tool entries (e.g. raw `function_declarations`
        # dicts or `google_search` dicts), include them unchanged. For legacy
        # flat function defs (dict with name/description/parameters), we keep the
        # prior behavior of grouping into Tool entries.
        if self._session_config.tools:
            flat_tools = getattr(self._session_config, "tools", []) or []
            # Collect simple function defs (non-tool-dict entries) to batch
            simple_funcs: list[dict[str, Any]] = []
            for tool in flat_tools:
                # If the tool is already a dict-shaped Tool (contains function_declarations or other top-level keys), pass it through unchanged
                if isinstance(tool, dict) and ("function_declarations" in tool or any(k in tool for k in ("google_search", "google_search")) or any(not isinstance(v, dict) for v in tool.values())):
                    # If this dict contains a `function_declarations` list, keep
                    # the full list together as a single tool entry instead of
                    # splitting it into multiple entries. Otherwise, pass the
                    # dict through unchanged.
                    if isinstance(tool.get("function_declarations"), (list, tuple)):
                        fdecls = tool.get("function_declarations") or []
                        try:
                            # Sanitize each declared function to avoid passing
                            # vendor-specific keys (annotations, inputSchema, etc.)
                            sanitized_fdecls = []
                            for fd in fdecls:
                                try:
                                    if isinstance(fd, dict):
                                        sanitized = _sanitize_fn_dict(fd)
                                        sanitized_fdecls.append(sanitized)
                                        # capture behavior if present
                                        beh = fd.get("behavior") or (fd.get("annotations") or {}).get("behavior")
                                        name_fd = fd.get("name") or fd.get("id") or ""
                                        if name_fd and isinstance(beh, str):
                                            self._function_behaviors[name_fd] = beh
                                        # store original raw function declaration
                                        try:
                                            if name_fd:
                                                self._raw_function_declarations[name_fd] = fd
                                        except Exception:
                                            pass
                                    else:
                                        sanitized_fdecls.append({"name": str(fd)})
                                except Exception:
                                    try:
                                        sanitized_fdecls.append({"name": "<error>"})
                                    except Exception:
                                        pass
                        except Exception:
                            sanitized_fdecls = list(fdecls)
                        tools_list.append({"function_declarations": sanitized_fdecls})
                        _LOGGER.debug("Added sanitized Tool dict from flat tools with %d functions", len(sanitized_fdecls))
                    else:
                        tools_list.append(tool)
                    continue

                # Otherwise treat as simple function definition (name/description/parameters)
                if isinstance(tool, dict) and "name" in tool:
                    simple_funcs.append(tool)
                else:
                    _LOGGER.debug("Ignoring unknown tool entry type in flat tools: %s", type(tool))

            # If we have simple function defs, include them as a single
            # tool entry so all functions are visible to the Live API.
            if simple_funcs:
                try:
                    # Sanitize and capture behaviors from simple function defs
                    sanitized_simple = []
                    for fd in simple_funcs:
                        try:
                            if isinstance(fd, dict):
                                sanitized = _sanitize_fn_dict(fd)
                                sanitized_simple.append(sanitized)
                                beh = fd.get("behavior") or (fd.get("annotations") or {}).get("behavior")
                                name_fd = fd.get("name") or fd.get("id") or ""
                                if name_fd and isinstance(beh, str):
                                    self._function_behaviors[name_fd] = beh
                                try:
                                    if name_fd:
                                        self._raw_function_declarations[name_fd] = fd
                                except Exception:
                                    pass
                            else:
                                sanitized_simple.append({"name": str(fd)})
                        except Exception:
                            sanitized_simple.append({"name": "<error>"})

                    tools_list.append({"function_declarations": sanitized_simple})
                    _LOGGER.debug("Added sanitized Tool dict with %d simple functions", len(sanitized_simple))
                except Exception:
                    _LOGGER.debug("Failed to add simple function defs; adding as single Tool dict", exc_info=True)
                    tools_list.append({"function_declarations": simple_funcs})

        # Log total tools size for debugging token usage
        try:
            tools_json = json.dumps(tools_list, ensure_ascii=False, default=str)
            tools_size = len(tools_json)
            # Rough token estimate: ~4 chars per token
            estimated_tokens = tools_size // 4
            _LOGGER.debug("_build_tools: total tools size=%d bytes, estimated_tokens=%d, num_tool_entries=%d", 
                        tools_size, estimated_tokens, len(tools_list))
        except Exception:
            pass

        return tools_list

    def _convert_parameters(self, params: dict) -> types.Schema | None:
        """Convert parameter schema to google-genai Schema format."""
        if not params:
            return None

        def schema_from_dict(d: dict) -> types.Schema:
            ptype = d.get("type", "object").upper()
            description = d.get("description", "")

            # Array type: convert items recursively
            if ptype == "ARRAY":
                items = d.get("items")
                item_schema = schema_from_dict(items) if isinstance(items, dict) else None
                return types.Schema(
                    type=getattr(types.Type, "ARRAY", types.Type.ARRAY),
                    items=item_schema,
                    description=description,
                )

            # Object type: convert properties recursively
            if ptype == "OBJECT":
                props = {}
                for k, v in d.get("properties", {}).items():
                    props[k] = schema_from_dict(v)
                return types.Schema(
                    type=getattr(types.Type, "OBJECT", types.Type.OBJECT),
                    properties=props,
                    required=d.get("required", []),
                    description=description,
                )

            # Primitive types
            return types.Schema(
                type=getattr(types.Type, ptype, types.Type.STRING),
                description=description,
            )

        try:
            return schema_from_dict(params)
        except Exception as e:
            _LOGGER.debug("Failed to convert parameters schema: %s", e)
            return None

    def _build_config(self) -> types.LiveConnectConfig:
        """Build LiveConnectConfig for the session."""
        # Build speech config
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=self._session_config.voice
                )
            )
        )

        # Build tools
        tools = self._build_tools()

        # Debug: compact preview of tools to help diagnose connect failures
        try:
            if tools:
                try:
                    preview = self._preview_tools(tools)
                    _LOGGER.debug("_build_config: tools preview=%s", preview)
                except Exception:
                    _LOGGER.debug("_build_config: failed to generate tools preview", exc_info=True)
        except Exception:
            pass

        # Build system instruction
        system_instruction = None
        if self._session_config.instructions:
            system_instruction = types.Content(
                parts=[types.Part.from_text(text=self._session_config.instructions)],
                role="user",
            )

        # Build context window compression
        context_compression = None
        if self._session_config.context_window_compression:
            context_compression = types.ContextWindowCompressionConfig(
                trigger_tokens=self._session_config.trigger_tokens,
                sliding_window=types.SlidingWindow(
                    target_tokens=self._session_config.target_tokens
                ),
            )

        # Build live input config with automatic activity detection
        realtime_input_config = types.RealtimeInputConfig(
            turn_coverage="TURN_INCLUDES_ALL_INPUT"
        )

        # Build config dict first
        config_kwargs = {
            "response_modalities": ["AUDIO"],
            "speech_config": speech_config,
            "realtime_input_config": realtime_input_config,
            "system_instruction": system_instruction,
        }

        # Add optional configs
        if self._session_config.media_resolution:
            config_kwargs["media_resolution"] = self._session_config.media_resolution

        if context_compression:
            config_kwargs["context_window_compression"] = context_compression

        if tools:
            config_kwargs["tools"] = tools

        # Add transcription configs
        if self._session_config.output_audio_transcription:
            config_kwargs["output_audio_transcription"] = {}

        if self._session_config.input_audio_transcription:
            config_kwargs["input_audio_transcription"] = {}

        # Add affective dialog (requires v1alpha API)
        if self._session_config.enable_affective_dialog:
            config_kwargs["enable_affective_dialog"] = True

        # Add proactive audio (requires v1alpha API)
        if self._session_config.enable_proactive_audio:
            config_kwargs["proactivity"] = {"proactive_audio": True}

        # Add session resumption for long-running sessions
        if self._session_config.enable_session_resumption:
            session_resumption_config = types.SessionResumptionConfig()
            # If we have a previous handle, use it to resume
            if self._session_config.session_resumption_handle:
                session_resumption_config = types.SessionResumptionConfig(
                    handle=self._session_config.session_resumption_handle
                )
            config_kwargs["session_resumption"] = session_resumption_config

        config = types.LiveConnectConfig(**config_kwargs)

        return config

    async def connect(self) -> bool:
        """Connect to Gemini Live API.

        Supports:
        - Session resumption via session_resumption_handle
        - Ephemeral tokens for client-side authentication
        - Automatic reconnection with stored resumption handle
        """
        # If already connected, increment connection count and return
        if self._connected:
            self._connection_count += 1
            _LOGGER.info("Additional client connected (count=%d)", self._connection_count)
            # Emit session started for new client (handlers are per-client if registered)
            await self._emit(EVENT_SESSION_STARTED, {"connected": True, "client_count": self._connection_count})
            return True

        try:
            # Determine API version - v1alpha needed for ephemeral tokens, affective dialog, proactive audio
            api_version = "v1beta"
            if (self._session_config.enable_affective_dialog or 
                self._session_config.enable_proactive_audio or
                self._session_config.ephemeral_token):
                api_version = "v1alpha"

            # Use ephemeral token if provided, otherwise use API key
            auth_key = self._session_config.ephemeral_token or self._api_key

            # Create client in executor to avoid blocking SSL operations
            def create_client():
                return genai.Client(
                    http_options={"api_version": api_version},
                    api_key=auth_key,
                )
            
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(None, create_client)

            # Build config
            realtime_config = self._build_config()

            # Get model name (without prefix for the SDK)
            model_name = self._session_config.model

            # Check if we're resuming a session
            is_resuming = bool(self._session_config.session_resumption_handle)

            # Connect using the async context manager
            # We need to keep the context manager reference for the session
            self._session_context = self._client.aio.live.connect(
                model=model_name,
                config=realtime_config,
            )
            self._session = await self._session_context.__aenter__()

            self._connected = True

            # Install frame logger immediately to capture all frames for debugging
            try:
                await self._install_ws_frame_logger(duration=600)  # 10 minutes
            except Exception as fl_err:
                _LOGGER.debug("Failed to install early frame logger: %s", fl_err)

            # Set initial connection count for the first successful connection
            self._connection_count = 1

            # Initialize audio receive queue
            self._audio_in_queue = asyncio.Queue()

            # Start background receive task
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Emit appropriate event
            if is_resuming:
                await self._emit(EVENT_SESSION_RESUMED, {
                    "connected": True,
                    "resumed_handle": self._session_config.session_resumption_handle,
                    "client_count": self._connection_count,
                })
                _LOGGER.info("Resumed Gemini Live API session (api_version=%s)", api_version)
            else:
                await self._emit(EVENT_SESSION_STARTED, {"connected": True, "client_count": self._connection_count})
                _LOGGER.info("Connected to Gemini Live API (api_version=%s)", api_version)

            return True

        except Exception as e:
            err_str = str(e)
            _LOGGER.error("Failed to connect to Gemini Live API: %s", err_str)
            await self._emit(EVENT_ERROR, {"error": err_str})

            # If connection failed and we have tools configured, try bisecting
            # to find any problematic tool declarations that may trigger
            # policy/invalid-frame rejections. Only run once per client
            # instance to avoid loops.
            try:
                if self._session_config.tools and not getattr(self, "_bisect_attempted", False):
                    self._bisect_attempted = True
                    _LOGGER.warning("Connect failed; attempting automatic bisect of %d tools to identify problematic declarations", len(self._session_config.tools))
                    try:
                        bisect_results = await self.bisect_tools(timeout=15.0)
                        _LOGGER.info("bisect_tools results: %s", bisect_results)
                        # Emit results so frontends get actionable info
                        await self._emit(EVENT_ERROR, {"error": "tool_bisect_results", "results": bisect_results})

                        # Filter out failing tools and retry once if any succeeded
                        successful = [t for t in (self._session_config.tools or []) if bisect_results.get((t.get("name") or t.get("id") or "<unnamed>"), False)]
                        if successful:
                            _LOGGER.warning("Some tools passed bisect; retrying connect with %d tools", len(successful))
                            self._session_config.tools = successful
                            return await self.connect()
                        else:
                            _LOGGER.warning("No tools passed bisect; disabling tools to allow connection")
                            # Disable tools to allow at least a basic connection
                            self._session_config.tools = []
                            return await self.connect()
                    except Exception as be:
                        _LOGGER.debug("Automatic bisect failed: %s", be, exc_info=True)
            except Exception:
                pass

            # If we attempted to resume a session but the server reports the
            # resumption handle/session is not found (policy violation 1008),
            # clear the handle and retry once without resumption to recover.
            try:
                if is_resuming and ("1008" in err_str or "session not found" in err_str.lower() or "policy violation" in err_str.lower()) and not getattr(self, "_retry_without_resumption_done", False):
                    _LOGGER.warning("Resumption handle appears invalid; retrying connect without resumption handle")
                    # Prevent further retries
                    self._retry_without_resumption_done = True
                    # Clear any configured resumption handle and retry
                    try:
                        self._session_config.session_resumption_handle = None
                    except Exception:
                        pass
                    return await self.connect()
            except Exception:
                pass

            # If underlying socket returned an invalid-frame (1007), treat as fatal
            if "1007" in err_str or "invalid frame" in err_str.lower() or "invalid frame payload" in err_str.lower():
                _LOGGER.warning("connect: invalid-frame error detected during connection")
                # Dump recent ws frames if available to aid diagnosis
                try:
                    ws = None
                    if self._session:
                        ws = getattr(self._session, "_ws", None) or getattr(self._session, "ws", None) or getattr(self._session, "_transport", None)
                    if ws and getattr(ws, "_frame_logger_history", None):
                        _LOGGER.error("connect: recent ws frames (most recent last):")
                        for idx, entry in enumerate(list(getattr(ws, "_frame_logger_history", []))):
                            try:
                                direction, opcode, length, preview_b64, full_b64 = entry
                                prefix = full_b64
                                _LOGGER.error("  [%02d] %s opcode=%s bytes=%d preview_b64=%s full_b64_prefix=%s", idx, direction, opcode, length, preview_b64, prefix)
                            except Exception:
                                _LOGGER.error("  [%02d] frame: (uninspectable)", idx)
                except Exception:
                    pass
            return False

    async def disconnect(self) -> None:
        """Disconnect from Gemini Live API."""
        # If not connected, nothing to do
        if not self._connected:
            return

        # If multiple clients are still using the session, only decrement the count
        if self._connection_count > 1:
            self._connection_count -= 1
            _LOGGER.info("Client disconnected, remaining clients=%d", self._connection_count)
            return

        # Last client disconnecting — tear down the session
        self._connection_count = 0
        self._connected = False

        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close session context properly
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                _LOGGER.debug("Error closing session context: %s", e)

        self._session = None
        self._session_context = None
        # Reset function call processing state
        self._processing_function_call = False
        self._last_turn_complete_time = 0.0
        _LOGGER.info("Disconnected from Gemini Live API (all clients disconnected)")

    async def _receive_loop(self) -> None:
        """Background task to receive responses from the API.
        
        According to the official API documentation:
        - response.server_content.model_turn.parts contains audio/text
        - response.server_content.output_transcription for output transcription
        - response.server_content.input_transcription for input transcription
        - response.server_content.interrupted indicates interruption
        - response.server_content.turn_complete indicates turn complete
        - response.tool_call contains function calls
        """
        import uuid

        while self._connected and self._session:
            try:
                # Initialize response for this turn
                self._current_response = LiveResponse(
                    id=str(uuid.uuid4()),
                    status="in_progress",
                )
                
                async for response in self._session.receive():
                    # Log response structure for debugging
                    _LOGGER.debug("Response received: %s", type(response))
                    if hasattr(response, "server_content") and response.server_content:
                        _LOGGER.debug("server_content attrs: came in maybe",)
                    # Handle server content
                    if hasattr(response, "server_content") and response.server_content:
                        server_content = response.server_content
                        
                        # Check for interruption - this is normal, not an error
                        if hasattr(server_content, "interrupted") and server_content.interrupted:
                            # Clear audio queue on interruption
                            while not self._audio_in_queue.empty():
                                self._audio_in_queue.get_nowait()
                            # Emit as dedicated interrupted event, not as error
                            await self._emit(EVENT_INTERRUPTED, {"interrupted": True})
                            _LOGGER.debug("Received interruption signal from API")
                            continue

                        # Handle model turn (audio/text output)
                        if hasattr(server_content, "model_turn") and server_content.model_turn:
                            model_turn = server_content.model_turn
                            _LOGGER.debug("Model turn received: maybe")
                            if hasattr(model_turn, "parts") and model_turn.parts:
                                _LOGGER.debug("Model turn has %d parts", len(model_turn.parts))
                                for part in model_turn.parts:
                                    # Log what the part actually contains
                                    try:
                                        _LOGGER.debug(
                                            "Part: inline_data=%s, inline_len=%s, text=%s, function_call=%s",
                                            getattr(part, "inline_data", None) is not None,
                                            len(getattr(part, "inline_data", {}).get("data", b"")) if getattr(part, "inline_data", None) else 0,
                                            getattr(part, "text", None) is not None,
                                            getattr(part, "function_call", None) is not None,
                                        )
                                    except Exception:
                                        _LOGGER.debug("Part: unable to determine inline/text/function attributes")
                                    
                                    # Handle inline audio data - check for both inline_data and data attributes
                                    inline_data = getattr(part, "inline_data", None)
                                    if inline_data:
                                        try:
                                            mime = getattr(inline_data, "mime_type", None)
                                            data_bytes = getattr(inline_data, "data", None)
                                            _LOGGER.debug("inline_data: mime_type=%s, data_type=%s, data_len=%s", mime, type(data_bytes), len(data_bytes) if data_bytes else 0)
                                            audio_data = data_bytes
                                            if audio_data and isinstance(audio_data, (bytes, bytearray)):
                                                try:
                                                    self._audio_in_queue.put_nowait(audio_data)
                                                except Exception:
                                                    _LOGGER.debug("Audio queue put_nowait failed; queue size may be overflowing")
                                                self._current_response.audio_data += bytes(audio_data)
                                                # Emit preview + full delta asynchronously. Preview helps diagnosing frame payloads.
                                                try:
                                                    preview = base64.b64encode(bytes(audio_data[:32])).decode("utf-8")
                                                except Exception:
                                                    preview = "<preview-unavailable>"
                                                audio_b64 = base64.b64encode(bytes(audio_data)).decode("utf-8")
                                                try:
                                                    asyncio.create_task(self._emit(EVENT_AUDIO_DELTA, {"audio": audio_b64}))
                                                    _LOGGER.debug(
                                                        "Scheduled EVENT_AUDIO_DELTA emit: %d bytes preview=%s",
                                                        len(audio_data),
                                                        preview,
                                                    )
                                                except Exception as e:
                                                    _LOGGER.error("Failed to schedule audio_delta emit: %s", e)
                                        except Exception as e:
                                            _LOGGER.error("Error processing inline_data part: %s", e)

                                    # Handle text
                                    text = getattr(part, "text", None)
                                    if text:
                                        self._current_response.text += text
                                        asyncio.create_task(self._emit(EVENT_TEXT_DELTA, {"text": text}))

                        # Handle output transcription
                        if hasattr(server_content, "output_transcription") and server_content.output_transcription:
                            output_trans = server_content.output_transcription
                            transcript = getattr(output_trans, "text", "")
                            if transcript:
                                self._current_response.audio_transcript += transcript
                                asyncio.create_task(self._emit(EVENT_OUTPUT_TRANSCRIPTION, {"text": transcript}))

                        # Handle input transcription
                        if hasattr(server_content, "input_transcription") and server_content.input_transcription:
                            input_trans = server_content.input_transcription
                            transcript = getattr(input_trans, "text", "")
                            if transcript:
                                asyncio.create_task(self._emit(EVENT_INPUT_TRANSCRIPTION, {"text": transcript}))

                        # Check for turn complete
                        if hasattr(server_content, "turn_complete") and server_content.turn_complete:
                            # Record turn complete time for input timing
                            import time
                            self._last_turn_complete_time = time.monotonic()
                            # Clear function call processing flag since turn is complete
                            self._processing_function_call = False
                            
                            # Turn complete
                            if self._current_response:
                                self._current_response.status = "completed"
                                response_id = self._current_response.id
                                if response_id in self._response_futures:
                                    future = self._response_futures.pop(response_id)
                                    if not future.done():
                                        future.set_result(self._current_response)

                            # Record completed response in conversation history
                            try:
                                if self._current_response:
                                    # Keep a shallow copy to preserve the snapshot
                                    self._conversation_history.append(self._current_response)
                            except Exception:
                                pass

                            await self._emit(EVENT_TURN_COMPLETE, {
                                "transcript": self._current_response.audio_transcript if self._current_response else ""
                            })

                    # Parse usage metadata to monitor token count (comes on response, not server_content)
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        usage = response.usage_metadata
                        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                        if prompt_tokens > 0:
                            self._last_prompt_token_count = prompt_tokens
                            # Log warning if approaching context limit
                            if prompt_tokens > self._token_warning_threshold:
                                _LOGGER.warning(
                                    "TOKEN WARNING: prompt_tokens=%d exceeds warning threshold %d. "
                                    "Context compression trigger is %d. Connection may be unstable.",
                                    prompt_tokens,
                                    self._token_warning_threshold,
                                    self._session_config.trigger_tokens
                                )
                            else:
                                _LOGGER.debug("Token usage: prompt_tokens=%d", prompt_tokens)

                        # Check for generation complete (new session management feature)
                        if hasattr(server_content, "generation_complete") and server_content.generation_complete:
                            self._session_resumable = True
                            asyncio.create_task(self._emit(EVENT_GENERATION_COMPLETE, {
                                "resumable": True,
                                "has_handle": bool(self._session_resumption_handle)
                            }))
                            _LOGGER.debug("Generation complete - session is now resumable")

                    # Handle session resumption update (new session management feature)
                    if hasattr(response, "session_resumption_update") and response.session_resumption_update:
                        update = response.session_resumption_update
                        new_handle = getattr(update, "new_handle", None)
                        if new_handle:
                            self._session_resumption_handle = new_handle
                            # Check if session is resumable
                            if hasattr(update, "resumable"):
                                self._session_resumable = update.resumable
                            asyncio.create_task(self._emit(EVENT_SESSION_RESUMPTION_UPDATE, {
                                "handle": new_handle,
                                "resumable": self._session_resumable
                            }))
                            _LOGGER.debug("Session resumption handle updated (resumable=%s)", self._session_resumable)

                    # Handle GoAway message (connection termination warning)
                    if hasattr(response, "go_away") and response.go_away:
                        go_away = response.go_away
                        time_left = None
                        if hasattr(go_away, "time_left") and go_away.time_left:
                            tl = go_away.time_left
                            # Handle various time_left formats: Duration, int, str, "50s"
                            if hasattr(tl, "seconds"):
                                time_left = tl.seconds
                            elif isinstance(tl, (int, float)):
                                time_left = int(tl)
                            elif isinstance(tl, str):
                                # Try parsing formats like "50s", "50", etc.
                                try:
                                    # Remove common suffixes like 's' for seconds
                                    cleaned = tl.rstrip('smSM').strip()
                                    time_left = int(cleaned)
                                except ValueError:
                                    _LOGGER.debug("Could not parse time_left: %s", tl)
                            self._go_away_time_left = time_left
                        # Emit initial go_away event so frontends get an immediate notification.
                        # This initial event is considered to originate from the
                        # Google API server itself.
                        await self._emit(
                            EVENT_GO_AWAY,
                            {
                                "time_left": time_left,
                                "resumption_handle": self._session_resumption_handle,
                                "from_google_api": True,
                            },
                        )
                        _LOGGER.warning(
                            "Received GoAway message - connection will terminate in %s seconds",
                            time_left,
                        )

                        # Cancel any existing emitter and start a fresh one for this
                        # time_left value. This ensures that if Google starts sending
                        # periodic GoAway messages (every N seconds) we always reset
                        # the countdown to the newest value.
                        try:
                            # If there's an existing emitter task, cancel it and
                            # await its cancellation to ensure we don't have two
                            # emitters running concurrently.
                            if self._go_away_task is not None:
                                try:
                                    self._go_away_task.cancel()
                                    try:
                                        await self._go_away_task
                                    except asyncio.CancelledError:
                                        pass
                                    except Exception as cerr:
                                        _LOGGER.debug("Error awaiting previous go_away task cancellation: %s", cerr)
                                except Exception:
                                    pass
                                self._go_away_task = None

                            if isinstance(time_left, int) and time_left > 0:
                                self._go_away_time_left = time_left

                                async def _go_away_emitter():
                                    try:
                                        # Emit subsequent updates once per second.
                                        while (
                                            self._go_away_time_left is not None
                                            and self._go_away_time_left > 0
                                        ):
                                            await asyncio.sleep(1)
                                            # Decrement then emit so the UI sees the
                                            # time remaining after the elapsed second.
                                            try:
                                                self._go_away_time_left -= 1
                                            except Exception:
                                                self._go_away_time_left = None
                                            # Periodic emitter updates are generated by
                                            # this integration (not directly from the
                                            # Google API). Mark them as not originating
                                            # from Google so frontends can distinguish
                                            # initial server-originated warnings.
                                            await self._emit(
                                                EVENT_GO_AWAY,
                                                {
                                                    "time_left": self._go_away_time_left,
                                                    "resumption_handle": self._session_resumption_handle,
                                                    "from_google_api": False,
                                                },
                                            )

                                        # Ensure a final zero emit if we ended on or below 0
                                        if self._go_away_time_left is not None and self._go_away_time_left <= 0:
                                            # Final zero emit from the emitter is also
                                            # integration-generated, mark as False.
                                            await self._emit(
                                                EVENT_GO_AWAY,
                                                {
                                                    "time_left": 0,
                                                    "resumption_handle": self._session_resumption_handle,
                                                    "from_google_api": False,
                                                },
                                            )

                                        self._go_away_time_left = None
                                    except asyncio.CancelledError:
                                        # Task cancelled due to new GoAway or normal shutdown
                                        pass
                                    except Exception:
                                        _LOGGER.debug(
                                            "GoAway emitter encountered an error",
                                            exc_info=True,
                                        )

                                # Start emitter for any positive time_left value. If the
                                # server later sends another GoAway we will cancel and
                                # restart this task with the new value above.
                                self._go_away_task = asyncio.create_task(_go_away_emitter())
                        except Exception:
                            _LOGGER.debug("Failed to start go_away emitter", exc_info=True)

                    # Handle tool calls
                    if hasattr(response, "tool_call") and response.tool_call:
                        try:
                            # Log a truncated raw representation to help debugging tool payloads
                            _LOGGER.debug("Raw tool_call: %s", str(response.tool_call)[:1000])
                        except Exception:
                            pass
                        await self._handle_tool_call(response.tool_call)

                    # Handle direct data (fallback for some SDK versions)
                    # Only use this if we haven't already processed model_turn audio
                    if not hasattr(response, "server_content") or not response.server_content:
                        if hasattr(response, "data") and response.data:
                            self._audio_in_queue.put_nowait(response.data)
                            self._current_response.audio_data += response.data
                            audio_b64 = base64.b64encode(response.data).decode("utf-8")
                            asyncio.create_task(self._emit(EVENT_AUDIO_DELTA, {"audio": audio_b64}))

                        # Handle direct text (some SDK versions)
                        if hasattr(response, "text") and response.text:
                            self._current_response.text += response.text
                            self._current_response.audio_transcript += response.text
                            asyncio.create_task(self._emit(EVENT_TEXT_DELTA, {"text": response.text}))

            except asyncio.CancelledError:
                break
            except Exception as e:
                error_str = str(e)
                # Treat common close codes as clean/fatal closures and stop the loop
                # 1000 = normal closure, 1001 = going away, 1007 = invalid frame payload, 1008 = policy violation, 1011 = internal error
                close_codes = ("1000", "1001", "1007", "1008", "1011")
                if any(code in error_str for code in close_codes) or "closed" in error_str.lower():
                    # Provide specialized logging for known codes
                    if "1007" in error_str or "invalid frame" in error_str.lower():
                        _LOGGER.error("WebSocket closed due to invalid frame payload (1007): %s", error_str)
                    elif "1011" in error_str:
                        _LOGGER.info("WebSocket connection terminated (deadline expired)")
                    elif "1008" in error_str or "policy violation" in error_str.lower():
                        _LOGGER.error("Policy error in receive loop; closing session: %s", error_str)
                    else:
                        _LOGGER.info("WebSocket connection closed: %s", error_str)

                    # If we have the frame logger history available, dump the most
                    # recent frames to aid diagnosing invalid-frame (1007) errors.
                    try:
                        ws = None
                        if self._session:
                            # Try multiple attribute names where SDK stores ws
                            ws = getattr(self._session, "_ws", None) or getattr(self._session, "ws", None) or getattr(self._session, "_transport", None)
                        if ws and getattr(ws, "_frame_logger_history", None):
                            _LOGGER.error("WebSocket frame logger captured recent frames (most recent last):")
                            try:
                                for idx, entry in enumerate(list(getattr(ws, "_frame_logger_history", []))):
                                    try:
                                        direction, opcode, length, preview_b64, full_b64 = entry
                                        # Decode the preview to show text content
                                        try:
                                            preview_bytes = base64.b64decode(preview_b64)
                                            preview_text = preview_bytes.decode("utf-8", errors="replace")
                                        except Exception:
                                            preview_text = f"(decode failed) b64={preview_b64}"
                                        _LOGGER.error("  [%02d] %s opcode=%s bytes=%d text=%s", idx, direction, opcode, length, preview_text)
                                    except Exception:
                                        _LOGGER.error("  [%02d] frame: (uninspectable)", idx)
                            except Exception:
                                _LOGGER.debug("Failed iterating _frame_logger_history", exc_info=True)
                    except Exception:
                        pass

                    # Include traceback/debug info for closure events
                    try:
                        _LOGGER.debug("Receive loop close exception", exc_info=True)
                    except Exception:
                        pass

                    # Clean up session context and mark disconnected
                    self._connected = False
                    try:
                        if self._session_context:
                            try:
                                await self._session_context.__aexit__(type(e), e, getattr(e, '__traceback__', None))
                            except Exception as cerr:
                                _LOGGER.debug("Error closing session context after close: %s", cerr)
                    except Exception:
                        pass
                    self._session = None
                    self._session_context = None
                    await self._emit(EVENT_ERROR, {"error": error_str})
                    break

                # Other unexpected errors - log and emit, may retry briefly if still connected
                _LOGGER.error("Error in receive loop: %s", e)
                # Log full traceback for unexpected errors
                try:
                    _LOGGER.exception("Receive loop unexpected exception")
                except Exception:
                    pass
                await self._emit(EVENT_ERROR, {"error": error_str})
                if self._connected:
                    await asyncio.sleep(1.0)
                else:
                    break

    async def _handle_tool_call(self, tool_call: Any) -> None:
        """Handle function/tool calls from the model."""
        try:
            function_calls = getattr(tool_call, "function_calls", [])
            for function_call in function_calls:
                call_id = getattr(function_call, "id", "")
                name = getattr(function_call, "name", "")
                args = dict(function_call.args) if hasattr(function_call, "args") and function_call.args else {}

                _LOGGER.info("Function call: %s with args: %s", name, args)

                # Set flag to block audio/input during function call processing
                # This prevents 1007 errors from sending input while Gemini processes function calls
                self._processing_function_call = True

                # Determine if this function was declared as non-blocking
                non_blocking = False
                try:
                    non_blocking = self._function_behaviors.get(name, "").upper() == "NON_BLOCKING"
                except Exception:
                    non_blocking = False

                # Store pending call with behavior metadata
                self._pending_function_calls[call_id] = {
                    "name": name,
                    "args": args,
                    "non_blocking": non_blocking,
                }
                _LOGGER.debug(
                    "Stored pending function call: id=%s name=%s args=%s non_blocking=%s (audio blocked)",
                    call_id,
                    name,
                    args,
                    non_blocking,
                )


                # Emit function call event; include non_blocking flag so callers
                # can decide whether to run the function asynchronously. Include
                # a sanitized preview of the original raw declaration when available.
                event_payload = {
                    "call_id": call_id,
                    "name": name,
                    "arguments": args,
                    "non_blocking": non_blocking,
                }
                
                await self._emit(EVENT_FUNCTION_CALL, event_payload)

        except Exception as e:
            _LOGGER.error("Error handling tool call: %s", e)

    def _extract_mcp_result(self, response: Any) -> Any:
        """Extract the actual result content from an MCP JSON-RPC response.

        MCP servers return responses in JSON-RPC format like:
        {"jsonrpc": "2.0", "id": 123, "result": {"content": [...], "isError": False}}

        The text field often contains a JSON string like:
        {"success": true, "result": "Live Context: ..."}

        Gemini's Live API expects a simple result object like {"result": "text"}.
        This method extracts the meaningful content and flattens nested JSON.
        """
        if not isinstance(response, dict):
            return {"result": str(response)}

        # Check if this is a JSON-RPC response (has jsonrpc key)
        if "jsonrpc" in response:
            # Extract the result field
            result = response.get("result")
            if result is None:
                # Check for error
                error = response.get("error")
                if error:
                    error_str = str(error) if not isinstance(error, str) else error
                    return {"error": error_str}
                return {"result": "No result returned"}

            # If result has MCP content format, extract text content
            if isinstance(result, dict):
                content = result.get("content")
                _LOGGER.debug("_extract_mcp_result: result is dict, content type=%s", 
                             type(content).__name__ if content else "None")
                if isinstance(content, list):
                    # Extract text from content items
                    texts = []
                    for i, item in enumerate(content):
                        if isinstance(item, dict):
                            text_value = item.get("text", "")
                            _LOGGER.debug("_extract_mcp_result: item[%d] text_value type=%s len=%d first100=%s", 
                                         i, type(text_value).__name__, len(text_value) if text_value else 0,
                                         repr(text_value[:100]) if text_value else "empty")
                            if text_value:
                                # The text might be a JSON string - try to parse and extract
                                extracted_text = self._unwrap_json_text(text_value)
                                _LOGGER.debug("_extract_mcp_result: extracted_text len=%d first100=%s",
                                             len(extracted_text), repr(extracted_text))
                                texts.append(extracted_text)
                    if texts:
                        # Join all text content with newlines
                        joined = "\n".join(texts)
                        _LOGGER.debug("_extract_mcp_result: returning joined result len=%d", len(joined))
                        return {"result": joined}

                # Return the result as a string representation
                return {"result": str(result)}

            # Simple result value
            return {"result": str(result)}

        # Not a JSON-RPC response, convert to simple result
        return {"result": str(response)}

    def _unwrap_json_text(self, text: str) -> str:
        """Unwrap text that might be a JSON string containing the actual content.
        
        MCP servers often return text like:
        '{"success": true, "result": "Live Context: ..."}'
        
        This extracts just the text content ("Live Context: ...") for Gemini.
        """
        if not isinstance(text, str):
            _LOGGER.debug("_unwrap_json_text: input not str, type=%s", type(text).__name__)
            return str(text)
        
        _LOGGER.debug("_unwrap_json_text: input len=%d, chars: %s", len(text), text)
        
        # Try to parse as JSON
        try:
            parsed = json.loads(text)
            _LOGGER.debug("_unwrap_json_text: parsed JSON type=%s keys=%s", 
                         type(parsed).__name__, 
                         list(parsed.keys()) if isinstance(parsed, dict) else "N/A")
            
            if isinstance(parsed, dict):
                # Look for common content keys
                for key in ("result", "text", "content", "message", "data"):
                    if key in parsed:
                        value = parsed[key]
                        _LOGGER.debug("_unwrap_json_text: found key '%s', value type=%s, len=%s", 
                                     key, type(value).__name__, 
                                     len(value) if isinstance(value, str) else "N/A")
                        if isinstance(value, str):
                            return value
                        elif isinstance(value, (dict, list)):
                            # If it's still structured, convert to readable string
                            return json.dumps(value, ensure_ascii=False, indent=2)
                # If no known key, just stringify the whole thing nicely
                _LOGGER.debug("_unwrap_json_text: no known key found, stringifying dict")
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            elif isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            else:
                return str(parsed)
        except (json.JSONDecodeError, TypeError) as e:
            # Not JSON, return as-is
            _LOGGER.debug("_unwrap_json_text: failed to parse as JSON: %s", e)
            return text

    def _sanitize_for_json(self, obj: Any, _depth: int = 0) -> Any:
        """Recursively convert an object to a JSON-serializable form.

        - bytes/bytearray -> base64 string
        - dict/list/tuple -> recurse
        - other non-serializable -> str(value)

        Limits recursion depth to avoid pathological structures.
        """
        if _depth > 10:
            return str(obj)

        # Bytes -> base64 string
        if isinstance(obj, (bytes, bytearray, memoryview)):
            try:
                return base64.b64encode(bytes(obj)).decode("utf-8")
            except Exception:
                return str(obj)

        # Primitive JSON types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Mapping
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    key = k if isinstance(k, str) else str(k)
                    out[key] = self._sanitize_for_json(v, _depth + 1)
                except Exception:
                    out[str(k)] = str(v)
            return out

        # Sequence
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(v, _depth + 1) for v in obj]

        # Pydantic models / objects with dict()/json()
        try:
            if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
                try:
                    return self._sanitize_for_json(obj.dict(), _depth + 1)
                except Exception:
                    pass
        except Exception:
            pass

        # Fallback: convert to string
        try:
            return str(obj)
        except Exception:
            return repr(obj)

    def _find_function_declaration(self, name: str) -> dict | None:
        """Find a function declaration by name from the configured tools.

        Returns the raw declaration dict when available, otherwise None.
        This is a best-effort lookup used for lightweight validation only.
        """
        try:
            tools = getattr(self._session_config, "tools", None) or []
            for t in tools:
                # Support both dict-shaped tool entries and SDK Tool objects
                fdecls = None
                if isinstance(t, dict):
                    fdecls = t.get("function_declarations") or t.get("functions") or []
                else:
                    fdecls = getattr(t, "function_declarations", None) or getattr(t, "functions", None) or []

                for fd in (fdecls or []):
                    try:
                        fname = fd.get("name") if isinstance(fd, dict) else getattr(fd, "name", None)
                        if fname == name:
                            return fd if isinstance(fd, dict) else fd.__dict__
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    def _validate_response_shape(self, fdecl: dict | None, response: Any) -> tuple[bool, str]:
        """Lightweight validator: check simple type expectations from declaration.

        Returns (is_valid, reason). This intentionally avoids full JSON Schema
        validation (keeps dependency-free). It only checks common type hints
        such as top-level `type` == 'object' vs primitive/string expectations.
        """
        if not fdecl:
            return True, "no_declaration"

        try:
            # Try common locations for output/return schema keys used by MCP
            schema = None
            for key in ("returns", "return", "response", "output_schema", "output", "result", "outputSchema", "parameters"):
                if isinstance(fdecl, dict) and key in fdecl and isinstance(fdecl.get(key), dict):
                    schema = fdecl.get(key)
                    break

            # If no explicit schema found, accept by default
            if not schema:
                return True, "no_schema"

            # Look for top-level type hint
            stype = schema.get("type") if isinstance(schema, dict) else None
            if not stype:
                return True, "no_type"

            stype_norm = str(stype).lower()
            # Map simple expectations
            if stype_norm == "object":
                if isinstance(response, dict):
                    return True, "ok"
                return False, "expected_object"
            if stype_norm == "array" or stype_norm == "list":
                if isinstance(response, (list, tuple)):
                    return True, "ok"
                return False, "expected_array"
            if stype_norm in ("string", "str"):
                if isinstance(response, str):
                    return True, "ok"
                return False, "expected_string"
            # For numeric/boolean be permissive
            if stype_norm in ("number", "integer", "int", "float"):
                if isinstance(response, (int, float)):
                    return True, "ok"
                return False, "expected_number"
            if stype_norm in ("boolean", "bool"):
                if isinstance(response, bool):
                    return True, "ok"
                return False, "expected_boolean"

            return True, "unknown_type"
        except Exception as e:
            return True, f"validation_error:{e}"

    async def _install_ws_frame_logger(self, duration: int = 10) -> None:
        """Install a short-lived wrapper around the underlying websocket's
        send/recv to log raw frame bytes (opcode + preview) for debugging.

        This is best-effort and will no-op if the underlying ws object isn't
        discoverable. It restores original methods after `duration` seconds.
        """
        try:
            if not self._session:
                _LOGGER.debug("_install_ws_frame_logger: no active session")
                return

            # Try common internal attribute names used by SDK to hold the ws
            ws = getattr(self._session, "_ws", None) or getattr(self._session, "ws", None) or getattr(self._session, "_transport", None)
            if not ws:
                _LOGGER.debug("_install_ws_frame_logger: underlying websocket object not found on session")
                return

            # Avoid double-install
            if getattr(ws, "_frame_logger_installed", False):
                _LOGGER.debug("_install_ws_frame_logger: already installed")
                return

            orig_send = getattr(ws, "send", None)
            orig_recv = getattr(ws, "recv", None)
            if not callable(orig_send) or not callable(orig_recv):
                _LOGGER.debug("_install_ws_frame_logger: ws.send or ws.recv not callable")
                return

            async def send_wrapper(data, *a, **kw):
                try:
                    if isinstance(data, str):
                        b = data.encode("utf-8", errors="replace")
                        opcode = "text"
                    else:
                        b = bytes(data)
                        opcode = "binary"
                    # Capture preview and store a history of recent frames for post-mortem
                    preview_b64 = base64.b64encode(b[:512]).decode("ascii", errors="ignore")
                    _LOGGER.debug("WS_FRAME_OUT opcode=%s bytes=%d preview_b64=%s", opcode, len(b), preview_b64)
                    try:
                        # Keep a short rolling history on the ws instance for later inspection
                        hist = getattr(ws, "_frame_logger_history", None)
                        if hist is None:
                            hist = deque(maxlen=64)
                            setattr(ws, "_frame_logger_history", hist)
                        # Store both a short preview and the full base64 payload
                        full_b64 = base64.b64encode(b).decode("ascii", errors="ignore")
                        hist.append(("OUT", opcode, len(b), preview_b64, full_b64))
                    except Exception:
                        pass
                except Exception:
                    _LOGGER.debug("WS_FRAME_OUT: failed to inspect outgoing frame")
                return await orig_send(data, *a, **kw)

            async def recv_wrapper(*a, **kw):
                raw = await orig_recv(*a, **kw)
                try:
                    if isinstance(raw, (bytes, bytearray)):
                        b = bytes(raw)
                        opcode = "binary"
                    else:
                        b = str(raw).encode("utf-8", errors="replace")
                        opcode = "text"
                    preview_b64 = base64.b64encode(b[:512]).decode("ascii", errors="ignore")
                    _LOGGER.debug("WS_FRAME_IN opcode=%s bytes=%d preview_b64=%s", opcode, len(b), preview_b64)
                    try:
                        hist = getattr(ws, "_frame_logger_history", None)
                        if hist is None:
                            hist = deque(maxlen=64)
                            setattr(ws, "_frame_logger_history", hist)
                        full_b64 = base64.b64encode(b).decode("ascii", errors="ignore")
                        hist.append(("IN", opcode, len(b), preview_b64, full_b64))
                    except Exception:
                        pass
                except Exception:
                    _LOGGER.debug("WS_FRAME_IN: failed to inspect incoming frame")
                return raw

            # Patch methods on the ws object instance
            try:
                ws.send = send_wrapper
                ws.recv = recv_wrapper
                setattr(ws, "_frame_logger_installed", True)
                _LOGGER.info("_install_ws_frame_logger: installed frame logger for %ds", duration)
            except Exception as e:
                _LOGGER.debug("_install_ws_frame_logger: failed to patch ws methods: %s", e, exc_info=True)
                return

            async def _uninstall_later():
                try:
                    await asyncio.sleep(duration)
                    try:
                        ws.send = orig_send
                        ws.recv = orig_recv
                        setattr(ws, "_frame_logger_installed", False)
                        _LOGGER.info("_install_ws_frame_logger: uninstalled frame logger")
                    except Exception as e:
                        _LOGGER.debug("_install_ws_frame_logger: failed to restore ws methods: %s", e, exc_info=True)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    _LOGGER.debug("_install_ws_frame_logger: uninstall task exception", exc_info=True)

            try:
                asyncio.create_task(_uninstall_later())
            except Exception:
                _LOGGER.debug("_install_ws_frame_logger: failed to create uninstall task", exc_info=True)
        except Exception as e:
            _LOGGER.debug("_install_ws_frame_logger: unexpected error: %s", e, exc_info=True)

    def _preview_tools(self, tools: list[Any]) -> dict[str, Any]:
        """Create a compact preview summary for tools to keep logs small.

        The preview includes counts and function names and a tiny parameter
        summary (top-level parameter keys) so we can identify problematic
        declarations without dumping large schemas.
        """
        try:
            previews: list[dict[str, Any]] = []
        except Exception:
            return {"tools": []}

        for t in tools:
            try:
                # Support both SDK `types.Tool` objects and plain dict-shaped tool entries
                if isinstance(t, dict):
                    fdecls = t.get("function_declarations") or t.get("functions") or []
                else:
                    fdecls = getattr(t, "function_declarations", None) or getattr(t, "functions", None) or []

                funcs: list[dict[str, Any]] = []
                for fd in (fdecls or []):
                    try:
                        if isinstance(fd, dict):
                            name = fd.get("name") or fd.get("id") or "<unnamed>"
                            params = fd.get("parameters") or fd.get("inputSchema") or fd.get("input_schema") or {}
                        else:
                            name = getattr(fd, "name", None) or getattr(fd, "id", None) or "<unnamed>"
                            params = getattr(fd, "parameters", None) or {}
                            
                        funcs.append({
                            "name": name,
                            "parameters": params,
                        })
                    except Exception:
                        funcs.append({"name": "<error>", "parameters": {}})

                # Preserve a clean, realistic shape: return tools as a list of
                # objects each containing a `functions` list with sanitized
                # function dicts. Do not invent synthetic `tool_count`/
                # `function_count` keys — callers expect standard tool shapes.
                previews.append({
                    "functions": funcs,
                })
            except Exception:
                previews.append({"error": "preview_failed"})

        return {"tools": previews}

    def _sanitize_strings_in_obj(self, obj: Any, _depth: int = 0) -> Any:
        """Recursively sanitize strings in an object.

        - Normalize Unicode to NFC
        - Remove problematic control characters (except common whitespace)
        - Limit recursion depth to avoid pathological structures
        """
        if _depth > 12:
            return obj

        # Strings: normalize and strip control chars except \n, \r, \t
        if isinstance(obj, str):
            try:
                s = unicodedata.normalize("NFC", obj)
            except Exception:
                s = obj
            try:
                s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
            except Exception:
                pass
            return s

        # Mapping
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    key = k if isinstance(k, str) else str(k)
                    out[key] = self._sanitize_strings_in_obj(v, _depth + 1)
                except Exception:
                    out[str(k)] = str(v)
            return out

        # Sequence
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_strings_in_obj(v, _depth + 1) for v in obj]

        return obj

    async def send_text(self, text: str, turn_complete: bool = True) -> LiveResponse:
        """Send text message to the API and wait for response.
        
        Uses session.send_client_content() as per the official API documentation.
        """
        if not self._connected or not self._session:
            raise RuntimeError("Not connected to Gemini Live API")

        import uuid
        import time as time_module
        
        # Add brief delay after turn complete to avoid sending during transitional state
        # This helps prevent 1007 errors when sending immediately after model response
        if self._last_turn_complete_time > 0:
            elapsed = time_module.monotonic() - self._last_turn_complete_time
            if elapsed < self._turn_complete_delay:
                delay_needed = self._turn_complete_delay - elapsed
                _LOGGER.debug("send_text: waiting %.3fs after turn complete", delay_needed)
                await asyncio.sleep(delay_needed)
        
        # Create response future
        self._current_response = LiveResponse(
            id=str(uuid.uuid4()),
            status="in_progress",
        )
        response_future: asyncio.Future[LiveResponse] = asyncio.get_event_loop().create_future()
        self._response_futures[self._current_response.id] = response_future

        # Send text using send_client_content (official API method)
        send_client_content = cast(Callable[..., Awaitable[Any]], getattr(self._session, "send_client_content"))
        turns_payload = {"role": "user", "parts": [{"text": text}]}
        # Install frame logger to capture what's actually being sent
        try:
            await self._install_ws_frame_logger(duration=5)
        except Exception:
            pass
        
        try:
            payload_dump = json.dumps(turns_payload, ensure_ascii=False)
            payload_bytes = len(payload_dump.encode("utf-8", errors="replace"))
        except Exception:
            payload_dump = None
            payload_bytes = 0
        _LOGGER.debug("send_text: sending turns bytes=%d sample=%s", payload_bytes, payload_dump[:512] if payload_dump else "<dump-failed>")
        await send_client_content(
            turns=turns_payload,
            turn_complete=turn_complete
        )

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(response_future, timeout=60.0)
            return response
        except asyncio.TimeoutError:
            _LOGGER.error("Timeout waiting for response")
            # Return partial response
            if self._current_response:
                self._current_response.status = "timeout"
                return self._current_response
            raise

    async def send_audio(self, audio_data: bytes, sample_rate: int = 16000) -> None:
        """Send audio data to the API using send_realtime_input.
        
        Audio should be 16-bit PCM, mono. Default sample rate is 16kHz.
        """
        if not self._connected or not self._session:
            return
        
        # Block audio during function call processing to avoid 1007 errors
        # The Gemini API rejects realtime input while processing function calls
        if self._processing_function_call:
            _LOGGER.debug("send_audio: skipping - function call in progress")
            return
        
        # Validate type
        if not isinstance(audio_data, (bytes, bytearray, memoryview)):
            _LOGGER.error("send_audio: audio_data must be bytes-like, got %s", type(audio_data))
            await self._emit(EVENT_ERROR, {"error": "send_audio: invalid audio_data type"})
            return

        # Prevent sending empty payloads
        if len(audio_data) == 0:
            _LOGGER.debug("send_audio: skipping empty audio payload")
            return

        # Basic validation: 16-bit PCM should have even byte length
        if len(audio_data) % 2 != 0:
            _LOGGER.warning("send_audio: audio payload length is odd (%d) — likely not 16-bit PCM", len(audio_data))

        # Prepare Blob for fallback
        blob = types.Blob(
            data=bytes(audio_data),
            mime_type=f"audio/pcm;rate={sample_rate}",
        )

        # Debug: log a short preview of the payload to help diagnose invalid-frame issues.
        try:
            preview_b64 = base64.b64encode(bytes(audio_data[:32])).decode("utf-8")
        except Exception:
            preview_b64 = "<preview-unavailable>"
        _LOGGER.debug(
            "send_audio: preparing to send audio bytes=%d sample_rate=%s mime=%s preview=%s",
            len(audio_data),
            sample_rate,
            blob.mime_type,
            preview_b64,
        )

        try:
            _LOGGER.debug("send_audio: sending %d bytes", len(audio_data))
            # Cast the session method to an awaitable callable so static checkers
            # (Pylance) treat it correctly.
            send_realtime = cast(Callable[..., Awaitable[Any]], getattr(self._session, "send_realtime_input"))
            
            # Send raw bytes as per Google's official example:
            # await session.send_realtime_input(audio={"data": data, "mime_type": "audio/pcm"})
            # The SDK handles serialization internally.
            try:
                await send_realtime(audio={"data": bytes(audio_data), "mime_type": blob.mime_type})
                _LOGGER.debug("send_audio: send successful via raw bytes dict %d bytes", len(audio_data))
            except Exception as dict_err:
                # Fallback to Blob object if dict format is rejected
                _LOGGER.debug("send_audio: raw bytes dict failed (%s); falling back to Blob for %d bytes", dict_err, len(audio_data))
                await send_realtime(audio=blob)
                _LOGGER.debug("send_audio: send successful via Blob %d bytes", len(audio_data))
        except Exception as e:
            err_str = str(e)
            # Log error summary
            _LOGGER.error("send_audio: error sending audio (%d bytes): %s", len(audio_data), err_str)
            # Log full exception repr and traceback for deeper diagnosis
            try:
                _LOGGER.debug("send_audio: exception repr=%r type=%s", e, type(e))
                _LOGGER.exception("send_audio: full exception")
            except Exception:
                pass
            # Emit error to handlers for UI/debug
            try:
                await self._emit(EVENT_ERROR, {"error": err_str})
            except Exception:
                pass

            # If underlying socket returned an invalid-frame (1007), treat as fatal and tear down session
            if "1007" in err_str or "invalid frame" in err_str.lower() or "invalid frame payload" in err_str.lower():
                # Try a safe fallback: some SDK/server combos accept base64-encoded
                # audio in a text-friendly dict instead of raw bytes.
                # Only attempt fallback if the session object still exists and is connected.
                if not self._session or not self._connected:
                    _LOGGER.debug("send_audio: session closed before fallback could run (session=%s connected=%s)", self._session is None, self._connected)
                else:
                    try:
                        _LOGGER.debug("send_audio: attempting fallback send as base64 text to avoid invalid-frame error")
                        audio_b64 = base64.b64encode(bytes(audio_data)).decode("utf-8")
                        # Use a plain dict payload which the SDK may serialize as JSON/text.
                        send_realtime = cast(Callable[..., Awaitable[Any]], getattr(self._session, "send_realtime_input"))
                        await send_realtime(audio={"data": audio_b64, "mime_type": blob.mime_type})
                        _LOGGER.info("send_audio: fallback send successful (%d bytes encoded)", len(audio_data))
                        return
                    except Exception as fallback_exc:
                        _LOGGER.warning("send_audio: fallback send failed: %s", fallback_exc)
                        _LOGGER.warning("send_audio: fatal frame error detected, closing session")
                        # Dump recent ws frames if available to aid diagnosis
                        try:
                            ws = None
                            if self._session:
                                ws = getattr(self._session, "_ws", None) or getattr(self._session, "ws", None) or getattr(self._session, "_transport", None)
                            if ws and getattr(ws, "_frame_logger_history", None):
                                _LOGGER.error("send_audio: recent ws frames (most recent last):")
                                for idx, entry in enumerate(list(getattr(ws, "_frame_logger_history", []))):
                                    try:
                                        direction, opcode, length, preview_b64, full_b64 = entry
                                        prefix = full_b64[:1024]
                                        _LOGGER.error("  [%02d] %s opcode=%s bytes=%d preview_b64=%s full_b64_prefix=%s", idx, direction, opcode, length, preview_b64, prefix)
                                    except Exception:
                                        _LOGGER.error("  [%02d] frame: (uninspectable)", idx)
                        except Exception:
                            pass
                        self._connected = False
                        try:
                            if self._session_context:
                                await self._session_context.__aexit__(type(fallback_exc), fallback_exc, getattr(fallback_exc, '__traceback__', None))
                        except Exception:
                            _LOGGER.debug("send_audio: error closing session context after frame error", exc_info=True)
                        self._session = None
                        self._session_context = None
            # Re-raise so callers can observe failure if they want
            raise

    async def send_audio_base64(self, audio_b64: str, sample_rate: int = 16000) -> None:
        """Send base64 encoded audio to the API."""
        audio_data = base64.b64decode(audio_b64)

        # Detect common container formats (WAV) and convert to raw 16-bit PCM
        try:
            # WAV files start with the ASCII 'RIFF' or 'RIFX' header
            if len(audio_data) >= 12 and (audio_data[:4] == b'RIFF' or audio_data[:4] == b'RIFX'):
                import io
                import wave
                import audioop

                _LOGGER.debug("send_audio_base64: detected WAV container, attempting to convert to raw PCM")
                with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                    nchannels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())

                # Convert sample width to 2 bytes (16-bit) if necessary
                if sampwidth != 2:
                    try:
                        frames = audioop.lin2lin(frames, sampwidth, 2)
                        sampwidth = 2
                    except Exception as e:
                        _LOGGER.debug("Failed to convert sample width: %s", e)

                # Convert to mono if needed (mixing channels)
                if nchannels != 1:
                    try:
                        frames = audioop.tomono(frames, 2, 1, 0)
                        nchannels = 1
                    except Exception as e:
                        _LOGGER.debug("Failed to convert to mono: %s", e)

                # Resample to target sample rate if needed
                if framerate != sample_rate:
                    try:
                        frames, _ = audioop.ratecv(frames, 2, 1, framerate, sample_rate, None)
                        framerate = sample_rate
                    except Exception as e:
                        _LOGGER.debug("Failed to resample audio: %s", e)

                audio_data = frames

        except Exception as e:
            # If conversion fails, fall back to sending raw decoded bytes and log
            _LOGGER.debug("send_audio_base64: WAV detection/conversion skipped: %s", e)

        # If payload is large, send in chunks to avoid oversized websocket frames
        try:
            max_chunk = 32 * 1024
            if len(audio_data) > max_chunk:
                _LOGGER.debug("send_audio_base64: large payload %d bytes, sending in %d-byte chunks", len(audio_data), max_chunk)
                # Send slices sequentially
                offset = 0
                while offset < len(audio_data):
                    chunk = audio_data[offset: offset + max_chunk]
                    try:
                        await self.send_audio(chunk, sample_rate)
                    except Exception as e:
                        _LOGGER.error("send_audio_base64: failed sending audio chunk at offset %d: %s", offset, e)
                        raise
                    offset += max_chunk
                    # Yield briefly to let event loop process incoming frames
                    await asyncio.sleep(0)
            else:
                await self.send_audio(audio_data, sample_rate)
        except Exception as e:
            _LOGGER.error("send_audio_base64: error sending audio payload (%d bytes): %s", len(audio_data), e)
            await self._emit(EVENT_ERROR, {"error": f"send_audio failed: {e}"})

    async def send_audio_stream_end(self) -> None:
        """Signal end of audio stream.
        
        When audio stream is paused for more than a second, send this to flush cached audio.
        """
        if not self._connected or not self._session:
            return

        try:
            send_realtime = cast(Callable[..., Awaitable[Any]], getattr(self._session, "send_realtime_input"))
            await send_realtime(audio_stream_end=True)
        except Exception as e:
            _LOGGER.debug("Error sending audio stream end: %s", e)

    async def send_image(self, image_data: bytes, mime_type: str = "image/jpeg") -> None:
        """Send image data to the API using send_realtime_input."""
        if not self._connected or not self._session:
            return

        send_realtime = cast(Callable[..., Awaitable[Any]], getattr(self._session, "send_realtime_input"))
        await send_realtime(
            media=types.Blob(
                data=image_data,
                mime_type=mime_type
            )
        )

    async def send_image_base64(self, image_b64: str, mime_type: str = "image/jpeg") -> None:
        """Send base64 encoded image to the API."""
        if not self._connected or not self._session:
            return

        image_data = base64.b64decode(image_b64)
        await self.send_image(image_data, mime_type)

    async def send_function_result(
        self,
        call_id: str,
        result: dict[str, Any],
    ) -> None:
        """Send function call result back to the API.
        
        Uses session.send_tool_response() as per official documentation.
        
        Note: MCP servers return JSON-RPC responses like:
        {"jsonrpc": "2.0", "id": 123, "result": {"content": [...]}}
        
        Gemini expects a simple result object, so we extract the content.
        The response is also truncated and made ASCII-safe to avoid 1007 errors.
        """
        if not self._connected or not self._session:
            return

        # Log function response count at entry
        _LOGGER.debug(
            "send_function_result: STARTING RESPONSE #%d call_id=%s result_type=%s result_size=%d",
            self._function_response_count + 1, call_id, type(result).__name__, len(str(result))
        )

        # Minimal, robust implementation: extract MCP result, sanitize,
        # and handle carefully to avoid 1007 errors.
        func_info = self._pending_function_calls.pop(call_id, {})
        func_name = func_info.get("name", "")
        
        # Log function info for debugging
        _LOGGER.debug("send_function_result: call_id=%s func_name=%s func_info=%s", call_id, func_name, func_info)
        
        # Validate we have required info
        if not func_name:
            _LOGGER.warning("send_function_result: missing func_name for call_id=%s, this may cause API errors", call_id)

        # First, extract actual result from MCP JSON-RPC envelope if present
        try:
            extracted = self._extract_mcp_result(result)
            _LOGGER.debug("send_function_result: extracted MCP result for call_id=%s: %s", call_id, str(extracted)[:500])
        except Exception as e:
            _LOGGER.warning("send_function_result: failed to extract MCP result, using original: %s", e)
            extracted = result

        try:
            sanitized = self._sanitize_for_json(extracted)
        except Exception as e:
            _LOGGER.error("send_function_result: failed to sanitize result for call_id=%s: %s", call_id, e)
            await self._emit(EVENT_ERROR, {"error": f"failed to sanitize function result: {e}"})
            return

        # Ensure top-level dict
        if not isinstance(sanitized, dict):
            sanitized = {"text": str(sanitized)}

        _LOGGER.debug("send_function_result: payload size=%d bytes for call_id=%s", 
                     len(json.dumps(sanitized, ensure_ascii=False)), call_id)

        # Prepare a function response using the SDK types
        # Pass the sanitized dict directly - the SDK should handle JSON serialization
        function_response = types.FunctionResponse(id=call_id, name=func_name, response=sanitized)

        send_tool = cast(Callable[..., Awaitable[Any]], getattr(self._session, "send_tool_response"))

        try:
            await self._install_ws_frame_logger(duration=8)
        except Exception:
            pass

        # Log the actual payload being sent for debugging
        try:
            payload_preview = json.dumps(sanitized, ensure_ascii=False, default=str)[:1000]
            _LOGGER.info("send_function_result: sending response call_id=%s name=%s payload_preview=%s", 
                        call_id, func_name, payload_preview)
        except Exception:
            _LOGGER.info("send_function_result: sending response call_id=%s name=%s (payload preview failed)", 
                        call_id, func_name)

        try:
            await send_tool(function_responses=[function_response])
            self._function_response_count += 1
            _LOGGER.debug(
                "send_function_result: RESPONSE #%d sent successfully call_id=%s func_name=%s",
                self._function_response_count, call_id, func_name
            )
            try:
                self._last_sent_was_function_response = True
            except Exception:
                pass
            return
        except Exception as e:
            err_str = str(e)
            _LOGGER.error("Error sending function result for call_id=%s: %s", call_id, e)
            _LOGGER.exception("send_function_result: full exception")
            
            # Try even more aggressive ASCII/text fallback for invalid-frame-like errors
            if "1007" in err_str or "invalid frame" in err_str.lower() or "invalid frame payload" in err_str.lower() or "invalid argument" in err_str.lower():
                try:
                    # Use a very conservative payload with ASCII-only content
                    ascii_dump = json.dumps(sanitized, ensure_ascii=True)
                    # Truncate further for fallback
                    if len(ascii_dump) > 2000:
                        ascii_dump = ascii_dump[:2000] + "...[truncated]"
                    fallback_payload = {"text": ascii_dump}
                    fallback_response = types.FunctionResponse(id=call_id, name=func_name, response=fallback_payload)
                    _LOGGER.info("send_function_result: attempting ASCII fallback for call_id=%s", call_id)
                    await send_tool(function_responses=[fallback_response])
                    try:
                        self._last_sent_was_function_response = True
                    except Exception:
                        pass
                    return
                except Exception as fallback_err:
                    _LOGGER.warning("send_function_result: ascii fallback failed for call_id=%s: %s", call_id, fallback_err)

            # If fallback failed or not applicable, close session on fatal frame errors
            if "1007" in err_str or "invalid frame" in err_str.lower() or "invalid frame payload" in err_str.lower():
                _LOGGER.warning("send_function_result: invalid-frame error detected, closing session (call_id=%s)", call_id)
                self._connected = False
                try:
                    if self._session_context:
                        await self._session_context.__aexit__(type(e), e, getattr(e, '__traceback__', None))
                except Exception:
                    _LOGGER.debug("send_function_result: error closing session context after frame error", exc_info=True)
                self._session = None
                self._session_context = None
            await self._emit(EVENT_ERROR, {"error": err_str})

    async def get_audio_chunk(self, timeout: float = 0.1) -> bytes | None:
        """Get an audio chunk from the receive queue."""
        if not self._audio_in_queue:
            return None

        try:
            return await asyncio.wait_for(
                self._audio_in_queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    def add_tool(self, tool: dict[str, Any]) -> None:
        """Add a function tool to the session."""
        _LOGGER.debug("Adding tool: %s", tool)
        self._session_config.tools.append(tool)

    async def bisect_tools(self, timeout: float = 20.0) -> dict[str, bool]:
        """Test configured tools one-by-one to find problematic declarations.

        This helper will temporarily replace the session tools with a single
        tool and attempt to `connect()` using the current session config.
        It records whether the connect succeeded for each tool. The original
        `self._session_config.tools` list is restored at the end.

        Returns a mapping of tool name -> bool (True if connect succeeded).
        """
        results: dict[str, bool] = {}

        orig_tools = list(self._session_config.tools)

        try:
            for tool in orig_tools:
                name = tool.get("name") or tool.get("id") or "<unnamed>"
                _LOGGER.info("bisect_tools: testing tool '%s'", name)
                # Replace tools with single candidate
                self._session_config.tools = [tool]

                try:
                    # Attempt to connect with a timeout
                    ok = await asyncio.wait_for(self.connect(), timeout=timeout)
                    if ok:
                        results[name] = True
                        # Immediately disconnect to reset server-side state
                        try:
                            await self.disconnect()
                        except Exception:
                            pass
                    else:
                        results[name] = False
                except Exception as e:
                    _LOGGER.debug("bisect_tools: connect for tool '%s' failed: %s", name, e, exc_info=True)
                    results[name] = False
                    try:
                        await self.disconnect()
                    except Exception:
                        pass

            return results
        finally:
            # Restore original tool list regardless of outcome
            self._session_config.tools = orig_tools

    def _preview_tool_dict(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Create a small preview from a tool dict (from config) for logging.

        Avoids dumping large parameter schemas but shows enough to identify
        the tool and top-level parameter keys.
        """
        try:
            name = tool.get("name") or tool.get("id") or tool.get("function_declarations") and "<tool-with-fdecl>" or "<unnamed>"
        except Exception:
            name = "<unnamed>"

        out: dict[str, Any] = {"name": name}

        try:
            if "function_declarations" in tool and isinstance(tool.get("function_declarations"), (list, tuple)):
                funcs = []
                for fd in tool.get("function_declarations", [])[:8]:
                    try:
                        fn = fd.get("name") if isinstance(fd, dict) else getattr(fd, "name", "<unnamed>")
                        params = fd.get("parameters") if isinstance(fd, dict) else getattr(fd, "parameters", None)
                        pkeys = list(params.get("properties", {}).keys())[:8] if isinstance(params, dict) and isinstance(params.get("properties", {}), dict) else []
                        funcs.append({"name": fn, "param_keys": pkeys})
                    except Exception:
                        funcs.append({"name": "<error>", "param_keys": []})
                out["function_declarations_preview"] = funcs
            else:
                # show top-level keys for non-fdecl tool dicts like {'google_search': {}}
                top_keys = list(tool.keys())[:8]
                out["keys"] = top_keys
        except Exception:
            pass

        return out

    async def bisect_tools_and_report(self, timeout: float = 20.0) -> dict[str, bool]:
        """Run bisect_tools() then log and emit detailed previews for failing tools.

        Returns the same mapping as `bisect_tools`.
        """
        try:
            results = await self.bisect_tools(timeout=timeout)
        except Exception as e:
            _LOGGER.error("bisect_tools_and_report: bisect run failed: %s", e, exc_info=True)
            await self._emit(EVENT_ERROR, {"error": "bisect_failed", "reason": str(e)})
            return {}

        # Emit results and detailed previews for failing tools to help debugging
        try:
            failing = [name for name, ok in results.items() if not ok]
            detailed: dict[str, Any] = {"failing": failing, "results": results}
            # Map failing names back to original tool dicts when possible
            previews: dict[str, Any] = {}
            for t in (self._session_config.tools or []):
                name = t.get("name") or t.get("id") or "<unnamed>"
                if name in failing:
                    previews[name] = self._preview_tool_dict(t)

            if previews:
                detailed["previews"] = previews

            _LOGGER.info("bisect_tools_and_report: detailed results=%s", detailed)
            await self._emit(EVENT_ERROR, {"error": "tool_bisect_detailed", "details": detailed})
        except Exception as e:
            _LOGGER.debug("bisect_tools_and_report: failed to prepare detailed report: %s", e, exc_info=True)

        return results

    def get_conversation_history(self) -> list[ConversationItem]:
        """Get the conversation history."""
        try:
            items: list[ConversationItem] = []
            for resp in self._conversation_history:
                parts = []
                if resp.text:
                    parts.append({"text": resp.text})
                if resp.audio_transcript and resp.audio_transcript != resp.text:
                    parts.append({"audio_transcript": resp.audio_transcript})
                items.append(ConversationItem(id=resp.id, role="assistant", parts=parts, status=resp.status))
            return items
        except Exception:
            return []

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self._current_response = None
        self._pending_function_calls.clear()
        self._processing_function_call = False
        self._last_turn_complete_time = 0.0

    def update_config(self, **kwargs) -> None:
        """Update session configuration."""
        for key, value in kwargs.items():
            if hasattr(self._session_config, key):
                setattr(self._session_config, key, value)

    # Session Management Methods

    def get_resumption_handle(self) -> str | None:
        """Get the current session resumption handle.
        
        This handle can be stored and used to resume the session
        after a disconnection. The handle should be saved when
        EVENT_SESSION_RESUMPTION_UPDATE is emitted.
        
        Returns:
            The resumption handle string, or None if not available.
        """
        return self._session_resumption_handle

    def is_session_resumable(self) -> bool:
        """Check if the current session is resumable.
        
        A session becomes resumable when generationComplete is True
        in the server content response.
        
        Returns:
            True if the session can be resumed, False otherwise.
        """
        return self._session_resumable

    def set_resumption_handle(self, handle: str | None) -> None:
        """Set the resumption handle for the next connection.
        
        Call this before connect() to resume a previous session.
        
        Args:
            handle: The resumption handle from a previous session.
        """
        self._session_config.session_resumption_handle = handle

    def get_go_away_time_left(self) -> int | None:
        """Get the time left before connection termination (if GoAway received).
        
        Returns:
            Seconds until termination, or None if no GoAway received.
        """
        return self._go_away_time_left

    def set_owner_ws(self, owner_ws: int | None) -> None:
        """Set the owning frontend websocket id for this client.

        This can be used by the websocket layer to tag a per-connection
        client with the frontend `id(connection)` that owns it. It is
        primarily used for routing/debugging purposes.
        """
        try:
            self._owner_ws = owner_ws
        except Exception:
            self._owner_ws = None

    def get_ws_id(self, user_ws: int | None = None) -> int | None:
        """Return this client's websocket id (object id) when queried.

        If `user_ws` is provided the method will return the client's id
        only if it matches the configured owner websocket (when set).
        When `user_ws` is None this returns the client object id.
        """
        try:
            if user_ws is None:
                return id(self)
            # If owner_ws is not set, we cannot verify ownership; return id
            if self._owner_ws is None:
                return id(self)
            return id(self) if user_ws == self._owner_ws else None
        except Exception:
            return None

    @staticmethod
    async def create_ephemeral_token(
        api_key: str,
        expire_time_seconds: int = 300,
    ) -> dict[str, Any] | None:
        """Create an ephemeral token for client-side authentication.
        
        Ephemeral tokens are short-lived tokens that can be safely used
        in client applications without exposing the main API key.
        
        Note: This requires the v1alpha API endpoint.
        
        Args:
            api_key: The Gemini API key to create the token with.
            expire_time_seconds: Token lifetime in seconds (default 300 = 5 minutes).
        
        Returns:
            Dictionary containing the token info, or None on error.
            {
                "name": "token_name",
                "displayName": "display_name",
                "expireTime": "2024-12-10T00:00:00Z"
            }
        """
        try:
            client = genai.Client(
                http_options={"api_version": "v1alpha"},
                api_key=api_key,
            )
            
            from datetime import datetime, timedelta
            
            expire_time = datetime.utcnow() + timedelta(seconds=expire_time_seconds)
            
            token = await client.aio.auth_tokens.create(
                config={
                    "uses": 1,  # Single use
                    "expire_time": expire_time.isoformat() + "Z",
                }
            )
            
            return {
                "name": getattr(token, "name", ""),
                "display_name": getattr(token, "display_name", ""),
                "expire_time": expire_time.isoformat() + "Z",
            }
            
        except Exception as e:
            _LOGGER.error("Failed to create ephemeral token: %s", e)
            return None
