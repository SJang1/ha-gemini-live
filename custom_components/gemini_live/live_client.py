"""Gemini Live API client using google-genai library.

This module provides a client for Google's Gemini Live API,
supporting real-time audio, video (camera/screen), function calling,
and Google Search.

Based on official documentation:
"""
from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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

        # Current response tracking
        self._current_response: LiveResponse | None = None
        self._response_futures: dict[str, asyncio.Future] = {}

        # Session resumption tracking
        self._session_resumption_handle: str | None = None
        self._session_resumable: bool = False

        # GoAway tracking
        self._go_away_time_left: int | None = None
        # Number of frontend/clients currently using this live session
        self._connection_count: int = 0

    @property
    def connected(self) -> bool:
        """Return connection status."""
        return self._connected

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
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
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
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

        # Add Google Search if enabled
        if self._session_config.enable_google_search:
            tools_list.append(types.Tool(google_search=types.GoogleSearch()))

        # Add function declarations from config
        if self._session_config.tools:
            function_declarations = []
            for tool in self._session_config.tools:
                # Convert our tool format to google-genai format
                func_decl = types.FunctionDeclaration(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    parameters=self._convert_parameters(tool.get("parameters", {})),
                )
                function_declarations.append(func_decl)

            if function_declarations:
                tools_list.append(types.Tool(function_declarations=function_declarations))

        return tools_list

    def _convert_parameters(self, params: dict) -> types.Schema | None:
        """Convert parameter schema to google-genai Schema format."""
        if not params:
            return None

        param_type = params.get("type", "object").upper()

        properties = {}
        for prop_name, prop_def in params.get("properties", {}).items():
            prop_type = prop_def.get("type", "string").upper()
            properties[prop_name] = types.Schema(
                type=getattr(types.Type, prop_type, types.Type.STRING),
                description=prop_def.get("description", ""),
            )

        return types.Schema(
            type=getattr(types.Type, param_type, types.Type.OBJECT),
            properties=properties,
            required=params.get("required", []),
        )

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
            _LOGGER.error("Failed to connect to Gemini Live API: %s", e)
            await self._emit(EVENT_ERROR, {"error": str(e)})
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
                        _LOGGER.debug("server_content attrs: %s", dir(response.server_content))
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
                            _LOGGER.debug("Model turn received: %s", type(model_turn))
                            if hasattr(model_turn, "parts") and model_turn.parts:
                                _LOGGER.debug("Model turn has %d parts", len(model_turn.parts))
                                for part in model_turn.parts:
                                    # Log what the part actually contains
                                    _LOGGER.debug("Part: inline_data=%s, text=%s, function_call=%s",
                                                 getattr(part, "inline_data", None) is not None,
                                                 getattr(part, "text", None) is not None,
                                                 getattr(part, "function_call", None) is not None)
                                    
                                    # Handle inline audio data - check for both inline_data and data attributes
                                    inline_data = getattr(part, "inline_data", None)
                                    if inline_data:
                                        _LOGGER.debug("inline_data: mime_type=%s, data_len=%s", 
                                                     getattr(inline_data, "mime_type", None),
                                                     len(getattr(inline_data, "data", b"")))
                                        audio_data = getattr(inline_data, "data", None)
                                        if audio_data and isinstance(audio_data, bytes):
                                            self._audio_in_queue.put_nowait(audio_data)
                                            self._current_response.audio_data += audio_data
                                            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                                            # Don't await here to avoid blocking the loop
                                            asyncio.create_task(self._emit(EVENT_AUDIO_DELTA, {"audio": audio_b64}))
                                            _LOGGER.debug("Emitted audio_delta: %d bytes", len(audio_data))

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
                            # Turn complete
                            if self._current_response:
                                self._current_response.status = "completed"
                                response_id = self._current_response.id
                                if response_id in self._response_futures:
                                    future = self._response_futures.pop(response_id)
                                    if not future.done():
                                        future.set_result(self._current_response)

                            await self._emit(EVENT_TURN_COMPLETE, {
                                "transcript": self._current_response.audio_transcript if self._current_response else ""
                            })

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
                        await self._emit(EVENT_GO_AWAY, {
                            "time_left": time_left,
                            "resumption_handle": self._session_resumption_handle
                        })
                        _LOGGER.warning("Received GoAway message - connection will terminate in %s seconds", time_left)

                    # Handle tool calls
                    if hasattr(response, "tool_call") and response.tool_call:
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
                # Check if this is a WebSocket close that should stop the loop
                # 1000 = normal closure, 1011 = internal error (deadline expired)
                if "1000" in error_str or "1011" in error_str or "closed" in error_str.lower():
                    if "1011" in error_str:
                        _LOGGER.info("WebSocket connection terminated (deadline expired)")
                    else:
                        _LOGGER.info("WebSocket connection closed normally")
                    self._connected = False
                    await self._emit(EVENT_ERROR, {"error": "Connection closed"})
                    break  # Exit the loop, don't retry
                else:
                    _LOGGER.error("Error in receive loop: %s", e)
                    await self._emit(EVENT_ERROR, {"error": error_str})
                    # Only retry for unexpected errors, with exponential backoff
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

                # Store pending call
                self._pending_function_calls[call_id] = {
                    "name": name,
                    "args": args,
                }

                await self._emit(
                    EVENT_FUNCTION_CALL,
                    {
                        "call_id": call_id,
                        "name": name,
                        "arguments": args,
                    },
                )

        except Exception as e:
            _LOGGER.error("Error handling tool call: %s", e)

    async def send_text(self, text: str, turn_complete: bool = True) -> LiveResponse:
        """Send text message to the API and wait for response.
        
        Uses session.send_client_content() as per the official API documentation.
        """
        if not self._connected or not self._session:
            raise RuntimeError("Not connected to Gemini Live API")

        import uuid
        
        # Create response future
        self._current_response = LiveResponse(
            id=str(uuid.uuid4()),
            status="in_progress",
        )
        response_future: asyncio.Future[LiveResponse] = asyncio.get_event_loop().create_future()
        self._response_futures[self._current_response.id] = response_future

        # Send text using send_client_content (official API method)
        await self._session.send_client_content(
            turns={"role": "user", "parts": [{"text": text}]},
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

        # Send audio using send_realtime_input (official API method)
        await self._session.send_realtime_input(
            audio=types.Blob(
                data=audio_data,
                mime_type=f"audio/pcm;rate={sample_rate}"
            )
        )

    async def send_audio_base64(self, audio_b64: str, sample_rate: int = 16000) -> None:
        """Send base64 encoded audio to the API."""
        audio_data = base64.b64decode(audio_b64)
        await self.send_audio(audio_data, sample_rate)

    async def send_audio_stream_end(self) -> None:
        """Signal end of audio stream.
        
        When audio stream is paused for more than a second, send this to flush cached audio.
        """
        if not self._connected or not self._session:
            return

        try:
            await self._session.send_realtime_input(audio_stream_end=True)
        except Exception as e:
            _LOGGER.debug("Error sending audio stream end: %s", e)

    async def send_image(self, image_data: bytes, mime_type: str = "image/jpeg") -> None:
        """Send image data to the API using send_realtime_input."""
        if not self._connected or not self._session:
            return

        await self._session.send_realtime_input(
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
        """
        if not self._connected or not self._session:
            return

        try:
            # Get the function name from pending calls
            func_info = self._pending_function_calls.pop(call_id, {})
            func_name = func_info.get("name", "")

            # Create function response
            function_response = types.FunctionResponse(
                id=call_id,
                name=func_name,
                response=result,
            )

            # Send the response using send_tool_response (official API method)
            await self._session.send_tool_response(
                function_responses=[function_response]
            )
            _LOGGER.info("Sent function result for call_id=%s", call_id)

        except Exception as e:
            _LOGGER.error("Error sending function result: %s", e)

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
        self._session_config.tools.append(tool)

    def get_conversation_history(self) -> list[ConversationItem]:
        """Get the conversation history."""
        return []

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self._current_response = None
        self._pending_function_calls.clear()

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
