"""Constants for the Gemini Live integration."""
from typing import Final

DOMAIN: Final = "gemini_live"

# Configuration keys
CONF_API_KEY: Final = "api_key"
CONF_MODEL: Final = "model"
CONF_VOICE: Final = "voice"
CONF_INSTRUCTIONS: Final = "instructions"
CONF_MCP_SERVERS: Final = "mcp_servers"
CONF_MCP_SERVER_URL: Final = "url"
CONF_MCP_SERVER_NAME: Final = "name"
CONF_MCP_SERVER_TOKEN: Final = "token"
CONF_MCP_SERVER_TYPE: Final = "server_type"
CONF_MCP_SERVER_COMMAND: Final = "command"
CONF_MCP_SERVER_ARGS: Final = "args"
CONF_MCP_SERVER_ENV: Final = "env"
CONF_MCP_SERVER_ENABLED: Final = "enabled"
CONF_MCP_SERVER_TIMEOUT: Final = "timeout"
CONF_TEMPERATURE: Final = "temperature"
CONF_ENABLE_AFFECTIVE_DIALOG: Final = "enable_affective_dialog"
CONF_ENABLE_PROACTIVE_AUDIO: Final = "enable_proactive_audio"
CONF_ENABLE_GOOGLE_SEARCH: Final = "enable_google_search"
CONF_ENABLE_PERSONALIZATION: Final = "enable_personalization"
CONF_MEDIA_RESOLUTION: Final = "media_resolution"

# MCP Server Types
MCP_SERVER_TYPE_SSE: Final = "sse"
MCP_SERVER_TYPE_STDIO: Final = "stdio"
MCP_SERVER_TYPES: Final = [
    MCP_SERVER_TYPE_SSE,
    MCP_SERVER_TYPE_STDIO,
]

# Default values
DEFAULT_MODEL: Final = "gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_VOICE: Final = "Kore"
DEFAULT_INSTRUCTIONS: Final = "I want you to act as smart home manager of Home Assistant.\nI will provide information of smart home along with a question, you will truthfully make correction or answer using information provided in one sentence in everyday language.\nUse available tools to control or check anything."
DEFAULT_TEMPERATURE: Final = 0.7
DEFAULT_MEDIA_RESOLUTION: Final = "MEDIA_RESOLUTION_MEDIUM"

# Available voices (from Gemini TTS)
# See: https://ai.google.dev/gemini-api/docs/speech-generation#voices
VOICES: Final = [
    "Kore",
    "Charon",
    "Fenrir",
    "Aoede",
    "Puck",
    "Zephyr",
    "Orus",
    "Leda",
]

# Media resolution options
MEDIA_RESOLUTIONS: Final = [
    "MEDIA_RESOLUTION_LOW",
    "MEDIA_RESOLUTION_MEDIUM",
    "MEDIA_RESOLUTION_HIGH",
]

# Available models
MODELS: Final = [
    "gemini-2.5-flash-native-audio-preview-12-2025"
]

# Audio settings
# Input: 16kHz, 16-bit PCM
# Output: 24kHz, 16-bit PCM
AUDIO_INPUT_SAMPLE_RATE: Final = 16000
AUDIO_OUTPUT_SAMPLE_RATE: Final = 24000
AUDIO_FORMAT: Final = "audio/pcm"
AUDIO_INPUT_MIME_TYPE: Final = "audio/pcm;rate=16000"

# Event types - Used internally
EVENT_SESSION_STARTED: Final = "session_started"
EVENT_SESSION_RESUMED: Final = "session_resumed"
EVENT_SESSION_RESUMPTION_UPDATE: Final = "session_resumption_update"
EVENT_GO_AWAY: Final = "go_away"
EVENT_GENERATION_COMPLETE: Final = "generation_complete"
EVENT_AUDIO_DELTA: Final = "audio_delta"
EVENT_TEXT_DELTA: Final = "text_delta"
EVENT_FUNCTION_CALL: Final = "function_call"
EVENT_TOOL_CALL: Final = "tool_call"
EVENT_TURN_COMPLETE: Final = "turn_complete"
EVENT_INPUT_TRANSCRIPTION: Final = "input_transcription"
EVENT_OUTPUT_TRANSCRIPTION: Final = "output_transcription"
EVENT_INTERRUPTED: Final = "interrupted"
EVENT_ERROR: Final = "error"

# Roles
ROLE_USER: Final = "user"
ROLE_MODEL: Final = "model"

# Response status
STATUS_IN_PROGRESS: Final = "in_progress"
STATUS_COMPLETED: Final = "completed"
STATUS_CANCELLED: Final = "cancelled"
STATUS_FAILED: Final = "failed"
