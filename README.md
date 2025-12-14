# Realtime AI Audio for Home Assistant

A Home Assistant custom component that integrates with **OpenAI's Realtime API** and **Google's Gemini Live API** for real-time voice and text conversations, with MCP (Model Context Protocol) server support.

## 🎯 Included Integrations

| Integration | API | Voice Model |
|-------------|-----|-------------|
| **OpenAI Realtime** | OpenAI Realtime API | GPT-4o Realtime |
| **Gemini Live** | Google Gemini Live API | Gemini 2.0 Flash |

Both integrations provide native speech-to-speech capabilities with minimal latency.

## Features

### Common Features (Both Integrations)
- **Real-time Conversations**: WebSocket-based low-latency responses
- **Native Speech-to-Speech**: Direct audio processing without separate STT/TTS pipeline
- **Voice Support**: Multiple voice options with configurable settings
- **Home Assistant Integration**: Built-in tools for controlling smart home devices
- **Conversation Agent**: Works as a Home Assistant conversation agent
- **Media Player Entity**: Control audio input/output directly
- **Binary Sensors**: Monitor connection, listening, speaking, and processing states
- **Custom Lovelace Card**: Browser-based microphone with real-time visualizer

### OpenAI Realtime Specific
- **MCP Server Integration**: Connect to external MCP servers for extended tool capabilities
- **Custom STT/TTS Providers**: Use Realtime API for speech recognition and synthesis

### Gemini Live Specific
- **Session Resumption**: Automatic session recovery on disconnection
- **Image/Audio File Input**: Send images and audio files for multimodal conversations
- **Google Search Integration**: Built-in Google Search tool

## Privacy & Personalization

This integration offers an optional "personalization" feature that can improve and tailor AI responses by using conversation content to adapt behavior over time. For privacy reasons, personalization is disabled by default.

- **What enabling personalization does:** The integration may send additional conversation content or metadata to the external AI service to allow it to provide more personalized responses.
- **Default:** Off. You must explicitly enable it during configuration and confirm that you accept the privacy implications.
- **Recommendation:** Keep personalization disabled unless you understand and accept the data handling implications and trust the service provider.

If you enable personalization, review your service provider's privacy policy and data retention practices.

## Architecture

Unlike the default Home Assistant voice pipeline (STT → AI → TTS), these integrations use native speech-to-speech APIs:

```
┌───────────────────────────────────────────────────────────┐
│ Default HA Pipeline                                       │
│ ┌─────┐    ┌────────────┐    ┌─────┐    ┌─────┐           │
│ │ Mic │───▶│ STT Engine │───▶│ AI  │───▶│ TTS │───▶🔊     │
│ └─────┘    └────────────┘    └─────┘    └─────┘           │
└───────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ OpenAI / Gemini Live Pipeline                        │
│ ┌─────┐    ┌───────────────────────────────┐            │
│ │ Mic │───▶│ Realtime API                  │───▶🔊      │
│ └─────┘    │ (Native Speech-to-Speech)     │            │
│            └───────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

## Requirements

- Home Assistant 2024.1.0 or later
- **For OpenAI Realtime**: OpenAI API key with access to the Realtime API
- **For Gemini Live**: Google AI API key (Gemini API)
- Python 3.11 or later

NOTE: Detailed configuration and platform-specific documentation has been moved into each integration folder. See:

- [custom_components/openai_realtime](custom_components/openai_realtime) — OpenAI Realtime integration files and component-specific notes.
- [custom_components/gemini_live/GOOGLE_DOC.md](custom_components/gemini_live/GOOGLE_DOC.md) — Gemini Live specific guidance and examples.

The sections below provide a concise quick-config summary for each integration. For full details and advanced options, open the files in the corresponding component folders above.

## Quick Configuration (Minimal examples)

Below are compact configuration snippets and key settings you may need when setting up each integration. These are meant as a quick reference — see the component folders for extended examples and edge-case options.

### OpenAI Realtime - Quick Config

Core options exposed in the integration UI or via YAML when applicable:

- `api_key`: Your OpenAI API key with Realtime access
- `model`: Realtime model (example: `gpt-4o-realtime-preview`)
- `voice`: Choose a voice (example: `alloy`)
- `temperature`: Float 0.0–2.0
- `mcp_servers`: List of MCP server configs (SSE or Stdio)

Example minimal YAML for an MCP server entry (SSE):

```yaml
# OpenAI Realtime MCP server example
- name: homeassistant
   url: http://localhost:8123/api/mcp
   type: sse
   token: YOUR_LONG_LIVED_ACCESS_TOKEN
```

When using the integration UI, supply your `api_key` and configure model/voice/temperature there. Add MCP servers through the integration options.

### Gemini Live - Quick Config

Core options exposed in the integration UI or via YAML when applicable:

- `api_key` / `google_api_key`: Your Google AI key for Gemini
- `model`: Gemini model (example: `gemini-2.0-flash-exp`)
- `voice`: Voice name (example: `Puck`)
- `ephemeral_token` (optional): Use for client-side auth
- `enable_session_resumption`: true/false
- `enable_affective_dialog`: true/false (v1alpha)
- `enable_proactive_audio`: true/false (v1alpha)

Example minimal settings (UI-oriented):

```yaml
# Gemini Live basic settings (example representation)
model: gemini-2.0-flash-exp
voice: Puck
enable_session_resumption: true
# optional: ephemeral_token: xxxxx
```

For advanced features (session resumption handles, proactive audio, image inputs), open the Gemini docs in the component folder: [custom_components/gemini_live/GOOGLE_DOC.md](custom_components/gemini_live/GOOGLE_DOC.md)


## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant
2. Click on "Integrations"
3. Click the three dots in the top right corner
4. Select "Custom repositories"
5. Add this repository URL: `https://github.com/your-username/ha-realtime-ai-audio`
6. Install "Realtime AI Audio for Home Assistant"
7. Restart Home Assistant

### Manual Installation

1. Download the repository
2. Copy both folders to your Home Assistant `custom_components` directory:
   - `custom_components/openai_realtime` - For OpenAI integration
   - `custom_components/gemini_live_audio` - For Gemini integration
3. Restart Home Assistant

---

# 🔵 OpenAI Realtime Integration

## Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "OpenAI Realtime"
3. Enter your OpenAI API key
4. Configure the settings:
   - **Model**: Select the Realtime model (default: `gpt-4o-realtime-preview`)
   - **Voice**: Choose the voice for audio responses
   - **Instructions**: Custom system instructions
   - **Temperature**: Response creativity (0.0 - 2.0)
   - **Max Output Tokens**: Maximum response length
5. Optionally add MCP servers for extended functionality

## MCP Server Configuration

MCP (Model Context Protocol) servers allow you to extend the AI's capabilities with external tools. This integration supports two types of MCP servers:

### MCP Server Types

| Type | Description | Use Case |
|------|-------------|----------|
| **SSE** | HTTP-based Server-Sent Events | Remote servers, cloud-hosted MCP services |
| **Stdio** | Local subprocess communication | Local tools, CLI-based MCP servers |

### SSE Servers (Recommended for HASSIO)

SSE servers communicate over HTTP/HTTPS and are passed directly to OpenAI's Realtime API. This is the recommended approach for Home Assistant OS (HASSIO) installations.

To add an SSE server:
1. Go to integration options → **Add SSE Server**
2. Configure:
   - **Server Name**: A unique identifier (letters, numbers, underscores, hyphens only)
   - **Server URL**: The HTTP/HTTPS endpoint (e.g., `http://localhost:8123/api/mcp`)
   - **Token** (optional): Authentication token if required

### Stdio Servers

Stdio servers run as local subprocesses and communicate via stdin/stdout. The integration connects to these servers locally and registers their tools as function calls.

> ⚠️ **Important: Stdio servers require the command to be available on the Home Assistant host system.**

To add a Stdio server:
1. Go to integration options → **Add Stdio Server**
2. Configure:
   - **Server Name**: A unique identifier
   - **Command**: The executable to run (e.g., `python`, `node`, `/usr/bin/my-mcp-server`)
   - **Arguments**: Comma-separated arguments (e.g., `-m,mcp_server,--port,3000`)
   - **Environment Variables**: Comma-separated key=value pairs (e.g., `API_KEY=xxx,DEBUG=true`)

### ⚠️ HASSIO / Home Assistant OS Limitations

**Node.js (`npx`, `node`) commands will NOT work on Home Assistant OS (HASSIO)** because:
- HASSIO is a minimal, containerized Linux environment
- Node.js is not pre-installed and cannot be easily added
- The host OS is read-only and doesn't support package installation

#### Workarounds for HASSIO Users:

1. **Use SSE mode instead of Stdio** (Recommended)
   
   Many MCP servers support both modes. Run the server on a separate machine with Node.js and connect via SSE:
   ```bash
   # On a machine with Node.js (not HASSIO)
   npx @anthropic/mcp-server-brightdata --transport sse --port 3000
   ```
   Then configure as an SSE server with URL `http://your-server-ip:3000/sse`

2. **Use `uvx` with Python-based MCP servers** ✅ **Works in HASSIO!**
   
   If `uv`/`uvx` is not already installed on your Home Assistant OS, you can use an add-on to install it on every boot:
   
   👉 **[ha-uv Add-on](https://github.com/SJang1/ha-uv)** - Installs uv/uvx on Home Assistant OS
   
   Once installed, this integration automatically configures the required environment variables for `uv` and `uvx` commands to work in HASSIO:
   
   ```
   Command: uvx
   Args: mcp-server-fetch
   ```
   
   The integration automatically sets:
   - `UV_TOOL_DIR=/config/.uv/tools`
   - `UV_CACHE_DIR=/config/.uv/cache`
   - `TMPDIR=/config/.uv/tmp`
   
   This ensures uvx uses the `/config` directory (which has exec permissions) instead of `/tmp` (which has noexec in HASSIO).

3. **Use Python module directly**
   
   If a package is installed in HA's Python environment:
   ```
   Command: python
   Args: -m,mcp_server_filesystem,/config
   ```

4. **Run MCP servers in Docker containers**
   
   If running HA in Docker (not HASSIO), add MCP server containers to your compose file:
   ```yaml
   mcp-server:
     image: node:20-alpine
     command: npx @anthropic/mcp-server-example --transport sse --port 3000
     ports:
       - "3000:3000"
   ```

5. **Create a Home Assistant Add-on**
   
   Build a custom add-on that includes the MCP server. The add-on runs in its own container with all dependencies.

### Example MCP Server Configurations

#### Home Assistant's Built-in MCP Server (SSE)

Home Assistant has a built-in MCP Server integration that exposes all your entities and services to MCP clients. This is the easiest way to give the AI full access to your smart home.

#### Step 1: Enable the MCP Server Integration

1. Add to your `configuration.yaml`:
   ```yaml
   mcp_server:
   ```

2. Restart Home Assistant

3. The MCP server will be available at:
   ```
   http://localhost:8123/api/mcp
   ```
   Or if using HTTPS:
   ```
   https://localhost:8123/api/mcp
   ```

For more details, see the [Home Assistant MCP Server documentation](https://www.home-assistant.io/integrations/mcp_server/).

#### Step 2: Configure OpenAI Realtime to Use It

When setting up or configuring the OpenAI Realtime integration:

1. Go to **Settings** → **Devices & Services** → **OpenAI Realtime** → **Configure**
2. Add MCP Server with:
   - **Name**: `homeassistant` (or any name you prefer)
   - **URL**: `http://localhost:8123/api/mcp`
   - **Token**: Create a Long-Lived Access Token:
     1. Go to your profile (click your name in sidebar)
     2. Scroll to "Long-Lived Access Tokens"
     3. Click "Create Token"
     4. Copy the token and paste it here

#### Example Configuration

```yaml
# MCP Server settings in OpenAI Realtime integration
name: homeassistant
url: http://localhost:8123/api/mcp
token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...  # Your long-lived access token
```

#### What This Enables

With the HA MCP Server connected, the AI gains access to:
- All entity states and attributes
- All available services
- Area and device information
- Much more comprehensive control than the built-in tools alone

> **Note**: The built-in tools (`get_entity_state`, `call_service`, etc.) still work alongside MCP servers. MCP servers provide additional capabilities.

#### Bright Data MCP Server (SSE - External Machine)

[Bright Data MCP Server](https://github.com/anthropics/mcp-servers) provides web scraping capabilities.

**On a machine with Node.js:**
```bash
npx @anthropic/mcp-server-brightdata --transport sse --port 3001
```

**In OpenAI Realtime integration:**
- **Type**: SSE
- **Name**: `bright_data`
- **URL**: `http://your-nodejs-machine:3001/sse`
- **Token**: Your Bright Data API key (if required)

#### Filesystem MCP Server (Stdio - Python)

For HA Core installations with Python available:

- **Type**: Stdio
- **Name**: `filesystem`
- **Command**: `python`
- **Args**: `-m,mcp_server_filesystem,/config`

#### Custom MCP Server (Stdio - Local Binary)

If you have a compiled MCP server binary:

- **Type**: Stdio
- **Name**: `my_custom_server`
- **Command**: `/usr/local/bin/my-mcp-server`
- **Args**: `--config,/config/mcp/config.yaml`
- **Env**: `DEBUG=true,LOG_LEVEL=info`

### Managing MCP Servers

You can manage MCP servers through the integration options:

1. Go to **Settings** → **Devices & Services** → **OpenAI Realtime** → **Configure**
2. Choose from:
   - **Add SSE Server**: Add a new HTTP-based MCP server
   - **Add Stdio Server**: Add a new subprocess-based MCP server
   - **Manage Existing Servers**: Edit, enable/disable, or delete servers
3. After making changes, the integration will reload automatically

### MCP Server Naming Rules

Server names must match the pattern `^[a-zA-Z0-9_-]+$`:
- ✅ `home_assistant`, `bright-data`, `myServer1`
- ❌ `Home Assistant`, `my server`, `서버이름`

Spaces and special characters in names will be automatically converted to underscores.

## Built-in Home Assistant Tools

The integration provides these built-in tools for controlling Home Assistant:

### get_entity_state
Get the current state of any Home Assistant entity.
```
"Turn on the living room light" → Checks light.living_room state
```

### call_service
Call any Home Assistant service.
```
"Set the thermostat to 72 degrees" → climate.set_temperature
```

### get_entities_by_domain
List all entities in a domain.
```
"What lights do I have?" → Lists all light entities
```

### get_area_entities
Get all entities in a specific area.
```
"What devices are in the bedroom?" → Lists entities in bedroom area
```

## Usage

### As Conversation Agent

1. Go to **Settings** → **Voice Assistants**
2. Create a new assistant or edit an existing one
3. Select "OpenAI Realtime" as the conversation agent
4. Use with any voice input method (Assist, voice satellites, etc.)

### Using the Media Player

The integration creates a media player entity for direct audio control:

- **Play**: Start listening for audio input
- **Stop**: Stop audio processing and cancel responses

### Binary Sensors

Monitor the state of the realtime connection:

| Sensor | Description |
|--------|-------------|
| `binary_sensor.openai_realtime_connected` | WebSocket connection status |
| `binary_sensor.openai_realtime_listening` | User is speaking (VAD detected) |
| `binary_sensor.openai_realtime_speaking` | Assistant is responding |
| `binary_sensor.openai_realtime_processing` | Request is being processed |

### Services

#### openai_realtime.send_message
Send a text message and get a response.
```yaml
service: openai_realtime.send_message
data:
  message: "What's the weather like?"
```

#### openai_realtime.send_audio
Send audio data directly to the API.
```yaml
service: openai_realtime.send_audio
data:
  audio_data: "<base64_encoded_pcm_audio>"
```

#### openai_realtime.start_listening
Start the audio session.
```yaml
service: openai_realtime.start_listening
```

#### openai_realtime.stop_listening
Stop audio processing.
```yaml
service: openai_realtime.stop_listening
```

#### openai_realtime.add_mcp_server
Add an MCP server at runtime.
```yaml
service: openai_realtime.add_mcp_server
data:
  name: "my_server"
  url: "https://mcp.example.com"
  token: "optional_token"
```

#### openai_realtime.clear_conversation
Clear the conversation history.
```yaml
service: openai_realtime.clear_conversation
```

### Example Commands

- "Turn on the kitchen lights"
- "What's the temperature in the living room?"
- "Set the bedroom thermostat to 68 degrees"
- "Lock all the doors"
- "What lights are on?"

## Lovelace Card (Browser Microphone)

This integration includes a custom Lovelace card that captures audio directly from your browser's microphone and streams it to the OpenAI Realtime API.

### Step 1: Add Lovelace Resource

The integration tries to register the card automatically, but you may need to add it manually:

1. Go to **Settings** → **Dashboards** → **⋮ (three dots)** → **Resources**
2. Click **Add Resource**
3. Enter:
   - **URL**: `/openai_realtime/openai-realtime-card.js?v=(random_int_for_debug/update)`
   - **Resource type**: JavaScript Module
4. Click **Create**

Alternatively, add to your `configuration.yaml`:

```yaml
lovelace:
  resources:
    - url: /openai_realtime/openai-realtime-card.js
      type: module
```

### Step 2: Add the Card to Dashboard

> **Note**: This card does not support the visual editor. When you see the error "Visual editor is not supported" or `setConfig is not a function`, use the YAML editor instead.

#### Using YAML Editor

1. Go to your dashboard and click **Edit** (pencil icon)
2. Click **+ Add Card**
3. Scroll down and select **Manual** (or click the three dots and choose "Edit in YAML")
4. Paste the following configuration:

```yaml
type: custom:openai-realtime-card
title: OpenAI Realtime Voice
show_transcript: true
show_waveform: true
mute_while_speaking: true
```

5. Click **Save**

#### Editing an Existing Card

If you need to edit the card later:
1. Click the three dots (⋮) on the card
2. Select **Edit** 
3. If you see "Visual editor is not supported", click **Edit in YAML**
4. Make your changes and save

### Card Features

- **Push-to-Talk**: Hold the microphone button to speak
- **Audio Visualization**: Real-time waveform display while speaking
- **Transcript**: Live display of your speech and AI responses
- **Audio Playback**: Automatic playback of AI voice responses

### Card Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title` | string | "OpenAI Realtime" | Card title |
| `show_transcript` | boolean | true | Show conversation transcript |
| `show_waveform` | boolean | true | Show audio waveform visualization |
| `mute_while_speaking` | boolean | true | Mute microphone while AI is speaking to prevent echo/feedback. Set to `false` to allow interrupting the AI (requires headphones or good hardware echo cancellation) |

### Browser Requirements

- Modern browser with Web Audio API support
- Microphone permissions granted
- HTTPS connection (required for microphone access)

### Using with Voice Satellites

For ESP-based voice satellites, configure them to use the custom STT/TTS providers created by this integration, or use the direct WebSocket API.

## Audio Configuration

The Realtime API uses PCM audio at 24kHz. The integration handles audio conversion automatically when used with Home Assistant's voice pipeline.

### Supported Audio Formats
- Input: PCM 16-bit, 24kHz
- Output: PCM 16-bit, 24kHz

## OpenAI Voice Options

Available voices:
- `alloy` - Neutral, balanced
- `echo` - Deep, resonant
- `fable` - Warm, storytelling
- `onyx` - Deep, authoritative
- `nova` - Youthful, energetic
- `shimmer` - Clear, expressive
- `coral` - Warm, engaging

## OpenAI Pricing

OpenAI Realtime API pricing (per 1M tokens):

| Type | Input | Cached Input | Output |
|------|-------|--------------|--------|
| Text | $4.00 | $0.50 | $16.00 |
| Audio | $32.00 | $0.50 | $64.00 |

---

# 🟢 Gemini Live Integration

## Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "Gemini Live"
3. Enter your Google AI API key
4. Configure the settings:
   - **Model**: Select the model (default: `gemini-2.0-flash-exp`)
   - **Voice**: Choose the voice for audio responses
   - **Instructions**: Custom system instructions

## Gemini Voice Options

Available voices:
- `Puck` - Playful, energetic
- `Charon` - Deep, mysterious
- `Kore` - Warm, friendly
- `Fenrir` - Strong, confident
- `Aoede` - Clear, melodic

## Gemini Lovelace Card

### Add Lovelace Resource

1. Go to **Settings** → **Dashboards** → **⋮ (three dots)** → **Resources**
2. Click **Add Resource**
3. Enter:
   - **URL**: `/gemini_live/gemini-live-card.js?v=1`
   - **Resource type**: JavaScript Module
4. Click **Create**

### Add the Card to Dashboard

```yaml
type: custom:gemini-live-card
title: Gemini Live Voice
```

### Card Features

- **Push-to-Talk**: Click the microphone button to start/stop speaking
- **Real-time Visualizer**: Live audio level visualization
- **Live Transcripts**: See your input and AI responses in real-time
- **Text Input**: Type messages instead of speaking
- **File Upload**: Send images and audio files for multimodal conversations
- **Mute Toggle**: Mute microphone while AI is speaking to prevent echo

### Card Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title` | string | "Gemini Live" | Card title |

## Gemini Services

#### gemini_live.send_message
Send a text message and get a response.
```yaml
service: gemini_live.send_message
data:
  message: "What's the weather like?"
```

#### gemini_live.send_audio
Send audio data directly to the API.
```yaml
service: gemini_live.send_audio
data:
  audio_data: "<base64_encoded_pcm_audio>"
```

#### gemini_live.start_listening
Start the audio session.
```yaml
service: gemini_live.start_listening
```

#### gemini_live.stop_listening
Stop audio processing.
```yaml
service: gemini_live.stop_listening
```

## Gemini Binary Sensors

| Sensor | Description |
|--------|-------------|
| `binary_sensor.gemini_live_connected` | WebSocket connection status |
| `binary_sensor.gemini_live_listening` | User is speaking |
| `binary_sensor.gemini_live_speaking` | Assistant is responding |
| `binary_sensor.gemini_live_processing` | Request is being processed |

## Gemini Pricing

Gemini 2.0 Flash is currently in preview with generous free tier limits. Check [Google AI pricing](https://ai.google.dev/pricing) for current rates.

---

# Troubleshooting

### Enable Debug Logging

Add to your `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.openai_realtime: debug
```

Then restart Home Assistant.

### Key Log Messages to Look For

| Log Message | Meaning |
|-------------|---------|
| `Connected to OpenAI Realtime API` | WebSocket connected successfully |
| `Session created: sess_XXX` | Session established with OpenAI |
| `Updating session with X tools` | Tools are being registered |
| `Session updated` | Tools registered successfully ✅ |
| `API Error: ...` | Something went wrong ❌ |
| `Function call received` | AI is calling a Home Assistant tool |
| `Function result sent to OpenAI` | Tool execution completed |
| `Registering event handlers including function_call handler` | WebSocket API handlers set up |

### Common Issues

#### API Error: Invalid type for 'session.max_response_output_tokens'
This was a known issue where the token value was sent as a decimal. Update to the latest version.

#### Tools not working / AI says it did something but nothing happened
1. Check logs for `API Error` messages after `Updating session with X tools`
2. Look for `Session updated` - if missing, session config failed and tools weren't registered
3. Verify entity IDs exist in Home Assistant
4. Check for `Function call received` in logs to confirm AI is trying to call tools

#### No audio playback
- Ensure your browser allows audio playback
- Check browser console for errors (F12 → Console)
- Try refreshing the page with Ctrl+Shift+R

#### Microphone not working
- Ensure HTTPS is enabled (required for microphone access)
- Check browser permissions for microphone access
- Try a different browser (Chrome recommended)

#### "Not connected to OpenAI Realtime API"
- Check your API key is valid
- Ensure you have access to the Realtime API (not all accounts have it)
- Check your internet connection

#### Audio playing multiple times / overlapping
- This was fixed in recent versions - update to latest
- Clear browser cache and reload

### MCP Server Issues

#### Stdio server not working on HASSIO
- **Cause**: Node.js (`npx`, `node`) is not available on Home Assistant OS
- **Solution**: Use SSE mode instead. Run the MCP server on a separate machine and connect via HTTP

#### MCP server "command not found"
- **Cause**: The command is not installed or not in PATH
- **Solution**: 
  - Use full path to executable (e.g., `/usr/bin/python3` instead of `python`)
  - For Python MCP servers, ensure the module is installed: `pip install mcp-server-xxx`

#### MCP tools not appearing in AI responses
1. Check logs for `Loading X MCP servers from config`
2. For stdio servers, look for `Connected to stdio MCP server X, found Y tools`
3. Verify the server is enabled in options
4. Check for connection errors in logs

#### MCP call succeeds but no audio response
- **Cause**: After MCP calls, OpenAI may need a trigger to generate audio
- **Solution**: This is handled automatically in recent versions. Update to latest and restart.

#### "Server not found" error when calling MCP tool
- **Cause**: Server name in function call doesn't match configured server
- **Solution**: Check server name for special characters. Names are sanitized (spaces → underscores)

### Browser Console Debugging

1. Open browser developer tools (F12)
2. Go to Console tab
3. Look for messages:
   - `Subscribing to OpenAI Realtime events...` - Card is connecting
   - `Subscribed successfully` - Connection established
   - `Received event:` - Events coming from backend
   - `Playing audio chunk` - Audio is being played

### Check Integration Status

1. Go to **Settings** → **Devices & Services**
2. Find "OpenAI Realtime"
3. Check if it shows any errors

### View Full Logs

```bash
# In Home Assistant terminal or SSH
tail -f /config/home-assistant.log | grep openai_realtime
```

### Test API Connection

Try sending a text message via Developer Tools → Services:
```yaml
service: openai_realtime.send_message
data:
  message: "Hello, can you hear me?"
```

## Updating

### Updating the Integration

1. **Via HACS**: 
   - Go to HACS → Integrations
   - Find "OpenAI Realtime" and click "Update"
   - Restart Home Assistant

2. **Manual Update**:
   - Replace the `custom_components/openai_realtime` folder with the new version
   - Restart Home Assistant

### Updating the JavaScript Card (Manual Cache Bust)

The JS card version is automatically updated based on file modification time. However, browsers may cache the old version. Here's how to force an update:

#### Method 1: Update Resource Version in Dashboard Settings (Recommended)

1. Go to **Settings** → **Dashboards**
2. Click the three-dot menu (⋮) → **Resources**
3. Find the resource containing `/openai_realtime/openai-realtime-card.js`
4. Click to edit it
5. Change the URL version parameter:
   ```
   Before: /openai_realtime/openai-realtime-card.js?v=1733000000
   After:  /openai_realtime/openai-realtime-card.js?v=1733100000
   ```
   (Just change the number to anything different)
6. Click **Update**
7. Hard refresh your browser: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)

#### Method 2: Hard Refresh Browser

- Windows/Linux: `Ctrl+Shift+R`
- Mac: `Cmd+Shift+R`
- Or open DevTools (F12) → Right-click Refresh → "Empty Cache and Hard Reload"

#### Method 3: Delete and Re-add Resource

1. Go to **Settings** → **Dashboards** → **Resources**
2. Delete the OpenAI Realtime card resource
3. Reload the integration:
   - **Settings** → **Devices & Services** → **OpenAI Realtime** → **⋮** → **Reload**
4. The resource will be re-added automatically with new version

#### Method 4: Clear Browser Cache Completely

1. Open browser settings
2. Clear cached images and files
3. Reload the dashboard

### After Updating

1. Clear browser cache or hard refresh (`Ctrl+Shift+R`)
2. Reload the dashboard
3. Check browser console (F12) for any errors
4. Test the microphone button

### Connection Issues
- Verify your API key has Realtime API access
- Check your network allows WebSocket connections
- Review Home Assistant logs for detailed error messages

### MCP Server Issues
- Ensure the MCP server URL is accessible from Home Assistant
- Verify authentication tokens are correct
- Check MCP server logs for connection issues

### Audio Issues
- Ensure audio is in the correct format (PCM 24kHz)
- Check voice assistant configuration
- Verify microphone/speaker setup

## Development

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/ha-realtime-ai-audio.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Link to Home Assistant custom_components
ln -s $(pwd)/custom_components/openai_realtime ~/.homeassistant/custom_components/
ln -s $(pwd)/custom_components/gemini_live ~/.homeassistant/custom_components/
```

### Running Tests

```bash
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Changelog

### 2.0.0
- Added Gemini Live Audio integration with Google's Gemini Live API
- Gemini features: Session resumption, image/audio input, Google Search
- Renamed project to "ha-realtime-ai-audio"
- Updated README for both integrations

### 1.0.0
- Initial release
- OpenAI Realtime API integration
- MCP server support
- Home Assistant conversation agent
- Built-in smart home tools
