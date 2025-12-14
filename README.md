# Realtime AI Audio for Home Assistant — Gemini Live

> **IMPORTANT:** This component uses a custom-made UI to enable Live speech functionality. READ this README thoroughly before installing or configuring the integration.

A Home Assistant custom component that integrates with **Google's Gemini Live API** for real-time voice and text conversations, with optional MCP (Model Context Protocol) server support.

## 🎯 Included Integration

| Integration | API | Voice Model |
|-------------|-----|-------------|
| **Gemini Live** | Google Gemini Live API | Gemini 2.0 Flash |

Provides native speech-to-speech capabilities with minimal latency.

## Features

- **Real-time Conversations:** WebSocket-based low-latency responses
- **Native Speech-to-Speech:** Direct audio processing without separate STT/TTS pipeline
- **Voice Support:** Multiple voice options with configurable settings
- **Home Assistant Integration:** Built-in tools for controlling smart home devices
- **Conversation Agent:** Works as a Home Assistant conversation agent
- **Media Player Entity:** Control audio input/output directly
- **Binary Sensors:** Monitor connection, listening, speaking, and processing states
- **Custom Lovelace Card:** Browser-based microphone with real-time visualizer

## Architecture

Unlike the default Home Assistant voice pipeline (STT → AI → TTS), this integration uses native speech-to-speech APIs:

```
Default HA Pipeline: Mic -> STT -> AI -> TTS -> Speaker

Gemini Live Pipeline: Mic -> Gemini Live Realtime API -> Speaker
```

## Requirements

- Home Assistant 2024.1.0 or later
- Google AI API key (Gemini API)
- Python 3.11 or later

See the Gemini-specific documentation for additional setup details:

- [custom_components/gemini_live/GOOGLE_DOC.md](custom_components/gemini_live/GOOGLE_DOC.md)

## Quick Configuration (Minimal examples)

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
# Gemini Live basic settings
model: gemini-2.5-flash-native-audio-preview-12-2025
voice: Kore
enable_session_resumption: true
```

For advanced features (session resumption handles, proactive audio, image inputs), open the Gemini docs in the component folder: [custom_components/gemini_live/GOOGLE_DOC.md](custom_components/gemini_live/GOOGLE_DOC.md)

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant
2. Click on "Integrations"
3. Click the three dots in the top right corner → "Custom repositories"
4. Add this repository URL: `https://github.com/SJang1/ha-gemini-live`
5. Install the "Realtime AI Audio for Home Assistant"
6. Restart Home Assistant

### Manual Installation

1. Download the repository
2. Copy the `custom_components/gemini_live` folder to your Home Assistant `custom_components` directory
3. Restart Home Assistant

## Gemini Live Integration

### Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "Gemini Live"
3. Enter your Google AI API key
4. Configure the settings:
   - **Model**: Select the model (default: `gemini-2.0-flash-exp`)
   - **Voice**: Choose the voice for audio responses
   - **Instructions**: Custom system instructions

### Gemini Voice Options

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
title: Gemini Live
show_transcript: true
keep_mic_when_hidden: true
```

```bash
# Clone the repository
git clone https://github.com/SJang1/ha-gemini-live.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Link to Home Assistant custom_components
ln -s $(pwd)/custom_components/gemini_live ~/.homeassistant/custom_components/
```
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

## Troubleshooting

### Enable Debug Logging

Add to your `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.gemini_live: debug
```

Then restart Home Assistant.

### Common Issues

- Ensure your `google_api_key` is valid and has access to Gemini.
- Microphone requires HTTPS and browser permissions.
- For session resumption and advanced features, consult [custom_components/gemini_live/GOOGLE_DOC.md](custom_components/gemini_live/GOOGLE_DOC.md).

## Development

```bash
# Clone the repository
git clone https://github.com/your-username/ha-realtime-ai-audio.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Link to Home Assistant custom_components
ln -s $(pwd)/custom_components/gemini_live ~/.homeassistant/custom_components/
```

## License

MIT License - see [LICENSE](LICENSE) for details.
