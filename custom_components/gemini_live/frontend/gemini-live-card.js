/**
 * Gemini Live Card for Home Assistant
 * 
 * A custom Lovelace card for interacting with the Gemini Live API
 * through the gemini_live integration.
 */

const DOMAIN = "gemini_live";

// Enable/disable debug logging
const DEBUG_LOG = true;

// Runtime load marker for debugging
try {
        if (DEBUG_LOG && window && window.console) {
        console.debug("GeminiLiveCard: module loaded (Gemini Live)");
    }
} catch (e) {
    /* ignore */
}

function logWS(direction, type, data) {
    if (!DEBUG_LOG) return;
    const timestamp = new Date().toISOString().substr(11, 12);
    const arrow = direction === 'send' ? '📤→' : '📥←';
    const color = direction === 'send' ? 'color: #4CAF50' : 'color: #2196F3';

    // For audio data, just log the size, not the content
    let logData = data;
    if (data && typeof data === 'object') {
        logData = { ...data };
        if (logData.audio && typeof logData.audio === 'string') {
            logData.audio = `[base64 audio: ${logData.audio.length} chars]`;
        }
    }

    console.log(`%c${arrow} [${timestamp}] ${type}`, color, logData);
}

class GeminiLiveCard extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: "open" });
        this._config = {};
        this._hass = null;
        this._connected = false;
        this._listening = false;
        this._speaking = false;
        this._messages = [];
        this._audioContext = null;
        this._mediaRecorder = null;
        this._audioChunks = [];
        this._subscription = null;

        // Transcript display
        this._inputTranscript = "";
        this._outputTranscript = "";

        // Options
        this._muteWhileSpeaking = true;
        // Keep microphone running when page/tab is hidden (default: true)
        this._keepMicWhenHidden = true;

        // File attachment
        this._selectedFile = null;

        // Session management
        this._sessionResumable = false;
        this._goAwayWarning = null;
        this._resumptionHandle = null;

        // Connection state
        this._connecting = false;  // True while connection is being established

        // Audio processing
        this._mediaStream = null;
        this._audioProcessor = null;

        // Index of the currently streaming assistant message in _messages (or null)
        this._currentStreamingIndex = null;

        // Audio playback queue for smoother playback
        this._audioPlaybackQueue = [];
        this._isPlayingAudio = false;
        this._playbackAudioContext = null;
        this._nextPlayTime = 0;  // For scheduled audio playback
        this._playbackStarted = false;  // Track if initial playback started

        // Pending listen request flag (true while awaiting backend confirmation)
        this._pendingListen = false;
        // Buffer for output_transcript chunks received before the user's final transcript
        this._bufferedOutput = "";
        // Audio initialization flag (true while setting up AudioContext/recorder)
        this._audioInitializing = false;
        // Buffer for user input while speaking (committed when output starts or when final)
        this._bufferedInput = "";
        // Whether assistant output has started for the current turn
        this._outputHasStarted = false;

        // Bind cleanup handlers
        this._handleBeforeUnload = this._handleBeforeUnload.bind(this);
        this._handleVisibilityChange = this._handleVisibilityChange.bind(this);
    }

    set hass(hass) {
        this._hass = hass;
        this._updateStates();
        this._render();
    }

    setConfig(config) {
        this._config = config;
        // Backwards-compatible default: keep mic active when hidden unless explicitly disabled
        this._keepMicWhenHidden = config.keep_mic_when_hidden !== undefined ? !!config.keep_mic_when_hidden : true;
        this._render();
    }

    getCardSize() {
        return 4;
    }

    static getConfigElement() {
        return document.createElement("gemini-live-card-editor");
    }

    static getStubConfig() {
        return {
            title: "Gemini Live",
            show_transcript: true,
            keep_mic_when_hidden: true,
        };
    }

    connectedCallback() {
        this._setupSubscription();

        // Add event listeners for auto-disconnect on page close/refresh
        window.addEventListener('beforeunload', this._handleBeforeUnload);
        document.addEventListener('visibilitychange', this._handleVisibilityChange);

        logWS('recv', 'CARD_CONNECTED', {});
    }

    disconnectedCallback() {
        logWS('recv', 'CARD_DISCONNECTED', {});

        // Remove event listeners
        window.removeEventListener('beforeunload', this._handleBeforeUnload);
        document.removeEventListener('visibilitychange', this._handleVisibilityChange);

        // Cleanup subscriptions and audio
        this._cleanupSubscription();
        this._stopRecording();
        this._cleanupPlaybackAudio();

        // Auto-disconnect from API
        this._forceDisconnect();
    }

    _handleBeforeUnload(e) {
        logWS('send', 'PAGE_UNLOAD', { action: 'disconnecting' });
        // Synchronously disconnect before page unload
        this._forceDisconnect();
    }

    _handleVisibilityChange() {
        if (document.visibilityState === 'hidden') {
            logWS('recv', 'PAGE_HIDDEN', { listening: this._listening, connected: this._connected });
            // Optionally stop listening when page is hidden (user switched tabs).
            // By default we keep the microphone running to allow long-running sessions
            // even when the user switches tabs; this is controlled by the
            // `keep_mic_when_hidden` card configuration (default: true).
            if (!this._keepMicWhenHidden && this._listening) {
                this._stopListening();
            } else {
                // Keep recording active; just log the state.
                logWS('recv', 'PAGE_HIDDEN_KEEPING_MIC', { listening: this._listening });
            }
        } else if (document.visibilityState === 'visible') {
            logWS('recv', 'PAGE_VISIBLE', { connected: this._connected });
        }
    }

    _forceDisconnect() {
        // Force disconnect from API - called on page close/refresh
        if (this._hass && this._connected) {
            try {
                // Use sendMessage instead of callWS for synchronous behavior
                this._hass.connection.sendMessage({
                    type: `${DOMAIN}/disconnect`,
                });
                logWS('send', 'FORCE_DISCONNECT', {});
            } catch (e) {
                console.error("Failed to force disconnect:", e);
            }
        }
        this._connected = false;
        this._listening = false;
    }

    _cleanupPlaybackAudio() {
        this._audioPlaybackQueue = [];
        this._isPlayingAudio = false;
        this._nextPlayTime = 0;
        this._playbackStarted = false;
        if (this._playbackAudioContext) {
            try {
                this._playbackAudioContext.close();
            } catch (e) { }
            this._playbackAudioContext = null;
        }
    }

    // Called when audio playback should stop (interruption)
    _stopAudioPlayback() {
        this._playbackStarted = false;
        this._nextPlayTime = 0;
        this._audioPlaybackQueue = [];
        // Close and recreate context to stop all scheduled audio
        if (this._playbackAudioContext && this._playbackAudioContext.state !== 'closed') {
            try {
                this._playbackAudioContext.close();
            } catch (e) { }
            this._playbackAudioContext = null;
        }
    }

    async _setupSubscription() {
        if (!this._hass) return;

        try {
            logWS('send', 'SUBSCRIBE', {});
            this._subscription = await this._hass.connection.subscribeMessage(
                (event) => {
                    logWS('recv', `EVENT:${event.type}`, event.data);
                    this._handleEvent(event);
                },
                {
                    type: `${DOMAIN}/subscribe`,
                }
            );
            logWS('recv', 'SUBSCRIBED', { success: true });
        } catch (e) {
            console.error("Failed to subscribe to Gemini Live events:", e);
            logWS('recv', 'SUBSCRIBE_ERROR', { error: e.message });
        }
    }

    _cleanupSubscription() {
        if (this._subscription) {
            this._subscription();
            this._subscription = null;
        }
    }

    _handleEvent(event) {
        switch (event.type) {
            case "audio_delta":
                this._handleAudioDelta(event.data);
                break;
            case "transcript":
                this._handleTranscript(event.data);
                break;
            case "output_transcript":
                this._handleOutputTranscript(event.data);
                break;
            case "turn_complete":
                this._handleTurnComplete(event.data);
                break;
            case "interrupted":
                this._handleInterruption(event.data);
                break;
            case "session_resumed":
                this._handleSessionResumed(event.data);
                break;
            case "session_resumption_update":
                this._handleSessionResumptionUpdate(event.data);
                break;
            case "go_away":
                this._handleGoAway(event.data);
                break;
            case "generation_complete":
                this._handleGenerationComplete(event.data);
                break;
            case "error":
                this._handleError(event.data);
                break;
        }
    }

    _handleAudioDelta(data) {
        // Handle incoming audio for playback
        if (data.audio) {
            // Don't log every audio chunk to reduce console spam
            this._playAudio(data.audio);
        }
    }

    _handleTranscript(data) {
        if (data.text) {
            // Check if this is a final transcript (user finished speaking)
            if (data.is_final && data.text.trim()) {
                // If assistant output already started for this turn, the buffered input
                // should have been committed when output started. Only commit now if
                // output has NOT started.
                if (!this._outputHasStarted) {
                    this._addMessage("user", data.text);
                }
                // Clear buffered input and live display
                this._bufferedInput = "";
                this._inputTranscript = "";

                // If assistant output arrived earlier while user transcript wasn't final,
                // flush buffered assistant output now so it appears after the user message.
                if (this._bufferedOutput && this._bufferedOutput.trim()) {
                    const msg = {
                        role: "assistant",
                        text: this._bufferedOutput,
                        timestamp: new Date().toISOString(),
                        streaming: true,
                    };
                    this._messages.push(msg);
                    this._currentStreamingIndex = this._messages.length - 1;
                    // Also set accumulated output for finalization
                    this._outputTranscript = this._bufferedOutput;
                    this._bufferedOutput = "";
                    this._outputHasStarted = true;
                }
            } else {
                // Buffer partial input while user is speaking; also show live transcript
                // Ignore empty interim transcripts to avoid clearing the buffer
                if (!data.text || !data.text.trim()) {
                    // keep existing buffer
                } else {
                    // If the new partial does not contain the previous buffer, append it
                    // otherwise replace with the new partial (covers both incremental and full replacements)
                    if (this._bufferedInput && this._bufferedInput.length > 0 && data.text.indexOf(this._bufferedInput) === -1) {
                        this._bufferedInput += data.text;
                    } else {
                        this._bufferedInput = data.text;
                    }
                    this._inputTranscript = this._bufferedInput;
                }
            }
            this._render();
        }
    }

    _handleOutputTranscript(data) {
        if (data.text) {
            // If user has buffered input (is speaking), commit it immediately as a final
            // user message when assistant output arrives, per requested behavior.
            if (this._bufferedInput && this._bufferedInput.trim()) {
                this._addMessage("user", this._bufferedInput);
                this._bufferedInput = "";
                this._inputTranscript = "";
            }

            // Mark that output has started for this turn
            this._outputHasStarted = true;

            // Maintain a single streaming assistant message by index
            if (this._currentStreamingIndex === null) {
                const msg = {
                    role: "assistant",
                    text: data.text,
                    timestamp: new Date().toISOString(),
                    streaming: true,
                };
                this._messages.push(msg);
                this._currentStreamingIndex = this._messages.length - 1;
            } else {
                const msg = this._messages[this._currentStreamingIndex];
                if (msg && msg.role === "assistant") {
                    msg.text += data.text;
                } else {
                    // Fallback: create a new streaming message
                    const fallback = {
                        role: "assistant",
                        text: data.text,
                        timestamp: new Date().toISOString(),
                        streaming: true,
                    };
                    this._messages.push(fallback);
                    this._currentStreamingIndex = this._messages.length - 1;
                }
            }
            // Also accumulate for finalization
            this._outputTranscript += data.text;
            this._render();
        }
    }

    _updateStreamingMessage(text) {
        // Find existing streaming message or create new one
        const lastMessage = this._messages[this._messages.length - 1];
        if (lastMessage && lastMessage.role === "assistant" && lastMessage.streaming) {
            // Update existing streaming message
            lastMessage.text = text;
        } else {
            // Create new streaming message
            this._messages.push({
                role: "assistant",
                text: text,
                timestamp: new Date(),
                streaming: true
            });
        }
    }

    _handleTurnComplete(data) {
        logWS('recv', 'TURN_COMPLETE_HANDLED', {
            hasInputTranscript: !!this._inputTranscript,
            hasDataTranscript: !!data.transcript,
            hasOutputTranscript: !!this._outputTranscript
        });

        // If we have a pending user transcript, add it as a message
        if (this._inputTranscript && this._inputTranscript.trim()) {
            this._addMessage("user", this._inputTranscript);
            this._inputTranscript = "";
        }

        // Mark speaking as done when turn completes
        this._speaking = false;
        this._playbackStarted = false;

        // Finalize the streaming message (prefer the tracked streaming index)
        const idx = this._currentStreamingIndex !== null ? this._currentStreamingIndex : this._messages.length - 1;
        const lastMessage = this._messages[idx];
        // Use the full transcript if available
        const finalText = data.transcript || this._outputTranscript;
        if (lastMessage && lastMessage.role === "assistant" && lastMessage.streaming) {
            lastMessage.text = finalText;
            lastMessage.streaming = false;
        } else if (finalText && finalText.trim()) {
            if (!(lastMessage && lastMessage.role === "assistant" && lastMessage.text === finalText)) {
                this._addMessage("assistant", finalText);
            }
        }
        this._outputTranscript = "";
        this._currentStreamingIndex = null;
        this._outputHasStarted = false;
        this._render();
    }

    _handleInterruption(data) {
        logWS('recv', 'INTERRUPTION', {});
        // Stop all scheduled audio on interruption
        this._stopAudioPlayback();
        this._speaking = false;
        this._render();
    }

    _handleError(data) {
        console.error("Gemini Live error:", data);
        logWS('recv', 'ERROR_HANDLED', data);
        this._addMessage("system", `Error: ${data.message || data.error || "Unknown error"}`);
        this._render();
    }

    _handleSessionResumed(data) {
        logWS('recv', 'SESSION_RESUMED_HANDLED', data);
        this._connected = true;
        this._addMessage("system", "Session resumed successfully");
        this._render();
    }

    _handleSessionResumptionUpdate(data) {
        logWS('recv', 'SESSION_RESUMPTION_UPDATE_HANDLED', data);
        // Store the resumption handle for later use
        if (data.handle) {
            this._resumptionHandle = data.handle;
            this._sessionResumable = data.resumable || false;
            // Persist to localStorage for page refresh recovery
            const storageKey = `gemini_live_handle`;
            localStorage.setItem(storageKey, data.handle);
        }
    }

    _handleGoAway(data) {
        logWS('recv', 'GO_AWAY_HANDLED', data);
        this._goAwayWarning = data.time_left;
        this._addMessage("system", `Warning: Connection will terminate in ${data.time_left || "unknown"} seconds. Save your work.`);
        this._render();

        // Auto-reconnect after termination if we have a resumption handle
        if (data.time_left && this._resumptionHandle) {
            setTimeout(() => {
                this._reconnectWithHandle();
            }, (data.time_left + 2) * 1000); // Wait a bit after termination
        }
    }

    _handleGenerationComplete(data) {
        logWS('recv', 'GENERATION_COMPLETE_HANDLED', data);
        this._sessionResumable = data.resumable || false;
    }

    _updateStates() {
        if (!this._hass) return;

        // Find the binary sensors by domain prefix
        const states = this._hass.states || {};
        for (const entityId of Object.keys(states)) {
            if (entityId.startsWith(`binary_sensor.${DOMAIN}_`) && entityId.endsWith('_listening')) {
                // Do not flip UI to listening if audio is still initializing locally.
                const listeningState = states[entityId].state === "on";
                this._listening = listeningState && !this._audioInitializing;
            } else if (entityId.startsWith(`binary_sensor.${DOMAIN}_`) && entityId.endsWith('_speaking')) {
                this._speaking = states[entityId].state === "on";
            } else if (entityId.startsWith(`binary_sensor.${DOMAIN}_`) && entityId.endsWith('_connected')) {
                this._connected = states[entityId].state === "on";
            }
        }
        // If HA reports disconnected while UI still thinks it's listening,
        // disable listening and clear buffered/input state so user input is not available.
        if (!this._connected && this._listening) {
            this._listening = false;
            this._pendingListen = false;
            this._audioInitializing = false;
            this._bufferedInput = "";
            this._inputTranscript = "";
            this._bufferedOutput = "";
            this._outputTranscript = "";
            this._currentStreamingIndex = null;
            // Optionally notify the user in UI
            this._addMessage("system", "Connection lost — input disabled until reconnected.");
            this._render();
        }
    }

    _addMessage(role, text) {
        this._messages.push({
            role,
            text,
            timestamp: new Date().toISOString(),
        });

        // Keep only last 50 messages
        if (this._messages.length > 50) {
            this._messages.shift();
        }
    }

    _formatTime(isoString) {
        try {
            const date = new Date(isoString);
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch {
            return '';
        }
    }

    _scrollChatToBottom() {
        // Use requestAnimationFrame to ensure DOM is updated before scrolling
        requestAnimationFrame(() => {
            const chatContainer = this.shadowRoot.getElementById("chat-container");
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });
    }

    async _connect() {
        if (!this._hass) return false;
        if (this._connected) return true;
        if (this._connecting) return false;  // Already connecting

        this._connecting = true;
        this._render();  // Show "Connecting..." state

        try {
            // Check for stored resumption handle
            const storageKey = `gemini_live_handle`;
            const storedHandle = localStorage.getItem(storageKey);

            const connectParams = {
                type: `${DOMAIN}/connect`,
            };

            // Include resumption handle if available
            if (storedHandle) {
                connectParams.resumption_handle = storedHandle;
                logWS('send', 'CONNECT (resume)', connectParams);
            } else {
                logWS('send', 'CONNECT', connectParams);
            }

            const result = await this._hass.callWS(connectParams);
            logWS('recv', 'CONNECT_RESULT', result);

            this._connected = true;
            this._connecting = false;
            this._goAwayWarning = null;
            this._render();
            return true;
        } catch (e) {
            console.error("Failed to connect:", e);
            logWS('recv', 'CONNECT_ERROR', { error: e.message });
            // Clear invalid handle if resume failed
            const storageKey = `gemini_live_handle`;
            localStorage.removeItem(storageKey);
            this._connecting = false;
            this._render();
            return false;
        }
    }

    async _reconnectWithHandle() {
        if (!this._hass || !this._resumptionHandle) return;

        console.log("Attempting automatic reconnection with resumption handle");
        try {
            await this._hass.callWS({
                type: `${DOMAIN}/connect`,
                resumption_handle: this._resumptionHandle,
            });
            this._connected = true;
            this._goAwayWarning = null;
            this._addMessage("system", "Reconnected successfully");
            this._render();
        } catch (e) {
            console.error("Failed to reconnect:", e);
            this._addMessage("system", "Failed to reconnect. Please connect manually.");
            this._render();
        }
    }

    async _disconnect() {
        if (!this._hass) return;

        try {
            logWS('send', 'DISCONNECT', {});
            const result = await this._hass.callWS({
                type: `${DOMAIN}/disconnect`,
            });
            logWS('recv', 'DISCONNECT_RESULT', result);
            this._connected = false;
            this._cleanupPlaybackAudio();
            this._render();
        } catch (e) {
            console.error("Failed to disconnect:", e);
            logWS('recv', 'DISCONNECT_ERROR', { error: e.message });
        }
    }

    async _startListening() {
        if (!this._hass) return;
        if (this._connecting) return;  // Already connecting, wait
        if (this._listening) return;  // Already listening
        // Mark that a listen request is pending until backend confirms
        this._pendingListen = true;
        this._render();

        try {
            // First, ensure we're connected - show connecting state to user
            if (!this._connected) {
                logWS('send', 'CONNECTING_BEFORE_LISTEN', {});
                const connected = await this._connect();
                if (!connected) {
                    this._addMessage("system", "Failed to connect. Please try again.");
                    this._render();
                    return;
                }
            }

            // Call backend start_listening FIRST - this ensures connection is ready
            // The backend will auto-connect if needed
            logWS('send', 'START_LISTENING', {});
            const result = await this._hass.callWS({
                type: `${DOMAIN}/start_listening`,
            });
            logWS('recv', 'START_LISTENING_RESULT', result);

            // Mark that audio initialization is starting so UI stays in loading state
            this._audioInitializing = true;
            this._render();

            // Only start recording AFTER backend confirms it's ready
            await this._startRecording();

            // Do not flip the UI to `listening` here — wait until local audio
            // initialization completes in `_setupAudioProcessing`. Keep `_pendingListen`
            // true until audio is ready so the UI shows a loading/connecting state.
            this._connected = true;  // Ensure frontend state is in sync
            // `_pendingListen` will be cleared in `_setupAudioProcessing` when ready
            this._render();
        } catch (e) {
            console.error("Failed to start listening:", e);
            logWS('recv', 'START_LISTENING_ERROR', { error: e.message });
            this._addMessage("system", `Failed to start: ${e.message}`);
            this._pendingListen = false;
            this._render();
        }
    }

    async _stopListening() {
        if (!this._hass) return;

        try {
            this._stopRecording();

            logWS('send', 'STOP_LISTENING', {});
            const result = await this._hass.callWS({
                type: `${DOMAIN}/stop_listening`,
            });
            logWS('recv', 'STOP_LISTENING_RESULT', result);
            this._listening = false;
            this._render();
        } catch (e) {
            console.error("Failed to stop listening:", e);
            logWS('recv', 'STOP_LISTENING_ERROR', { error: e.message });
        }
    }

    async _startRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("getUserMedia is not supported");
        }

        // Get microphone access FIRST - don't specify sample rate, use browser's native rate
        this._mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });

        // Set up audio processing AFTER getting stream
        await this._setupAudioProcessing();
    }

    async _setupAudioProcessing() {
        // Create AudioContext WITHOUT specifying sample rate - it will use system default
        // This ensures it matches the MediaStream's sample rate
        this._audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Wait for context to be ready
        if (this._audioContext.state === "suspended") {
            await this._audioContext.resume();
        }

        const nativeSampleRate = this._audioContext.sampleRate;
        const targetSampleRate = 16000; // Gemini expects 16kHz

        console.log(`Audio: native=${nativeSampleRate}Hz, target=${targetSampleRate}Hz`);

        const source = this._audioContext.createMediaStreamSource(this._mediaStream);

        // Use ScriptProcessor for wider compatibility
        const bufferSize = 4096;
        const processor = this._audioContext.createScriptProcessor(bufferSize, 1, 1);

        processor.onaudioprocess = (e) => {
            if (!this._listening) return;

            // Optionally mute mic while AI is speaking to prevent feedback
            if (this._muteWhileSpeaking && this._speaking) return;

            const inputData = e.inputBuffer.getChannelData(0);

            // Update visualizer
            this._updateVisualizer(inputData);

            // Resample from native rate to target rate (16kHz)
            const resampledData = this._resampleTo16kHz(inputData, nativeSampleRate, targetSampleRate);

            // Convert Float32 to Int16 PCM
            const pcmData = this._floatTo16BitPCM(resampledData);

            // Convert to base64 and send
            const base64Audio = this._arrayBufferToBase64(pcmData);
            this._sendAudio(base64Audio);
        };

        source.connect(processor);
        processor.connect(this._audioContext.destination);

        this._audioProcessor = processor;
        // Audio is now initialized and ready; clear initializing flag
        this._audioInitializing = false;

        // If a listen request is pending (we started the flow), mark listening active now
        if (this._pendingListen) {
            this._listening = true;
            this._pendingListen = false;
        }

        // Ensure connected state is in sync and re-render UI
        this._connected = this._connected || false;
        this._render();
    }

    _resampleTo16kHz(inputData, fromRate, toRate) {
        if (fromRate === toRate) {
            return inputData;
        }

        const ratio = fromRate / toRate;
        const outputLength = Math.round(inputData.length / ratio);
        const output = new Float32Array(outputLength);

        for (let i = 0; i < outputLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, inputData.length - 1);
            const t = srcIndex - srcIndexFloor;

            // Linear interpolation
            output[i] = inputData[srcIndexFloor] * (1 - t) + inputData[srcIndexCeil] * t;
        }

        return output;
    }

    _stopRecording() {
        if (this._audioProcessor) {
            this._audioProcessor.disconnect();
            this._audioProcessor = null;
        }

        if (this._audioContext) {
            this._audioContext.close();
            this._audioContext = null;
        }

        if (this._mediaStream) {
            this._mediaStream.getTracks().forEach(track => track.stop());
            this._mediaStream = null;
        }
    }

    _floatTo16BitPCM(float32Array) {
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        return buffer;
    }

    _arrayBufferToBase64(buffer) {
        let binary = "";
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    async _sendAudio(base64Audio) {
        if (!this._hass) return;

        try {
            // Log audio send (without the actual base64 data for brevity)
            // logWS('send', 'SEND_AUDIO', { size: base64Audio.length });
            await this._hass.callWS({
                type: `${DOMAIN}/send_audio`,
                audio: base64Audio,
            });
        } catch (e) {
            console.error("Failed to send audio:", e);
            logWS('recv', 'SEND_AUDIO_ERROR', { error: e.message });
        }
    }

    async _sendText(text) {
        if (!this._hass || !text.trim()) return;

        try {
            this._addMessage("user", text);
            this._render();

            logWS('send', 'SEND_TEXT', { text: text });
            const result = await this._hass.callWS({
                type: `${DOMAIN}/send_text`,
                text: text,
            });
            logWS('recv', 'SEND_TEXT_RESULT', result);

            if (result.text || result.audio_transcript) {
                const respText = result.text || result.audio_transcript;
                if (this._currentStreamingIndex !== null) {
                    const msg = this._messages[this._currentStreamingIndex];
                    if (msg && msg.role === "assistant") {
                        msg.text = respText;
                        msg.streaming = false;
                    } else {
                        this._addMessage("assistant", respText);
                    }
                } else {
                    this._addMessage("assistant", respText);
                }
                this._render();
            }
        } catch (e) {
            console.error("Failed to send text:", e);
            logWS('recv', 'SEND_TEXT_ERROR', { error: e.message });
            this._addMessage("system", `Error: ${e.message}`);
            this._render();
        }
    }

    async _playAudio(base64Audio) {
        // Use streaming playback with scheduled audio for smooth, gapless playback
        // This schedules audio chunks to play back-to-back without waiting

        // Create or reuse playback context - MUST be 24kHz for Gemini output
        if (!this._playbackAudioContext || this._playbackAudioContext.state === 'closed') {
            this._playbackAudioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 24000,
            });
            this._nextPlayTime = 0;
            this._playbackStarted = false;
        }

        // Resume if suspended (needed for autoplay policies)
        if (this._playbackAudioContext.state === 'suspended') {
            await this._playbackAudioContext.resume();
        }

        try {
            // Decode base64 to raw bytes
            const audioData = atob(base64Audio);
            const arrayBuffer = new ArrayBuffer(audioData.length);
            const view = new Uint8Array(arrayBuffer);
            for (let i = 0; i < audioData.length; i++) {
                view[i] = audioData.charCodeAt(i);
            }

            // Convert 16-bit signed PCM (little-endian) to Float32
            // Gemini outputs 24kHz, 16-bit, mono PCM
            const samples = new Int16Array(arrayBuffer);
            const floatSamples = new Float32Array(samples.length);
            for (let i = 0; i < samples.length; i++) {
                floatSamples[i] = samples[i] / 32768.0;
            }

            // Create audio buffer at 24kHz (Gemini output sample rate)
            const audioBuffer = this._playbackAudioContext.createBuffer(1, floatSamples.length, 24000);
            audioBuffer.getChannelData(0).set(floatSamples);

            // Calculate duration of this chunk
            const chunkDuration = audioBuffer.length / 24000;

            // Get current time
            const currentTime = this._playbackAudioContext.currentTime;

            // If this is the first chunk or we've fallen behind, reset timing
            if (!this._playbackStarted || this._nextPlayTime < currentTime) {
                this._nextPlayTime = currentTime + 0.02; // Small buffer for smooth start
                this._playbackStarted = true;
            }

            // Create and schedule the audio source
            const source = this._playbackAudioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this._playbackAudioContext.destination);

            // Schedule to play at the next available slot
            source.start(this._nextPlayTime);

            // Update next play time for the next chunk
            this._nextPlayTime += chunkDuration;

            // Update speaking state
            if (!this._speaking) {
                this._speaking = true;
                this._render();
            }

        } catch (e) {
            console.error("Audio playback error:", e);
            logWS('recv', 'AUDIO_PLAYBACK_ERROR', { error: e.message });
        }
    }

    _render() {
        const title = this._config.title || "Gemini Live";
        const showTranscript = this._config.show_transcript !== false;

        // Determine status
        let statusText = "Disconnected";
        let statusClass = "";
        // If we're actively connecting, waiting for start_listening confirmation,
        // or initializing audio, show connecting
        if (this._connecting || this._pendingListen || this._audioInitializing) {
            statusText = "Connecting...";
            statusClass = "connecting";
        } else if (this._speaking) {
            statusText = "Speaking...";
            statusClass = "speaking";
        } else if (this._listening) {
            statusText = "Listening...";
            statusClass = "listening";
        } else if (this._connected) {
            statusText = "Connected";
            statusClass = "connected";
        }

        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                }
                
                ha-card {
                    padding: 16px;
                }
                
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 16px;
                }
                
                .title {
                    font-size: 1.2em;
                    font-weight: 500;
                }
                
                .status {
                    display: flex;
                    gap: 8px;
                    align-items: center;
                }
                
                .status-dot {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background-color: var(--disabled-color, #9e9e9e);
                }
                
                .status-dot.connected {
                    background-color: var(--success-color, #4caf50);
                }
                
                .status-dot.connecting {
                    background-color: var(--warning-color, #ff9800);
                    animation: pulse 0.8s infinite;
                }
                
                .status-dot.listening {
                    background-color: var(--warning-color, #ff9800);
                    animation: pulse 1s infinite;
                }
                
                .status-dot.speaking {
                    background-color: var(--info-color, #2196f3);
                    animation: pulse 0.5s infinite;
                }
                
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                
                @keyframes pulse-button {
                    0%, 100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); }
                    50% { box-shadow: 0 0 0 20px rgba(244, 67, 54, 0); }
                }
                
                .visualizer {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 40px;
                    gap: 3px;
                    margin: 8px 0;
                }
                
                .visualizer-bar {
                    width: 4px;
                    background-color: var(--primary-color);
                    border-radius: 2px;
                    transition: height 0.1s ease;
                }
                
                .controls {
                    display: flex;
                    justify-content: center;
                    margin: 16px 0;
                }
                
                .mic-button {
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    border: none;
                    background-color: var(--primary-color);
                    color: white;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                }
                
                .mic-button:hover {
                    transform: scale(1.05);
                }
                
                .mic-button:active {
                    transform: scale(0.95);
                }
                
                .mic-button.connecting {
                    background-color: var(--warning-color, #ff9800);
                    cursor: wait;
                    animation: pulse-button 1s infinite;
                }
                
                .mic-button.connecting:hover {
                    transform: none;
                }
                
                .mic-button.listening {
                    background-color: var(--error-color, #f44336);
                    animation: pulse-button 1s infinite;
                }
                
                .mic-button.processing {
                    background-color: var(--warning-color, #ff9800);
                }
                
                .mic-button svg {
                    width: 32px;
                    height: 32px;
                }
                
                .chat-container {
                    margin-top: 16px;
                    max-height: 300px;
                    overflow-y: auto;
                    padding: 12px;
                    background-color: var(--secondary-background-color);
                    border-radius: 12px;
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                
                .chat-message {
                    max-width: 80%;
                    padding: 10px 14px;
                    border-radius: 18px;
                    font-size: 0.95em;
                    line-height: 1.4;
                    word-wrap: break-word;
                }
                
                .chat-message.user {
                    align-self: flex-end;
                    background-color: var(--primary-color);
                    color: white;
                    border-bottom-right-radius: 4px;
                }
                
                .chat-message.assistant {
                    align-self: flex-start;
                    background-color: var(--card-background-color, #fff);
                    color: var(--primary-text-color);
                    border: 1px solid var(--divider-color, #e0e0e0);
                    border-bottom-left-radius: 4px;
                }
                
                .chat-message.system {
                    align-self: center;
                    background-color: transparent;
                    color: var(--secondary-text-color);
                    font-size: 0.85em;
                    font-style: italic;
                    padding: 4px 8px;
                }
                
                .chat-message .timestamp {
                    font-size: 0.7em;
                    opacity: 0.7;
                    margin-top: 4px;
                    display: block;
                }
                
                .chat-message.user .timestamp {
                    text-align: right;
                }
                
                .chat-message.assistant .timestamp {
                    text-align: left;
                }
                
                .typing-indicator {
                    align-self: flex-start;
                    padding: 10px 14px;
                    background-color: var(--card-background-color, #fff);
                    border: 1px solid var(--divider-color, #e0e0e0);
                    border-radius: 18px;
                    border-bottom-left-radius: 4px;
                    display: flex;
                    gap: 4px;
                }
                
                .typing-indicator span {
                    width: 8px;
                    height: 8px;
                    background-color: var(--secondary-text-color);
                    border-radius: 50%;
                    animation: typing 1.4s infinite ease-in-out;
                }
                
                .typing-indicator span:nth-child(1) { animation-delay: 0s; }
                .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
                .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
                
                @keyframes typing {
                    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
                    30% { transform: translateY(-4px); opacity: 1; }
                }
                
                .streaming-cursor {
                    animation: blink 1s infinite;
                    color: var(--primary-color);
                }
                
                @keyframes blink {
                    0%, 50% { opacity: 1; }
                    51%, 100% { opacity: 0; }
                }
                
                .chat-message.streaming {
                    opacity: 0.9;
                }
                
                .live-transcript {
                    align-self: flex-end;
                    max-width: 80%;
                    padding: 10px 14px;
                    background-color: var(--primary-color);
                    opacity: 0.7;
                    color: white;
                    border-radius: 18px;
                    border-bottom-right-radius: 4px;
                    font-size: 0.95em;
                    font-style: italic;
                }
                
                .input-row {
                    display: flex;
                    gap: 8px;
                    margin-top: 16px;
                }
                
                .text-input {
                    flex: 1;
                    padding: 10px 14px;
                    border: 1px solid var(--divider-color, #e0e0e0);
                    border-radius: 24px;
                    outline: none;
                    font-size: 14px;
                    background-color: var(--card-background-color, #fff);
                    color: var(--primary-text-color, #000);
                }
                
                .text-input:focus {
                    border-color: var(--primary-color);
                }
                
                .send-btn {
                    padding: 10px 20px;
                    background-color: var(--primary-color);
                    color: white;
                    border: none;
                    border-radius: 24px;
                    cursor: pointer;
                    font-weight: 500;
                    transition: all 0.2s;
                }
                
                .send-btn:hover {
                    opacity: 0.9;
                }
                
                .send-btn:disabled {
                    background-color: var(--disabled-color, #9e9e9e);
                    cursor: not-allowed;
                }
                
                .file-input-row {
                    display: flex;
                    gap: 8px;
                    margin-top: 12px;
                    align-items: center;
                }
                
                .file-label {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding: 8px 16px;
                    background-color: var(--secondary-background-color);
                    border: 1px solid var(--divider-color, #e0e0e0);
                    border-radius: 20px;
                    cursor: pointer;
                    font-size: 13px;
                    color: var(--primary-text-color);
                    transition: all 0.2s;
                }
                
                .file-label:hover {
                    background-color: var(--divider-color);
                }
                
                .file-label svg {
                    width: 18px;
                    height: 18px;
                }
                
                .file-input {
                    display: none;
                }
                
                .file-name {
                    flex: 1;
                    font-size: 13px;
                    color: var(--secondary-text-color);
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }
                
                .options-row {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    margin-top: 16px;
                    font-size: 0.85em;
                    color: var(--secondary-text-color);
                }
                
                .toggle-switch {
                    position: relative;
                    width: 40px;
                    height: 22px;
                    background-color: var(--disabled-color, #ccc);
                    border-radius: 11px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }
                
                .toggle-switch.active {
                    background-color: var(--primary-color);
                }
                
                .toggle-switch::after {
                    content: '';
                    position: absolute;
                    width: 18px;
                    height: 18px;
                    background-color: white;
                    border-radius: 50%;
                    top: 2px;
                    left: 2px;
                    transition: transform 0.3s;
                }
                
                .toggle-switch.active::after {
                    transform: translateX(18px);
                }
                
                .option-label {
                    cursor: pointer;
                    user-select: none;
                }
                
                .error-message {
                    color: var(--error-color);
                    font-size: 0.9em;
                    margin-top: 12px;
                    text-align: center;
                }
                
                .go-away-warning {
                    background-color: var(--warning-color, #ff9800);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 8px;
                    text-align: center;
                    margin-bottom: 12px;
                    font-weight: 500;
                    animation: pulse 1s infinite;
                }
            </style>
            
            <ha-card>
                <div class="header">
                    <div class="title">${title}</div>
                    <div class="status">
                        <div class="status-dot ${statusClass}"></div>
                        <span>${statusText}</span>
                    </div>
                </div>
                
                ${this._goAwayWarning ? `
                    <div class="go-away-warning">
                        ⚠️ Connection terminating in ${this._goAwayWarning}s
                    </div>
                ` : ""}
                
                <div class="visualizer" id="visualizer" style="${this._listening ? '' : 'display: none;'}">
                    ${Array(20).fill(0).map(() => '<div class="visualizer-bar" style="height: 5px;"></div>').join('')}
                </div>
                
                <div class="controls">
                    <button class="mic-button ${this._connecting || this._pendingListen || this._audioInitializing ? 'connecting' : this._listening ? 'listening' : ''}" id="mic-button" ${this._connecting || this._pendingListen || this._audioInitializing ? 'disabled' : ''}>
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            ${this._connecting ? `
                                <!-- Loading/connecting icon -->
                                <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="31.4" stroke-dashoffset="0">
                                    <animate attributeName="stroke-dashoffset" values="0;62.8" dur="1s" repeatCount="indefinite"/>
                                </circle>
                            ` : this._listening ? `
                                <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1.2-9.1c0-.66.54-1.2 1.2-1.2.66 0 1.2.54 1.2 1.2l-.01 6.2c0 .66-.53 1.2-1.19 1.2-.66 0-1.2-.54-1.2-1.2V4.9zm6.5 6.1c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/>
                            ` : `
                                <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/>
                            `}
                        </svg>
                    </button>
                </div>
                
                ${showTranscript ? `
                    <div class="chat-container" id="chat-container">
                        ${this._messages.length === 0 ? `
                            <div class="chat-message system">Start a conversation...</div>
                        ` : this._messages.map(msg => `
                            <div class="chat-message ${msg.role}${msg.streaming ? ' streaming' : ''}">
                                ${msg.text}${msg.streaming ? '<span class="streaming-cursor">▌</span>' : ''}
                                ${msg.role !== 'system' ? `<span class="timestamp">${this._formatTime(msg.timestamp)}</span>` : ''}
                            </div>
                        `).join('')}
                        ${this._inputTranscript && this._listening ? `
                            <div class="live-transcript">${this._inputTranscript}...</div>
                        ` : ''}
                    </div>
                ` : ""}
                
                <div class="input-row">
                    <input type="text" class="text-input" id="text-input" ${this._connected ? 'placeholder="Type a message..."' : 'placeholder="Not connected" disabled'}>
                    <button class="send-btn" id="send-btn" ${this._connected ? '' : 'disabled'}>Send</button>
                </div>
                
                <div class="file-input-row">
                    <label class="file-label" for="file-input">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/>
                        </svg>
                        Attach file
                    </label>
                    <input type="file" class="file-input" id="file-input" accept="image/*,audio/*">
                    <span class="file-name" id="file-name"></span>
                </div>
                
                <div class="options-row">
                    <span class="option-label" id="mute-label">Mute while AI speaks</span>
                    <div class="toggle-switch ${this._muteWhileSpeaking ? 'active' : ''}" id="mute-toggle"></div>
                </div>
                
                <div class="error-message" id="error-message" style="display: none;"></div>
            </ha-card>
        `;

        // Add event listeners
        this._addEventListeners();

        // Scroll chat to bottom
        this._scrollChatToBottom();
    }

    _addEventListeners() {
        const micButton = this.shadowRoot.getElementById("mic-button");
        const textInput = this.shadowRoot.getElementById("text-input");
        const sendBtn = this.shadowRoot.getElementById("send-btn");
        const fileInput = this.shadowRoot.getElementById("file-input");
        const muteToggle = this.shadowRoot.getElementById("mute-toggle");
        const muteLabel = this.shadowRoot.getElementById("mute-label");

        // Mic button - toggle listening
        if (micButton) {
            micButton.addEventListener("click", async () => {
                if (this._listening) {
                    await this._stopListening();
                } else {
                    await this._startListening();
                }
            });
        }

        // Text input - send on Enter
        if (textInput) {
            textInput.addEventListener("keypress", (e) => {
                if (e.key === "Enter") {
                    this._sendText(textInput.value);
                    textInput.value = "";
                }
            });
        }

        // Send button
        if (sendBtn) {
            sendBtn.addEventListener("click", () => {
                const input = this.shadowRoot.getElementById("text-input");
                if (input && input.value.trim()) {
                    this._sendText(input.value);
                    input.value = "";
                }
            });
        }

        // File input
        if (fileInput) {
            fileInput.addEventListener("change", (e) => {
                const file = e.target.files[0];
                if (file) {
                    this._selectedFile = file;
                    const fileName = this.shadowRoot.getElementById("file-name");
                    if (fileName) {
                        fileName.textContent = file.name;
                    }
                    // Send the file
                    this._sendFile(file);
                }
            });
        }

        // Mute toggle
        if (muteToggle) {
            const toggleMute = () => {
                this._muteWhileSpeaking = !this._muteWhileSpeaking;
                muteToggle.classList.toggle("active", this._muteWhileSpeaking);
                console.log("Mute while speaking:", this._muteWhileSpeaking);
            };
            muteToggle.addEventListener("click", toggleMute);
            if (muteLabel) {
                muteLabel.addEventListener("click", toggleMute);
            }
        }
    }

    async _sendFile(file) {
        if (!this._hass || !file) return;

        try {
            // Connect if not connected
            if (!this._connected) {
                await this._connect();
            }

            // Read file as base64
            const reader = new FileReader();
            reader.onload = async (e) => {
                const base64 = e.target.result.split(",")[1]; // Remove data:... prefix
                const mimeType = file.type;

                try {
                    if (mimeType.startsWith("image/")) {
                        // Send image
                        await this._hass.callWS({
                            type: `${DOMAIN}/send_image`,
                            image: base64,
                            mime_type: mimeType,
                        });
                        this._inputTranscript = `[Sent image: ${file.name}]`;
                    } else if (mimeType.startsWith("audio/")) {
                        // Send audio
                        await this._hass.callWS({
                            type: `${DOMAIN}/send_audio`,
                            audio: base64,
                        });
                        this._inputTranscript = `[Sent audio: ${file.name}]`;
                    }
                    this._render();
                } catch (err) {
                    console.error("Failed to send file:", err);
                    this._showError(err.message);
                }
            };
            reader.readAsDataURL(file);
        } catch (e) {
            console.error("Failed to read file:", e);
            this._showError(e.message);
        }
    }

    _showError(message) {
        const errorEl = this.shadowRoot.getElementById("error-message");
        if (errorEl) {
            errorEl.textContent = message;
            errorEl.style.display = "block";
            setTimeout(() => {
                errorEl.style.display = "none";
            }, 5000);
        }
    }

    _updateVisualizer(inputData) {
        const visualizer = this.shadowRoot.getElementById("visualizer");
        if (!visualizer) return;

        const bars = visualizer.querySelectorAll(".visualizer-bar");
        const bufferLength = inputData.length;
        const step = Math.floor(bufferLength / bars.length);

        bars.forEach((bar, index) => {
            let sum = 0;
            for (let i = 0; i < step; i++) {
                sum += Math.abs(inputData[index * step + i] || 0);
            }
            const average = sum / step;
            const height = Math.max(5, Math.min(40, average * 200));
            bar.style.height = `${height}px`;
        });
    }
}

// Card Editor - Simplified, entry_id auto-detected
class GeminiLiveCardEditor extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: "open" });
        this._config = {};
    }

    set hass(hass) {
        // Not needed - entry_id is auto-detected by the card
    }

    setConfig(config) {
        this._config = config;
        this._render();
    }

    _fireConfigChanged() {
        const event = new CustomEvent("config-changed", {
            detail: { config: this._config },
            bubbles: true,
            composed: true,
        });
        this.dispatchEvent(event);
    }

    _render() {
        this.shadowRoot.innerHTML = `
            <style>
                .form-row {
                    margin-bottom: 16px;
                }
                label {
                    display: block;
                    margin-bottom: 4px;
                    font-weight: 500;
                }
                input, select {
                    width: 100%;
                    padding: 8px;
                    border: 1px solid var(--divider-color, #e0e0e0);
                    border-radius: 4px;
                    background-color: var(--card-background-color, #fff);
                    color: var(--primary-text-color, #000);
                    box-sizing: border-box;
                }
                .info {
                    color: var(--secondary-text-color, #666);
                    padding: 8px;
                    background-color: rgba(33, 150, 243, 0.1);
                    border-radius: 4px;
                    margin-bottom: 16px;
                    font-size: 0.9em;
                }
            </style>
            
            <div class="info">
                The card will automatically detect your Gemini Live integration.
            </div>
            
            <div class="form-row">
                <label>Title</label>
                <input type="text" id="title" value="${this._config.title || "Gemini Live"}">
            </div>
            
            <div class="form-row">
                <label>
                    <input type="checkbox" id="show_transcript" 
                           ${this._config.show_transcript !== false ? "checked" : ""}>
                    Show Transcript
                </label>
            </div>
            
            <div class="form-row">
                <label>
                    <input type="checkbox" id="keep_mic_when_hidden" 
                           ${this._config.keep_mic_when_hidden !== false ? "checked" : ""}>
                    Keep microphone active when tab is hidden
                </label>
            </div>
        `;

        this._addEventListeners();
    }

    _addEventListeners() {
        const title = this.shadowRoot.getElementById("title");
        const showTranscript = this.shadowRoot.getElementById("show_transcript");

        if (title) {
            title.addEventListener("change", () => this._updateConfig());
        }
        if (showTranscript) {
            showTranscript.addEventListener("change", () => this._updateConfig());
        }
        const keepMicEl = this.shadowRoot.getElementById("keep_mic_when_hidden");
        if (keepMicEl) {
            keepMicEl.addEventListener("change", () => this._updateConfig());
        }
    }

    _updateConfig() {
        const titleEl = this.shadowRoot.getElementById("title");
        const showTranscriptEl = this.shadowRoot.getElementById("show_transcript");
        const keepMicEl = this.shadowRoot.getElementById("keep_mic_when_hidden");

        this._config = {
            ...this._config,
            title: titleEl ? titleEl.value : this._config.title,
            show_transcript: showTranscriptEl ? showTranscriptEl.checked : this._config.show_transcript,
            keep_mic_when_hidden: keepMicEl ? keepMicEl.checked : (this._config.keep_mic_when_hidden !== undefined ? this._config.keep_mic_when_hidden : true),
        };

        this._fireConfigChanged();
    }
}

// Register custom elements only if not already registered. Wrap in
// try/catch and log so browser console shows why registration may fail.
try {
        if (!customElements.get("gemini-live-card")) {
        customElements.define("gemini-live-card", GeminiLiveCard);
        console.debug("GeminiLiveCard: custom element defined");
    } else {
        console.debug("GeminiLiveCard: custom element already defined");
    }
} catch (e) {
    console.error("GeminiLiveCard: failed to define custom element:", e);
}

try {
    if (!customElements.get("gemini-live-card-editor")) {
        customElements.define("gemini-live-card-editor", GeminiLiveCardEditor);
        console.debug("GeminiLiveCard: editor element defined");
    } else {
        console.debug("GeminiLiveCard: editor element already defined");
    }
} catch (e) {
    console.error("GeminiLiveCard: failed to define editor element:", e);
}

// Register with Home Assistant (prevent duplicates)
try {
    window.customCards = window.customCards || [];
        if (!window.customCards.find(card => card.type === "gemini-live-card")) {
        window.customCards.push({
            type: "gemini-live-card",
            name: "Gemini Live Card",
            description: "A card for interacting with Gemini Live API",
            preview: true,
        });
        console.debug("GeminiLiveCard: pushed to window.customCards");
    } else {
        console.debug("GeminiLiveCard: already in window.customCards");
    }
} catch (e) {
    console.error("GeminiLiveCard: failed to register customCards:", e);
}
