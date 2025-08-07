# --- HTML Content for Client ---


html_content2="""<!DOCTYPE html>
<html>
<head>
    <title>FastAPI Mic & Tab Audio WebSocket</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        button { padding: 10px 15px; margin: 5px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:disabled { background-color: #cccccc; cursor: not-allowed; }
        input, select { padding: 5px; margin-right: 10px; }

        .tab-content {
            border: 1px solid #ccc;
            padding: 10px;
            min-height: 200px;
            background-color: #e0e0e0;
            border-radius: 5px;
            overflow-y: auto;
            max-height: 400px;
        }
        .message-item {
            padding: 8px;
            margin-bottom: 5px;
            background-color: #f9f9f9;
            border-radius: 5px;
            border-left: 3px solid #007bff;
            word-wrap: break-word;
        }
        .suggestion-item {
            padding: 8px;
            margin-bottom: 5px;
            background-color: #e6ffe6;
            border-radius: 5px;
            border-left: 3px solid #28a745;
            word-wrap: break-word;
        }
        .tab-container {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .tab-panel {
            flex: 1;
        }
        .input-group {
            margin-bottom: 15px;
        }
        .input-row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <h1>Zoom Transcriptions and Suggestions</h1>
 
    <!-- Input selection -->
    <div class="input-group">
        <div class="input-row">
            <select id="inputType">
                <option value="meeting_url">Meeting URL</option>
                <option value="bot_id">Bot ID</option>
            </select>
            <input type="text" id="meetingUrlInput" placeholder="Enter Meeting URL" style="width: 400px;" />
            <input type="text" id="botIdInput" placeholder="Enter Bot ID" style="width: 200px; display: none;" />
            <button id="startBotButton">Start Bot</button>
        </div>
    </div>

    <div id="micTab" class="tab-content" style="display: block;">
        <h2>Microphone Transcriptions & Suggestions</h2>
        
        <div class="tab-container">
            <div class="tab-panel">
                <h3>Transcriptions</h3>
                <ul id="micTranscriptionMessages" class="tab-content">
                    <!-- Transcription messages will appear here -->
                </ul>
            </div>
            
            <div class="tab-panel">
                <h3>Suggestions</h3>
                <ul id="micSuggestionMessages" class="tab-content">
                    <!-- Suggestion messages will appear here -->
                </ul>
            </div>
        </div>
    </div>

    <script>
        const startBotButton = document.getElementById('startBotButton');
        const meetingUrlInput = document.getElementById('meetingUrlInput');
        const botIdInput = document.getElementById('botIdInput');
        const inputType = document.getElementById('inputType');
        const micTranscriptionMessages = document.getElementById('micTranscriptionMessages');
        const micSuggestionMessages = document.getElementById('micSuggestionMessages');
   
        let wsMic = null; // Keep WebSocket reference

        // Toggle between meeting URL and bot ID inputs
        inputType.addEventListener('change', function() {
            if (this.value === 'meeting_url') {
                meetingUrlInput.style.display = 'inline-block';
                botIdInput.style.display = 'none';
            } else {
                meetingUrlInput.style.display = 'none';
                botIdInput.style.display = 'inline-block';
            }
        });

        startBotButton.onclick = async function() {
            const inputValue = inputType.value === 'meeting_url' 
                ? meetingUrlInput.value.trim()
                : botIdInput.value.trim();

            if (!inputValue) {
                alert(`Please enter a valid ${inputType.value === 'meeting_url' ? 'Meeting URL' : 'Bot ID'}`);
                return;
            }

            try {
                // Close previous WebSocket if needed
                if (wsMic && wsMic.readyState === WebSocket.OPEN) {
                    wsMic.close();
                }

                let wsUrl;
                if (inputType.value === 'meeting_url') {
                    // Call API to start new bot
                    const encodedMeetingUrl = encodeURIComponent(inputValue);
                    const response = await fetch(`https://transcribe.testir.xyz/start-recall-bot?meeting_url=${encodedMeetingUrl}`, {
                        method: "POST"
                    });

                    if (!response.ok) {
                        throw new Error(`API error: ${response.status}`);
                    }

                    const data = await response.json();
                    const botId = data.id;

                    if (!botId) {
                        throw new Error('No bot_id found in API response.');
                    }

                    wsUrl = `wss://transcribe.testir.xyz/ws_mic/${botId}`;
                } else {
                    // Connect directly using bot ID
                    wsUrl = `wss://transcribe.testir.xyz/ws_mic/${inputValue}`;
                }

                wsMic = new WebSocket(wsUrl);

                wsMic.onopen = function(event) {
                    console.log("Mic WebSocket connection opened:", event);
                    addMessageToTab(micTranscriptionMessages, `<em>Connected to microphone service</em>`, 'message-item');
                };

                wsMic.onmessage = function(event) {
                    handleWsMessage(event.data);
                };

                wsMic.onclose = function(event) {
                    console.log("WebSocket connection closed:", event);
                    addMessageToTab(micTranscriptionMessages, '<em>Disconnected from service</em>', 'message-item');
                };

                wsMic.onerror = function(event) {
                    console.error("WebSocket error:", event);
                    addMessageToTab(micTranscriptionMessages, '<em style="color: red;">WebSocket error!</em>', 'message-item');
                };

            } catch (err) {
                console.error('Error starting bot or connecting WebSocket:', err);
                alert(`Error: ${err.message}`);
            }
        };

        function handleWsMessage(message) {
            console.log("Received message:", message); // Debug log
            
            if (message.includes('Transcription')) {
                // Handle transcription messages
                const isFinal = message.includes('[False]');
                const cleanedMsg = message
                    .replace('MicTranscription[False]:', '')
                    .replace('MicTranscription[True]:', '')
                    .trim();
                
                addMessageToTab(
                    micTranscriptionMessages, 
                    cleanedMsg, 
                    'message-item', 
                    isFinal
                );
            } 
            else if (message.includes('Suggestion')) {
                // Handle suggestion messages
                const cleanedMsg = message
                    .replace('Suggestion:', '')
                    .replace('Suggestion:', '')
                    .trim();
                addSuggestionToTab(micSuggestionMessages, cleanedMsg, 'suggestion-item');
            }
            else {
                // Fallback for other messages
                addMessageToTab(micTranscriptionMessages, message, 'message-item');
            }
        }

        function addSuggestionToTab(tabElement, msg, className) {
            const li = document.createElement('li');
            li.className = className;
            li.innerHTML = `<p>${msg}</p>`;
            tabElement.prepend(li);
         //   tabElement.scrollTop = tabElement.scrollHeight;
        }

        function addMessageToTab(tabElement, msg, className, isFinal = false) {
             if (!isFinal ) {
        // ✅ Update the most recent transcription
        tabElement.firstChild.innerHTML = `<p>${msg}</p>`;
    } else {
        // ➕ Add new transcription message
        const li = document.createElement('li');
        li.className = className;
        li.innerHTML = `<p>${msg.replace('MicTranscription[False]', '').trim()}</p>`;
        
        tabElement.prepend(li);
          
        li.className = className;
        li.innerHTML = `<p></p>`;
        tabElement.prepend(li);
        // tabElement.scrollTop = tabElement.scrollHeight;
    }
        }
    </script>
</body>
</html>
"""
# --- AudioWorklet Processor: `audio-processor.js` ---
# This script runs in a separate thread in the browser to handle audio capture and pre-processing.
audio_processor_js = """
// audio-processor.js
// This AudioWorkletProcessor handles raw audio data from the microphone/tab,
// performs resampling, basic Voice Activity Detection (VAD), and buffers
// audio to send to the server at regular intervals.
class AudioDataProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sampleRate = 16000; // Target sample rate for Whisper
        this.buffer = []; // Buffer to accumulate audio data
        this.lastUpdateTime = 0;
        this.sendInterval = 1000; // Send audio chunks to server every 1000ms (1 second)

        this.inputSampleRate = 0; // Actual sample rate of the input audio stream
        this.resampler = null; // Function to resample audio if needed

        this.silenceThreshold = 0.006; // RMS threshold for Voice Activity Detection (VAD)
        this.minSilentChunks = 1;    // Number of consecutive silent chunks before stopping sending
        this.consecutiveSilentChunks = 2;
        this.wasSendingAudio = false; // Flag to track if we were just sending voice

        // Utility function to determine if an audio buffer is "really silent"
        // This is a stricter check used just before sending to WebSocket.
        this.isReallySilent = (audioBuffer, threshold = 0.001) => {
            if (!audioBuffer || audioBuffer.length === 0) return true;
            let energy = 0;
            for (let i = 0; i < audioBuffer.length; i++) {
                energy += audioBuffer[i] * audioBuffer[i];
            }
            // Calculate RMS (Root Mean Square)
            return Math.sqrt(energy / audioBuffer.length) < threshold;
        };
    }

    // Initializes the resampler function if the input sample rate doesn't match the target
    initResampler(inputSampleRate) {
        if (inputSampleRate === this.sampleRate) {
            this.resampler = null; // No resampling needed
        } else {
            // Simple linear interpolation resampler
            this.resampler = (inputBuffer) => {
                const outputBuffer = new Float32Array(Math.ceil(inputBuffer.length * (this.sampleRate / inputSampleRate)));
                const ratio = inputSampleRate / this.sampleRate;
                for (let i = 0; i < outputBuffer.length; i++) {
                    const index = i * ratio;
                    const lower = Math.floor(index);
                    const upper = Math.ceil(index);
                    const weight = index - lower;

                    if (upper < inputBuffer.length) {
                        outputBuffer[i] = inputBuffer[lower] * (1 - weight) + inputBuffer[upper] * weight;
                    } else {
                        outputBuffer[i] = inputBuffer[lower];
                    }
                }
                return outputBuffer;
            };
        }
    }

    // Main processing loop for incoming audio data
    process(inputs, outputs, parameters) {
        const input = inputs[0]; // Get the first input stream
        if (input.length === 0) {
            return true; // No input data, continue processing
        }

        const inputChannelData = input[0]; // Get the audio data from the first channel

        // Initialize resampler if not already done
        if (this.inputSampleRate === 0) {
            this.inputSampleRate = sampleRate; // `sampleRate` is globally available in AudioWorklet
            this.initResampler(this.inputSampleRate);
        }

        let processedData = inputChannelData;
        if (this.resampler) {
            processedData = this.resampler(inputChannelData); // Resample if necessary
        }

        // Calculate RMS for VAD
        const rms = Math.sqrt(processedData.reduce((sum, value) => sum + value * value, 0) / processedData.length);
        const isVoice = rms > this.silenceThreshold; // Determine if current chunk contains voice

        if (isVoice) {
            this.consecutiveSilentChunks = 5; // Reset silent counter
            this.buffer.push(processedData); // Buffer the voice data
            this.wasSendingAudio = true; // Mark as actively sending voice
        } else {
            this.consecutiveSilentChunks++; // Increment silent counter
            if (this.wasSendingAudio && this.buffer.length > 0) {
                // If we were just sending voice and now detected silence,
                // send the remaining buffered audio to capture the tail end of speech.
                const totalLength = this.buffer.reduce((acc, val) => acc + val.length, 0);
                const combinedBuffer = new Float32Array(totalLength);
                let offset = 0;
                for (const array of this.buffer) {
                    combinedBuffer.set(array, offset);
                    offset += array.length;
                }
                this.port.postMessage(combinedBuffer.buffer, [combinedBuffer.buffer]); // Send the ArrayBuffer
                this.buffer = []; // Clear buffer after sending
                this.wasSendingAudio = false; // Reset flag
            }

            if (this.consecutiveSilentChunks > this.minSilentChunks) {
                // If prolonged silence, clear buffer and reset time to stop sending
                // This prevents sending empty data during long pauses.
                this.buffer = [];
                this.lastUpdateTime = Date.now();
                return true; // No audio to send in this case
            } else {
                // If not prolonged silence (i.e., a brief pause within an utterance),
                // continue to buffer silent chunks to maintain context.
                this.buffer.push(processedData);
            }
        }

        const currentTime = Date.now();
        // If buffer has accumulated enough data OR enough time has passed, send the buffered data.
        if (this.buffer.length > 0 && currentTime - this.lastUpdateTime > this.sendInterval) {
            const totalLength = this.buffer.reduce((acc, val) => acc + val.length, 0);
            const combinedBuffer = new Float32Array(totalLength);
            let offset = 0;
            for (const array of this.buffer) {
                combinedBuffer.set(array, offset);
                offset += array.length;
            }

            this.port.postMessage(combinedBuffer.buffer, [combinedBuffer.buffer]); // Send the ArrayBuffer
            this.buffer = []; // Clear buffer after sending
            this.lastUpdateTime = currentTime;
            this.wasSendingAudio = false; // Reset after sending a chunk
        }

        return true; // Keep the processor active
    }
}

registerProcessor('audio-data-processor', AudioDataProcessor);
"""