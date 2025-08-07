# 🎤 Meet Live Transcription

Meet Live Transcription is a ⚡ FastAPI-based service that provides **real-time transcription** and 🤖 AI-powered question suggestions for meetings. It integrates with 🧑‍💻 Zoom and supports both microphone and tab audio sources.

## ✨ Features

- 📝 Real-time audio transcription from meetings
- 💡 AI-generated question suggestions using OpenAI GPT-4o
- 🔗 Zoom integration (meeting join, signature generation)
- 🪙 Token-based access control
- 🔌 WebSocket endpoints for live audio streaming
- 🖥️ Simple web interface for testing

## 🚀 Getting Started

### 🛠️ Prerequisites

- 🐍 Python 3.8+
- 🔑 OpenAI API key
- 🔑 Zoom SDK credentials

### 📦 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/meet_live_transcription.git
    cd meet_live_transcription
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your `.env` file with the required API keys:**
    ```env
    OPENAI_API_KEY=your_openai_key
    RECALL_KEY=your_recall_key
    XAPI_KEY=your_xapi_key
    ZOOM_CLIENT_ID=your_zoom_client_id
    ZOOM_CLIENT_SECRET=your_zoom_client_secret
    ```

### ▶️ Running the Server

```bash
python main.py
```

The server will start on `http://localhost:8000` by default.

## 📋 API Documentation

### REST Endpoints

#### 🏁 Create Zoom Meeting
**POST** `/create-zoom-meeting`

Creates a new Zoom meeting and automatically starts a Recall bot for transcription.

**Request Body:**
```json
{
    "topic": "Coaching Session"  // Optional, defaults to "Coaching Session"
}
```

**Response:**
```json
{
    "meetingNumber": "123456789",
    "joinUrl": "https://zoom.us/j/123456789",
    "password": "123456",
    "recallBot": {
        "id": "bot_id_here",
        // ... other bot details
    }
}
```

#### 🤖 Start Recall Bot
**POST** `/start-recall-bot`

Starts a Recall bot for an existing meeting with token-based access control.

**Request Parameters:**
- `meeting_url` (required): The meeting URL to join
- `user_id` (required): User identifier for token management
- `user_email` (optional): User email for tracking

**Response:**
```json
{
    "id": "bot_id",
    "status": "joining",
    // ... other bot details
}
```

**Error Responses:**
- `500`: Insufficient tokens or token deduction failed
- `500`: Bot already active for user

#### ✍️ Generate Zoom Signature
**POST** `/generate-zoom-signature`

Generates a signature for Zoom SDK authentication.

**Request Body:**
```json
{
    "meetingNumber": "123456789",
    "role": 1  // 0 for participant, 1 for host
}
```

**Response:**
```json
{
    "signature": "base64_encoded_signature"
}
```

### WebSocket Endpoints

#### 🎤 Microphone Audio WebSocket
**WebSocket** `/ws_mic/{bot_id}`

Real-time WebSocket endpoint for processing microphone audio transcriptions from Recall bots.

**Parameters:**
- `bot_id`: The Recall bot identifier (defaults to "default_bot")

**Message Format:**
Sends JSON events containing transcription data:
```json
{
    "event": "transcript.data",
    "data": {
        "bot": {"id": "bot_id"},
        "recording": {"id": "recording_id"},
        "data": {
            "participant": {
                "name": "Speaker Name",
                "is_host": false
            },
            "words": [
                {
                    "text": "word",
                    "start_timestamp": 12345
                }
            ]
        }
    }
}
```

**Features:**
- 🎯 Real-time AI-powered question suggestions
- 🔄 Automatic token deduction based on usage
- 📝 Conversation history tracking
- ⚡ Smart suggestion throttling and debouncing

#### 🖥️ Tab Audio WebSocket
**WebSocket** `/ws_tab`

WebSocket endpoint for processing tab audio (currently available but less used).

### 🔐 Authentication & Security

The API uses multiple authentication methods:

1. **Environment Variables**: API keys stored securely
2. **Token-based Access Control**: Integration with AI District Agents API for usage tracking
3. **CORS**: Configured to allow cross-origin requests

### 📊 Token Management

The system automatically:
- ✅ Deducts tokens when starting a bot (1/600 minutes initially)
- ⏱️ Tracks usage time and deducts tokens periodically
- ⚠️ Automatically disconnects bots when tokens are insufficient
- 🛡️ Prevents multiple active bots per user

### 🤖 AI Features

The API includes advanced AI capabilities:

#### Question Suggestions
- 🧠 Powered by GPT-4o
- 🎯 Context-aware coaching questions
- ⚡ Real-time generation during conversations
- 🔄 Smart throttling (0.5-second cooldown)
- ⏱️ Debounced processing (1-second delay)

#### Transcription Processing
- 📝 Real-time speech-to-text via AssemblyAI
- 👥 Speaker identification and host detection
- 🔄 Partial and final transcript handling
- 📊 Conversation history management

### 🌐 Web Interface

Access the testing interface at:
- **Main Interface**: `GET /`
- **Audio Processor Script**: `GET /audio-processor.js`

### ⚙️ Configuration

Required environment variables:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# Recall API Configuration
RECALL_API_KEY=your_recall_key

# Token Management
XAPI_KEY=your_xapi_key

# Zoom SDK Configuration
ZOOM_CLIENT_ID=your_zoom_client_id
ZOOM_CLIENT_SECRET=your_zoom_client_secret
ZOOM_ACCOUNT_ID=your_zoom_account_id
```

### 🚀 Usage Examples

#### Creating a Meeting with cURL
```bash
curl -X POST "http://localhost:8000/create-zoom-meeting" \
  -H "Content-Type: application/json" \
  -d '{"topic": "Weekly Coaching Session"}'
```

#### Starting a Bot for Existing Meeting
```bash
curl -X POST "http://localhost:8000/start-recall-bot" \
  -H "Content-Type: application/json" \
  -d 'meeting_url=https://zoom.us/j/123456789&user_id=user123'
```

#### WebSocket Connection (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws_mic/your_bot_id');
ws.onmessage = (event) => {
    console.log('Received:', event.data);
};
```
