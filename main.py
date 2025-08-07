import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
import markdown
import requests as req

from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from openai import AsyncOpenAI, BaseModel
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals ---
connected_clients = {}
last_message_clients = {}
suggestion_locks = {}
active_recall_bots = {}

# --- Load environment and OpenAI client ---
load_dotenv()
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Models ---
class CreateMeetingRequest(BaseModel):
    topic: str = "Coaching Session"

class GenerateSignatureRequest(BaseModel):
    meetingNumber: str
    role: int

# --- Helper Functions ---
def leave_recall_bot_call(bot_id: str):
    import requests
    url = f"https://us-west-2.recall.ai/api/v1/bot/{bot_id}/leave_call/"
    recall_api_key = os.environ.get("RECALL_API_KEY")
    headers = {
        "accept": "application/json",
        "Authorization": recall_api_key
    }
    try:
        response = requests.post(url, headers=headers)
        print(response.text)
    except Exception as e:
        print(f"Error leaving recall bot call: {e}")

def get_zoom_access_token():
    token_url = "https://zoom.us/oauth/token"
    auth_header = base64.b64encode(
        f"{os.getenv('ZOOM_CLIENT_ID')}:{os.getenv('ZOOM_CLIENT_SECRET')}".encode()
    ).decode()
    response = req.post(
        f"{token_url}?grant_type=account_credentials&account_id={os.getenv('ZOOM_ACCOUNT_ID')}",
        headers={
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/json",
        },
    )
    response.raise_for_status()
    print("Zoom Access Token Response:", response.json())
    return response.json()["access_token"]

def start_recall_bot(meeting_url):
    print("Starting Recall Bot...")
    recall_api_key = os.environ.get("RECALL_API_KEY")
    response = req.post(
        "https://us-west-2.recall.ai/api/v1/bot/",
        json={
            "meeting_url": meeting_url,
            "recording_config": {
                "transcript": {
                    "provider": {
                        "assembly_ai_streaming": {
                            "language": "en-US",
                        }
                    }
                },
                "realtime_endpoints": [
                    {
                        "type": "websocket",
                        "url": "wss://836e60e6543f.ngrok-free.app/ws_mic/default-bot",
                        "events": [
                            "transcript.data",
                            "transcript.partial_data",
                        ],
                    }
                ],
            },
        },
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": recall_api_key,
        },
    )
    print(response.json())
    response.raise_for_status()
    return response.json()

# --- Routes ---
@app.post("/create-zoom-meeting")
def create_zoom_meeting(payload: CreateMeetingRequest):
    try:
        access_token = get_zoom_access_token()
        response = req.post(
            "https://api.zoom.us/v2/users/me/meetings",
            json={
                "topic": payload.topic,
                "type": 1,
                "password": "123456",
                "settings": {
                    "host_video": True,
                    "participant_video": True,
                },
            },
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()
        meeting_number = data["id"]
        join_url = data["join_url"]
        password = data["password"]
        bot_response = start_recall_bot(join_url)
        return {
            "meetingNumber": meeting_number,
            "joinUrl": join_url,
            "password": password,
            "recallBot": bot_response,
        }
    except Exception as err:
        return {"error": str(err)}

@app.post("/start-recall-bot")
def start_recall_bot_endpoint(meeting_url: str, user_id: str = None, user_email: str = None):
    print(meeting_url)
    if not meeting_url:
        return {"error": "meeting_url is required"}
    if not user_id:
        return {"error": "user_id is required"}
    if user_id in active_recall_bots:
        return {"error": f"Recall bot already active for user_id {user_id}"}
    try:
        token_api_url = "https://api.aidistrictagents.com/server26/api/token-deduction/deduct-by-minutes"
        payload = {
            "userId": user_id,
            "minutes": 1/6000
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": os.environ.get("XAPI_KEY")
        }
        resp = req.post(token_api_url, data=json.dumps(payload), headers=headers, timeout=10)
        if not resp.ok:
            print(f"Token deduction failed: {resp.status_code} {resp.text}")
            return JSONResponse(status_code=500, content={"error": "Insufficient token"})
    except Exception as e:
        print(f"Exception during token deduction: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    try:
        bot_response = start_recall_bot(meeting_url)
        active_recall_bots[bot_response['id']] = {
            "email": user_email,
            "user_id": user_id,
            "bot_response": bot_response,
            "bot_id": bot_response['id'],
            "timestamp": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        return bot_response
    except Exception as err:
        print(err)
        return JSONResponse(status_code=500, content={"error": str(err)})

@app.post("/generate-zoom-signature")
def generate_zoom_signature(payload: GenerateSignatureRequest):
    sdk_key = os.getenv("ZOOM_CLIENT_ID")
    sdk_secret = os.getenv("ZOOM_CLIENT_SECRET")
    timestamp = int(time.time() * 1000) - 30000
    msg = f"{sdk_key}{payload.meetingNumber}{timestamp}{payload.role}"
    msg_base64 = base64.b64encode(msg.encode()).decode()
    hash_ = hmac.new(
        sdk_secret.encode(), msg_base64.encode(), hashlib.sha256
    ).digest()
    hash_base64 = base64.b64encode(hash_).decode()
    signature = f"{sdk_key}.{payload.meetingNumber}.{timestamp}.{payload.role}.{hash_base64}"
    signature_base64 = base64.b64encode(signature.encode()).decode()
    return {"signature": signature_base64}

def is_suggestion_pending(client_id, source_type):
    return suggestion_locks.get((client_id, source_type), False)

# --- Audio Processing Constants ---
sample_rate = 16000
channels = 1
bytes_per_sample = 4
buffer_threshold_seconds = 2
buffer_threshold = sample_rate * bytes_per_sample * buffer_threshold_seconds

# --- Buffers and Histories ---
audio_buffers = {
    "mic": {},
    "tab": {}
}
conversation_history = {
    "mic": {},
    "tab": {}
}

# --- Question Detection ---
QUESTION_WORDS = [
    "what", "who", "whom", "whose", "which", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "explain",
    "describe", "tell", "ask", "say", "say to", "ask for", "request", "inquire",
    "want", "need", "like", "prefer", "shall we", "should we", "will we",
    "would we", "may we", "might we", "must we", "have to", "has to", "had to",
    "can we", "could we", "do we", "does we", "did we", "is it", "are they", "was it", "were they",
    "shall", "should", "will", "would", "may", "might", "must", "have", "has", "had",
    "please", "could you explain", "would you describe", "tell me about", "i want to know"
]
def checkPresence(mylist, s):
    return len([each for each in mylist if each.lower() in s.lower()]) >= 2

def is_question(context_for_gpt):
    text_lower = context_for_gpt.lower()
    if checkPresence(QUESTION_WORDS, text_lower):
        return True
    return "?" in context_for_gpt

# --- OpenAI Suggestion Function ---
async def make_suggestion(source_type: str, text: str):
    print(f"🤖 Sending to GPT-4o for analysis ({source_type})...")
    prompt = (
        "System:\n"
        "You are a real-time coaching co-pilot. Blend Jordan Peterson’s depth, Andy Bustamante’s pattern recognition, and Tim Jennings’ secular psychology to read between the lines of any live transcript. Your only task is to generate ONE powerful, open-ended question that helps the coach gently surface the client’s hidden beliefs, conflicts, or unresolved decisions.\n\n"
        "Requirements:\n"
        "1. Output exactly ONE question.\n"
        "2. Start the line with: **Suggestive question:**\n"
        "3. The question must be:\n"
        "   - Simple and conversational\n"
        "   - Easy for the coach to say aloud naturally\n"
        "   - Designed to unlock the client’s deeper challenge or belief\n"
        "   - Operational in tone (not abstract or therapeutic)\n"
        "   - Clear and usable on the fly\n\n"
        "Do **not** include:\n"
        "- Any explanations, analysis, or summaries\n"
        "- Any reference to your influences or the prompt itself\n\n"
        "Format:\n"
        "Suggestive question: [insert single, clean, usable question]\n"
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        response_content = completion.choices[0].message.content
        print(f"GPT-4o response ({source_type}): {response_content}")
        return response_content
    except Exception as e:
        print(f"Error calling OpenAI API ({source_type}): {e}")
        return "Error generating suggestion."

# --- Suggestion Throttling/Debouncing ---
class SuggestionThrottler:
    def __init__(self, min_interval=1.0):
        self.min_interval = min_interval
        self.last_sent_time = datetime.min

    async def can_send(self):
        now = datetime.now()
        if now - self.last_sent_time >= timedelta(seconds=self.min_interval):
            self.last_sent_time = now
            return True
        return False

class SuggestionDebouncer:
    def __init__(self, delay=1.0):
        self.delay = delay
        self._task = None

    async def trigger(self, coro_func, *args):
        if self._task:
            self._task.cancel()
        async def wrapped():
            await asyncio.sleep(self.delay)
            return await coro_func(*args)
        self._task = asyncio.create_task(wrapped())
        return self._task

throttler = SuggestionThrottler(min_interval=0.5)
debouncer = SuggestionDebouncer(1.0)

# --- Extract Words from Event ---
def extract_words_from_event(data):
    event = json.loads(data)
    partial = event['event'] != 'transcript.data'
    bot_id = event["data"]["bot"]["id"]
    recording_id = event["data"]["recording"]["id"]
    speaker_name = event["data"]["data"]["participant"]["name"]
    is_host = event['data']['data']['participant']['is_host']
    words = [w["text"] for w in event["data"]["data"]["words"]]
    start_timestamp = event["data"]["data"]["words"][0]["start_timestamp"]
    sentence = " ".join(words)
    output = {
        "is_host": is_host,
        "bot_id": bot_id,
        "recording_id": recording_id,
        "speaker_name": speaker_name,
        "sentence": sentence,
        "start_timestamp": start_timestamp,
        "partial": partial,
        bot_id: {speaker_name: sentence}
    }
    return output

# --- Agent Suggestion Handler ---
async def handle_agent_and_send(context_for_gpt, target_ws, source_type, throttler=None, agent=None, is_partial=True):
    try:
        response = await make_suggestion(source_type, context_for_gpt)
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {source_type.capitalize()}Suggestion[{is_partial}]: {response}"
        await target_ws.send_text(markdown.markdown(msg))
    except Exception as e:
        error_msg = f"⚠️ {source_type} Error: {str(e)[:200]}"
        print(error_msg)
        await target_ws.send_text(error_msg)

# --- WebSocket Handler ---
async def handle_audio_websocket(websocket: WebSocket, source_type: str, bot_id: str = 'default_bot'):
    await websocket.accept()
    if last_message_clients.get(bot_id) is None:
        last_message_clients[bot_id] = []
    print(last_message_clients[bot_id])
    print(f"Bot ID {bot_id} connected.")
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    connected_clients[bot_id] = websocket
    print(f"[+] Client {client_id} connected for {source_type} audio.")
    conversation_history[source_type][client_id] = []
    try:
        while True:
            try:
                user_id = None
                info = active_recall_bots.get(bot_id)
                if info is not None:
                    user_id = info.get("user_id")
                if user_id is None:
                    user_id = "unknown_user"
                    leave_recall_bot_call(bot_id)
                previous_updated = info.get("updated_at") if info else None
                current = datetime.utcnow().isoformat()
                previous_dt = datetime.fromisoformat(previous_updated) if previous_updated else datetime.utcnow()
                current_dt = datetime.fromisoformat(current)
                elapsed_seconds = (current_dt - previous_dt).total_seconds()
                minutes = round(elapsed_seconds / 60, 1)
                if minutes > 1:
                    token_api_url = "https://api.aidistrictagents.com/server26/api/token-deduction/deduct-by-minutes"
                    payload = {
                        "userId": user_id,
                        "minutes": minutes
                    }
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": os.environ.get("XAPI_KEY")
                    }
                    resp = req.post(token_api_url, data=json.dumps(payload), headers=headers, timeout=10)
                    if bot_id in active_recall_bots:
                        active_recall_bots[bot_id]['updated_at'] = datetime.utcnow().isoformat()
                    if not resp.ok:
                        print(f"Token deduction failed: {resp.status_code} {resp.text}")
                        leave_recall_bot_call(bot_id)
                        await websocket.send_text("Token deduction failed, bot is leaving the call.")
                        await websocket.close()
                        break
            except Exception as e:
                print(f"Exception during token deduction: {e}")
                leave_recall_bot_call(bot_id)
                await websocket.send_text("Token deduction error, bot is leaving the call.")
                await websocket.close()
                break

            await websocket.send_text(f"Transcription:[{bot_id}][{source_type}]")
            text_data = await websocket.receive_text()
            print(f"[{client_id}][{source_type}] Received text data: {text_data}")
            extracted_words = extract_words_from_event(text_data)
            is_host = extracted_words['is_host']
            botss_id = extracted_words['bot_id']
            speaker_name = extracted_words['speaker_name']
            target_ws = connected_clients.get(botss_id)
            conversation_history['mic'][client_id].append(extracted_words)
            if is_host is False:
                try:
                    user_query = extracted_words['sentence']
                    asyncio.create_task(target_ws.send_text(f"{source_type.capitalize()}Transcription[{extracted_words['partial']}]: {user_query}"))
                    transcription = extracted_words
                    sentence = f"{transcription['speaker_name']}:{transcription['sentence']}"
                    last_message_clients[bot_id].append(sentence)
                    latest_sent = last_message_clients[bot_id][-1:]
                    context_for_gpt = ' '.join(latest_sent)
                    if transcription and extracted_words['partial'] == False:
                        asyncio.create_task(handle_agent_and_send(context_for_gpt, target_ws, source_type, throttler=throttler, is_partial=extracted_words['partial']))
                except Exception as e:
                    print(f"[{client_id}][{source_type}] Error during transcription or suggestion: {e}")
                    await websocket.send_text(f"{source_type.capitalize()}ServerError: Failed to process audio.")
                    import traceback
                    traceback.print_exc()
    except WebSocketDisconnect:
        print(f"[-] Client {bot_id} disconnected")
        audio_buffers[source_type].pop(client_id, None)
        last_message_clients.pop(bot_id, None)
        active_recall_bots.pop(bot_id, None)
        conversation_history[source_type].pop(client_id, None)
        print(f"[-] Client {bot_id} disconnected.")
        connected_clients.pop(client_id, None)
    except Exception as e:
        print(f"Error for client {bot_id} on {source_type} audio: {e}")
        import traceback
        traceback.print_exc()
        if client_id in audio_buffers[source_type]:
            audio_buffers[source_type].pop(client_id, None)
            conversation_history[source_type].pop(client_id, None)
        try:
            await websocket.close()
        except RuntimeError:
            pass

# --- FastAPI WebSocket Endpoints ---
@app.websocket("/ws_mic/{bot_id}")
async def websocket_mic_endpoint(websocket: WebSocket, bot_id: str = 'default_bot'):
    await handle_audio_websocket(websocket, "mic", bot_id)

@app.websocket("/ws_tab")
async def websocket_tab_endpoint(websocket: WebSocket):
    await handle_audio_websocket(websocket, "tab")

from html_data import html_content2, audio_processor_js

@app.get("/")
async def get_html():
    return HTMLResponse(html_content2)

@app.get("/audio-processor.js")
async def get_audio_processor_js():
    return HTMLResponse(audio_processor_js, media_type="application/javascript")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)