import asyncio
import base64
from datetime import datetime, timedelta
import hashlib
import hmac
import io
import json
import os
import time
import markdown
import requests as req

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, requests
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from fastapi import FastAPI, Response, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

import uvicorn
from openai import AsyncOpenAI, BaseModel
# security = HTTPBearer()
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store connected clients (can be improved with rooms/groups)
connected_clients = {}
last_message_clients = {}
# --- Initialize API Client and Load Model ---
load_dotenv()
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------
suggestion_locks = {}

# Models
class CreateMeetingRequest(BaseModel):
    topic: str = "Coaching Session"

class GenerateSignatureRequest(BaseModel):
    meetingNumber: str
    role: int

def leave_recall_bot_call(bot_id: str):
    import requests

    url = f"https://us-west-2.recall.ai/api/v1/bot/{bot_id}/leave_call/"

    recall_api_key = os.environ.get("RECALL_API_KEY")
    headers = {
        "accept": "application/json",
        "Authorization": recall_api_key
    }

    response = requests.post(url, headers=headers)

    print(response.text)

# Helpers
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
                        # "rev_streaming": {"language": 'en-Us'}
                        "assembly_ai_streaming": {
                            "language": "en-US",
                            # "disable_partial_transcripts": True
                        }
                    }
                },
                "realtime_endpoints": [
                    {
                        "type": "websocket",
                        "url": "wss://transcribe.testir.xyz/ws_mic/default-bot",  # url: "wss://fb65-111-119-49-101.ngrok-free.app/ws_mic/default-bot",
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

# Routes
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

        # Start Recall Bot
        bot_response = start_recall_bot(join_url)

        return {
            "meetingNumber": meeting_number,
            "joinUrl": join_url,
            "password": password,
            "recallBot": bot_response,
        }

    except Exception as err:
        return {"error": str(err)}
from datetime import datetime

# Dictionary to track active recall bots: {user_id: {"bot_response": ..., "timestamp": ...}}
active_recall_bots = {}

@app.post("/start-recall-bot")
def start_recall_bot_endpoint(meeting_url: str, user_id: str = None, user_email: str = None):
    """
    Endpoint to start the Recall Bot with a given meeting URL.
    """
    if not meeting_url:
        return {"error": "meeting_url is required"}
    if not user_id:
        return {"error": "user_id is required"}
    # Check if user_id already has an active recall bot and return error if so
    if user_id in active_recall_bots:
        return {"error": f"Recall bot already active for user_id {user_id}"}

    # Add API call before creating bot (token deduction)
    try:
        token_api_url = "https://api.aidistrictagents.com/server26/api/token-deduction/deduct-by-minutes"
        payload = {
            "userId": user_id,
            "minutes": 1/600  # Deduct 1 minute on bot creation (or adjust as needed)
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": os.environ.get("XAPI_KEY")
        }
        resp = req.post(token_api_url, data=json.dumps(payload), headers=headers, timeout=10)
        print(resp.json())
        print(resp.status_code)
        if not resp.ok:
            print(f"Token deduction failed: {resp.status_code} {resp.text}")
            return JSONResponse(status_code=500,content={"error": "Insufficient token"})
    except Exception as e:
        print(f"Exception during token deduction: {e}")
        return JSONResponse(status_code=500,content={"error": str(e)})

    try:
        bot_response = start_recall_bot(meeting_url)

        # Save user_id and bot_response with timestamp
        active_recall_bots[user_id] = {
            "email": user_email,
            "bot_response": bot_response,
            "bot_id": bot_response['id'],
            "timestamp": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        return bot_response
    except Exception as err:
        print(err)
        return JSONResponse(status_code=500,content={"error": str(err)})
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
bytes_per_sample = 4 # float32 uses 4 bytes per sample
buffer_threshold_seconds = 2 # Process audio in 5-second chunks
buffer_threshold = sample_rate * bytes_per_sample * buffer_threshold_seconds
avatar = """Avatar Preview
Andrea Chen
Mid-level Manager
reserved and thoughtful mid-level manager who is stressed and anxious about leadership development and better team management. neutral about coaching but willing to engage."""
# --- Buffers and Histories (per client, per source) ---
# Each client will have separate buffers/histories for mic and tab audio
audio_buffers = {
    "mic": {}, # {client_id: BytesIO}
    "tab": {}  # {client_id: BytesIO}
}
conversation_history = {
    "mic": {}, # {client_id: [utterances]}
    "tab": {}  # {client_id: [utterances]}
}
# -----------------------------------------------
checkPresence = lambda mylist, s: len([each for each in mylist if each.lower() in s.lower()]) >= 2
def is_question(context_for_gpt):
    text_lower = context_for_gpt.lower()
    if checkPresence(QUESTION_WORDS, text_lower):
        return True
    return "?" in context_for_gpt  # also consider if the text contains '?'
# --- OpenAI Suggestion Function ---
async def make_suggestion(source_type: str, text: str):
    print(f"🤖 Sending to GPT-4o for analysis ({source_type})...")
    prompt = f"""System:
You are a real-time coaching co-pilot. Blend Jordan Peterson’s depth, Andy Bustamante’s pattern recognition, and Tim Jennings’ secular psychology to read between the lines of any live transcript. Your only task is to generate ONE powerful, open-ended question that helps the coach gently surface the client’s hidden beliefs, conflicts, or unresolved decisions.

Requirements:
1. Output exactly ONE question.
2. Start the line with: **Suggestive question:**
3. The question must be:
   - Simple and conversational
   - Easy for the coach to say aloud naturally
   - Designed to unlock the client’s deeper challenge or belief
   - Operational in tone (not abstract or therapeutic)
   - Clear and usable on the fly

Do **not** include:
- Any explanations, analysis, or summaries
- Any reference to your influences or the prompt itself

Format:
Suggestive question: [insert single, clean, usable question]

 """
    client_id = "default_client"  # Placeholder, replace with actual client ID logic

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
        print(f"GPT-4o response ({source_type}) for {client_id}: {response_content}")
        return response_content
    except Exception as e:
        print(f"Error calling OpenAI API ({source_type}) for {client_id}: {e}")
        return "Error generating suggestion."
# --- Import run_in_threadpool ---
from fastapi.concurrency import run_in_threadpool
import asyncio
async def send_suggestion_later(client_id, source_type, transcription, websocket):
    key = (client_id, source_type)

    # Skip if already processing a suggestion for this client/source
    if suggestion_locks.get(key):
        print(f"Skipping suggestion for {key}, already in progress.")
        return

    suggestion_locks[key] = True
    try:
        suggestion = await make_suggestion(client_id, source_type, transcription)

        await websocket.send_text(f"Transcription:{source_type.capitalize()}Suggestion: {suggestion}")
    except Exception as e:
        print(f"Failed to generate/send suggestion for {key}: {e}")
    finally:
        suggestion_locks[key] = False  # Clear lock
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

# Function to extract words from the event data
def extract_words_from_event(data):
    # Extracting bot ID, speaker name, and words
    event = json.loads(data)
    partial = False

    if (event['event'] != 'transcript.data'):
        partial = True
    bot_id = event["data"]["bot"]["id"]
    recording_id = event["data"]["recording"]["id"]
    speaker_name = event["data"]["data"]["participant"]["name"]
    is_host = event['data']['data']['participant']['is_host']
    words = [w["text"] for w in event["data"]["data"]["words"]]
    start_timestamp = event["data"]["data"]["words"][0]["start_timestamp"]
    # Combine the words into a single sentence
    sentence = " ".join(words)

    # Construct the output JSON
    output = {
        "is_host": is_host,
        "bot_id": bot_id,
        "recording_id": recording_id,
        "speaker_name": speaker_name,
        "sentence": sentence,
        "start_timestamp": start_timestamp,
        "partial": partial,
        bot_id:
            {
                speaker_name: sentence
            }
    }

    return output
class SuggestionThrottler:
    """Handles rate-limiting for suggestions"""
    def __init__(self, min_interval=1.0):  # 1 second between suggestions
        self.min_interval = min_interval
        self.last_sent_time = datetime.min

    async def can_send(self):
        now = datetime.now()
        if now - self.last_sent_time >= timedelta(seconds=self.min_interval):
            self.last_sent_time = now
            return True
        return False

class SuggestionDebouncer:
    """Waits for pause before executing"""
    def __init__(self, delay=1.0):
        self.delay = delay
        self._task = None

    async def trigger(self, coro_func, *args):
        """Cancel previous call and schedule new one"""
        if self._task:
            self._task.cancel()

        async def wrapped():
            await asyncio.sleep(self.delay)
            return await coro_func(*args)

        self._task = asyncio.create_task(wrapped())
        return self._task

throttler = SuggestionThrottler(min_interval=.5)  # 1.5 second cooldown
# Process final transcription after 1 second of silence
debouncer = SuggestionDebouncer(1.0)

# Track latest task and debounce timer per source_type
latest_tasks = {}
debounce_timers = {}

# # Async wrapper that runs sync agent.run in background
async def handle_agent_and_send(context_for_gpt, target_ws, source_type, throttler=None, agent=None, is_partial=True):
    """Process agent suggestion and send with rate-limiting"""
    try:
        # Throttle check (skip if sending too frequently)
        # if throttler and not await throttler.can_send():
        #     await debouncer.trigger(handle_agent_and_send, context_for_gpt, target_ws, source_type, throttler,agent)
        #     return
        # # Debounce to wait for pause before sending
        # if debouncer:
        #     await debouncer.trigger(handle_agent_and_send, context_for_gpt, target_ws, source_type, throttler,agent)
        #     return
        # Get AI suggestion (async)
        response = await make_suggestion(source_type, context_for_gpt)
        # response=agent.run(context_for_gpt,session_id=session)
        # Format message with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"[{timestamp}] {source_type.capitalize()}Suggestion[{is_partial}]: {response}"
            if "final" in source_type.lower()
            else f"[{timestamp}] {source_type.capitalize()}Suggestion[{is_partial}]: {response}"
        )

        # Send non-blocking with error fallback
        await target_ws.send_text(markdown.markdown(msg))

    except Exception as e:
        error_msg = f"⚠️ {source_type} Error: {str(e)[:200]}"
        print(error_msg)
        await target_ws.send_text(error_msg)

# Async function that does the sending
async def send_suggestions(ws, msg1):
    try:
        await ws.send_text(msg1)
    except Exception as e:
        print(f"Error sending suggestion: {e}")
# --- WebSocket Processing Handler (reusable for both mic and tab) ---
async def handle_audio_websocket(websocket: WebSocket, source_type: str, bot_id: str = 'default_bot'):
    await websocket.accept()
    if last_message_clients.get(bot_id) is None:
        last_message_clients[bot_id] = []
    print(last_message_clients[bot_id])
    print(f"Bot ID {bot_id} connected.")
    # response = agent.run("Let's start")
    # print(f"Agent response: {response.data.output}")
    # session = response.data.session_id
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    connected_clients[bot_id] = websocket
    print(f"[+] Client {client_id} connected for {source_type} audio.")

    conversation_history[source_type][client_id] = []
    try:
        while True:
            # --- Add this call before each transcription and leave bot if throws error ---
            try:
                # Call the token deduction API before each transcription
                # You may want to get userId and minutes from context or event, here hardcoded for demo
                # Find user_id from active_recall_bots where bot_id matches
                user_id = None
                info = None
                for uid, bot_info in active_recall_bots.items():
                    if str(bot_info.get("bot_id")) == str(bot_id):
                        user_id = uid
                        info = bot_info
                        break
                if user_id is None:
                    user_id = "unknown_user"
                    leave_recall_bot_call(bot_id)
                print(updated_at)
                # Calculate minutes from current timestamp (rounded to nearest 0.5 min)
                previous_updated = info.get("updated_at")
                current = datetime.utcnow().isoformat(),
                # Calculate minutes elapsed between previous_updated and current
                from datetime import datetime
                previous_dt = datetime.fromisoformat(previous_updated) if previous_updated else datetime.utcnow()
                # 'current' is a tuple (because of the trailing comma above), so we extract the first element if needed
                if isinstance(current, tuple):
                    current_dt = datetime.fromisoformat(current[0])
                else:
                    current_dt = datetime.fromisoformat(current)
                # Calculate the number of seconds elapsed between the previous and current timestamps
                elapsed_seconds = (current_dt - previous_dt).total_seconds()
                # Convert seconds to minutes (rounded to 1 decimal place)
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
                    print(payload)
                    resp = req.post(token_api_url, data=json.dumps(payload), headers=headers, timeout=10)
                    print(resp.json())
                    # Update the 'updated_at' field for the active recall bot
                    if user_id in active_recall_bots:
                        active_recall_bots[user_id]['updated_at'] = datetime.utcnow().isoformat()
                    #  "updated_at":
                    if not resp.ok:
                        print(f"Token deduction failed: {resp.status_code} {resp.text}")
                        # Leave the recall bot and close websocket
                        leave_recall_bot_call(bot_id)
                        print("bot leaving ")
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

            # print(extracted_words)
            botss_id = extracted_words['bot_id']
            speaker_name = extracted_words['speaker_name']

            target_ws = connected_clients.get(botss_id)
            conversation_history['mic'][client_id].append(extracted_words)

            if is_host is False:
                try:

                    # -----------------------------------------------------

                    user_query = extracted_words['sentence']
                    asyncio.create_task(target_ws.send_text(f"{source_type.capitalize()}Transcription[{extracted_words['partial']}]: {user_query}"))

                    transcription = (extracted_words)
                    sentence = transcription["speaker_name"] + ":" + transcription['sentence']
                    last_message_clients[bot_id].append(sentence)

                    latest_sent = last_message_clients[bot_id][-1:]
                    context_for_gpt = ' '.join(latest_sent)

                    if transcription:

                        if extracted_words['partial'] == False:
                            # if(sentence.split().__len__() > 3 and is_question(context_for_gpt)):
                            asyncio.create_task(handle_agent_and_send(context_for_gpt, target_ws, source_type, throttler=throttler, is_partial=extracted_words['partial']))
                except Exception as e:
                    print(f"[{client_id}][{source_type}] Error during transcription or suggestion: {e}")
                    await websocket.send_text(f"{source_type.capitalize()}ServerError: Failed to process audio.")
                    import traceback
                    traceback.print_exc()

                # audio_buffers[source_type][client_id].seek(0)
                # audio_buffers[source_type][client_id].truncate(0)

    except WebSocketDisconnect:
        print(f"[-] Client {bot_id} disconnected")
        audio_buffers[source_type].pop(client_id, None)
        last_message_clients.pop(bot_id, None)
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
# --- FastAPI Routes ---
@app.get("/")
async def get_html():
    return HTMLResponse(html_content2)

@app.get("/audio-processor.js")
async def get_audio_processor_js():
    return HTMLResponse(audio_processor_js, media_type="application/javascript")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)