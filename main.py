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
import threading
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, requests, BackgroundTasks
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from fastapi import FastAPI, Response, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

import uvicorn
from openai import AsyncOpenAI, BaseModel
from token_validator import TokenReservationClient

# Initialize token client
token_client = TokenReservationClient()
app = FastAPI()

# Set up comprehensive logging
def setup_logging():
    """Set up logging for token deduction monitoring"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set up token deduction logger
    token_logger = logging.getLogger("token_deduction")
    token_logger.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        "logs/token_deduction.log", 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    token_logger.addHandler(file_handler)
    token_logger.addHandler(console_handler)
    
    return token_logger

# Initialize logger
token_logger = setup_logging()

SALES_COACH_PLATFORM_ID = "P1013"
NEXTGEN_COACH_PLATFORM_ID = "P1033"


def get_platform_id(is_outreach: bool) -> str:
    """Outreach → sales coach platform; default live coaching → nextgen coach platform."""
    return SALES_COACH_PLATFORM_ID if is_outreach else NEXTGEN_COACH_PLATFORM_ID

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

class ModuleResponsesRequest(BaseModel):
    module1_response: str = None
    module2_response: str = None

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
@app.post("/end-bot")
def end_bot(bot_id:str):
    try:
        leave_recall_bot_call(bot_id=bot_id)
        
        # Stop the token deduction timer
        stop_token_deduction_timer(bot_id)
        
        # Clean up active recall bots
        active_recall_bots.pop(bot_id, None)
        
        return {"status": "Bot ended"}
    except Exception as err:
        JSONResponse(status_code=400,content={"error": str(err)})
        return {"status": "Error ending bot"}

@app.get("/bot-status")
def get_bot_status():
    """Get status of all active bots and their timers"""
    try:
        status = {
            "active_bots": len(active_recall_bots),
            "active_timers": len(token_deduction_timers),
            "bots": {}
        }
        
        for bot_id, bot_info in active_recall_bots.items():
            status["bots"][bot_id] = {
                "user_id": bot_info.get("user_id"),
                "email": bot_info.get("email"),
                "timestamp": bot_info.get("timestamp"),
                "updated_at": bot_info.get("updated_at"),
                "has_timer": bot_id in token_deduction_timers,
                "timer_running": bot_id in token_deduction_timers and token_deduction_timers[bot_id].is_alive()
            }
        
        return status
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": str(err)})

@app.post("/test-timer")
def test_timer_creation(user_id: str = "test_user"):
    """Test endpoint to debug timer creation"""
    try:
        # Create a fake bot entry
        test_bot_id = f"test_bot_{int(time.time())}"
        active_recall_bots[test_bot_id] = {
            "email": "test@example.com",
            "bot_response": {"id": test_bot_id},
            "bot_id": test_bot_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        print(f"🔧 DEBUG: Created test bot {test_bot_id}")
        print(f"🔧 DEBUG: active_recall_bots now contains: {list(active_recall_bots.keys())}")
        
        # Start timer
        start_token_deduction_timer(test_bot_id, user_id, is_outreach=True)  # Use outreach to avoid actual token deduction
        
        return {
            "test_bot_id": test_bot_id,
            "active_bots": len(active_recall_bots),
            "active_timers": len(token_deduction_timers),
            "timer_created": test_bot_id in token_deduction_timers
        }
    except Exception as err:
        print(f"❌ Test timer error: {err}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(err)})

@app.post("/cleanup-test-bots")
def cleanup_test_bots():
    """Clean up test bots"""
    try:
        test_bots = [bot_id for bot_id in active_recall_bots.keys() if bot_id.startswith("test_bot_")]
        cleaned_count = 0
        
        for bot_id in test_bots:
            stop_token_deduction_timer(bot_id)
            active_recall_bots.pop(bot_id, None)
            cleaned_count += 1
        
        return {
            "cleaned_bots": cleaned_count,
            "remaining_bots": len(active_recall_bots),
            "remaining_timers": len(token_deduction_timers)
        }
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": str(err)})

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
                        "assembly_ai_streaming": {
                            "language": "en-US",
                        }
                    }
                },
                "realtime_endpoints": [
                    {
                        "type": "websocket",
                        "url": "wss://transcrbe.testir.xyz/ws_mic/default-bot",
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

# Dictionary to track active recall bots: {bot_id: {"bot_response": ..., "timestamp": ...}}
active_recall_bots = {}

# Dictionary to track timer tasks for token deduction: {bot_id: threading.Thread}
token_deduction_timers = {}

# Dictionary to track stop flags for token deduction threads: {bot_id: threading.Event}
token_deduction_stop_flags = {}

# Thread pool executor for running async tasks from sync context
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# Continuous token deduction function with per minute deduction
async def continuous_token_deduction(bot_id: str, user_id: str, is_outreach: bool = False, interval_minutes: float = 1):
    """
    Continuously deduct tokens at regular intervals until the bot is stopped.
    
    Args:
        bot_id: The recall bot ID
        user_id: The user ID for token deduction
        is_outreach: Whether this is an outreach call
        interval_minutes: How often to deduct tokens (default: 1 minutes = 60 seconds)
    """
    token_logger.info(f"🔄 STARTING continuous token deduction for bot {bot_id}, user {user_id}, outreach: {is_outreach}")
    token_logger.info(f"Bot in active_recall_bots: {bot_id in active_recall_bots}")
    token_logger.info(f"Current active_recall_bots: {list(active_recall_bots.keys())}")
    
    cycle_count = 0
    while bot_id in active_recall_bots:
        try:
            cycle_count += 1
            token_logger.info(f"💤 CYCLE {cycle_count}: Waiting {interval_minutes} minutes for bot {bot_id}")
            
            # Wait for the interval
            await asyncio.sleep(interval_minutes * 60)  # Convert minutes to seconds
            
            # Check if bot is still active
            if bot_id not in active_recall_bots:
                token_logger.info(f"⏹️ Bot {bot_id} no longer active, stopping token deduction after {cycle_count} cycles")
                break
                
            # Perform token deduction using TokenReservationClient
            token_logger.info(f"💰 CYCLE {cycle_count}: Attempting to deduct {interval_minutes} minutes for user {user_id}, bot {bot_id}")
            
            deduct_result = token_client.deduct_tokens(
                user_id=user_id,
                feature_id="sales-coaching",
                custom_deduction=interval_minutes,
                platform_id=get_platform_id(is_outreach=is_outreach),
            )
            
            if not deduct_result['success']:
                token_logger.error(f"❌ CYCLE {cycle_count}: Token deduction failed: {deduct_result.get('error')}")
                token_logger.error(f"⏹️ Stopping bot {bot_id} due to token deduction failure")
                
                # End the bot due to token deduction failure
                leave_recall_bot_call(bot_id)
                
                # Notify connected clients
                target_ws = connected_clients.get(bot_id)
                if target_ws:
                    try:
                        await target_ws.send_text("❌ Insufficient tokens, bot is leaving the call.")
                        await target_ws.close()
                        token_logger.info(f"Notified client about token deduction failure for bot {bot_id}")
                    except Exception as e:
                        token_logger.error(f"Error notifying client: {e}")
                
                # Clean up
                active_recall_bots.pop(bot_id, None)
                token_deduction_timers.pop(bot_id, None)
                break
            else:
                # Update the last successful deduction time
                if bot_id in active_recall_bots:
                    active_recall_bots[bot_id]['updated_at'] = datetime.utcnow().isoformat()
                token_logger.info(f"✅ CYCLE {cycle_count}: Token deduction successful for user {user_id}, deducted {interval_minutes} minutes")
                
        except asyncio.CancelledError:
            print(f"⏹️ Token deduction timer cancelled for bot {bot_id}")
            break
        except Exception as e:
            token_logger.error(f"❌ CYCLE {cycle_count}: Error in continuous token deduction for bot {bot_id}: {e}")
            import traceback
            token_logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue the loop unless it's a critical error
            continue
    
    token_logger.info(f"🏁 Token deduction timer ended for bot {bot_id} after {cycle_count} cycles")

def run_async_in_thread(coro):
    """Run an async coroutine in a new thread with its own event loop"""
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    return threading.Thread(target=run_in_thread, daemon=True)

def start_token_deduction_timer(bot_id: str, user_id: str, is_outreach: bool = False):
    """Start the continuous token deduction timer for a bot"""
    try:
        token_logger.info(f"🔧 STARTING TIMER: Attempting to start timer for bot {bot_id}, user {user_id}, outreach: {is_outreach}")
        
        # Stop existing timer if any
        if bot_id in token_deduction_timers:
            token_logger.info(f"🔧 Stopping existing timer for bot {bot_id}")
            stop_token_deduction_timer(bot_id)
        
        # Check if bot is in active_recall_bots
        if bot_id not in active_recall_bots:
            token_logger.error(f"❌ Bot {bot_id} not found in active_recall_bots!")
            return
        
        # Create and start the timer thread
        timer_thread = run_async_in_thread(continuous_token_deduction(bot_id, user_id, is_outreach))
        timer_thread.start()
        
        # Store the thread reference
        token_deduction_timers[bot_id] = timer_thread
        
        token_logger.info(f"⏰ TIMER STARTED: Token deduction timer thread for bot {bot_id}")
        token_logger.info(f"Timer thread: {timer_thread}")
        token_logger.info(f"Active timers count: {len(token_deduction_timers)}")
            
    except Exception as e:
        token_logger.error(f"❌ Error starting token deduction timer for bot {bot_id}: {e}")
        import traceback
        token_logger.error(f"Traceback: {traceback.format_exc()}")
        # If timer fails to start, we should still allow the bot to proceed
        # but log the error for debugging

def stop_token_deduction_timer(bot_id: str):
    """Stop the continuous token deduction timer for a bot"""
    if bot_id in token_deduction_timers:
        timer_thread = token_deduction_timers.pop(bot_id, None)
        if timer_thread and timer_thread.is_alive():
            # The thread will stop naturally when the bot is removed from active_recall_bots
            token_logger.info(f"⏹️ STOPPING TIMER: Marked token deduction timer for stopping: bot {bot_id}")
        else:
            token_logger.info(f"⏹️ Token deduction timer already stopped for bot {bot_id}")
        
        # Remove from active bots to stop the timer loop
        if bot_id in active_recall_bots:
            active_recall_bots.pop(bot_id, None)
            token_logger.info(f"⏹️ Removed bot {bot_id} from active_recall_bots")
    else:
        token_logger.warning(f"⚠️ No timer found for bot {bot_id} when trying to stop")

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

    # Check token availability using TokenReservationClient
    try:
        token_logger.info(f"Checking token availability for user: {user_id}")
        check_result = token_client.check_availability(
            user_id=user_id,
            feature_id="sales-coaching",
            platform_id=get_platform_id(is_outreach=False),
        )
        
        if not check_result['success']:
            token_logger.error(f"Token availability check failed for user {user_id}: {check_result.get('error')}")
            return JSONResponse(status_code=400, content={"error": "Token availability check failed"})
        
        # Check if user can afford the operation
        check_data = check_result['data']
        if not check_data.get('canAfford', False):
            token_logger.warning(f"User {user_id} cannot afford bot call - insufficient tokens")
            return JSONResponse(status_code=403, content={"error": "Insufficient tokens for bot call"})
        
        token_logger.info(f"Token availability confirmed for user {user_id}")
        
    except Exception as e:
        token_logger.error(f"Exception during token availability check for user {user_id}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Token validation error: {str(e)}"})

    # Perform initial token deduction using TokenReservationClient
    try:
        token_logger.info(f"Performing initial token deduction for user: {user_id}")
        deduct_result = token_client.deduct_tokens(
            user_id=user_id,
            feature_id="sales-coaching",
            custom_deduction=1,  # Deduct minimal amount on bot creation
            platform_id=get_platform_id(is_outreach=False),
        )
        
        if not deduct_result['success']:
            token_logger.error(f"Initial token deduction failed for user {user_id}: {deduct_result.get('error')}")
            return JSONResponse(status_code=400, content={"error": "Initial token deduction failed"})
        
        token_logger.info(f"Initial token deduction successful for user {user_id}")

        # Log the generation attempt (nextgen coach platform for default live bot)
        try:
            log_result = token_client.log_generation_attempt(
                user_id=user_id,
                email=user_email or "unknown@example.com",
                platform_id=get_platform_id(is_outreach=False),
                status="success"  
            )
            if log_result.get('success'):
                token_logger.info(f"Successfully logged nextgen-coach generation attempt for user {user_id}")
            else:
                token_logger.warning(f"Failed to log generation attempt: {log_result.get('error')}")
        except Exception as e:
            token_logger.warning(f"Exception during generation logging: {e}")
        
    except Exception as e:
        token_logger.error(f"Exception during initial token deduction for user {user_id}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Token deduction error: {str(e)}"})

    try:
        bot_response = start_recall_bot(meeting_url)

        # Save user_id and bot_response with timestamp
        bot_id=bot_response['id']
        active_recall_bots[bot_id] = {
            "email": user_email,
            "bot_response": bot_response,
            "bot_id": bot_response['id'],
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Start continuous token deduction timer
        start_token_deduction_timer(bot_id, user_id, is_outreach=False)
        
        return bot_response
    except Exception as err:
        print(err)
        return JSONResponse(status_code=500,content={"error": str(err)})


@app.post("/start-call-bot-outreach")
def start_recall_bot_outreach(meeting_url: str, user_id: str = None, user_email: str = None, module_responses: ModuleResponsesRequest = None):
    """
    Endpoint to start the Recall Bot for outreach calls with token validation.
    Uses the new AI Portal API for token management.
    Accepts module responses for outreach-specific coaching.
    """
    # Extract module responses from request body
    module1_response = module_responses.module1_response if module_responses else None
    module2_response = module_responses.module2_response if module_responses else None
    if not meeting_url:
        return {"error": "meeting_url is required"}
    if not user_id:
        return {"error": "user_id is required"}
    
    # Check if user_id already has an active recall bot and return error if so
    if user_id in active_recall_bots:
        return {"error": f"Recall bot already active for user_id {user_id}"}

    # Check token availability using the new API
    try:
        token_logger.info(f"Checking token availability for outreach user: {user_id}")
        check_result = token_client.check_availability(
            user_id=user_id,
            feature_id="sales-coaching",
            platform_id=get_platform_id(is_outreach=True),
        )
        
        if not check_result['success']:
            token_logger.error(f"Token availability check failed for user {user_id}: {check_result.get('error')}")
            return JSONResponse(status_code=400, content={"error": "Token availability check failed"})
        
        # Check if user can afford the operation
        check_data = check_result['data']
        if not check_data.get('canAfford', False):
            token_logger.warning(f"User {user_id} cannot afford outreach call - insufficient tokens")
            return JSONResponse(status_code=403, content={"error": "Insufficient tokens for outreach call"})
        
        token_logger.info(f"Token availability confirmed for user {user_id}")
        
    except Exception as e:
        token_logger.error(f"Exception during token availability check for user {user_id}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Token validation error: {str(e)}"})

    # Perform initial token deduction for outreach
    try:
        token_logger.info(f"Performing initial token deduction for outreach user: {user_id}")
        deduct_result = token_client.deduct_tokens(
            user_id=user_id,
            feature_id="sales-coaching",
            custom_deduction=1,
            platform_id=get_platform_id(is_outreach=True),
        )

        if not deduct_result['success']:
            token_logger.error(
                f"Initial outreach token deduction failed for user {user_id}: {deduct_result.get('error')}"
            )
            return JSONResponse(status_code=400, content={"error": "Initial outreach token deduction failed"})

        token_logger.info(f"Initial outreach token deduction successful for user {user_id}")
    except Exception as e:
        token_logger.error(f"Exception during initial outreach token deduction for user {user_id}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Outreach token deduction error: {str(e)}"})

    try:
        bot_response = start_recall_bot(meeting_url)

        # Save user_id and bot_response with timestamp
        bot_id = bot_response['id']
        active_recall_bots[bot_id] = {
            "email": user_email,
            "bot_response": bot_response,
            "bot_id": bot_response['id'],
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "module1_response": module1_response,
            "module2_response": module2_response,
            "is_outreach": True,
        }
        
        # Start continuous token deduction timer
        start_token_deduction_timer(bot_id, user_id, is_outreach=True)
        
        # Log the generation attempt
        try:
            log_result = token_client.log_generation_attempt(
                user_id=user_id,
                email=user_email or "unknown@example.com",
                platform_id=get_platform_id(is_outreach=True),
                status="success"
            )
            if log_result['success']:
                token_logger.info(f"Successfully logged outreach (sales platform) generation for user {user_id}")
            else:
                token_logger.warning(f"Failed to log outreach generation attempt: {log_result.get('error')}")
        except Exception as e:
            token_logger.warning(f"Exception during generation logging: {e}")
        
        return bot_response
    except Exception as err:
        token_logger.error(f"Error starting outreach bot: {err}")
        return JSONResponse(status_code=400, content={"error": str(err)})

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
            model="gpt-3.5-turbo",  # Use "gpt-4o" if available in your environment
            messages=messages
        )
        response_content = completion.choices[0].message.content
        print(f"GPT-4o response ({source_type}) for {client_id}: {response_content}")
        return response_content
    except Exception as e:
        print(f"Error calling OpenAI API ({source_type}) for {client_id}: {e}")
        return "Error generating suggestion."

# --- Outreach-specific Suggestion Function ---
async def make_outreach_suggestion(source_type: str, text: str, module1_response: str = None, module2_response: str = None):
    print(f"🎯 Sending to GPT-4o for outreach analysis ({source_type})...")
    
    # Import the outreach prompt from the dedicated file
    try:
        from outreach_prompt import prompt as outreach_prompt_template
    except ImportError:
        print("⚠️ Could not import outreach_prompt.py, using fallback prompt")
        outreach_prompt_template = """You are a real-time sales co-pilot. Generate ONE short, actionable line I can say immediately in the middle of a sales call.
    
    Requirements:
    1. Output exactly ONE line.
    2. The response must be:
       - Conversational, natural, and under 20 words
       - Focused on moving toward the sale
       - Confident but not pushy, easy to say aloud
    """
    
    # Replace placeholders in the prompt with actual module responses
    prompt = outreach_prompt_template.replace("${module1Result}", module1_response or "our product")
    prompt = prompt.replace("${module2Result}", module2_response or "your target audience")
    
    # Add current conversation context
    prompt += f"\n\nCurrent conversation context: {text}"
    
    client_id = "outreach_client"  # Specific client ID for outreach

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Generate a suggested response for this sales conversation."}
    ]

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        
        response_content = completion.choices[0].message.content.strip()
        print(f"🎯 Outreach suggestion generated: {response_content}")
        return response_content
    except Exception as e:
        print(f"Error calling OpenAI API for outreach ({source_type}): {e}")
        return "Error generating outreach suggestion."

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
        # Check if this is an outreach call by looking up the bot_id in active_recall_bots
        bot_info = None
        for bot_id, info in active_recall_bots.items():
            if info.get('user_id') == client_id:
                bot_info = info
                break
        
        # Use outreach suggestion if this is an outreach call
        if bot_info and bot_info.get('is_outreach', False):
            print(f"🎯 Generating outreach suggestion for {client_id}")
            suggestion = await make_outreach_suggestion(
                source_type, 
                transcription, 
                bot_info.get('module1_response'),
                bot_info.get('module2_response')
            )
        else:
            # Use regular suggestion for non-outreach calls
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
        # Check if this is an outreach call by looking up the bot_id in active_recall_bots
        bot_info = None
        for bot_id, info in active_recall_bots.items():
            if info.get('is_outreach', False):
                bot_info = info
                break
        
        # Use outreach suggestion if this is an outreach call
        if bot_info and bot_info.get('is_outreach', False):
            print(f"🎯 Generating outreach suggestion for agent")
            response = await make_outreach_suggestion(
                source_type, 
                context_for_gpt, 
                bot_info.get('module1_response'),
                bot_info.get('module2_response')
            )
        else:
            # Use regular suggestion for non-outreach calls
            response = await make_suggestion(source_type, context_for_gpt)
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"[{timestamp}] {source_type.capitalize()}Suggestion[{is_partial}]: {response}"
            if "final" in source_type.lower()
            else f"[{timestamp}] {source_type.capitalize()}Suggestion[{is_partial}]: {response}"
        )
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
async def handle_audio_websocket(websocket: WebSocket, source_type: str, bot_id: str = 'default_bot',is_outreach:bool=False):
    await websocket.accept()
    try:
        # Ensure last_message_clients and conversation_history are initialized for this bot/client
        if last_message_clients.get(bot_id) is None:
            last_message_clients[bot_id] = []
        print(last_message_clients[bot_id])
        print(f"Bot ID {bot_id} connected.")

        client_id = f"{websocket.client.host}:{websocket.client.port}"
        connected_clients[bot_id] = websocket
        print(f"[+] Client {client_id} connected for {source_type} audio.")

        if client_id not in conversation_history[source_type]:
            conversation_history[source_type][client_id] = []

        while True:
            await websocket.send_text(f"Transcription:[{bot_id}][{source_type}]")
            try:
                text_data = await websocket.receive_text()
            except WebSocketDisconnect:
                print(f"[-] Client {bot_id} disconnected (WebSocketDisconnect on receive_text)")
                break
            except Exception as e:
                print(f"Error receiving text from client {client_id}: {e}")
                break

            print(f"[{client_id}][{source_type}] Received text data: {text_data}")
            try:
                extracted_words = extract_words_from_event(text_data)
            except Exception as e:
                print(f"Error extracting words from event: {e}")
                continue

            is_host = extracted_words['is_host']
            botss_id = extracted_words['bot_id']
            speaker_name = extracted_words['speaker_name']

            # DEBUG: Add connection debugging
            print(f"Connected clients: {list(connected_clients.keys())}")
            print(f"Looking for bot_id: {botss_id}")
            print(f"Active recall bots: {list(active_recall_bots.keys())}")

            target_ws = connected_clients.get(botss_id)
            if target_ws is None:
                print(f"No target_ws found for botss_id {botss_id}")
                # Try to find WebSocket by looking in active_recall_bots
                for active_bot_id, bot_info in active_recall_bots.items():
                    if active_bot_id == botss_id:
                        # Look for any connected client (fallback)
                        if connected_clients:
                            target_ws = list(connected_clients.values())[0]
                            print(f"Using fallback WebSocket for bot_id {botss_id}")
                            break
                
                if target_ws is None:
                    print(f"No fallback WebSocket available, skipping processing for {botss_id}")
                    continue

            if client_id not in conversation_history['mic']:
                conversation_history['mic'][client_id] = []
            conversation_history['mic'][client_id].append(extracted_words)

            if is_host is False:
                try:
                    user_query = extracted_words['sentence']
                    asyncio.create_task(target_ws.send_text(f"{source_type.capitalize()}Transcription[{extracted_words['partial']}]: {user_query}"))

                    transcription = extracted_words
                    sentence = transcription["speaker_name"] + ":" + transcription['sentence']
                    last_message_clients[bot_id].append(sentence)

                    latest_sent = last_message_clients[bot_id][-1:]
                    context_for_gpt = ' '.join(latest_sent)

                    if transcription and extracted_words['partial'] == False:
                        asyncio.create_task(handle_agent_and_send(context_for_gpt, target_ws, source_type, throttler=throttler, is_partial=extracted_words['partial']))

                    # Token deduction logic
                    # info = active_recall_bots.get(botss_id)
                    # if info is not None:
                    #     user_id = info.get("user_id")
                    #     if user_id is None:
                    #         user_id = "unknown_user"
                    #         leave_recall_bot_call(botss_id)
                    #     previous_updated = info.get("updated_at")
                    #     current = datetime.utcnow().isoformat()

                    #     previous_dt = datetime.fromisoformat(previous_updated) if previous_updated else datetime.utcnow()
                    #     current_dt = datetime.fromisoformat(current)
                    #     elapsed_seconds = (current_dt - previous_dt).total_seconds()
                    #     print(elapsed_seconds)

                    #     minutes = round(elapsed_seconds / 60, 2)
                    #     print(f"Elapsed minutes: {minutes}")
                    #     if minutes >= 0.1:  # Deduct tokens for every 0.1 minute (6 seconds) or more
                    #         token_api_url = "https://api.aidistrictagents.com/server26/api/token-deduction/deduct-by-minutes"
                    #         payload = {
                    #             "userId": user_id,
                    #             "minutes": minutes
                    #         }
                    #         headers = {
                    #             "Content-Type": "application/json",
                    #             "x-api-key": os.environ.get("XAPI_KEY")
                    #         }
                    #         print(f"Token deduction payload: {payload}")
                    #         resp = req.post(token_api_url, data=json.dumps(payload), headers=headers, timeout=10)
                    #         print(f"Token API Response Status: {resp.status_code}")
                    #         print(f"Token API Response: {resp.json()}")
                            
                    #         # For non-outreach calls, check if deduction succeeded
                    #         if is_outreach == False:
                    #             if not resp.ok:
                    #                 print(f"Token deduction failed: {resp.status_code} {resp.text}")
                    #                 leave_recall_bot_call(botss_id)
                    #                 print("Bot leaving due to token deduction failure")
                    #                 await target_ws.send_text("Token deduction failed, bot is leaving the call.")
                    #                 await target_ws.close()
                    #                 break
                    #             else:
                    #                 # SUCCESS: Update timestamp only after successful deduction
                    #                 active_recall_bots[botss_id]['updated_at'] = datetime.utcnow().isoformat()
                    #                 print(f"✅ Token deduction successful for user {user_id}, deducted {minutes} minutes")
                    #         else:
                    #             # Outreach calls - no token deduction needed, just log
                    #             print(f"Outreach call - no token deduction required for bot {botss_id}")
                except Exception as e:
                    print(f"Exception during transcription processing: {e}")
                    # print(f"Exception during token deduction or suggestion: {e}")
                    # leave_recall_bot_call(botss_id)
                    # try:
                    #     await target_ws.send_text("Token deduction error, bot is leaving the call.")
                    #     await target_ws.close()
                    # except Exception as close_exc:
                    #     print(f"Error closing websocket: {close_exc}")
                    # break

    except WebSocketDisconnect:
        print(f"[-] Client {bot_id} disconnected (outer WebSocketDisconnect)")
    except Exception as e:
        print(f"Error for client {bot_id} on {source_type} audio: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        try:
            audio_buffers[source_type].pop(client_id, None)
        except Exception:
            pass
        try:
            last_message_clients.pop(bot_id, None)
        except Exception:
            pass
        try:
            conversation_history[source_type].pop(client_id, None)
        except Exception:
            pass
        try:
            connected_clients.pop(bot_id, None)
        except Exception:
            pass
        try:
            # Stop token deduction timer when WebSocket disconnects (this also cleans up active_recall_bots)
            stop_token_deduction_timer(bot_id)
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        print(f"[-] Client {bot_id} cleanup complete.")

# --- FastAPI WebSocket Endpoints ---
@app.websocket("/ws_mic/{bot_id}")
async def websocket_mic_endpoint(websocket: WebSocket, bot_id: str = 'default_bot'):
    await handle_audio_websocket(websocket, "mic", bot_id)

@app.websocket("/ws_outreach/{bot_id}")
async def websocket_outreach_endpoint(websocket: WebSocket, bot_id: str = 'default_bot'):
    await handle_audio_websocket(websocket, "mic", bot_id,True)



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
    port = int(os.environ.get("PORT", 9019))
    # Disable reload in production/Docker environments
    reload_mode = os.environ.get("RELOAD", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload_mode)
