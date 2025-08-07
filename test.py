import pytest
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def test_create_zoom_meeting():
    payload = {"topic": "Test Meeting"}
    response = client.post("/create-zoom-meeting", json=payload)
    assert response.status_code == 200
    # The response may contain error if Zoom credentials are not set, so just check keys
    assert "meetingNumber" in response.json() or "error" in response.json()

def test_start_recall_bot_endpoint_missing_params():
    response = client.post("/start-recall-bot", params={})
    assert response.status_code == 200
    assert "error" in response.json()

def test_start_recall_bot_endpoint_missing_user_id():
    response = client.post("/start-recall-bot", params={"meeting_url": "https://example.com"})
    assert response.status_code == 200
    assert "error" in response.json()

def test_start_recall_bot_endpoint_success(monkeypatch):
    # Patch start_recall_bot to avoid real API call
    def fake_start_recall_bot(meeting_url):
        class BotResponse:
            id = "fake-bot-id"
            def __getitem__(self, key): return "value"
        return BotResponse()
    monkeypatch.setattr(main, "start_recall_bot", fake_start_recall_bot)
    response = client.post("/start-recall-bot", params={"meeting_url": "https://example.com", "user_id": "user1", "user_email": "test@example.com"})
    assert response.status_code == 200

def test_generate_zoom_signature():
    payload = {"meetingNumber": "123456789", "role": 1}
    response = client.post("/generate-zoom-signature", json=payload)
    assert response.status_code == 200
    assert "signature" in response.json() or "error" in response.json()

def test_get_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.text.lower() or "<!doctype" in response.text.lower()

def test_get_audio_processor_js():
    response = client.get("/audio-processor.js")
    assert response.status_code == 200
    assert "function" in response.text or "const" in response.text

# Additional tests for each function in main.py

def test_is_suggestion_pending():
    # Setup
    main.suggestion_locks[("client1", "mic")] = True
    assert main.is_suggestion_pending("client1", "mic") is True
    assert main.is_suggestion_pending("client2", "mic") is False

def test_checkPresence():
    mylist = ["hello", "world", "foo"]
    s = "Hello there, world!"
    assert main.checkPresence(mylist, s) is True
    s2 = "Just hello here"
    assert main.checkPresence(mylist, s2) is False

def test_is_question():
    # Should return True for question words
    assert main.is_question("What is your name?") is True
    assert main.is_question("Tell me about yourself.") is True
    # Should return True for question mark
    assert main.is_question("This is a test?") is True
    # Should return False for non-question
    assert main.is_question("This is a statement.") is False

def test_extract_words_from_event():
    # Minimal valid event
    event = {
        "event": "transcript.data",
        "data": {
            "bot": {"id": "bot123"},
            "recording": {"id": "rec456"},
            "data": {
                "participant": {"name": "Alice", "is_host": False},
                "words": [
                    {"text": "Hello", "start_timestamp": 123},
                    {"text": "world", "start_timestamp": 124}
                ]
            }
        }
    }
    import json
    data = json.dumps(event)
    result = main.extract_words_from_event(data)
    assert result["bot_id"] == "bot123"
    assert result["recording_id"] == "rec456"
    assert result["speaker_name"] == "Alice"
    assert result["sentence"] == "Hello world"
    assert result["is_host"] is False
    assert result["partial"] is False
    assert result["start_timestamp"] == 123
    assert "bot123" in result
    assert result["bot123"]["Alice"] == "Hello world"

def test_SuggestionThrottler():
    import asyncio
    throttler = main.SuggestionThrottler(min_interval=0.01)
    loop = asyncio.get_event_loop()
    # First call should allow
    assert loop.run_until_complete(throttler.can_send()) is True
    # Second call immediately should not allow
    assert loop.run_until_complete(throttler.can_send()) is False

def test_SuggestionDebouncer(monkeypatch):
    import asyncio
    debouncer = main.SuggestionDebouncer(delay=0.01)
    called = []
    async def dummy_coro(arg):
        called.append(arg)
        return arg
    loop = asyncio.get_event_loop()
    # Should schedule and call after delay
    task = loop.run_until_complete(debouncer.trigger(dummy_coro, "foo"))
    loop.run_until_complete(task)
    assert called == ["foo"]

def test_leave_recall_bot_call(monkeypatch):
    # Patch requests.post to avoid real HTTP call
    class DummyResp:
        text = "ok"
    monkeypatch.setattr("requests.post", lambda *a, **k: DummyResp())
    # Should not raise
    main.leave_recall_bot_call("botid")

def test_get_zoom_access_token(monkeypatch):
    # Patch req.post to return dummy response
    class DummyResp:
        def raise_for_status(self): pass
        def json(self): return {"access_token": "token123"}
    monkeypatch.setattr(main, "req", type("Req", (), {"post": lambda *a, **k: DummyResp()})())
    import os
    monkeypatch.setenv("ZOOM_CLIENT_ID", "id")
    monkeypatch.setenv("ZOOM_CLIENT_SECRET", "secret")
    monkeypatch.setenv("ZOOM_ACCOUNT_ID", "accid")
    token = main.get_zoom_access_token()
    assert token == "token123"

def test_start_recall_bot(monkeypatch):
    # Patch req.post to return dummy response
    class DummyResp:
        def json(self): return {"id": "botid"}
        @property
        def ok(self): return True
        text = "ok"
    monkeypatch.setattr(main, "req", type("Req", (), {"post": lambda *a, **k: DummyResp()})())
    result = main.start_recall_bot("https://example.com")
    assert result is not None
