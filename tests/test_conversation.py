# tests/test_conversation.py
import asyncio
import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.conversation import ConversationManager
from src.realtime_client import RealtimeClient

@pytest.fixture
def mock_realtime_client():
    """Create a mock Realtime client"""
    client = MagicMock(spec=RealtimeClient)
    client.update_session = AsyncMock()
    client.create_conversation_item = AsyncMock(return_value="event_id_1")
    client.create_response = AsyncMock(return_value="event_id_2")
    client.submit_function_call_output = AsyncMock(return_value="event_id_3")
    
    # Register event callbacks
    callback_registry = {}
    
    def register_callback(event_type, callback):
        if event_type not in callback_registry:
            callback_registry[event_type] = []
        callback_registry[event_type].append(callback)
    
    client.register_event_callback = register_callback
    client.event_callbacks = callback_registry
    
    return client

@pytest.fixture
def conversation(mock_realtime_client):
    """Create a ConversationManager with a mock client"""
    manager = ConversationManager(mock_realtime_client)
    return manager

@pytest.mark.asyncio
async def test_configure_session(conversation, mock_realtime_client):
    """Test configuring the conversation session"""
    result = await conversation.configure_session(
        instructions="Test instructions",
        tools=[{"type": "function", "name": "test_function"}],
        voice="alloy",
        vad_enabled=True,
        auto_response=True
    )
    
    # Check that the configuration was successful
    assert result is True
    
    # Check that the client method was called with the right parameters
    expected_config = {
        "instructions": "Test instructions",
        "tools": [{"type": "function", "name": "test_function"}],
        "voice": "alloy"
    }
    
    mock_realtime_client.update_session.assert_called_once()
    call_args = mock_realtime_client.update_session.call_args[0][0]
    for key, value in expected_config.items():
        assert call_args[key] == value

@pytest.mark.asyncio
async def test_send_text_message(conversation, mock_realtime_client):
    """Test sending a text message"""
    result = await conversation.send_text_message("Hello, world!")
    
    # Check that the message was sent successfully
    assert result == "event_id_1"
    
    # Check that the client method was called with the right parameters
    mock_realtime_client.create_conversation_item.assert_called_once()
    call_args = mock_realtime_client.create_conversation_item.call_args[0][0]
    assert call_args["type"] == "message"
    assert call_args["role"] == "user"
    assert call_args["content"][0]["type"] == "input_text"
    assert call_args["content"][0]["text"] == "Hello, world!"

@pytest.mark.asyncio
async def test_request_response(conversation, mock_realtime_client):
    """Test requesting a response"""
    result = await conversation.request_response(
        modalities=["text", "audio"],
        instructions="Test instructions",
        metadata={"test": "data"}
    )
    
    # Check that the response was requested successfully
    assert result == "event_id_2"
    
    # Check that the client method was called
    mock_realtime_client.create_response.assert_called_once()

@pytest.mark.asyncio
async def test_process_text_delta(conversation):
    """Test processing text delta events"""
    # Create a test callback
    callback_called = asyncio.Event()
    test_data = {"delta": "", "full_text": "", "response_id": ""}
    
    async def test_callback(delta, full_text, response_id):
        test_data["delta"] = delta
        test_data["full_text"] = full_text
        test_data["response_id"] = response_id
        callback_called.set()
    
    # Register the callback
    conversation.register_event_handler("on_text_response", test_callback)
    
    # Create a response
    response_id = "resp_123"
    conversation.responses[response_id] = {"id": response_id, "text": "Hello", "status": "created"}
    
    # Process a text delta event
    event_data = {
        "response": {"id": response_id},
        "delta": ", world!"
    }
    
    await conversation._handle_text_delta(event_data)
    
    # Wait for the callback to be called
    await asyncio.wait_for(callback_called.wait(), timeout=1.0)
    
    # Check that the callback was called with the right data
    assert test_data["delta"] == ", world!"
    assert test_data["full_text"] == "Hello, world!"
    assert test_data["response_id"] == response_id
    assert conversation.responses[response_id]["text"] == "Hello, world!"

@pytest.mark.asyncio
async def test_function_call(conversation, mock_realtime_client):
    """Test function call handling"""
    # Create a test callback
    callback_called = asyncio.Event()
    test_data = {"name": "", "arguments": {}, "call_id": "", "response_id": ""}
    
    async def test_callback(name, arguments, call_id, response_id):
        test_data["name"] = name
        test_data["arguments"] = arguments
        test_data["call_id"] = call_id
        test_data["response_id"] = response_id
        callback_called.set()
    
    # Register the callback
    conversation.register_event_handler("on_function_call", test_callback)
    
    # Create a response
    response_id = "resp_123"
    conversation.responses[response_id] = {
        "id": response_id,
        "text": "",
        "status": "created",
        "function_calls": []
    }
    
    # Process a function call arguments delta event
    call_id = "call_456"
    event_data = {
        "response": {"id": response_id},
        "call_id": call_id,
        "name": "test_function",
        "delta": '{"arg":"value"}'
    }
    
    await conversation._handle_function_call_delta(event_data)
    
    # Wait for the callback to be called
    await asyncio.wait_for(callback_called.wait(), timeout=1.0)
    
    # Check that the callback was called with the right data
    assert test_data["name"] == "test_function"
    assert test_data["arguments"] == {"arg": "value"}
    assert test_data["call_id"] == call_id
    assert test_data["response_id"] == response_id
    
    # Check that the function call was tracked
    assert call_id in conversation.function_calls
    assert conversation.function_calls[call_id]["name"] == "test_function"
    assert conversation.function_calls[call_id]["arguments_json"] == {"arg": "value"}
    
    # Test submitting a function result
    result = await conversation.submit_function_result(call_id, {"result": "test"})
    
    # Check that the result was submitted successfully
    assert result == "event_id_3"
    
    # Check that the client method was called
    mock_realtime_client.submit_function_call_output.assert_called_once_with(call_id, {"result": "test"})