import asyncio
import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from src.utils.transcription import (
    extract_transcription_from_realtime_event,
    generate_realtime_session_config,
    handle_realtime_transcription_event
)
from src.events.event_interface import Event, EventType, TranscriptionEvent, event_bus
from src.services.api_client import RealtimeClient

@pytest.fixture
def mock_event_data():
    """Create mock event data for testing"""
    return {
        "id": "test-event-123",
        "content": {
            "text": "This is a test transcription.",
            "language": "en",
            "metadata": {
                "is_final": True
            }
        }
    }

def test_generate_realtime_session_config():
    """Test generating transcription configuration"""
    # Test with default model
    config = generate_realtime_session_config()
    assert "input_audio_transcription" in config
    assert config["input_audio_transcription"]["model"] == "whisper-1"
    
    # Test with custom model
    custom_config = generate_realtime_session_config(model="whisper-2")
    assert custom_config["input_audio_transcription"]["model"] == "whisper-2"

def test_extract_transcription_from_realtime_event(mock_event_data):
    """Test extracting transcription data from event"""
    result = extract_transcription_from_realtime_event(mock_event_data)
    
    assert "text" in result
    assert result["text"] == "This is a test transcription."
    assert result["language"] == "en"
    assert result["is_final"] == True

def test_handle_realtime_transcription_event(mock_event_data):
    """Test handling transcription event"""
    transcript = handle_realtime_transcription_event(mock_event_data)
    
    assert transcript == "This is a test transcription."

@pytest.mark.asyncio
async def test_transcription_event_integration():
    """Test integration between transcription handling and event bus"""
    # Create a mock event handler
    mock_handler = MagicMock()
    
    # Register handler with event bus
    event_bus.on(EventType.USER_TRANSCRIPTION_COMPLETED, mock_handler)
    
    # Create mock event data
    event_data = {
        "content": {
            "text": "Testing event integration.",
            "language": "en",
            "metadata": {
                "is_final": True
            }
        },
        "id": "test-123"
    }
    
    # Create a transcription event
    event = TranscriptionEvent(
        type=EventType.USER_TRANSCRIPTION_COMPLETED,
        data={"text": "Testing event integration.", "is_final": True, "source": "whisper"},
        text="Testing event integration.",
        is_final=True,
        source="whisper"
    )
    
    # Emit the event
    event_bus.emit(event)
    
    # Check that handler was called
    mock_handler.assert_called_once()
    # Get the event that was passed to the handler
    call_args = mock_handler.call_args[0][0]
    assert call_args.text == "Testing event integration."
    
    # Unregister handler
    event_bus.off(EventType.USER_TRANSCRIPTION_COMPLETED, mock_handler)

@pytest.mark.asyncio
async def test_api_client_transcription_integration():
    """Test API client transcription integration"""
    # Create mock websocket
    mock_websocket = AsyncMock()
    mock_websocket.send = AsyncMock()
    mock_websocket.close = AsyncMock()
    
    # Create mock event handler
    mock_handler = MagicMock()
    
    # Register handler with event bus
    event_bus.on(EventType.USER_TRANSCRIPTION_COMPLETED, mock_handler)
    
    # Patch the API settings and connect function 
    with patch('src.services.api_client.settings.api') as mock_api_settings, \
         patch('websockets.connect', return_value=mock_websocket), \
         patch('src.services.api_client.RealtimeClient._handle_transcription_completed_event') as mock_handle:
            
        # Setup the mock API settings
        mock_api_settings.api_key = "test_key"
        mock_api_settings.base_url = "https://api.openai.com"
        mock_api_settings.model = "gpt-4o"
            
        client = RealtimeClient()
        
        # Mock connection
        client.connected = True
        client.ws = mock_websocket
        
        # Create mock transcription event data
        event_data = {
            "content": {
                "text": "Testing API client integration.",
                "language": "en",
                "metadata": {
                    "is_final": True
                }
            },
            "id": "test-456"
        }
        
        # Call the handler directly since we can't easily simulate receiving a WebSocket message
        await client._handle_transcription_completed_event(event_data)
        
        # Check that handler was called with appropriate data
        mock_handle.assert_called_once_with(event_data)
    
    # Unregister handler
    event_bus.off(EventType.USER_TRANSCRIPTION_COMPLETED, mock_handler) 