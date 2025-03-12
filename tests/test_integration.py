"""
Integration test for the OpenAI Realtime Assistant.

This script tests the integration between the transcription module, API client,
and conversation manager without requiring real API credentials.
"""

import asyncio
import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from src.utils.transcription import (
    extract_transcription_from_realtime_event,
    generate_realtime_session_config
)
from src.events.event_interface import Event, EventType, TranscriptionEvent, event_bus
from src.services.api_client import RealtimeClient
from src.domain.conversation.manager import ConversationManager, ConversationState

@pytest.mark.asyncio
async def test_transcription_conversation_flow():
    """Test the flow of transcription events through the system"""
    
    # Mock API and audio settings and services
    with patch('src.services.api_client.settings.api') as mock_api_settings, \
         patch('src.services.audio_service.settings.audio') as mock_audio_settings, \
         patch('websockets.connect', return_value=AsyncMock()) as mock_websocket_connect, \
         patch('src.services.audio_service.AudioService') as mock_audio_service:
         
        # Setup API settings
        mock_api_settings.api_key = "test_key"
        mock_api_settings.base_url = "https://api.openai.com"
        mock_api_settings.model = "gpt-4o"
        
        # Setup audio settings
        mock_audio_settings.sample_rate = 24000
        mock_audio_settings.channels = 1
        mock_audio_settings.sample_width = 2
        mock_audio_settings.chunk_size = 1024
        mock_audio_settings.input_device = None
        mock_audio_settings.output_device = None
        
        # Setup audio service mock
        mock_audio_service_instance = MagicMock()
        mock_audio_service.return_value = mock_audio_service_instance
        mock_audio_service_instance.start_recording = AsyncMock()
        mock_audio_service_instance.stop_recording = AsyncMock()
        mock_audio_service_instance.stop_playback = AsyncMock()
        
        # Create a conversation manager
        conversation_manager = ConversationManager(
            assistant_id="test-assistant-id",
            instructions="This is a test assistant"
        )
        
        # Mock the connection
        conversation_manager.api_client.connect = AsyncMock(return_value=True)
        conversation_manager.api_client.connected = True
        conversation_manager.api_client.ws = AsyncMock()
        
        # Track captured transcription events
        captured_events = []
        
        def capture_transcription_event(event):
            captured_events.append(event)
            print(f"Captured transcription event: {event.text}")
        
        # Register our event handler
        event_bus.on(EventType.USER_TRANSCRIPTION_COMPLETED, capture_transcription_event)
        
        try:
            # Start the conversation
            await conversation_manager.start()
            
            # Verify the conversation is in the READY state
            assert conversation_manager.state == ConversationState.READY
            
            # Simulate a transcription event
            mock_event_data = {
                "content": {
                    "text": "This is a simulated transcription from the Realtime API.",
                    "language": "en",
                    "metadata": {
                        "is_final": True
                    }
                },
                "id": "test-transcription-event"
            }
            
            # Send it directly to the handler
            await conversation_manager.api_client._handle_transcription_completed_event(mock_event_data)
            
            # Wait briefly for events to be processed
            await asyncio.sleep(0.1)
            
            # Verify we received a transcription event
            assert len(captured_events) > 0
            assert captured_events[0].text == "This is a simulated transcription from the Realtime API."
            
            # Verify the transcription was added to conversation history
            assert len(conversation_manager.messages) > 0
            assert conversation_manager.messages[0]["content"] == "This is a simulated transcription from the Realtime API."
            assert conversation_manager.messages[0]["role"] == "user"
            assert conversation_manager.messages[0]["is_transcription"] == True
            
            print("Integration test successful!")
            
        finally:
            # Clean up
            await conversation_manager.stop()
            event_bus.off(EventType.USER_TRANSCRIPTION_COMPLETED, capture_transcription_event)

if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_transcription_conversation_flow()) 