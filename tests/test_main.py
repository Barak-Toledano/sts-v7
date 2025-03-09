# tests/test_main.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.main import (
    handle_text_response,
    handle_transcript,
    handle_function_call,
    handle_response_complete,
    handle_error,
    setup_conversation
)

@pytest.mark.asyncio
async def test_handle_text_response():
    """Test text response handler"""
    with patch('builtins.print') as mock_print:
        await handle_text_response("Hello", "Hello world", "resp_123")
        mock_print.assert_called_once_with("Hello", end="", flush=True)

@pytest.mark.asyncio
async def test_handle_function_call():
    """Test function call handler"""
    # Create mock conversation manager
    mock_conversation = MagicMock()
    mock_conversation.submit_function_result = AsyncMock()
    
    with patch('builtins.print') as mock_print:
        # Test check_availability function
        await handle_function_call(
            "check_availability",
            {"date": "2023-01-01", "service_type": "Consultation"},
            "call_123",
            "resp_456",
            mock_conversation
        )
        
        # Check that the function was called
        mock_print.assert_called()
        
        # Check that function result was submitted
        mock_conversation.submit_function_result.assert_called_once()
        call_args = mock_conversation.submit_function_result.call_args[0]
        assert call_args[0] == "call_123"
        assert "available_slots" in call_args[1]

@pytest.mark.asyncio
async def test_setup_conversation():
    """Test conversation setup"""
    mock_conversation = MagicMock()
    mock_conversation.configure_session = AsyncMock()
    mock_conversation.send_text_message = AsyncMock()
    mock_conversation.request_response = AsyncMock()
    
    await setup_conversation(mock_conversation)
    
    # Check that the conversation was configured
    mock_conversation.configure_session.assert_called_once()
    
    # Check that the initial message was sent
    mock_conversation.send_text_message.assert_called_once()
    
    # Check that a response was requested
    mock_conversation.request_response.assert_called_once()