"""
OpenAI Realtime API Event Handler.

This module centralizes the handling of all events from the OpenAI Realtime API.
It maps event types to handler functions and ensures consistent event processing.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import (
    AudioSpeechEvent,
    ErrorEvent,
    Event,
    EventType,
    TranscriptionEvent,
    event_bus,
)
from src.utils.error_handling import ApiError, ErrorSeverity
from src.utils.transcription import extract_transcription_from_realtime_event

logger = get_logger(__name__)

# Comprehensive list of all server events based on OpenAI's documentation
SERVER_EVENTS = {
    "session.created": "Session creation confirmation",
    "session.updated": "Session update confirmation",
    "conversation.created": "New conversation creation",
    "conversation.item.created": "New conversation item creation",
    "conversation.item.input_audio_transcription.completed": "Input audio transcription completion",
    "conversation.item.input_audio_transcription.failed": "Input audio transcription failure",
    "conversation.item.truncated": "Conversation items truncation",
    "conversation.item.deleted": "Conversation item deletion",
    "input_audio_buffer.committed": "Audio buffer commit acknowledgement",
    "input_audio_buffer.cleared": "Audio buffer clear acknowledgement",
    "input_audio_buffer.speech_started": "Speech detection in audio input",
    "input_audio_buffer.speech_stopped": "Speech end detection in audio input",
    "response.created": "Response creation confirmation",
    "response.done": "Response generation completion",
    "response.output_item.added": "New output item addition",
    "response.output_item.done": "Output items completion",
    "response.content_part.added": "New content part addition", 
    "response.content_part.done": "Content parts completion",
    "response.text.delta": "Incremental text data",
    "response.text.done": "Text data completion",
    "response.audio_transcript.delta": "Incremental audio transcript",
    "response.audio_transcript.done": "Audio transcript completion",
    "response.audio.delta": "Incremental audio data",
    "response.audio.done": "Audio data completion",
    "response.function_call_arguments.delta": "Incremental function call arguments",
    "response.function_call_arguments.done": "Function call arguments completion",
    "rate_limits.updated": "Rate limits update",
    "error": "Error notification"
}

# Standard client events from OpenAI's documentation
STANDARD_CLIENT_EVENTS = {
    "session.update": "Update session settings",
    "input_audio_buffer.append": "Append audio data to input buffer",
    "input_audio_buffer.commit": "Signal end of audio input",
    "input_audio_buffer.clear": "Clear current audio buffer",
    "conversation.item.create": "Add new conversation item",
    "conversation.item.truncate": "Remove conversation items",
    "conversation.item.delete": "Delete conversation item",
    "response.create": "Request response generation",
    "response.cancel": "Cancel response generation"
}

# Custom client events implemented by our application
CUSTOM_CLIENT_EVENTS = {
    "audio": "Send audio data directly (custom implementation)",
    "interrupt": "Interrupt assistant response (custom implementation)",
    "heartbeat": "Keep connection alive (custom implementation)"
}

# Combined client events dictionary for validation
CLIENT_EVENTS = {**STANDARD_CLIENT_EVENTS, **CUSTOM_CLIENT_EVENTS}


class RealtimeEventHandler:
    """
    Handler for processing OpenAI Realtime API events.
    
    This class centralizes the processing of all event types and 
    maps them to appropriate handler functions.
    """
    
    def __init__(self, client_ref=None):
        """
        Initialize the event handler.
        
        Args:
            client_ref: Optional reference to the RealtimeClient instance
        """
        self.client = client_ref
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """Set up mapping between event types and handler functions."""
        # Map event types to their respective handlers
        self.event_handlers = {
            # Session events
            "session.created": self.handle_session_created,
            "session.updated": self.handle_session_updated,
            
            # Conversation events
            "conversation.created": self.handle_conversation_created,
            "conversation.item.created": self.handle_conversation_item_created,
            
            # Transcription events
            "conversation.item.input_audio_transcription.completed": self.handle_transcription_completed,
            "conversation.item.input_audio_transcription.failed": self.handle_transcription_failed,
            
            # Input audio buffer events
            "input_audio_buffer.committed": self.handle_buffer_committed,
            "input_audio_buffer.cleared": self.handle_buffer_cleared,
            "input_audio_buffer.speech_started": self.handle_speech_started,
            "input_audio_buffer.speech_stopped": self.handle_speech_stopped,
            
            # Response events
            "response.created": self.handle_response_created,
            "response.done": self.handle_response_done,
            "response.text.delta": self.handle_text_delta,
            "response.audio.delta": self.handle_audio_delta,
            
            # Error events
            "error": self.handle_error
        }
    
    def set_client(self, client_ref) -> None:
        """
        Set the client reference.
        
        Args:
            client_ref: Reference to the RealtimeClient instance
        """
        self.client = client_ref
    
    def _log_custom_event_usage(self, event_type: str) -> None:
        """
        Log when a custom event is used.
        
        Args:
            event_type: The type of event being used
        """
        if event_type in CUSTOM_CLIENT_EVENTS:
            logger.info(f"Using custom client event: {event_type} - {CUSTOM_CLIENT_EVENTS[event_type]}")
    
    async def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Process an event from the OpenAI Realtime API.
        
        This is the main entry point for event handling. It validates
        the event type and dispatches it to the appropriate handler.
        
        Args:
            event_type: Type of the event
            event_data: Event data payload
        """
        # Validate event type
        if event_type not in SERVER_EVENTS:
            if settings.debug_mode:
                logger.warning(f"Received unknown event type: {event_type}")
            return
        
        # Log event for debugging
        log_level = logging.DEBUG
        if event_type == "error":
            log_level = logging.ERROR
        elif event_type.startswith("session."):
            log_level = logging.INFO
            
        logger.log(log_level, f"Processing event: {event_type}")
        
        # Get the appropriate handler
        handler = self.event_handlers.get(event_type)
        
        if handler:
            try:
                # Call the handler
                await handler(event_data)
            except Exception as e:
                logger.error(f"Error handling event {event_type}: {str(e)}")
                
                # Create an error event
                error_event = ErrorEvent(
                    type=EventType.ERROR,
                    data={
                        "error": {
                            "message": f"Error handling event {event_type}: {str(e)}",
                            "type": "event_handler_error"
                        }
                    },
                    error={
                        "message": f"Error handling event {event_type}: {str(e)}",
                        "type": "event_handler_error"
                    }
                )
                
                # Emit the error event
                event_bus.emit(error_event)
        else:
            # Default handling for events without specific handlers
            if settings.debug_mode:
                logger.debug(f"No specific handler for event type: {event_type}")
            
            # Create a generic event
            event = Event.from_dict({
                "type": event_type,
                "data": event_data
            })
            
            # Emit the event to the event bus
            event_bus.emit(event)
    
    def is_valid_event_type(self, event_type: str, is_client_event: bool = False) -> bool:
        """
        Check if an event type is valid according to the API documentation.
        
        Args:
            event_type: Type of the event to validate
            is_client_event: Whether this is a client event (vs server event)
            
        Returns:
            bool: True if the event type is valid
        """
        if is_client_event:
            # Check if it's a standard or custom client event
            is_standard = event_type in STANDARD_CLIENT_EVENTS
            is_custom = event_type in CUSTOM_CLIENT_EVENTS
            
            if is_custom and settings.debug_mode:
                logger.debug(f"Using custom client event: {event_type}")
                
            return is_standard or is_custom
        else:
            return event_type in SERVER_EVENTS
    
    # Session event handlers
    
    async def handle_session_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle session created event.
        
        Args:
            event_data: Event data
        """
        session_id = event_data.get("id")
        
        if session_id and self.client:
            # Update client's session ID
            self.client.session_id = session_id
        
        logger.info(f"Session created: {session_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.SESSION_CREATED,
            data=event_data,
            id=event_data.get("id")
        )
        event_bus.emit(event)
    
    async def handle_session_updated(self, event_data: Dict[str, Any]) -> None:
        """
        Handle session updated event.
        
        Args:
            event_data: Event data
        """
        logger.debug(f"Session updated: {json.dumps(event_data)}")
        
        # Create and emit event
        event = Event(
            type=EventType.SESSION_UPDATED,
            data=event_data,
            id=event_data.get("id")
        )
        event_bus.emit(event)
    
    # Conversation event handlers
    
    async def handle_conversation_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle conversation created event.
        
        Args:
            event_data: Event data
        """
        conversation_id = event_data.get("id")
        logger.info(f"Conversation created: {conversation_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.CONVERSATION_CREATED,
            data=event_data,
            id=event_data.get("id")
        )
        event_bus.emit(event)
    
    async def handle_conversation_item_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle conversation item created event.
        
        Args:
            event_data: Event data
        """
        item_id = event_data.get("id")
        item_type = event_data.get("type")
        
        logger.debug(f"Conversation item created: {item_type} (ID: {item_id})")
        
        # Create and emit event
        event = Event(
            type=EventType.MESSAGE_CREATED,
            data=event_data,
            id=item_id
        )
        event_bus.emit(event)
    
    # Transcription event handlers
    
    async def handle_transcription_completed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle transcription completed event.
        
        Args:
            event_data: Event data containing the transcription
        """
        try:
            # Process the transcription using our utilities
            transcription_data = extract_transcription_from_realtime_event(event_data)
            
            if transcription_data["text"]:
                # Create a TranscriptionEvent object
                transcription_event = TranscriptionEvent(
                    type=EventType.USER_TRANSCRIPTION_COMPLETED,
                    data={
                        "text": transcription_data["text"],
                        "is_final": transcription_data["is_final"],
                        "language": transcription_data.get("language", "en"),
                        "timestamp": transcription_data.get("timestamp", time.time()),
                        "item_id": event_data.get("id"),
                        "source": "whisper",
                    },
                    text=transcription_data["text"],
                    is_final=transcription_data["is_final"],
                    source="whisper"
                )
                
                # Emit the event
                event_bus.emit(transcription_event)
                
                logger.info(f"Transcription completed: '{transcription_data['text']}'")
            else:
                logger.warning("Received empty transcription from Whisper")
            
        except Exception as e:
            logger.error(f"Error handling transcription event: {str(e)}")
            logger.debug(f"Raw transcription event data: {json.dumps(event_data)}")
            
            # Create an error event
            error_event = ErrorEvent(
                type=EventType.ERROR,
                data={
                    "error": {
                        "message": f"Error processing transcription: {str(e)}",
                        "type": "transcription_error"
                    }
                },
                error={
                    "message": f"Error processing transcription: {str(e)}",
                    "type": "transcription_error"
                }
            )
            
            # Emit the error event
            event_bus.emit(error_event)
    
    async def handle_transcription_failed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle transcription failed event.
        
        Args:
            event_data: Event data
        """
        error = event_data.get("error", {})
        error_message = error.get("message", "Unknown error")
        error_type = error.get("type", "unknown")
        
        logger.error(f"Transcription failed: {error_type} - {error_message}")
        
        # Create and emit error event
        error_event = ErrorEvent(
            type=EventType.ERROR,
            data=event_data,
            id=event_data.get("id"),
            error=error
        )
        event_bus.emit(error_event)
    
    # Input audio buffer event handlers
    
    async def handle_buffer_committed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle buffer committed event.
        
        Args:
            event_data: Event data
        """
        logger.debug(f"Audio buffer committed: {json.dumps(event_data)}")
        
        # Create and emit event
        event = Event(
            type=EventType.AUDIO_BUFFER_COMMITTED,
            data=event_data,
            id=event_data.get("id")
        )
        event_bus.emit(event)
    
    async def handle_buffer_cleared(self, event_data: Dict[str, Any]) -> None:
        """
        Handle buffer cleared event.
        
        Args:
            event_data: Event data
        """
        logger.debug(f"Audio buffer cleared: {json.dumps(event_data)}")
        
        # Create and emit event
        event = Event(
            type=EventType.AUDIO_BUFFER_CLEARED,
            data=event_data,
            id=event_data.get("id")
        )
        event_bus.emit(event)
    
    async def handle_speech_started(self, event_data: Dict[str, Any]) -> None:
        """
        Handle speech started event.
        
        Args:
            event_data: Event data
        """
        logger.debug(f"Speech started: {json.dumps(event_data)}")
        
        # Create and emit event
        event = Event(
            type=EventType.USER_SPEECH_STARTED,
            data=event_data,
            id=event_data.get("id")
        )
        event_bus.emit(event)
    
    async def handle_speech_stopped(self, event_data: Dict[str, Any]) -> None:
        """
        Handle speech stopped event.
        
        Args:
            event_data: Event data
        """
        logger.debug(f"Speech stopped: {json.dumps(event_data)}")
        
        # Create and emit event
        event = Event(
            type=EventType.USER_SPEECH_FINISHED,
            data=event_data,
            id=event_data.get("id")
        )
        event_bus.emit(event)
    
    # Response event handlers
    
    async def handle_response_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response created event.
        
        Args:
            event_data: Event data
        """
        response_id = event_data.get("id")
        logger.debug(f"Response created: {response_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.RESPONSE_CREATED,
            data=event_data,
            id=response_id
        )
        event_bus.emit(event)
    
    async def handle_response_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response done event.
        
        Args:
            event_data: Event data
        """
        response_id = event_data.get("id")
        logger.debug(f"Response completed: {response_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.MESSAGE_COMPLETED,
            data=event_data,
            id=response_id
        )
        event_bus.emit(event)
    
    async def handle_text_delta(self, event_data: Dict[str, Any]) -> None:
        """
        Handle text delta event.
        
        Args:
            event_data: Event data
        """
        text = event_data.get("delta", {}).get("text", "")
        
        if text:
            # Create and emit event with text data
            event = Event(
                type=EventType.TEXT_CREATED,
                data=event_data,
                id=event_data.get("id")
            )
            event_bus.emit(event)
    
    async def handle_audio_delta(self, event_data: Dict[str, Any]) -> None:
        """
        Handle audio delta event.
        
        Args:
            event_data: Event data
        """
        # Extract audio chunk
        audio_data = event_data.get("delta", {}).get("audio", "")
        
        # Decode base64 audio data if present
        if audio_data:
            try:
                audio_bytes = base64.b64decode(audio_data)
                
                # Create and emit audio speech event
                event = AudioSpeechEvent(
                    type=EventType.AUDIO_SPEECH_CREATED,
                    data=event_data,
                    id=event_data.get("id"),
                    chunk=audio_bytes
                )
                event_bus.emit(event)
            except Exception as e:
                logger.error(f"Error decoding audio data: {str(e)}")
    
    # Error event handler
    
    async def handle_error(self, event_data: Dict[str, Any]) -> None:
        """
        Handle error event.
        
        Args:
            event_data: Event data
        """
        error = event_data.get("error", {})
        error_message = error.get("message", "Unknown error")
        error_type = error.get("type", "unknown")
        
        logger.error(f"API error: {error_type} - {error_message}")
        
        # Create and emit error event
        error_event = ErrorEvent(
            type=EventType.ERROR,
            data=event_data,
            id=event_data.get("id"),
            error=error
        )
        event_bus.emit(error_event)


# Create a singleton instance
event_handler = RealtimeEventHandler() 