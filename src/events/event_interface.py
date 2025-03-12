"""
Event interface for the OpenAI Realtime API.

This module defines the base event classes and handlers for working with
the OpenAI Realtime API event system, ensuring consistent event processing
throughout the application.
"""

import abc
import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from src.config.logging_config import get_logger
from src.utils.error_handling import ApiError

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events that can be emitted by the event bus."""
    
    # System events
    INITIALIZED = "initialized"
    SHUTDOWN = "shutdown"
    ERROR = "error"
    
    # Conversation events
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"
    CONVERSATION_STATE_CHANGED = "conversation_state_changed"
    
    # User speech events - using consistent naming convention
    USER_SPEECH_STARTED = "user.speech.started"
    USER_SPEECH_ONGOING = "user.speech.ongoing"
    USER_SPEECH_FINISHED = "user.speech.finished"
    USER_SPEECH_CANCELLED = "user.speech.cancelled"
    
    # Transcription events
    USER_TRANSCRIPTION_COMPLETED = "user_transcription_completed"
    
    # Assistant events
    ASSISTANT_MESSAGE_STARTED = "assistant_message_started"
    ASSISTANT_MESSAGE_CONTENT = "assistant_message_content"
    ASSISTANT_MESSAGE_COMPLETED = "assistant_message_completed"
    
    # Audio events
    AUDIO_SPEECH_CREATED = "audio_speech_created"
    AUDIO_PLAYBACK_STARTED = "audio_playback_started"
    AUDIO_PLAYBACK_STOPPED = "audio_playback_stopped"
    AUDIO_PLAYBACK_COMPLETED = "audio_playback_completed"
    
    # API events
    API_REQUEST_STARTED = "api_request_started"
    API_REQUEST_COMPLETED = "api_request_completed"
    API_RESPONSE_RECEIVED = "api_response_received"
    MESSAGE_CREATED = "message_created"
    MESSAGE_COMPLETED = "message_completed"
    
    # Function call events
    FUNCTION_CALL_RECEIVED = "function_call_received"
    FUNCTION_CALL_EXECUTED = "function_call_executed"
    FUNCTION_CALL_FAILED = "function_call_failed"
    
    # Session events
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.update"
    SESSION_DELETED = "session.deleted"
    
    # Message events
    MESSAGE_INCOMPLETE = "thread.message.incomplete"
    MESSAGE_DELTA = "thread.message.delta"
    
    # Run events
    RUN_CREATED = "thread.run.created"
    RUN_QUEUED = "thread.run.queued"
    RUN_IN_PROGRESS = "thread.run.in_progress"
    RUN_COMPLETED = "thread.run.completed"
    RUN_FAILED = "thread.run.failed"
    RUN_CANCELLED = "thread.run.cancelled"
    RUN_EXPIRED = "thread.run.expired"
    RUN_REQUIRES_ACTION = "thread.run.requires_action"
    
    # Catch-all for unknown events
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, event_type_str: str) -> 'EventType':
        """Convert a string to an EventType enum value."""
        try:
            # Try to match directly by value
            return next(e for e in cls if e.value == event_type_str)
        except StopIteration:
            logger.warning(f"Unknown event type: {event_type_str}")
            return cls.UNKNOWN


@dataclass
class Event:
    """Base class for all events in the system."""
    
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    created_at: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary."""
        result = asdict(self)
        # Convert EventType enum to string value
        result['type'] = self.type.value
        return result
    
    def to_json(self) -> str:
        """Convert the event to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create an event from a dictionary."""
        # Convert string to EventType enum
        event_type = EventType.from_string(data.get('type', 'unknown'))
        
        # Create appropriate event subclass based on type
        if event_type in SESSION_EVENT_TYPES:
            return SessionEvent(
                type=event_type,
                data=data.get('data', {}),
                id=data.get('id'),
                created_at=data.get('created_at'),
                session_id=data.get('data', {}).get('id')
            )
        elif event_type in MESSAGE_EVENT_TYPES:
            return MessageEvent(
                type=event_type,
                data=data.get('data', {}),
                id=data.get('id'),
                created_at=data.get('created_at'),
                message_id=data.get('data', {}).get('id'),
                thread_id=data.get('data', {}).get('thread_id')
            )
        elif event_type in RUN_EVENT_TYPES:
            return RunEvent(
                type=event_type,
                data=data.get('data', {}),
                id=data.get('id'),
                created_at=data.get('created_at'),
                run_id=data.get('data', {}).get('id'),
                thread_id=data.get('data', {}).get('thread_id')
            )
        elif event_type == EventType.AUDIO_SPEECH_CREATED:
            return AudioSpeechEvent(
                type=event_type,
                data=data.get('data', {}),
                id=data.get('id'),
                created_at=data.get('created_at'),
                chunk=data.get('data', {}).get('chunk', b'')
            )
        elif event_type == EventType.USER_TRANSCRIPTION_COMPLETED:
            return TranscriptionEvent(
                type=event_type,
                data=data.get('data', {}),
                id=data.get('id'),
                created_at=data.get('created_at'),
                text=data.get('data', {}).get('text', ''),
                is_final=data.get('data', {}).get('is_final', True),
                confidence=data.get('data', {}).get('confidence', 1.0),
                source=data.get('data', {}).get('source', 'whisper')
            )
        elif event_type == EventType.ERROR:
            return ErrorEvent(
                type=event_type,
                data=data.get('data', {}),
                id=data.get('id'),
                created_at=data.get('created_at'),
                error=data.get('data', {}).get('error', {})
            )
        else:
            # Default case, return base Event
            return cls(
                type=event_type,
                data=data.get('data', {}),
                id=data.get('id'),
                created_at=data.get('created_at')
            )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Create an event from a JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON event: {e}")
            return ErrorEvent(
                type=EventType.ERROR,
                data={"error": {"message": f"Invalid JSON: {str(e)}", "type": "json_decode_error"}},
            )


@dataclass
class SessionEvent(Event):
    """Events related to session lifecycle."""
    
    session_id: Optional[str] = None


@dataclass
class MessageEvent(Event):
    """Events related to messages."""
    
    message_id: Optional[str] = None
    thread_id: Optional[str] = None


@dataclass
class RunEvent(Event):
    """Events related to runs."""
    
    run_id: Optional[str] = None
    thread_id: Optional[str] = None


@dataclass
class AudioSpeechEvent(Event):
    """Events for audio speech output."""
    
    chunk: bytes = field(default_factory=bytes)


@dataclass
class ErrorEvent(Event):
    """Error events."""
    
    error: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSpeechEvent(Event):
    """Events related to user speech input."""
    
    audio_data: Optional[bytes] = None
    duration: float = 0.0
    is_final: bool = False


@dataclass
class TranscriptionEvent(Event):
    """Events related to speech transcription."""
    
    text: str = ""
    is_final: bool = True
    confidence: float = 1.0
    source: str = "whisper"  # Source of the transcription (e.g., "whisper", "other")


# Define event type categories for easier handling
SESSION_EVENT_TYPES = {
    EventType.SESSION_CREATED,
    EventType.SESSION_UPDATED,
    EventType.SESSION_DELETED
}

MESSAGE_EVENT_TYPES = {
    EventType.MESSAGE_CREATED,
    EventType.MESSAGE_COMPLETED,
    EventType.MESSAGE_INCOMPLETE,
    EventType.MESSAGE_DELTA
}

RUN_EVENT_TYPES = {
    EventType.RUN_CREATED,
    EventType.RUN_QUEUED,
    EventType.RUN_IN_PROGRESS,
    EventType.RUN_COMPLETED,
    EventType.RUN_FAILED,
    EventType.RUN_CANCELLED,
    EventType.RUN_EXPIRED,
    EventType.RUN_REQUIRES_ACTION
}

USER_SPEECH_EVENT_TYPES = {
    EventType.USER_SPEECH_STARTED,
    EventType.USER_SPEECH_ONGOING,
    EventType.USER_SPEECH_FINISHED,
    EventType.USER_SPEECH_CANCELLED
}


# Type for event handlers
EventHandlerType = Callable[[Event], None]


class EventEmitter:
    """
    Event emitter for publishing and subscribing to events.
    
    This class provides methods for registering event handlers and
    emitting events to all registered handlers.
    """
    
    def __init__(self):
        """Initialize the event emitter."""
        self._handlers: Dict[EventType, List[EventHandlerType]] = {}
        self._wildcard_handlers: List[EventHandlerType] = []
    
    def on(self, event_type: Union[EventType, str], handler: EventHandlerType) -> None:
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: The type of event to handle
            handler: The callback function to invoke when the event occurs
        """
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            event_type = EventType.from_string(event_type)
        
        # Initialize handler list if needed
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        # Add handler to list
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for event type: {event_type.value}")
    
    def on_any(self, handler: EventHandlerType) -> None:
        """
        Register a handler for all event types.
        
        Args:
            handler: The callback function to invoke when any event occurs
        """
        self._wildcard_handlers.append(handler)
        logger.debug("Registered wildcard event handler")
    
    def off(self, event_type: Union[EventType, str], handler: Optional[EventHandlerType] = None) -> None:
        """
        Remove a handler for a specific event type.
        
        Args:
            event_type: The type of event
            handler: The handler to remove. If None, removes all handlers for the event type.
        """
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            event_type = EventType.from_string(event_type)
        
        # Remove specific handler or all handlers for event type
        if event_type in self._handlers:
            if handler is None:
                self._handlers[event_type] = []
                logger.debug(f"Removed all handlers for event type: {event_type.value}")
            else:
                try:
                    self._handlers[event_type].remove(handler)
                    logger.debug(f"Removed handler for event type: {event_type.value}")
                except ValueError:
                    logger.warning(f"Handler not found for event type: {event_type.value}")
    
    def off_any(self, handler: Optional[EventHandlerType] = None) -> None:
        """
        Remove a wildcard handler.
        
        Args:
            handler: The handler to remove. If None, removes all wildcard handlers.
        """
        if handler is None:
            self._wildcard_handlers = []
            logger.debug("Removed all wildcard handlers")
        else:
            try:
                self._wildcard_handlers.remove(handler)
                logger.debug("Removed wildcard handler")
            except ValueError:
                logger.warning("Wildcard handler not found")
    
    def emit(self, event: Event) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event: The event to emit
        """
        # Call specific handlers for this event type
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.type.value}: {str(e)}")
        
        # Call wildcard handlers
        for handler in self._wildcard_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in wildcard event handler for {event.type.value}: {str(e)}")


# Global event emitter instance
event_bus = EventEmitter()


class EventHandler(abc.ABC):
    """Base class for event handlers."""
    
    def __init__(self):
        """Initialize the event handler."""
        self.register_handlers()
    
    @abc.abstractmethod
    def register_handlers(self) -> None:
        """Register event handlers with the event bus."""
        pass
    
    def unregister_handlers(self) -> None:
        """Unregister all event handlers."""
        # This method can be overridden in subclasses to provide
        # specific unregistration logic
        pass 