"""
Conversation manager for the OpenAI Realtime Assistant.

This module orchestrates the conversation flow between the user and the
assistant, handling state transitions, events, and coordinating the 
API client and audio services.
"""

import asyncio
import logging
import uuid
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Set, TypeVar, Union, Any, Callable

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import (
    AudioSpeechEvent,
    Event,
    EventType,
    TranscriptionEvent,
    UserSpeechEvent,
    event_bus,
)
from src.services.api_client import RealtimeClient
from src.services.audio_service import AudioService, AudioState
from src.utils.async_helpers import TaskManager, debounce
from src.utils.error_handling import AppError, ErrorSeverity, safe_execute

logger = get_logger(__name__)


class ConversationState(Enum):
    """Possible states for a conversation."""
    
    IDLE = auto()
    CONNECTING = auto()
    READY = auto()
    USER_SPEAKING = auto()
    ASSISTANT_SPEAKING = auto()
    THINKING = auto()
    ERROR = auto()
    DISCONNECTED = auto()


class ConversationManager:
    """
    Manager for coordinating conversation between user and assistant.
    
    This class is responsible for:
    - Managing the conversation state machine
    - Coordinating audio input/output with API communication
    - Handling speech and response events
    - Maintaining conversation context and history
    """
    
    def __init__(
        self,
        assistant_id: str,
        instructions: Optional[str] = None,
        temperature: float = 1.0,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ):
        """
        Initialize the conversation manager.
        
        Args:
            assistant_id: ID of the OpenAI assistant to use
            instructions: Optional custom instructions for the assistant
            temperature: Temperature parameter for generation (0.0-2.0)
            input_device: Optional audio input device index
            output_device: Optional audio output device index
        """
        # Initialize services
        self.api_client = RealtimeClient()
        self.audio_service = AudioService(
            input_device_index=input_device,
            output_device_index=output_device
        )
        
        # Assistant configuration
        self.assistant_id = assistant_id
        self.instructions = instructions
        self.temperature = temperature
        
        # State management
        self.state = ConversationState.IDLE
        self.task_manager = TaskManager()
        self.session_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        
        # Track active responses and function calls
        self.active_responses: Set[str] = set()
        self.pending_responses: Set[str] = set()
        self.pending_function_calls: Dict[str, Dict[str, Any]] = {}
        
        # Conversation history tracking
        self.messages: List[Dict[str, Any]] = []
        
        # Voice activity detection state
        self.user_speaking = False
        self.assistant_speaking = False
        self.interrupt_requested = False
        
        # Automatic response behavior
        self.auto_respond = True
        
        # Register event handlers
        self._register_event_handlers()
    
    async def start(self) -> bool:
        """
        Start the conversation session.
        
        This method connects to the OpenAI API and starts audio recording.
        
        Returns:
            bool: True if the session started successfully
        """
        if self.state != ConversationState.IDLE:
            logger.warning(f"Cannot start conversation in {self.state} state")
            return False
        
        logger.info("Starting conversation session")
        self.state = ConversationState.CONNECTING
        
        try:
            # Connect to the OpenAI Realtime API
            self.session_id = f"session_{uuid.uuid4()}"
            
            connection_successful = await self.api_client.connect(
                assistant_id=self.assistant_id,
                session_id=self.session_id,
                temperature=self.temperature,
                instructions=self.instructions,
                enable_transcription=True  # Enable transcription by default
            )
            
            if not connection_successful:
                raise AppError(
                    "Failed to connect to OpenAI Realtime API",
                    severity=ErrorSeverity.ERROR
                )
            
            # Start audio recording
            await self.audio_service.start_recording()
            
            # Update state
            self.state = ConversationState.READY
            logger.info("Conversation session started successfully")
            
            return True
            
        except Exception as e:
            error = AppError(
                f"Failed to start conversation: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            self.state = ConversationState.ERROR
            return False
    
    async def stop(self) -> None:
        """
        Stop the conversation session.
        
        This method disconnects from the API and stops audio recording.
        """
        logger.info("Stopping conversation session")
        
        # Stop audio services
        await self.audio_service.stop_recording()
        await self.audio_service.stop_playback()
        
        # Disconnect from API
        await self.api_client.disconnect()
        
        # Cancel all tasks
        await self.task_manager.cancel_all()
        
        # Update state
        self.state = ConversationState.DISCONNECTED
        logger.info("Conversation session stopped")
    
    async def request_response(self, instructions: Optional[str] = None) -> bool:
        """
        Request a response from the assistant.
        
        Args:
            instructions: Optional custom instructions for this response
        
        Returns:
            bool: True if the request was successful
        """
        if not self.api_client.is_connected:
            logger.error("Cannot request response: not connected to API")
            return False
        
        # Send the response request
        success = await self.api_client.request_response(instructions=instructions)
        
        if success:
            logger.info("Response requested from assistant")
            self.pending_responses.add("manual_request")
            self.state = ConversationState.THINKING
        else:
            logger.error("Failed to request response from assistant")
        
        return success
    
    async def send_text_message(self, text: str) -> bool:
        """
        Send a text message to the assistant.
        
        This bypasses speech recognition and sends text directly.
        
        Args:
            text: Text message to send
            
        Returns:
            bool: True if the message was sent successfully
        """
        if not self.api_client.is_connected:
            logger.error("Cannot send message: not connected to API")
            return False
        
        # TODO: Implement text message sending when API supports it
        logger.warning("Text messaging not yet implemented in OpenAI Realtime API")
        return False
    
    async def interrupt(self) -> bool:
        """
        Interrupt the assistant's current response.
        
        Returns:
            bool: True if the interrupt was successful
        """
        if not self.assistant_speaking:
            logger.warning("Cannot interrupt: assistant is not speaking")
            return False
        
        # Set interrupt flag
        self.interrupt_requested = True
        
        # Stop audio playback
        await self.audio_service.stop_playback()
        
        # Send interrupt signal to API
        success = await self.api_client.interrupt()
        
        if success:
            logger.info("Assistant interrupted")
            self.assistant_speaking = False
            self.state = ConversationState.READY
        else:
            logger.error("Failed to interrupt assistant")
            self.interrupt_requested = False
        
        return success
    
    async def set_transcription_enabled(self, enabled: bool = True) -> bool:
        """
        Enable or disable transcription for the session.
        
        When enabled, the system will use Whisper to generate transcriptions
        for user speech, which will be processed by the _handle_transcription_completed
        handler.
        
        Args:
            enabled: Whether to enable transcription
            
        Returns:
            bool: True if the operation was successful
        """
        if not self.api_client.is_connected:
            logger.error("Cannot set transcription: not connected to API")
            return False
            
        # Build session config update
        session_config = {}
        
        if enabled:
            # Import here to avoid circular imports
            from src.utils.transcription import generate_realtime_session_config
            transcription_config = generate_realtime_session_config()
            session_config.update(transcription_config)
            logger.info("Enabling transcription with Whisper")
        else:
            # Set to empty config to disable
            session_config["input_audio_transcription"] = None
            logger.info("Disabling transcription")
            
        # Update session configuration
        success = await self.api_client.update_session(session_config)
        
        if success:
            logger.info(f"Transcription {'enabled' if enabled else 'disabled'} successfully")
        else:
            logger.error(f"Failed to {'enable' if enabled else 'disable'} transcription")
            
        return success
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for API and audio events."""
        # Register handlers for user speech events
        event_bus.on(EventType.USER_SPEECH_STARTED, self._handle_user_speech_started)
        event_bus.on(EventType.USER_SPEECH_ONGOING, self._handle_user_speech_ongoing)
        event_bus.on(EventType.USER_SPEECH_FINISHED, self._handle_user_speech_finished)
        event_bus.on(EventType.USER_SPEECH_CANCELLED, self._handle_user_speech_cancelled)
        
        # Register handlers for transcription events
        event_bus.on(EventType.USER_TRANSCRIPTION_COMPLETED, self._handle_transcription_completed)
        
        # Register handlers for assistant events
        event_bus.on(EventType.AUDIO_SPEECH_CREATED, self._handle_assistant_speech)
        event_bus.on(EventType.MESSAGE_CREATED, self._handle_message_created)
        event_bus.on(EventType.MESSAGE_COMPLETED, self._handle_message_completed)
        
        # Register handler for errors
        event_bus.on(EventType.ERROR, self._handle_error)
    
    def _handle_user_speech_started(self, event: UserSpeechEvent) -> None:
        """
        Handle the start of user speech.
        
        Args:
            event: User speech event
        """
        if self.state == ConversationState.ASSISTANT_SPEAKING:
            # User is interrupting the assistant
            self.task_manager.create_task(self.interrupt(), "interrupt_assistant")
        
        # Set user speaking flag
        self.user_speaking = True
        self.state = ConversationState.USER_SPEAKING
        
        logger.debug("User started speaking")
    
    def _handle_user_speech_ongoing(self, event: UserSpeechEvent) -> None:
        """
        Handle ongoing user speech.
        
        Args:
            event: User speech event
        """
        if not event.audio_data:
            return
        
        # Send the audio chunk to the API
        self.task_manager.create_task(
            self.api_client.send_audio(event.audio_data, is_final=False),
            "send_audio_chunk"
        )
    
    def _handle_user_speech_finished(self, event: UserSpeechEvent) -> None:
        """
        Handle the end of user speech.
        
        Args:
            event: User speech event
        """
        if not event.audio_data:
            return
        
        # Send the final audio chunk to the API
        self.task_manager.create_task(
            self.api_client.send_audio(event.audio_data, is_final=True),
            "send_final_audio"
        )
        
        # Reset user speaking flag
        self.user_speaking = False
        
        # Automatically request a response if enabled
        if self.auto_respond:
            response_id = f"auto_response_{uuid.uuid4().hex[:8]}"
            self.pending_responses.add(response_id)
            self.task_manager.create_task(
                self.api_client.request_response(),
                f"auto_request_response_{response_id}"
            )
            self.state = ConversationState.THINKING
            logger.debug("Automatically requested response after user speech")
        else:
            self.state = ConversationState.READY
        
        logger.debug("User finished speaking")
    
    def _handle_user_speech_cancelled(self, event: UserSpeechEvent) -> None:
        """
        Handle cancelled user speech (too short or invalid).
        
        Args:
            event: User speech event
        """
        # Reset user speaking flag
        self.user_speaking = False
        self.state = ConversationState.READY
        
        logger.debug("User speech cancelled (too short or invalid)")
    
    def _handle_transcription_completed(self, event: Event) -> None:
        """
        Handle user speech transcription events.
        
        This method processes transcription from the Whisper model via the Realtime API
        and updates the conversation history with the transcript.
        
        Args:
            event: Transcription event
        """
        text = event.data.get("text", "")
        is_final = event.data.get("is_final", True)
        
        if not text:
            logger.debug("Received empty transcription")
            return
        
        # Process transcription for goodbye detection
        # Convert to lowercase and strip whitespace for better matching
        normalized_text = text.lower().strip()
        
        # Print the exact transcription for debugging
        print(f"\nTranscription received: '{normalized_text}'")
        
        # Check for exit command variations with more variations and partial matching
        exit_phrases = ["goodbye", "good bye", "bye", "exit", "quit", "end session", "stop session", "goodbye computer"]
        exit_partial_words = ["bye", "quit", "exit"]
        
        # Check for exact matches first
        is_exit_command = any(phrase in normalized_text for phrase in exit_phrases)
        
        # If no exact match, check for words containing our exit terms
        if not is_exit_command:
            words = normalized_text.split()
            is_exit_command = any(
                any(partial in word for partial in exit_partial_words)
                for word in words
            )
        
        if is_final and is_exit_command:
            # Extract which phrase was detected
            matched_phrase = next(
                (phrase for phrase in exit_phrases if phrase in normalized_text),
                next(
                    (f"partial:{word}" for word in words if any(partial in word for partial in exit_partial_words)),
                    "goodbye"
                )
            )
            
            # Log both to console and logger for visibility
            exit_msg = f"EXIT COMMAND DETECTED: '{matched_phrase}' in '{text}'"
            print(f"\n{exit_msg}")
            logger.warning(exit_msg)
            
            # Immediately attempt to send a response before shutting down
            try:
                asyncio.create_task(self.api_client.send_text("Goodbye! Shutting down the application."))
            except Exception as e:
                logger.error(f"Error sending goodbye response: {e}")
            
            # Respond to the user and schedule exit (ensure these are awaited)
            self.task_manager.create_task(self._respond_to_exit_command(), "exit_response")
            self.task_manager.create_task(self._schedule_exit(), "exit_schedule")
            
            # Directly emit a shutdown event as well, for redundancy
            event_bus.emit(EventType.SHUTDOWN, {
                "reason": "user_exit_command",
                "command": matched_phrase,
                "transcript": text
            })
            
            return

        # Log the transcript
        logger.info(f"Transcription: '{text}'")
        
        # Update conversation history with the transcript
        if is_final:
            # Add transcription to conversation history
            transcription = {
                "role": "user",
                "content": text,
                "is_transcription": True,
                "timestamp": event.data.get("timestamp", 0),
                "language": event.data.get("language", "en"),
                "source": event.data.get("source", "whisper")
            }
            
            # Add to messages if we don't already have a message for this speech
            if (not self.messages or 
                self.messages[-1].get("role") != "user" or 
                self.messages[-1].get("is_transcription", False) == False):
                self.messages.append(transcription)
            
            # Emit an event with the updated conversation state
            event_bus.emit(
                EventType.CONVERSATION_STATE_CHANGED, 
                {
                    "state": self.state.name,
                    "latest_message": transcription
                }
            )
    
    def _handle_assistant_speech(self, event: AudioSpeechEvent) -> None:
        """
        Handle assistant speech audio.
        
        Args:
            event: Audio speech event
        """
        if not event.chunk:
            return
        
        # First chunk of a response
        if not self.assistant_speaking:
            self.assistant_speaking = True
            self.state = ConversationState.ASSISTANT_SPEAKING
            logger.debug("Assistant started speaking")
        
        # Play the audio chunk
        self.task_manager.create_task(
            self.audio_service.play_audio(event.chunk),
            "play_audio_chunk"
        )
    
    def _handle_message_created(self, event: Event) -> None:
        """
        Handle message created event.
        
        Args:
            event: Message created event
        """
        message_data = event.data
        message_id = message_data.get("id")
        
        if message_id:
            logger.debug(f"New message created: {message_id}")
            
            # Add to conversation history
            self.messages.append(message_data)
    
    def _handle_message_completed(self, event: Event) -> None:
        """
        Handle message completed event.
        
        Args:
            event: Message completed event
        """
        # Reset assistant speaking flag
        self.assistant_speaking = False
        self.state = ConversationState.READY
        
        # Remove from pending responses if it was auto-requested
        if self.pending_responses and len(self.pending_responses) > 0:
            # Just remove the most recently added pending response
            self.pending_responses.pop()
        
        # Update message in history if it exists
        message_data = event.data
        message_id = message_data.get("id")
        
        if message_id:
            for i, message in enumerate(self.messages):
                if message.get("id") == message_id:
                    self.messages[i] = message_data
                    break
        
        logger.debug("Assistant finished speaking")
    
    def _handle_error(self, event: Event) -> None:
        """
        Handle error events.
        
        Args:
            event: Error event
        """
        error_data = event.data.get("error", {})
        error_type = error_data.get("type", "unknown")
        error_message = error_data.get("message", "Unknown error")
        
        logger.error(f"Error from OpenAI API: {error_type} - {error_message}")
        
        # Update state if in an active state
        if self.state in (ConversationState.THINKING, ConversationState.ASSISTANT_SPEAKING):
            self.state = ConversationState.READY
            
        # Reset flags
        self.assistant_speaking = False
        self.interrupt_requested = False 

    async def _respond_to_exit_command(self) -> None:
        """Respond to the user before exiting the application."""
        try:
            # Add a goodbye message to the conversation
            goodbye_message = {
                "role": "assistant",
                "content": "Goodbye! Shutting down the application.",
                "timestamp": time.time()
            }
            
            # Add to messages
            self.messages.append(goodbye_message)
            
            # Send audio response if possible
            if self.state != ConversationState.ERROR and self.api_client and self.api_client.is_connected():
                await self.api_client.send_text(
                    "Goodbye! Shutting down the application."
                )
            
            # Emit an event for UI updates
            event_bus.emit(
                EventType.ASSISTANT_MESSAGE_COMPLETED,
                {
                    "message": goodbye_message,
                    "is_final": True
                }
            )
            
            logger.info("Sent goodbye response to user")
        except Exception as e:
            logger.error(f"Error responding to exit command: {e}")
    
    async def _schedule_exit(self) -> None:
        """Schedule the application to exit after a short delay."""
        try:
            # Give time for the goodbye message to be processed
            await asyncio.sleep(3)
            
            # Log the shutdown at a higher level to ensure visibility
            logger.warning("Executing exit sequence from 'goodbye' command")
            
            # Stop the conversation session
            await self.stop()
            
            # First emit a signal for any listeners
            event_bus.emit(EventType.SHUTDOWN, {
                "reason": "user_exit_command", 
                "initiated_at": time.time(),
                "command": "goodbye",
                "source": "conversation_manager"
            })
            
            # Wait a moment to ensure the event is processed
            await asyncio.sleep(1)
            
            # For redundancy, emit the event again with a different source
            # This increases chances of it being captured
            event_bus.emit(EventType.SHUTDOWN, {
                "reason": "user_exit_command",
                "initiated_at": time.time(),
                "command": "goodbye",
                "source": "conversation_manager_final"
            })
            
            logger.info("Application exit sequence completed")
        except Exception as e:
            logger.error(f"Error scheduling exit: {e}") 