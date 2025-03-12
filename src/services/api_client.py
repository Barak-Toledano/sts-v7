"""
API client for the OpenAI Realtime API.

This module handles communication with the OpenAI Realtime API,
including initializing and maintaining WebSocket connections,
sending events, and dispatching received events.
"""

import asyncio
import base64
import json
import logging
import ssl
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    WebSocketException,
)

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import (
    AudioSpeechEvent,
    ErrorEvent,
    Event,
    EventType,
    event_bus,
    TranscriptionEvent,
)
from src.services.realtime_event_handler import event_handler
from src.utils.async_helpers import TaskManager, run_with_timeout
from src.utils.error_handling import ApiError, ErrorSeverity, safe_execute

logger = get_logger(__name__)


class RealtimeClient:
    """
    Client for interacting with the OpenAI Realtime API via WebSockets.
    
    This class handles:
    - Establishing and maintaining WebSocket connections
    - Sending events (audio, control messages, etc.)
    - Processing received events and dispatching to event handlers
    - Auto-reconnection and session management
    """
    
    def __init__(self):
        """Initialize the Realtime API client."""
        self.api_key = settings.api.api_key
        self.api_base = settings.api.api_base
        
        # Session state
        self.session_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.run_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.connected = False
        self.reconnecting = False
        self.first_connect = True
        
        # WebSocket connection
        self.ws: Optional[WebSocketClientProtocol] = None
        self.task_manager = TaskManager()
        
        # Connection settings
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0  # Initial delay in seconds
        self.max_reconnect_delay = 30.0  # Maximum delay in seconds
        self.connection_timeout = 10.0  # Timeout for connection attempts
        
        # Event sequence tracking
        self.event_sequence = 0
        self.pending_events: Set[str] = set()
        
        # Set reference to this client in the event handler
        event_handler.set_client(self)
    
    async def connect(
        self,
        assistant_id: str,
        session_id: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = 1.0,
        enable_transcription: bool = True,
    ) -> bool:
        """
        Connect to the OpenAI Realtime API.
        
        Args:
            assistant_id: ID of the OpenAI assistant to use
            session_id: Optional session ID (generated if not provided)
            instructions: Optional system instructions
            temperature: Temperature parameter for generation (0.0-2.0)
            enable_transcription: Whether to enable Whisper transcription
            
        Returns:
            bool: True if the connection was successful
        """
        if self.connected:
            logger.warning("Already connected to OpenAI Realtime API")
            return True
        
        logger.info(f"Connecting to OpenAI Realtime API with assistant {assistant_id}")
        
        # Store the connection parameters for potential reconnection
        self.connect_params = {
            "assistant_id": assistant_id,
            "session_id": session_id,
            "instructions": instructions,
            "temperature": temperature,
            "enable_transcription": enable_transcription
        }
        
        try:
            # Generate a session ID if not provided
            if not session_id:
                session_id = f"session_{uuid.uuid4().hex}"
            
            self.thread_id = assistant_id
            self.session_id = session_id
            
            # Create URL with query parameters
            url = f"{settings.api.base_url}/v1/realtime?model={settings.api.model}"
            
            # Setup headers
            headers = {
                "Authorization": f"Bearer {settings.api.key}",
                "OpenAI-Beta": "realtime=v1",
            }
            
            # Connect to WebSocket
            self.ws = await websockets.connect(
                url,
                extra_headers=headers,
                max_size=None,  # No limit on message size
                ping_interval=30,  # 30 seconds ping interval
                ping_timeout=10,  # 10 seconds ping timeout
            )
            
            # Start message handling task
            self.task_manager.create_task(
                self._message_handler(),
                "websocket_message_handler"
            )
            
            # Wait for session.created event
            session_created = await self._wait_for_session_created()
            
            if not session_created:
                raise ApiError(
                    "Failed to receive session.created event",
                    severity=ErrorSeverity.ERROR
                )
            
            # Configure session with provided parameters
            session_config = {
                "voice": "alloy",  # Default voice
                "instructions": instructions or "",
                "temperature": temperature,
                "turn_detection": {"type": "server_vad"},
                "modalities": ["audio", "text"],
            }
            
            # Add transcription configuration if enabled
            if enable_transcription:
                from src.utils.transcription import generate_realtime_session_config
                transcription_config = generate_realtime_session_config()
                session_config.update(transcription_config)
            
            # Update session configuration
            success = await self.update_session(session_config)
            
            if not success:
                raise ApiError(
                    "Failed to update session configuration",
                    severity=ErrorSeverity.ERROR
                )
            
            self.connected = True
            logger.info("Successfully connected to OpenAI Realtime API")
            
            return True
        
        except Exception as e:
            # Clean up websocket if it was created
            if self.ws:
                await self.ws.close()
                self.ws = None
            
            error = ApiError(
                f"Failed to connect to OpenAI Realtime API: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            
            # Emit error event
            event_bus.emit(EventType.ERROR, {"error": error.to_dict()})
            
            return False
    
    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to the OpenAI Realtime API.
        
        Returns:
            bool: True if reconnection was successful
        """
        if not hasattr(self, 'connect_params'):
            logger.error("Cannot reconnect: no previous connection parameters")
            return False
        
        self.reconnecting = True
        
        # Close existing connection if it exists
        await self.disconnect(clear_session=False)
        
        # Attempt reconnection with exponential backoff
        attempt = 0
        delay = self.reconnect_delay
        
        while attempt < self.max_reconnect_attempts:
            attempt += 1
            logger.info(f"Reconnection attempt {attempt}/{self.max_reconnect_attempts}")
            
            try:
                result = await self.connect(**self.connect_params)
                if result:
                    logger.info(f"Reconnected successfully on attempt {attempt}")
                    return True
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt} failed: {str(e)}")
            
            # Wait before next attempt with exponential backoff
            logger.info(f"Waiting {delay:.1f}s before next reconnection attempt")
            await asyncio.sleep(delay)
            
            # Increase delay with exponential backoff (2x), capped at max_reconnect_delay
            delay = min(delay * 2, self.max_reconnect_delay)
        
        logger.error(f"Failed to reconnect after {self.max_reconnect_attempts} attempts")
        self.reconnecting = False
        return False
    
    async def disconnect(self, clear_session: bool = True) -> None:
        """
        Disconnect from the OpenAI Realtime API.
        
        Args:
            clear_session: Whether to clear session IDs
        """
        # Cancel all background tasks
        self.task_manager.cancel_all()
        
        # Close WebSocket connection
        if self.ws:
            try:
                await self.ws.close()
                logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {str(e)}")
            finally:
                self.ws = None
        
        # Reset connection state
        self.connected = False
        
        # Optionally clear session information
        if clear_session:
            self.session_id = None
            self.run_id = None
            self.message_id = None
    
    async def send_audio(self, audio_data: bytes, is_final: bool = False) -> bool:
        """
        Send audio data to the OpenAI Realtime API.
        
        Note: This uses a custom "audio" event type that is not part of the standard
        OpenAI Realtime API specification but is implemented in our client.
        
        Args:
            audio_data: Audio data as bytes (16-bit PCM, 24kHz, mono)
            is_final: Whether this is the final chunk in the stream
            
        Returns:
            bool: True if audio was sent successfully
        """
        if not self.connected or not self.ws:
            logger.error("Cannot send audio: not connected")
            return False
        
        if not audio_data:
            logger.warning("Cannot send empty audio data")
            return False
        
        try:
            # Encode audio data as base64
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')
            
            # Prepare audio event
            event = {
                "type": "audio",  # Custom event type
                "data": {
                    "audio": encoded_audio,
                    "is_final": is_final
                }
            }
            
            # Send event
            await self._send_event(event)
            
            if is_final:
                logger.debug(f"Sent final audio chunk ({len(audio_data)} bytes)")
            else:
                logger.debug(f"Sent audio chunk ({len(audio_data)} bytes)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending audio data: {str(e)}")
            return False
    
    async def request_response(self, instructions: Optional[str] = None) -> bool:
        """
        Request a response from the OpenAI Realtime API.
        
        Args:
            instructions: Optional instructions to override assistant's default instructions
            
        Returns:
            bool: True if request was sent successfully
        """
        if not self.connected or not self.ws:
            logger.error("Cannot request response: not connected")
            return False
        
        try:
            # Prepare request
            request = {
                "type": "response.create",
                "data": {}
            }
            
            # Add instructions if provided
            if instructions:
                request["data"]["instructions"] = instructions
            
            # Send request
            event_id = await self._send_event(request)
            if event_id:
                logger.info("Requested response from assistant")
                return True
            else:
                logger.error("Failed to send response request")
                return False
            
        except Exception as e:
            logger.error(f"Error requesting response: {str(e)}")
            return False
    
    async def request_run(
        self,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> bool:
        """
        Request a run from the OpenAI Realtime API.
        
        Args:
            instructions: Optional instructions to override assistant's default instructions
            temperature: Optional temperature setting for the model
            top_p: Optional top_p setting for the model
            
        Returns:
            bool: True if request was sent successfully
        """
        if not self.connected or not self.ws:
            logger.error("Cannot request run: not connected")
            return False
        
        if not self.thread_id:
            logger.error("Cannot request run: no thread_id available")
            return False
        
        try:
            # Prepare run request
            run_data = {
                "thread_id": self.thread_id,
            }
            
            # Add optional parameters
            if instructions:
                run_data["instructions"] = instructions
            
            if temperature is not None:
                run_data["temperature"] = temperature
            
            if top_p is not None:
                run_data["top_p"] = top_p
            
            # Create run request
            request = {
                "type": "thread.run.create",
                "data": run_data
            }
            
            # Send request
            event_id = await self._send_event(request)
            if event_id:
                logger.info(f"Requested run for thread {self.thread_id}")
                return True
            else:
                logger.error("Failed to send run request")
                return False
            
        except Exception as e:
            logger.error(f"Error requesting run: {str(e)}")
            return False
    
    async def submit_tool_outputs(
        self,
        tool_outputs: List[Dict[str, Any]],
    ) -> bool:
        """
        Submit tool outputs for a run that requires action.
        
        Args:
            tool_outputs: List of tool outputs
            
        Returns:
            bool: True if submission was successful
        """
        if not self.connected or not self.ws:
            logger.error("Cannot submit tool outputs: not connected")
            return False
        
        if not self.thread_id or not self.run_id:
            logger.error("Cannot submit tool outputs: missing thread_id or run_id")
            return False
        
        try:
            # Prepare submission
            request = {
                "type": "thread.run.tool_outputs.create",
                "data": {
                    "thread_id": self.thread_id,
                    "run_id": self.run_id,
                    "tool_outputs": tool_outputs
                }
            }
            
            # Send request
            event_id = await self._send_event(request)
            if event_id:
                logger.info(f"Submitted {len(tool_outputs)} tool outputs for run {self.run_id}")
                return True
            else:
                logger.error("Failed to send tool outputs")
                return False
            
        except Exception as e:
            logger.error(f"Error submitting tool outputs: {str(e)}")
            return False
    
    async def cancel_run(self) -> bool:
        """
        Cancel the current run.
        
        Returns:
            bool: True if cancellation request was successful
        """
        if not self.connected or not self.ws:
            logger.error("Cannot cancel run: not connected")
            return False
        
        if not self.thread_id or not self.run_id:
            logger.error("Cannot cancel run: missing thread_id or run_id")
            return False
        
        try:
            # Prepare cancellation request
            request = {
                "type": "thread.run.cancel",
                "data": {
                    "thread_id": self.thread_id,
                    "run_id": self.run_id
                }
            }
            
            # Send request
            event_id = await self._send_event(request)
            if event_id:
                logger.info(f"Requested cancellation of run {self.run_id}")
                return True
            else:
                logger.error("Failed to send cancellation request")
                return False
            
        except Exception as e:
            logger.error(f"Error cancelling run: {str(e)}")
            return False
    
    async def interrupt(self) -> bool:
        """
        Send an interrupt signal to stop the assistant mid-response.
        
        Note: This uses a custom "interrupt" event type that is not part of the standard
        OpenAI Realtime API specification but is implemented in our client.
        
        Returns:
            bool: True if interrupt signal was sent successfully
        """
        if not self.connected or not self.ws:
            logger.error("Cannot send interrupt: not connected")
            return False
        
        try:
            # Prepare interrupt signal
            interrupt = {
                "type": "interrupt"  # Custom event type
            }
            
            # Send interrupt
            event_id = await self._send_event(interrupt)
            if event_id:
                logger.info("Sent interrupt signal to stop assistant response")
                return True
            else:
                logger.error("Failed to send interrupt signal")
                return False
            
        except Exception as e:
            logger.error(f"Error sending interrupt: {str(e)}")
            return False
    
    async def _send_json_message(self, data: Dict[str, Any]) -> None:
        """
        Send a JSON message over the WebSocket connection.
        
        Args:
            data: Dictionary to send as JSON
        """
        if not self.ws:
            raise ApiError(
                "Cannot send message: WebSocket connection not established",
                severity=ErrorSeverity.ERROR,
                error_code="no_connection",
            )
        
        try:
            # Serialize and send
            message = json.dumps(data)
            await self.ws.send(message)
        except Exception as e:
            logger.error(f"Error sending JSON message: {str(e)}")
            raise ApiError(
                f"Failed to send message: {str(e)}",
                severity=ErrorSeverity.ERROR,
                error_code="send_failed",
                cause=e,
            )
    
    async def _send_event(self, event: Dict[str, Any]) -> Optional[str]:
        """
        Send an event to the OpenAI Realtime API with tracking ID.
        
        Args:
            event: Event data to send
            
        Returns:
            Optional[str]: Event ID if sent successfully
        """
        # Generate event ID if not already present
        if "id" not in event:
            event["id"] = f"event_{uuid.uuid4()}"
        
        # Add sequence number
        if "sequence" not in event:
            event["sequence"] = self.event_sequence
            self.event_sequence += 1
        
        # Track pending event
        self.pending_events.add(event["id"])
        
        try:
            # Send event
            await self._send_json_message(event)
            
            # Debug log based on event type
            event_type = event.get("type", "unknown")
            
            if event_type == "audio":
                # Don't log audio data as it's too verbose
                logger.debug(
                    f"Sent event: id={event['id']}, type={event_type}, "
                    f"is_final={event.get('data', {}).get('is_final', False)}"
                )
            elif event_type == "session.update":
                # Log session updates with relevant details
                logger.debug(
                    f"Sent event: id={event['id']}, type={event_type}, "
                    f"data={json.dumps({k: v for k, v in event.get('data', {}).items() if k != 'instructions'})}"
                )
            else:
                # Log other events with full data
                logger.debug(f"Sent event: {json.dumps(event)}")
            
            return event["id"]
            
        except Exception as e:
            logger.error(f"Error sending event: {str(e)}")
            if event["id"] in self.pending_events:
                self.pending_events.remove(event["id"])
            return None
    
    async def _listen_for_messages(self) -> None:
        """
        Listen for incoming messages from the WebSocket connection.
        
        This coroutine runs in a background task and processes
        all incoming messages from the OpenAI Realtime API.
        """
        if not self.ws:
            logger.error("Cannot listen for messages: WebSocket connection not established")
            return
        
        try:
            async for message in self.ws:
                await self._process_message(message)
                
        except ConnectionClosedOK:
            logger.info("WebSocket connection closed normally")
            self.connected = False
            
        except ConnectionClosedError as e:
            logger.error(f"WebSocket connection closed with error: {e.code} {e.reason}")
            self.connected = False
            
            # Try to reconnect if appropriate
            if not self.reconnecting and e.code not in (1000, 1001):  # Not normal closure
                logger.info("Attempting to reconnect...")
                self.task_manager.create_task(self.reconnect(), "reconnect")
                
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {str(e)}")
            self.connected = False
            
            # Try to reconnect if appropriate
            if not self.reconnecting:
                logger.info("Attempting to reconnect due to error...")
                self.task_manager.create_task(self.reconnect(), "reconnect")
    
    async def _send_heartbeat(self) -> None:
        """
        Send periodic heartbeat messages to keep the connection alive.
        
        Note: This uses a custom "heartbeat" event type that is not part of the standard
        OpenAI Realtime API specification but is implemented in our client.
        """
        while self.connected and self.ws:
            try:
                # Send heartbeat event
                await self._send_event({"type": "heartbeat"})  # Custom event type
                
                # Wait for next heartbeat interval
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Error sending heartbeat: {str(e)}")
                
                # If we lost connection, try to reconnect
                if not self.reconnecting and not self.connected:
                    logger.info("Attempting to reconnect due to heartbeat failure...")
                    self.task_manager.create_task(self.reconnect(), "reconnect")
                    break
    
    async def _process_message(self, message: Union[str, bytes]) -> None:
        """
        Process a message received from the WebSocket connection.
        
        Args:
            message: Raw message from WebSocket
        """
        try:
            # Parse JSON message
            if isinstance(message, bytes):
                data = json.loads(message.decode('utf-8'))
            else:
                data = json.loads(message)
            
            # Extract message details
            message_id = data.get('id')
            message_type = data.get('type')
            
            # Process using the centralized event handler
            await event_handler.handle_event(message_type, data.get('data', {}))
            
            # Remove from pending events if it was a response to one
            if message_id in self.pending_events:
                self.pending_events.remove(message_id)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON message: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected."""
        return self.connected and self.ws is not None

    async def update_session(self, session_config: Dict[str, Any]) -> bool:
        """
        Update the session configuration.
        
        Args:
            session_config: New session configuration
            
        Returns:
            bool: True if session configuration was updated successfully
        """
        if not self.connected or not self.ws:
            logger.error("Cannot update session: not connected")
            return False
        
        try:
            # Send session update request
            request = {
                "type": "session.update",
                "data": session_config
            }
            
            # Send request
            event_id = await self._send_event(request)
            if event_id:
                logger.info("Session configuration updated successfully")
                return True
            else:
                logger.error("Failed to send session update request")
                return False
            
        except Exception as e:
            logger.error(f"Error updating session configuration: {str(e)}")
            return False

    async def _message_handler(self) -> None:
        """
        Handle incoming messages from the WebSocket connection.
        """
        while self.connected and self.ws:
            try:
                message = await self.ws.recv()
                await self._process_message(message)
            except Exception as e:
                logger.error(f"Error receiving message: {str(e)}")
                self.connected = False
                break

    async def _wait_for_session_created(self) -> bool:
        """
        Wait for the session.created event to be received.
        
        Returns:
            bool: True if session.created event was received
        """
        while not self.session_id:
            await asyncio.sleep(0.1)
        
        return True 