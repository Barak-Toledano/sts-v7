import json
import asyncio
import base64
import uuid
import websockets
from typing import Dict, Any, Optional, Callable, List, Union, Set
import logging
from datetime import datetime
import time

from src.utils.logging_utils import setup_logger
from src.config import settings

# Set up logger
logger = setup_logger("realtime_client")

class RealtimeClient:
    """
    Client for interacting with OpenAI's Realtime API via WebSockets
    """
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_REALTIME_MODEL
        self.websocket_url = f"{settings.WEBSOCKET_URL}?model={self.model}"
        self.websocket = None
        self.session_id = None
        self.session_state = {}
        self.conversation_items = {}
        self.event_callbacks = {}
        self.responses = {}
        self.connected = False
        self.connection_error = None
        self.heartbeat_task = None
        self.receiver_task = None
        self.max_audio_chunk_size = 15 * 1024 * 1024  # 15MB limit per OpenAI docs
        
        # Track request event IDs to correlate with errors
        self.pending_events = {}
        
        # Track voice activity detection state
        self.vad_enabled = True
        self.speech_active = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
        
    async def connect(self) -> bool:
        """
        Establish a WebSocket connection to the OpenAI Realtime API
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if not self.api_key:
            logger.error("API key not found. Please set OPENAI_API_KEY in your environment.")
            return False
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"  # Required beta header
        }
        
        try:
            logger.info(f"Connecting to {self.websocket_url}")
            self.websocket = await websockets.connect(
                self.websocket_url,
                extra_headers=headers,
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=10,   # Wait 10 seconds for pong before timeout
                close_timeout=5    # Allow 5 seconds for clean close
            )
            self.connected = True
            logger.info("Successfully connected to OpenAI Realtime API")
            
            # Start the message receiver loop
            self.receiver_task = asyncio.create_task(self._receive_messages())
            
            # Start the heartbeat monitoring loop
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            return True
            
        except Exception as e:
            self.connection_error = str(e)
            logger.error(f"Failed to connect to OpenAI Realtime API: {e}")
            return False
        
    async def ensure_connected(self) -> bool:
        """
        Ensure the client is connected, attempting to reconnect if necessary
        with exponential backoff for retries.
        
        Returns:
            bool: True if connected, False if failed to connect
        """
        if self.connected and self.websocket:
            return True
        
        # Reconnection parameters
        max_retries = 3
        base_delay = 0.5  # Start with 0.5 second delay
        
        # Try to reconnect with exponential backoff
        logger.info("Connection lost, attempting to reconnect...")
        
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                # Calculate backoff delay (0.5s, 1s, 2s)
                delay = base_delay * (2 ** (attempt - 1))
                logger.info(f"Waiting {delay:.1f}s before reconnection attempt {attempt}/{max_retries}...")
                await asyncio.sleep(delay)
            
            # Attempt to connect
            if await self.connect():
                logger.info(f"Successfully reconnected on attempt {attempt}")
                return True
            
            logger.warning(f"Reconnection attempt {attempt}/{max_retries} failed")
        
        logger.error(f"Failed to reconnect after {max_retries} attempts")
        return False
    
    async def disconnect(self) -> None:
        """Close the WebSocket connection and cleanup resources"""
        # Set connection status to false first to prevent new operations
        self.connected = False
        
        # Cancel any running tasks
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self.receiver_task and not self.receiver_task.done():
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass
        
        # Close the WebSocket connection
        if self.websocket:
            try:
                await self.websocket.close(code=1000, reason="Client disconnecting")
                logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                logger.warning(f"Error during WebSocket closure: {e}")
            finally:
                self.websocket = None
    
    # In the RealtimeClient class, let's improve the event handling:
    # Add this method for waiting for specific events
    async def wait_for_event(self, event_type: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """
        Wait for a specific event type to be received
        
        Args:
            event_type: The event type to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Optional[Dict[str, Any]]: The event data if received, None if timed out
        """
        event_received = asyncio.Event()
        event_data = [None]  # Use a list to store the event data
        
        async def event_handler(data):
            event_data[0] = data
            event_received.set()
        
        # Register the temporary handler
        self.register_event_callback(event_type, event_handler)
        
        try:
            # Wait for the event or timeout
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
            return event_data[0]
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for event: {event_type}")
            return None
        finally:
            # Clean up the temporary handler
            if event_type in self.event_callbacks and event_handler in self.event_callbacks[event_type]:
                self.event_callbacks[event_type].remove(event_handler)

    async def _heartbeat_loop(self) -> None:
        """
        Monitor the connection health
        This method no longer sends custom heartbeat events since they're not in the API spec
        Instead, it relies on the built-in ping/pong from websockets
        """
        try:
            while self.connected and self.websocket:
                await asyncio.sleep(25)  # Check connection periodically
                if not self.connected or not self.websocket:
                    break
        except asyncio.CancelledError:
            # Task was cancelled, just exit
            pass
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")
        
    async def _send_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Send an event to the OpenAI Realtime API
        
        Args:
            event_type: Type of event to send
            data: Event data payload
            
        Returns:
            str: Event ID for tracking
        """
        # Check if connected and try to reconnect if needed
        if not self.connected or not self.websocket:
            logger.warning("Not connected, attempting to reconnect...")
            connected = await self.ensure_connected()
            if not connected:
                logger.error("Cannot send event: not connected and reconnection failed")
                raise ConnectionError("Not connected to the Realtime API")
        
        # Create a unique event ID using UUID for reliability
        event_id = f"evt_{uuid.uuid4().hex}"
        
        # Create the full event payload
        event = {
            "type": event_type,
            "event_id": event_id,
            **data
        }
        
        # Track this event for error correlation
        self.pending_events[event_id] = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        try:
            event_json = json.dumps(event)
            # Additional debugging - what we're actually sending to the API
            if event_type == "session.update" and "session" in data and "instructions" in data["session"]:
                print(f"DEBUG - SENDING TO API: type={event_type}, instructions={data['session']['instructions'][:50]}...")
                print(f"DEBUG - FULL EVENT JSON: {event_json[:200]}...")
            
            logger.debug(f"Sending event: {event_type} ({event_id})")
            await self.websocket.send(event_json)
            return event_id
        except websockets.exceptions.ConnectionClosedOK as e:
            logger.info(f"Cannot send event: WebSocket already closed normally: {e}")
            self.connected = False
            raise
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Cannot send event: WebSocket closed with error: {e}")
            self.connected = False
            raise
        except Exception as e:
            logger.error(f"Error sending event: {e}")
            raise
    
    async def _receive_messages(self) -> None:
        """
        Continuously receive and process messages from the WebSocket
        """
        if not self.websocket:
            logger.error("Cannot receive messages: WebSocket is not connected")
            return
            
        try:
            async for message in self.websocket:
                await self._process_message(message)
        except websockets.exceptions.ConnectionClosedOK as e:
            # This is a normal closure code (1000)
            logger.info(f"WebSocket connection closed normally: {e}")
            self.connected = False
        except websockets.exceptions.ConnectionClosedError as e:
            # This is an abnormal closure
            logger.warning(f"WebSocket connection closed unexpectedly: {e}")
            self.connected = False
        except asyncio.CancelledError:
            # Task was cancelled, just exit
            pass
        except Exception as e:
            logger.error(f"Error in message receiver: {e}")
            self.connected = False
    
    async def _process_message(self, message: str) -> None:
        """
        Process a message received from the WebSocket
        
        Args:
            message: Raw message received from WebSocket
        """
        try:
            data = json.loads(message)
            event_type = data.get("type", "unknown")
            
            # Debug log all received events
            logger.debug(f"Received event: {event_type}")
            
            # Handle error events
            if event_type == "error":
                self._handle_error(data)
                return
            
            # Update session info if this is a session event
            if event_type == "session.created":
                self.session_id = data.get("session", {}).get("id")
                self.session_state = data.get("session", {})
                logger.info(f"Session created with ID: {self.session_id}")
            elif event_type == "session.updated":
                self.session_state.update(data.get("session", {}))
                logger.debug("Session state updated")
            
            # Track voice activity detection events
            if event_type == "input_audio_buffer.speech_started":
                self.speech_active = True
                logger.debug("Speech started detected")
            elif event_type == "input_audio_buffer.speech_stopped":
                self.speech_active = False
                logger.debug("Speech stopped detected")
            
            # Track conversation items for state management
            if event_type == "conversation.item.created":
                item = data.get("item", {})
                item_id = item.get("id")
                if item_id:
                    self.conversation_items[item_id] = item
                    logger.debug(f"Added conversation item: {item_id}")
            
            # Track response status
            if event_type == "response.created":
                response = data.get("response", {})
                response_id = response.get("id")
                if response_id:
                    self.responses[response_id] = response
                    logger.debug(f"Response created: {response_id}")
            elif event_type == "response.done":
                response = data.get("response", {})
                response_id = response.get("id")
                if response_id and response_id in self.responses:
                    self.responses[response_id].update(response)
                    logger.debug(f"Response completed: {response_id}")
            
            # Call any registered callbacks for this event type
            if event_type in self.event_callbacks:
                for callback in self.event_callbacks[event_type]:
                    asyncio.create_task(callback(data))
            
            # Call the general event callback if registered
            if "all" in self.event_callbacks:
                for callback in self.event_callbacks["all"]:
                    asyncio.create_task(callback(data))
                    
        except json.JSONDecodeError:
            logger.error(f"Received invalid JSON: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _handle_error(self, error_data: Dict[str, Any]) -> None:
        """
        Handle error events from the API
        
        Args:
            error_data: Error event data
        """
        event_id = error_data.get("event_id")
        error_type = error_data.get("type", "unknown_error")
        error_message = error_data.get("message", "No error message provided")
        error_code = error_data.get("code", "unknown_code")
        
        # Format a detailed error message
        error_msg = f"API Error: {error_type}"
        if error_code:
            error_msg += f" ({error_code})"
        if error_message:
            error_msg += f": {error_message}"
            
        # Log the error with appropriate severity
        if error_code in ["rate_limit_exceeded", "token_limit_exceeded"]:
            logger.warning(error_msg)
        else:
            logger.error(error_msg)
        
        # If this error is related to an event we sent, log that relationship
        if event_id and event_id in self.pending_events:
            original_event = self.pending_events[event_id]
            event_type = original_event.get('type', 'unknown')
            logger.error(
                f"Error caused by event: {event_type} sent at {original_event['timestamp']}"
            )
        
        # Attempt to recover from the error
        asyncio.create_task(self._attempt_error_recovery(error_data))
            
        # Call error callbacks if registered
        for callback in self.event_callbacks.get("error", []):
            try:
                asyncio.create_task(callback(error_data))
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
                
    async def _attempt_error_recovery(self, error_data: Dict[str, Any]) -> None:
        """
        Attempt to recover from an API error
        
        Args:
            error_data: Error event data
        """
        try:
            # Try to recover using our recovery strategies
            recovery_attempted = await self._handle_api_error_recovery(error_data)
            
            if recovery_attempted:
                logger.info("Recovery strategy applied for API error")
            else:
                logger.warning("No recovery strategy available for this error")
                
        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}")
            
        # Clean up the pending event if it exists
        event_id = error_data.get("event_id")
        if event_id and event_id in self.pending_events:
            del self.pending_events[event_id]
    
    def register_event_callback(
        self, 
        event_type: str, 
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback function for a specific event type
        
        Args:
            event_type: The event type to listen for
            callback: Function to call when this event type is received
        """
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def get_session_state(self) -> Dict[str, Any]:
        """Get the current session state"""
        return self.session_state.copy()
    
    def is_speech_active(self) -> bool:
        """Check if speech is currently active (based on VAD)"""
        return self.speech_active
        
    async def update_session(self, session_data: Dict[str, Any]) -> str:
        """
        Update the session configuration
        
        Args:
            session_data: Session configuration parameters
            
        Returns:
            str: Event ID
        """
        # Create a clean copy of the data to avoid modifying the input
        validated_data = {}
        
        # Process and validate specific fields
        if "instructions" in session_data:
            validated_data["instructions"] = session_data["instructions"]
            # Print for debugging
            print(f"DEBUG - Sending instructions: {session_data['instructions'][:50]}...")
            
        if "voice" in session_data:
            # Validate voice parameter
            valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            voice = session_data["voice"]
            if voice in valid_voices:
                validated_data["voice"] = voice
            else:
                logger.warning(f"Invalid voice: {voice}. Valid options are: {', '.join(valid_voices)}")
        
        # Handle turn detection (VAD) settings
        if "turn_detection" in session_data:
            turn_detection = session_data.get("turn_detection")
            if turn_detection is None:
                self.vad_enabled = False
                validated_data["turn_detection"] = None
            else:
                self.vad_enabled = True
                # Copy turn_detection settings
                validated_data["turn_detection"] = turn_detection
        
        # Handle tools configuration
        if "tools" in session_data:
            validated_data["tools"] = session_data["tools"]
            
        if "tool_choice" in session_data:
            valid_choices = ["auto", "none"]
            tool_choice = session_data["tool_choice"]
            if isinstance(tool_choice, str) and tool_choice in valid_choices:
                validated_data["tool_choice"] = tool_choice
            elif isinstance(tool_choice, dict):
                validated_data["tool_choice"] = tool_choice
            else:
                logger.warning(f"Invalid tool_choice: {tool_choice}. Using default.")
        
        # Copy any other fields that don't need special validation
        for key, value in session_data.items():
            if key not in validated_data and key not in ["event_id", "type"]:
                validated_data[key] = value
        
        # Print the full event for debugging
        event_data = {"session": validated_data}
        print(f"DEBUG - Full session.update event: {json.dumps(event_data)[:100]}...")
        
        logger.debug(f"Updating session with: {json.dumps(validated_data)}")
        return await self._send_event("session.update", {"session": validated_data})
    
    async def create_conversation_item(self, item_data: Dict[str, Any]) -> str:
        """
        Create a new conversation item
        
        Args:
            item_data: Conversation item data
            
        Returns:
            str: Event ID
        """
        return await self._send_event("conversation.item.create", {"item": item_data})
    
    async def append_audio(self, audio_bytes: bytes) -> str:
        """
        Append audio to the input audio buffer
        
        Args:
            audio_bytes: Raw audio bytes to append
            
        Returns:
            str: Event ID
            
        Raises:
            ValueError: If audio chunk exceeds the maximum size
        """
        # Check chunk size
        if len(audio_bytes) > self.max_audio_chunk_size:
            raise ValueError(f"Audio chunk size ({len(audio_bytes)} bytes) exceeds maximum allowed size (15MB)")
        
        base64_audio = base64.b64encode(audio_bytes).decode('ascii')
        return await self._send_event("input_audio_buffer.append", {"audio": base64_audio})
    
    async def append_audio_chunks(self, audio_chunks: List[bytes]) -> List[str]:
        """
        Append multiple audio chunks to the input audio buffer
        
        Args:
            audio_chunks: List of audio byte chunks to append
            
        Returns:
            List[str]: List of event IDs
        """
        event_ids = []
        for chunk in audio_chunks:
            event_id = await self.append_audio(chunk)
            event_ids.append(event_id)
        return event_ids
    
    async def commit_audio_buffer(self) -> str:
        """
        Commit the input audio buffer to create a user message
        
        Returns:
            str: Event ID
        """
        return await self._send_event("input_audio_buffer.commit", {})
    
    async def clear_audio_buffer(self) -> str:
        """
        Clear the input audio buffer
        
        Returns:
            str: Event ID
        """
        return await self._send_event("input_audio_buffer.clear", {})
    
    async def create_response(
        self, 
        modalities: List[str] = None, 
        instructions: str = None,
        input_audio_format: Dict[str, Any] = None,
        output_audio_format: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        conversation: str = None,
        tools: List[Dict[str, Any]] = None,
        input_items: List[Dict[str, Any]] = None
    ) -> str:
        """
        Request the model to generate a response
        
        Args:
            modalities: List of response modalities ("text", "audio")
            instructions: Special instructions for this response
            input_audio_format: Format specification for input audio
            output_audio_format: Format specification for output audio
            metadata: Custom metadata for this response
            conversation: Conversation context (or "none" for out-of-band)
            tools: Tools available for this response
            input_items: Custom input items for this response
            
        Returns:
            str: Event ID
        """
        # Validate modalities
        valid_modalities = ["text", "audio"]
        if modalities:
            for modality in modalities:
                if modality not in valid_modalities:
                    logger.warning(f"Invalid modality: {modality}. Using default modalities.")
                    modalities = None
                    break
        
        response_data = {}
        
        # Only add non-None fields to prevent API errors
        if modalities:
            response_data["modalities"] = modalities
        if instructions:
            response_data["instructions"] = instructions
            # Print debug info
            print(f"DEBUG - Including instructions in response.create: {instructions[:50]}...")
        if input_audio_format:
            response_data["input_audio_format"] = input_audio_format
        if output_audio_format:
            response_data["output_audio_format"] = output_audio_format
        if metadata:
            response_data["metadata"] = metadata
        if conversation:
            response_data["conversation"] = conversation
        if tools:
            response_data["tools"] = tools
        if input_items:
            response_data["input"] = input_items
            
        # Print full response.create data for debugging
        print(f"DEBUG - FULL response.create: {json.dumps(response_data)[:200]}...")
        logger.debug(f"Creating response with data: {json.dumps(response_data)}")
        return await self._send_event("response.create", {"response": response_data})
    
    async def submit_function_call_output(self, call_id: str, output: Dict[str, Any]) -> str:
        """
        Submit the result of a function call back to the conversation
        
        Args:
            call_id: The ID of the function call
            output: The result of the function call
            
        Returns:
            str: Event ID
        """
        # Convert the output to a JSON string
        output_str = json.dumps(output)
        
        # Create a function call output item
        item_data = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_str
        }
        
        return await self.create_conversation_item(item_data)
    
    async def _handle_api_error_recovery(self, error_data: Dict[str, Any]) -> bool:
        """
        Attempt to recover from specific API errors with appropriate strategies
        
        Args:
            error_data: Error event data from the API
            
        Returns:
            bool: True if recovery was attempted, False otherwise
        """
        error_code = error_data.get("code", "unknown_code")
        
        # Handle rate limit errors with exponential backoff
        if error_code == "rate_limit_exceeded":
            # Get retry-after if available, otherwise use default backoff
            retry_after = error_data.get("retry_after", 1.0)
            logger.warning(f"Rate limit exceeded. Waiting {retry_after}s before retrying...")
            await asyncio.sleep(float(retry_after))
            return True
            
        # Handle token limit errors by clearing conversation context
        elif error_code == "token_limit_exceeded":
            logger.warning("Token limit exceeded. Consider truncating the conversation history.")
            return True
            
        # Handle invalid request errors
        elif error_code == "invalid_request_error":
            # Check if this is related to a specific event we can retry
            event_id = error_data.get("event_id")
            if event_id and event_id in self.pending_events:
                original_event = self.pending_events[event_id]
                event_type = original_event.get('type')
                
                # For certain event types, we might want to retry with modified parameters
                if event_type == "response.create":
                    logger.warning("Invalid request when creating response. Check response parameters.")
                    
                # Clean up the tracked event
                del self.pending_events[event_id]
            return True
        
        # Handle unknown errors
        elif error_code == "unknown_code" or not error_code:
            # Log the entire error data for debugging
            logger.warning(f"Received unknown error from API. Full error data: {json.dumps(error_data, indent=2)}")
            
            # Check if this is related to a specific event we can identify
            event_id = error_data.get("event_id")
            if event_id and event_id in self.pending_events:
                original_event = self.pending_events[event_id]
                event_type = original_event.get('type')
                
                logger.warning(f"Unknown error related to event type: {event_type}")
                logger.warning(f"Original event data: {json.dumps(original_event, indent=2)}")
                
                # For response.create events, retry with simplified parameters
                if event_type == "response.create":
                    logger.warning("Will attempt to continue with conversation despite unknown error")
                    return True
            
            # Continue running despite the error
            logger.warning("Continuing with operation despite unknown error")
            return True
            
        # No specific recovery strategy for this error
        return False
    
    async def truncate_conversation(self, keep_last_n: int = 10) -> str:
        """
        Truncate the conversation history, keeping only the most recent items
        This is useful for managing token limits in long conversations
        
        Args:
            keep_last_n: Number of most recent conversation items to keep
            
        Returns:
            str: Event ID
        """
        # Get the IDs of all conversation items
        item_ids = list(self.conversation_items.keys())
        
        # If we have fewer items than the limit, no need to truncate
        if len(item_ids) <= keep_last_n:
            logger.info(f"Conversation has only {len(item_ids)} items, no truncation needed")
            return ""
        
        # Calculate how many items to remove
        items_to_remove = len(item_ids) - keep_last_n
        
        # Get the IDs of items to keep (the most recent ones)
        items_to_keep = item_ids[-keep_last_n:]
        
        logger.info(f"Truncating conversation: removing {items_to_remove} items, keeping {keep_last_n} most recent")
        
        # Send the truncate event
        return await self._send_event("conversation.item.truncate", {
            "before_id": items_to_keep[0]  # Keep everything after this ID
        })