import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union, Set

from src.utils.logging_utils import setup_logger
from src.realtime_client import RealtimeClient
from src.config import settings

# Set up logger
logger = setup_logger("conversation")

class ConversationManager:
    """
    Manages conversation state and interactions with the Realtime API
    Provides a higher-level interface for working with conversations
    """
    
    def __init__(self, realtime_client: RealtimeClient):
        """
        Initialize a new conversation manager
        
        Args:
            realtime_client: Connected Realtime API client
        """
        self.client = realtime_client
        self.session_id = None
        self.active_responses = {}  # Response ID -> Response Data
        self.active_function_calls = {}  # Call ID -> Call Data
        self.event_handlers = {
            "on_text_response": [],
            "on_transcript": [],
            "on_function_call": [],
            "on_response_complete": [],
            "on_error": []
        }  # Event type -> List of handlers
        self.pending_responses = set()  # Set to track pending responses
        self.tasks = []  # Track async tasks for cleanup
        
        # Track conversation content
        self.conversation_items = {}  # Item ID -> Item Data
        self.conversation_history = []  # List of conversation items in order
        
        # For tracking transcript data
        self.current_text_response = ""
        self.current_transcript = {}  # Chunk Index -> Text
        
        # Register internal event handlers
        self._register_callbacks()
    
    def _register_callbacks(self) -> None:
        """Register event callbacks with the Realtime client"""
        # Text response events
        self.client.register_event_callback("response.text.delta", self._handle_text_delta)
        self.client.register_event_callback("response.text.done", self._handle_text_done)
        
        # Audio transcript events
        self.client.register_event_callback(
            "response.audio_transcript.delta", 
            self._handle_audio_transcript_delta
        )
        self.client.register_event_callback(
            "response.audio_transcript.done", 
            self._handle_audio_transcript_done
        )
        
        # Function call events
        self.client.register_event_callback(
            "response.function_call_arguments.delta", 
            self._handle_function_call_delta
        )
        
        # Response lifecycle events
        self.client.register_event_callback("response.created", self._handle_response_created)
        self.client.register_event_callback("response.done", self._handle_response_done)
        
        # Session events
        self.client.register_event_callback("session.created", self._handle_session_created)
        
        # Conversation events
        self.client.register_event_callback("conversation.item.created", self._handle_conversation_item_created)
        
        # Error events
        self.client.register_event_callback("error", self._handle_error)
    
    async def _handle_session_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle session.created event
        
        Args:
            event_data: Event data
        """
        try:
            session = event_data.get("session", {})
            self.session_id = session.get("id")
            logger.info(f"Conversation session created: {self.session_id}")
        except Exception as e:
            logger.error(f"Error handling session created event: {e}")
            await self._notify_error("Error handling session created event", e)
    
    async def _handle_conversation_item_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle conversation.item.created event
        
        Args:
            event_data: Event data
        """
        try:
            item = event_data.get("item", {})
            item_id = item.get("id")
            
            if item_id:
                self.conversation_items[item_id] = item
                
                # Add to conversation history
                history_entry = {
                    "id": item_id,
                    "type": item.get("type"),
                    "role": item.get("role"),
                    "created_at": time.time(),
                    "item": item
                }
                self.conversation_history.append(history_entry)
                
                logger.debug(f"Conversation item created: {item_id} (type: {item.get('type')})")
        except Exception as e:
            logger.error(f"Error handling conversation item created event: {e}")
            await self._notify_error("Error handling conversation item created", e)
    
    async def _handle_response_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response.created event
        
        Args:
            event_data: Event data
        """
        try:
            response = event_data.get("response", {})
            response_id = response.get("id")
            
            if response_id:
                self.active_responses[response_id] = {
                    "id": response_id,
                    "text": "",
                    "transcripts": {},
                    "function_calls": [],
                    "status": "created",
                    "metadata": response.get("metadata", {}),
                    "created_at": time.time()
                }
                logger.debug(f"Response created: {response_id}")
        except Exception as e:
            logger.error(f"Error handling response created event: {e}")
            await self._notify_error("Error handling response created", e)
    
    
    # Improve the _handle_text_delta method:
    async def _handle_text_delta(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response.text.delta event
        
        Args:
            event_data: Event data
        """
        try:
            response_id = event_data.get("response_id")
            delta = event_data.get("delta", "")
            
            if response_id and delta:
                # Ensure response exists in our tracking
                if response_id not in self.active_responses:
                    logger.warning(f"Received delta for unknown response: {response_id}")
                    self.active_responses[response_id] = {
                        "id": response_id,
                        "text": "",
                        "status": "in_progress",
                        "created_at": time.time()
                    }
                    
                # Update the stored response text
                if "text" not in self.active_responses[response_id]:
                    self.active_responses[response_id]["text"] = ""
                
                self.active_responses[response_id]["text"] += delta
                full_text = self.active_responses[response_id]["text"]
                
                # Log for debugging (only first few characters to avoid spam)
                logger.debug(f"Text delta: '{delta[:20]}{'...' if len(delta) > 20 else ''}'")
                
                # Call any registered text response handlers
                for handler in self.event_handlers["on_text_response"]:
                    try:
                        await handler(delta, full_text, response_id)
                    except Exception as e:
                        logger.error(f"Error in text response handler: {e}")
        except Exception as e:
            logger.error(f"Error handling text delta event: {e}")
            await self._notify_error("Error handling text delta", e)


    async def _handle_text_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response.text.done event
        
        Args:
            event_data: Event data
        """
        try:
            response_id = event_data.get("response", {}).get("id")
            text = event_data.get("text", "")
            
            if response_id and response_id in self.active_responses:
                # Update with the final text
                self.active_responses[response_id]["text"] = text
                logger.debug(f"Text response complete for {response_id}")
        except Exception as e:
            logger.error(f"Error handling text done event: {e}")
            await self._notify_error("Error handling text done", e)
    
    async def _handle_audio_transcript_delta(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response.audio_transcript.delta event
        
        Args:
            event_data: Event data
        """
        try:
            response_id = event_data.get("response", {}).get("id")
            delta = event_data.get("delta", "")
            part_id = event_data.get("part_id", "unknown")
            
            if response_id and response_id in self.active_responses and delta:
                # Initialize transcript for this part if needed
                if part_id not in self.active_responses[response_id]["transcripts"]:
                    self.active_responses[response_id]["transcripts"][part_id] = ""
                
                # Update the transcript
                self.active_responses[response_id]["transcripts"][part_id] += delta
                full_transcript = self.active_responses[response_id]["transcripts"][part_id]
                
                # Call any registered transcript handlers
                for handler in self.event_handlers["on_transcript"]:
                    try:
                        await handler(delta, full_transcript, part_id, response_id)
                    except Exception as e:
                        logger.error(f"Error in transcript handler: {e}")
        except Exception as e:
            logger.error(f"Error handling audio transcript delta event: {e}")
            await self._notify_error("Error handling audio transcript delta", e)
    
    async def _handle_audio_transcript_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response.audio_transcript.done event
        
        Args:
            event_data: Event data
        """
        try:
            response_id = event_data.get("response", {}).get("id")
            transcript = event_data.get("transcript", "")
            part_id = event_data.get("part_id", "unknown")
            
            if response_id and response_id in self.active_responses:
                # Update with the final transcript
                self.active_responses[response_id]["transcripts"][part_id] = transcript
                logger.debug(f"Audio transcript complete for part {part_id} in response {response_id}")
        except Exception as e:
            logger.error(f"Error handling audio transcript done event: {e}")
            await self._notify_error("Error handling audio transcript done", e)
    
    async def _handle_function_call_delta(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response.function_call_arguments.delta event
        
        Args:
            event_data: Event data
        """
        try:
            response_id = event_data.get("response", {}).get("id")
            call_id = event_data.get("call_id")
            delta = event_data.get("delta", "")
            function_name = event_data.get("name", "unknown")
            
            if not (response_id and call_id and delta):
                return
                
            # Initialize function call if needed
            if call_id not in self.active_function_calls:
                self.active_function_calls[call_id] = {
                    "id": call_id,
                    "response_id": response_id,
                    "name": function_name,
                    "arguments": "",
                    "arguments_json": None,
                    "complete_json": False,
                    "status": "in_progress"
                }
                
                # Add to response's function calls
                if response_id in self.active_responses:
                    if "function_calls" not in self.active_responses[response_id]:
                        self.active_responses[response_id]["function_calls"] = []
                    self.active_responses[response_id]["function_calls"].append(call_id)
            
            # Update arguments
            self.active_function_calls[call_id]["arguments"] += delta
            
            # Only try to parse JSON if we haven't already got valid JSON
            # and we have at least one curly brace
            args = self.active_function_calls[call_id]["arguments"]
            if not self.active_function_calls[call_id]["complete_json"] and "{" in args:
                try:
                    args_json = json.loads(args)
                    self.active_function_calls[call_id]["arguments_json"] = args_json
                    self.active_function_calls[call_id]["complete_json"] = True
                    
                    # Call any registered function call handlers
                    for handler in self.event_handlers["on_function_call"]:
                        try:
                            await handler(
                                function_name,
                                args_json,
                                call_id,
                                response_id
                            )
                        except Exception as e:
                            logger.error(f"Error in function call handler: {e}")
                except json.JSONDecodeError:
                    # Incomplete JSON, still accumulating
                    pass
        except Exception as e:
            logger.error(f"Error handling function call delta event: {e}")
            await self._notify_error("Error handling function call delta", e)
    
    async def _handle_response_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response.done event
        
        Args:
            event_data: Event data
        """
        try:
            response = event_data.get("response", {})
            response_id = response.get("id")
            
            if response_id and response_id in self.active_responses:
                # Update response status
                self.active_responses[response_id]["status"] = "completed"
                
                # Store output items
                self.active_responses[response_id]["output"] = response.get("output", [])
                
                # Store usage information
                self.active_responses[response_id]["usage"] = response.get("usage", {})
                
                # Save response data before removing it
                response_data = self.active_responses[response_id].copy()
                
                # Remove from pending responses if present
                if response_id in self.pending_responses:
                    self.pending_responses.remove(response_id)
                
                # Remove from active responses
                if response_id in self.active_responses:
                    del self.active_responses[response_id]
                
                # Handle any function calls that completed
                for output_item in response.get("output", []):
                    if output_item.get("type") == "function_call":
                        call_id = output_item.get("call_id")
                        if call_id and call_id in self.active_function_calls:
                            self.active_function_calls[call_id]["status"] = "completed"
                            self.active_function_calls[call_id]["arguments"] = output_item.get("arguments", "")
                            
                            # Try to parse the arguments as JSON
                            try:
                                args_json = json.loads(self.active_function_calls[call_id]["arguments"])
                                self.active_function_calls[call_id]["arguments_json"] = args_json
                                self.active_function_calls[call_id]["complete_json"] = True
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse function call arguments as JSON: {self.active_function_calls[call_id]['arguments']}")
                
                # Add final response to history
                history_entry = {
                    "id": response_id,
                    "type": "response",
                    "text": response_data.get("text", ""),
                    "created_at": response_data.get("created_at"),
                    "completed_at": time.time(),
                    "response": response_data
                }
                self.conversation_history.append(history_entry)
                
                # Call any registered response complete handlers
                for handler in self.event_handlers["on_response_complete"]:
                    try:
                        await handler(response_data)
                    except Exception as e:
                        logger.error(f"Error in response complete handler: {e}")
                        
                logger.info(f"Response completed: {response_id}")
            elif response_id:
                # Response ID exists but not in active_responses
                logger.warning(f"Received done event for unknown response: {response_id}")
            else:
                # No response ID in the event
                logger.warning("Received done event with no response ID")
        except Exception as e:
            logger.error(f"Error handling response done event: {e}")
            await self._notify_error("Error handling response done", e)
    
    async def _handle_error(self, event_data: Dict[str, Any]) -> None:
        """
        Handle error events
        
        Args:
            event_data: Error event data
        """
        try:
            error_type = event_data.get("type", "unknown_error")
            error_message = event_data.get("message", "Unknown error")
            error_code = event_data.get("code", "unknown_code")
            
            logger.error(f"Realtime API error: {error_type} ({error_code}): {error_message}")
            
            # Notify any error handlers
            await self._notify_error(f"{error_type}: {error_message}", None, event_data)
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
    
    async def _notify_error(
        self, 
        message: str, 
        exception: Optional[Exception] = None,
        error_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Notify error handlers
        
        Args:
            message: Error message
            exception: Exception object (if any)
            error_data: Raw error data (if any)
        """
        error_info = {
            "message": message,
            "exception": exception,
            "error_data": error_data,
            "timestamp": time.time()
        }
        
        for handler in self.event_handlers["on_error"]:
            try:
                await handler(error_info)
            except Exception as e:
                logger.error(f"Error in error notification handler: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register a handler function for conversation events
        
        Args:
            event_type: Type of event to handle (on_transcript, on_text_response, 
                        on_function_call, on_response_complete, on_error)
            handler: Function to call when this event occurs
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def unregister_event_handler(self, event_type: str, handler: Callable) -> bool:
        """
        Unregister a handler function
        
        Args:
            event_type: Type of event
            handler: Handler function to remove
            
        Returns:
            bool: True if handler was removed, False otherwise
        """
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            return True
        return False
    
    async def configure_session(
        self,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        voice: Optional[str] = None,
        vad_enabled: bool = True,
        auto_response: bool = True
    ) -> bool:
        """
        Configure the conversation session
        
        Args:
            instructions: System instructions for the model
            tools: List of tools (functions) the model can call
            voice: Voice ID for audio responses
            vad_enabled: Whether to enable voice activity detection
            auto_response: Whether to automatically generate responses when user stops speaking
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            session_config = {}
            
            if instructions is not None:
                session_config["instructions"] = instructions
            
            if tools is not None:
                session_config["tools"] = tools
            
            if voice is not None:
                session_config["voice"] = voice
            
            # Configure turn detection (VAD)
            if not vad_enabled:
                session_config["turn_detection"] = None
            else:
                # Custom VAD settings with better parameters for interruption
                session_config["turn_detection"] = {
                    "type": "server_vad",
                    "threshold": 0.85,  # Higher number means less sensitive (0-1 range)
                    "prefix_padding_ms": 500,  # More padding before speech
                    "silence_duration_ms": 1200,  # Even longer wait for silence to ensure user is done speaking
                    "create_response": auto_response,
                    "interrupt_response": True  # Explicitly enable interruption
                }
            
            # Update the session
            if session_config:
                await self.client.update_session(session_config)
                logger.info("Session configuration updated")
                return True
            
            return True
        except Exception as e:
            logger.error(f"Error configuring session: {e}")
            await self._notify_error("Error configuring session", e)
            return False
    
    async def send_text_message(self, text: str) -> Optional[str]:
        """
        Send a text message to the conversation
        
        Args:
            text: Message text
            
        Returns:
            Optional[str]: Event ID if successful, None otherwise
        """
        try:
            if not text or not text.strip():
                logger.warning("Attempted to send empty text message")
                return None
                
            item_data = {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
            
            event_id = await self.client.create_conversation_item(item_data)
            logger.info(f"Sent text message: {text[:50]}{'...' if len(text) > 50 else ''}")
            return event_id
        except Exception as e:
            logger.error(f"Error sending text message: {e}")
            await self._notify_error("Error sending text message", e)
            return None
    
    async def request_response(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Request the model to generate a response
        
        Args:
            modalities: List of response modalities ("text", "audio")
            instructions: Optional instructions for this specific response
            metadata: Optional metadata for this response
            
        Returns:
            str: Response ID
        """
        if not self.client:
            logger.error("Cannot request response: Client not available")
            return ""
            
        try:
            # Import instructions
            from src.system_instructions import APPOINTMENT_SCHEDULER
            
            response_data = {}
            
            if modalities:
                response_data["modalities"] = modalities
            else:
                # Default to text and audio if not specified
                response_data["modalities"] = ["text", "audio"]
                
            # ALWAYS include instructions - either custom or our default
            if instructions:
                response_data["instructions"] = instructions
            else:
                # Make a copy of the instructions string to avoid any reference issues
                appointment_instructions = str(APPOINTMENT_SCHEDULER)
                response_data["instructions"] = appointment_instructions
                print(f"DEBUG - Using default instructions: {appointment_instructions}")
                
            if metadata:
                response_data["metadata"] = metadata
                
            # Print the formatted response data
            print(f"DEBUG - RESPONSE CREATE DATA: {json.dumps(response_data)[:200]}...")
                
            # Request a response from the model
            response_id = await self.client.create_response(**response_data)
            
            if response_id:
                self.pending_responses.add(response_id)
                logger.info(f"Requested model response (modalities: {response_data['modalities']})")
            else:
                logger.error("Failed to request model response")
                
            return response_id
                
        except Exception as e:
            logger.error(f"Error requesting response: {e}")
            return ""
    
    async def submit_function_result(self, call_id: str, result: Dict[str, Any]) -> Optional[str]:
        """
        Submit the result of a function call back to the conversation
        
        Args:
            call_id: Function call ID
            result: Result data
            
        Returns:
            Optional[str]: Event ID if successful, None otherwise
        """
        try:
            if call_id not in self.active_function_calls:
                logger.warning(f"Function call ID not found: {call_id}")
                return None
                
            event_id = await self.client.submit_function_call_output(call_id, result)
            logger.info(f"Submitted result for function call: {call_id}")
            return event_id
        except Exception as e:
            logger.error(f"Error submitting function result: {e}")
            await self._notify_error("Error submitting function result", e)
            return None
    
    async def get_last_response_text(self) -> str:
        """
        Get the text of the last model response
        
        Returns:
            str: Response text
        """
        try:
            # Find the most recent completed response
            latest_response_id = None
            latest_timestamp = 0
            
            for response_id, response in self.active_responses.items():
                if response.get("status") == "completed" and response.get("text"):
                    timestamp = response.get("created_at", 0)
                    if timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_response_id = response_id
            
            if latest_response_id:
                return self.active_responses[latest_response_id].get("text", "")
                
            return ""
        except Exception as e:
            logger.error(f"Error getting last response text: {e}")
            return ""
    
    async def get_last_transcript(self) -> Dict[str, str]:
        """
        Get the transcript of the last model response
        
        Returns:
            Dict[str, str]: Transcript by part ID
        """
        try:
            # Find the most recent response with transcripts
            latest_response_id = None
            latest_timestamp = 0
            
            for response_id, response in self.active_responses.items():
                if response.get("transcripts") and response.get("created_at", 0) > latest_timestamp:
                    latest_timestamp = response.get("created_at", 0)
                    latest_response_id = response_id
            
            if latest_response_id:
                return self.active_responses[latest_response_id].get("transcripts", {})
                
            return {}
        except Exception as e:
            logger.error(f"Error getting last transcript: {e}")
            return {}
    
    async def get_conversation_history(
        self, 
        limit: Optional[int] = None, 
        include_system: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            limit: Maximum number of items to return (newest first)
            include_system: Whether to include system messages
            
        Returns:
            List[Dict[str, Any]]: Conversation history
        """
        try:
            if not include_system:
                filtered_history = [
                    item for item in self.conversation_history 
                    if not (item.get("type") == "message" and item.get("role") == "system")
                ]
            else:
                filtered_history = self.conversation_history.copy()
                
            # Sort by timestamp, newest first
            sorted_history = sorted(
                filtered_history, 
                key=lambda x: x.get("created_at", 0), 
                reverse=True
            )
            
            # Apply limit if specified
            if limit is not None and limit > 0:
                return sorted_history[:limit]
                
            return sorted_history
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def get_response_by_id(self, response_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a response by ID
        
        Args:
            response_id: Response ID
            
        Returns:
            Optional[Dict[str, Any]]: Response data or None if not found
        """
        return self.active_responses.get(response_id)
    
    async def get_function_call_by_id(self, call_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a function call by ID
        
        Args:
            call_id: Function call ID
            
        Returns:
            Optional[Dict[str, Any]]: Function call data or None if not found
        """
        return self.active_function_calls.get(call_id)
    
        
    # In ConversationManager, improve the wait_for_all_responses method:
    async def wait_for_all_responses(self, timeout: float = 30.0) -> bool:
        """
        Wait for all active responses to complete
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if all responses completed, False if timed out
        """
        try:
            if not self.active_responses:
                return True
                
            start_time = time.time()
            
            # Log what we're waiting for
            logger.info(f"Waiting for {len(self.active_responses)} active responses to complete")
            for resp_id in self.active_responses:
                logger.info(f"  - Response ID: {resp_id}")
            
            while self.active_responses and (time.time() - start_time) < timeout:
                # Log remaining every 5 seconds
                if int(time.time() - start_time) % 5 == 0:
                    logger.info(f"Still waiting for {len(self.active_responses)} responses. ({int(timeout - (time.time() - start_time))}s remaining)")
                
                await asyncio.sleep(0.1)
                    
            success = len(self.active_responses) == 0
            if success:
                logger.info("All responses completed successfully")
            else:
                logger.warning(f"Timed out waiting for responses: {self.active_responses}")
                
            return success
        except Exception as e:
            logger.error(f"Error waiting for responses: {e}")
            return False
    
    async def interrupt_response(self, response_id: str = None) -> bool:
        """
        Interrupt an in-progress response
        
        Args:
            response_id: Specific response ID to interrupt, or None for all active responses
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # TODO: Implement when OpenAI adds this capability to the API
            logger.warning("Response interruption not yet implemented in the OpenAI Realtime API")
            return False
        except Exception as e:
            logger.error(f"Error interrupting response: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Cancel any running tasks
            if hasattr(self, 'tasks') and self.tasks:
                for task in self.tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            
            # Clear all event handlers to prevent memory leaks
            for event_type in self.event_handlers:
                self.event_handlers[event_type] = []
                
            logger.info("Conversation manager resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")