"""
Domain logic for managing conversations.
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

from ..data.models import Conversation, Message
from ..services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation state and interaction with the OpenAI service.
    
    This class is responsible for coordinating the conversation flow,
    handling events from the AI service, and managing message history.
    """
    
    def __init__(self, openai_service: OpenAIService):
        """Initialize the conversation manager.
        
        Args:
            openai_service: Service for OpenAI API communication
        """
        self.service = openai_service
        self.conversation: Optional[Conversation] = None
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize tracking collections
        self.active_responses: Set[str] = set()
        self.active_function_calls: Dict[str, Dict[str, Any]] = {}
        self.pending_responses: Set[str] = set()
        self.last_response_text: Dict[str, str] = {}
        self.last_transcript: Dict[str, Dict[str, str]] = {}
        
        # Register event handlers
        self._register_callbacks()
    
    def _register_callbacks(self) -> None:
        """Register handlers for OpenAI API events."""
        # Session events
        self.service.register_event_handler("session.created", self._handle_session_created)
        
        # Message events
        self.service.register_event_handler("conversation.message.created", 
                                           self._handle_conversation_item_created)
        
        # Response events
        self.service.register_event_handler("response.created", self._handle_response_created)
        self.service.register_event_handler("response.text.delta", self._handle_text_delta)
        self.service.register_event_handler("response.text.done", self._handle_text_done)
        self.service.register_event_handler("response.audio.transcript.delta", 
                                           self._handle_audio_transcript_delta)
        self.service.register_event_handler("response.audio.transcript.done", 
                                           self._handle_audio_transcript_done)
        self.service.register_event_handler("response.function_call.delta", 
                                           self._handle_function_call_delta)
        self.service.register_event_handler("response.done", self._handle_response_done)
        
        # Error events
        self.service.register_event_handler("error", self._handle_error)
    
    async def _handle_session_created(self, event_data: Dict[str, Any]) -> None:
        """Handle session creation event.
        
        Args:
            event_data: Session event data
        """
        session_id = event_data.get("session_id", "")
        
        # Create a new conversation record
        self.conversation = Conversation(session_id=session_id)
        logger.info(f"Conversation session created: {session_id}")
    
    async def _handle_conversation_item_created(self, event_data: Dict[str, Any]) -> None:
        """Handle new conversation item event.
        
        Args:
            event_data: Conversation item event data
        """
        item_data = event_data.get("conversation_item", {})
        item_id = item_data.get("id", "")
        
        # Update conversation history
        self.conversation_history.append(item_data)
        logger.debug(f"Conversation item created: {item_id}")
    
    async def _handle_response_created(self, event_data: Dict[str, Any]) -> None:
        """Handle response creation event.
        
        Args:
            event_data: Response event data
        """
        response_id = event_data.get("response_id", "")
        self.active_responses.add(response_id)
        self.pending_responses.add(response_id)
        logger.debug(f"Response created: {response_id}")
        
        # Initialize empty text for this response
        self.last_response_text[response_id] = ""
    
    async def _handle_text_delta(self, event_data: Dict[str, Any]) -> None:
        """Handle text delta event.
        
        Args:
            event_data: Text delta event data
        """
        response_id = event_data.get("response_id", "")
        delta = event_data.get("delta", "")
        
        # Update the accumulated text for this response
        if response_id in self.last_response_text:
            self.last_response_text[response_id] += delta
        else:
            self.last_response_text[response_id] = delta
        
        # Forward to event handlers
        event_type = "text_delta"
        for handler in self._get_event_handlers(event_type):
            await handler(
                delta=delta,
                full_text=self.last_response_text[response_id],
                response_id=response_id
            )
    
    async def _handle_text_done(self, event_data: Dict[str, Any]) -> None:
        """Handle text completion event.
        
        Args:
            event_data: Text completion event data
        """
        response_id = event_data.get("response_id", "")
        logger.debug(f"Text response completed for: {response_id}")
    
    async def _handle_audio_transcript_delta(self, event_data: Dict[str, Any]) -> None:
        """Handle audio transcript delta event.
        
        Args:
            event_data: Audio transcript delta event data
        """
        response_id = event_data.get("response_id", "")
        part_id = event_data.get("part_id", "")
        delta = event_data.get("delta", "")
        
        # Initialize transcript for this response if needed
        if response_id not in self.last_transcript:
            self.last_transcript[response_id] = {}
            
        # Initialize or update this part's transcript
        if part_id in self.last_transcript[response_id]:
            self.last_transcript[response_id][part_id] += delta
        else:
            self.last_transcript[response_id][part_id] = delta
        
        # Forward to event handlers
        event_type = "transcript_delta"
        full_transcript = self.last_transcript[response_id][part_id]
        
        for handler in self._get_event_handlers(event_type):
            await handler(
                delta=delta,
                full_transcript=full_transcript,
                part_id=part_id,
                response_id=response_id
            )
    
    async def _handle_audio_transcript_done(self, event_data: Dict[str, Any]) -> None:
        """Handle audio transcript completion event.
        
        Args:
            event_data: Audio transcript completion event data
        """
        response_id = event_data.get("response_id", "")
        part_id = event_data.get("part_id", "")
        logger.debug(f"Audio transcript completed for part: {part_id}, response: {response_id}")
    
    async def _handle_function_call_delta(self, event_data: Dict[str, Any]) -> None:
        """Handle function call delta event.
        
        Args:
            event_data: Function call delta event data
        """
        response_id = event_data.get("response_id", "")
        call_id = event_data.get("call_id", "")
        delta = event_data.get("delta", {})
        
        # Initialize or update function call data
        if call_id not in self.active_function_calls:
            self.active_function_calls[call_id] = {
                "response_id": response_id,
                "function_name": "",
                "arguments": "",
                "completed": False
            }
        
        # Update with delta data
        if "function_name" in delta:
            self.active_function_calls[call_id]["function_name"] += delta["function_name"]
            
        if "arguments" in delta:
            self.active_function_calls[call_id]["arguments"] += delta["arguments"]
        
        # If we've received an "end" marker, parse the arguments from JSON to dict
        if delta.get("type") == "function_call_end":
            self.active_function_calls[call_id]["completed"] = True
            
            # Parse arguments if they're in JSON format
            args_str = self.active_function_calls[call_id]["arguments"]
            try:
                import json
                args_dict = json.loads(args_str)
                self.active_function_calls[call_id]["arguments_dict"] = args_dict
            except Exception as e:
                logger.error(f"Error parsing function arguments: {e}")
                self.active_function_calls[call_id]["arguments_dict"] = {}
                
            # Notify function call handlers
            function_name = self.active_function_calls[call_id]["function_name"]
            arguments_dict = self.active_function_calls[call_id].get("arguments_dict", {})
            
            event_type = "function_call"
            for handler in self._get_event_handlers(event_type):
                await handler(
                    function_name=function_name,
                    arguments=arguments_dict,
                    call_id=call_id,
                    response_id=response_id,
                    conversation_manager=self
                )
    
    async def _handle_response_done(self, event_data: Dict[str, Any]) -> None:
        """Handle response completion event.
        
        Args:
            event_data: Response completion event data
        """
        response_id = event_data.get("response_id", "")
        
        # Update response tracking
        if response_id in self.pending_responses:
            self.pending_responses.remove(response_id)
        
        if response_id in self.active_responses:
            self.active_responses.remove(response_id)
            
        # Calculate usage statistics
        usage = event_data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        
        logger.info(f"Response completed: {response_id}")
        
        # Forward to event handlers
        event_type = "response_complete"
        for handler in self._get_event_handlers(event_type):
            await handler(response=event_data)
    
    async def _handle_error(self, event_data: Dict[str, Any]) -> None:
        """Handle error event.
        
        Args:
            event_data: Error event data
        """
        error_type = event_data.get("type", "unknown")
        error_message = event_data.get("message", "Unknown error")
        logger.error(f"API error: {error_type} - {error_message}")
        
        # Forward to event handlers
        event_type = "error"
        for handler in self._get_event_handlers(event_type):
            await handler(error_info=event_data)
    
    async def _notify_error(
        self, 
        message: str, 
        exception: Optional[Exception] = None,
        error_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify handlers of an error.
        
        Args:
            message: Error message
            exception: Exception that occurred
            error_data: Additional error information
        """
        error_info = {
            "type": "client_error",
            "message": message
        }
        
        if exception:
            error_info["exception"] = str(exception)
            
        if error_data:
            error_info.update(error_data)
            
        # Forward to event handlers
        event_type = "error"
        for handler in self._get_event_handlers(event_type):
            await handler(error_info=error_info)
    
    # Event handling registration
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for an event type.
        
        Args:
            event_type: Type of event to handle
            handler: Callback function for the event
        """
        if not hasattr(self, "_event_handlers"):
            self._event_handlers = {}
            
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
            
        self._event_handlers[event_type].append(handler)
    
    def unregister_event_handler(self, event_type: str, handler: Callable) -> bool:
        """Unregister a handler for an event type.
        
        Args:
            event_type: Type of event
            handler: Handler to remove
            
        Returns:
            bool: True if handler was found and removed
        """
        if not hasattr(self, "_event_handlers") or event_type not in self._event_handlers:
            return False
            
        if handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
            return True
            
        return False
    
    def _get_event_handlers(self, event_type: str) -> List[Callable]:
        """Get all handlers for an event type.
        
        Args:
            event_type: Type of event
            
        Returns:
            List[Callable]: List of handler functions
        """
        if not hasattr(self, "_event_handlers"):
            self._event_handlers = {}
            
        return self._event_handlers.get(event_type, [])
    
    # API Methods
    async def configure_session(
        self,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        voice: Optional[str] = None,
        vad_enabled: bool = True,
        auto_response: bool = True
    ) -> bool:
        """Configure the conversation session.
        
        Args:
            instructions: System instructions for the AI
            tools: Function definitions for the AI
            voice: Voice ID for audio responses
            vad_enabled: Whether to use voice activity detection
            auto_response: Whether to auto-generate responses
            
        Returns:
            bool: True if successful
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
                success = await self.service.update_session(session_config)
                if success:
                    logger.info("Session configuration updated")
                return success
            
            return True
        except Exception as e:
            logger.error(f"Error configuring session: {e}")
            await self._notify_error("Error configuring session", e)
            return False
    
    async def send_text_message(self, text: str) -> Optional[str]:
        """Send a text message to the conversation.
        
        Args:
            text: Message text
            
        Returns:
            Optional[str]: Message ID if successful
        """
        try:
            message_id = await self.service.send_text_message(text)
            return message_id
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self._notify_error("Error sending message", e)
            return None
    
    async def request_response(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Request a response from the AI.
        
        Args:
            modalities: Response modalities (text, audio)
            instructions: Additional instructions for this response
            metadata: Additional metadata for the request
            
        Returns:
            Optional[str]: Response ID if successful
        """
        try:
            # Default to text and audio if not specified
            if modalities is None:
                modalities = ["text", "audio"]
                
            response_id = await self.service.request_response(
                modalities=modalities,
                instructions=instructions,
                metadata=metadata
            )
            
            return response_id
        except Exception as e:
            logger.error(f"Error requesting response: {e}")
            await self._notify_error("Error requesting response", e)
            return None
    
    async def submit_function_result(self, call_id: str, result: Dict[str, Any]) -> Optional[str]:
        """Submit the result of a function call.
        
        Args:
            call_id: Function call ID
            result: Function result data
            
        Returns:
            Optional[str]: Message ID if successful
        """
        if call_id not in self.active_function_calls:
            logger.error(f"Function call ID not found: {call_id}")
            return None
        
        try:
            # Use the client to submit the result
            # This would need to be implemented in the OpenAIService
            message_id = "function_result_" + str(time.time())  # Placeholder
            return message_id
        except Exception as e:
            logger.error(f"Error submitting function result: {e}")
            await self._notify_error("Error submitting function result", e)
            return None
    
    async def get_last_response_text(self) -> str:
        """Get the text of the most recent response.
        
        Returns:
            str: Last response text
        """
        if not self.last_response_text:
            return ""
            
        # Find the most recent response ID
        latest_response_id = next(iter(self.last_response_text.keys()))
        
        # If there are active responses, prefer those
        for response_id in self.active_responses:
            if response_id in self.last_response_text:
                latest_response_id = response_id
                break
                
        return self.last_response_text.get(latest_response_id, "")
    
    async def get_last_transcript(self) -> Dict[str, str]:
        """Get the transcript of the most recent audio.
        
        Returns:
            Dict[str, str]: Mapping of part IDs to transcript text
        """
        if not self.last_transcript:
            return {}
            
        # Find the most recent response ID
        latest_response_id = next(iter(self.last_transcript.keys()))
        
        # If there are active responses, prefer those
        for response_id in self.active_responses:
            if response_id in self.last_transcript:
                latest_response_id = response_id
                break
                
        return self.last_transcript.get(latest_response_id, {})
    
    async def get_conversation_history(
        self, 
        limit: Optional[int] = None, 
        include_system: bool = False
    ) -> List[Dict[str, Any]]:
        """Get the conversation history.
        
        Args:
            limit: Maximum number of items to return
            include_system: Whether to include system messages
            
        Returns:
            List[Dict[str, Any]]: Conversation history items
        """
        if not include_system:
            filtered_history = [
                item for item in self.conversation_history 
                if item.get("role", "") != "system"
            ]
        else:
            filtered_history = self.conversation_history.copy()
            
        if limit and limit > 0:
            return filtered_history[-limit:]
            
        return filtered_history
    
    async def get_response_by_id(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Get a response by its ID.
        
        Args:
            response_id: Response ID
            
        Returns:
            Optional[Dict[str, Any]]: Response data if found
        """
        # This would require implementing a response cache or retrieval mechanism
        logger.warning("get_response_by_id not fully implemented")
        return None
    
    async def get_function_call_by_id(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get a function call by its ID.
        
        Args:
            call_id: Function call ID
            
        Returns:
            Optional[Dict[str, Any]]: Function call data if found
        """
        return self.active_function_calls.get(call_id)
    
    async def wait_for_all_responses(self, timeout: float = 30.0) -> bool:
        """Wait for all pending responses to complete.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            bool: True if all responses completed, False if timed out
        """
        start_time = time.time()
        
        while self.pending_responses and time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
            
        return len(self.pending_responses) == 0
    
    async def interrupt_response(self, response_id: str = None) -> bool:
        """Interrupt the current or specified response.
        
        Args:
            response_id: Response ID to interrupt, or None for current
            
        Returns:
            bool: True if successful
        """
        try:
            if response_id is None and self.active_responses:
                # If no specific ID provided, interrupt the most recent response
                response_id = next(iter(self.active_responses))
                
            if not response_id:
                logger.warning("No active response to interrupt")
                return False
            
            # This would require implementing interruption in the OpenAIService
            logger.info(f"Interrupting response: {response_id}")
            return True
        except Exception as e:
            logger.error(f"Error interrupting response: {e}")
            await self._notify_error("Error interrupting response", e)
            return False
    
    async def cleanup(self) -> None:
        """Clean up resources used by the conversation manager."""
        if self.conversation:
            # Mark the conversation as inactive
            self.conversation.is_active = False
            self.conversation.end_time = datetime.now()
            
        # Clear tracking collections
        self.active_responses.clear()
        self.active_function_calls.clear()
        self.pending_responses.clear()
        
        logger.info("Conversation manager resources cleaned up") 