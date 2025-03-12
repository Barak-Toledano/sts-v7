"""
OpenAI service implementation for API communication.
"""
import logging
from typing import Any, Dict, List, Optional, Callable

from .base_service import BaseService
from ..realtime_client import RealtimeClient

logger = logging.getLogger(__name__)

class OpenAIService(BaseService):
    """Service for interfacing with OpenAI APIs.
    
    This service handles communication with both the standard OpenAI API
    and the Realtime API for audio processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI service.
        
        Args:
            config: Configuration including API keys and endpoints
        """
        super().__init__(config)
        self.realtime_client = None
        self.event_handlers = {}
    
    def _validate_config(self) -> None:
        """Validate the service configuration.
        
        Raises:
            ValueError: If API key or other required config is missing
        """
        if "api_key" not in self.config:
            raise ValueError("OpenAI API key is required")
    
    async def connect(self) -> bool:
        """Connect to the OpenAI Realtime API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.realtime_client = RealtimeClient(self.config.get("api_key"))
            await self.realtime_client.connect()
            
            # Register any pre-defined event handlers
            for event_type, handler in self.event_handlers.items():
                self.realtime_client.register_event_handler(event_type, handler)
                
            logger.info("Connected to OpenAI Realtime API")
            return True
        except Exception as e:
            error_info = await self.handle_error(e, "connect")
            logger.error(f"Failed to connect to OpenAI Realtime API: {error_info}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the OpenAI Realtime API."""
        if self.realtime_client:
            await self.realtime_client.disconnect()
            logger.info("Disconnected from OpenAI Realtime API")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the OpenAI API connection.
        
        Returns:
            Dict[str, Any]: Connection status information
        """
        is_connected = self.realtime_client and self.realtime_client.is_connected()
        
        return {
            "service": "OpenAI",
            "connected": is_connected,
            "session_id": self.realtime_client.session_id if is_connected else None
        }
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for a specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Callback function to handle the event
        """
        self.event_handlers[event_type] = handler
        
        # If already connected, register the handler immediately
        if self.realtime_client:
            self.realtime_client.register_event_handler(event_type, handler)
    
    async def create_session(self) -> Optional[str]:
        """Create a new session with the OpenAI Realtime API.
        
        Returns:
            Optional[str]: Session ID if successful, None otherwise
        """
        if not self.realtime_client:
            logger.error("Cannot create session: Not connected to OpenAI API")
            return None
            
        try:
            session_id = await self.realtime_client.create_session()
            logger.info(f"Created OpenAI session: {session_id}")
            return session_id
        except Exception as e:
            await self.handle_error(e, "create_session")
            return None
    
    async def update_session(self, config: Dict[str, Any]) -> bool:
        """Update the current session with new configuration.
        
        Args:
            config: Session configuration parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.realtime_client:
            logger.error("Cannot update session: Not connected to OpenAI API")
            return False
            
        try:
            await self.realtime_client.update_session(config)
            logger.info("Session configuration updated")
            return True
        except Exception as e:
            await self.handle_error(e, "update_session")
            return False
            
    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio data to the OpenAI Realtime API.
        
        Args:
            audio_data: Audio data in PCM16 format
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.realtime_client:
            logger.error("Cannot send audio: Not connected to OpenAI API")
            return False
            
        try:
            await self.realtime_client.send_audio(audio_data)
            return True
        except Exception as e:
            await self.handle_error(e, "send_audio")
            return False
            
    async def send_text_message(self, text: str) -> Optional[str]:
        """Send a text message to the OpenAI Realtime API.
        
        Args:
            text: Text message to send
            
        Returns:
            Optional[str]: Message ID if successful, None otherwise
        """
        if not self.realtime_client:
            logger.error("Cannot send message: Not connected to OpenAI API")
            return None
            
        try:
            message_id = await self.realtime_client.send_message(text)
            return message_id
        except Exception as e:
            await self.handle_error(e, "send_text_message")
            return None
            
    async def request_response(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Request a response from the OpenAI Realtime API.
        
        Args:
            modalities: Response modalities (text, audio)
            instructions: Additional instructions for this response
            metadata: Additional metadata for the request
            
        Returns:
            Optional[str]: Response ID if successful, None otherwise
        """
        if not self.realtime_client:
            logger.error("Cannot request response: Not connected to OpenAI API")
            return None
            
        try:
            response_id = await self.realtime_client.request_response(
                modalities=modalities or ["text", "audio"],
                instructions=instructions,
                metadata=metadata
            )
            return response_id
        except Exception as e:
            await self.handle_error(e, "request_response")
            return None 