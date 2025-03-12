"""
Base service interface for external API integrations.
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BaseService(ABC):
    """Base class for all external service integrations.
    
    This abstract class defines the interface that all service
    implementations should follow. It provides common functionality
    like error handling and logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the service with configuration.
        
        Args:
            config: Configuration dictionary for the service
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the service configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the service.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the service."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the service connection.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        pass
    
    async def handle_error(
        self, 
        error: Exception, 
        operation: str, 
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle service errors in a consistent way.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            additional_info: Any additional context information
            
        Returns:
            Dict[str, Any]: Error information in a structured format
        """
        error_info = {
            "service": self.__class__.__name__,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        if additional_info:
            error_info["additional_info"] = additional_info
            
        logger.error(f"Service error: {error_info}")
        return error_info 