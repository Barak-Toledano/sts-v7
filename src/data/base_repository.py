"""
Base repository interface for database operations.
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, TypeVar, Generic

T = TypeVar('T')
logger = logging.getLogger(__name__)

class BaseRepository(Generic[T], ABC):
    """Base class for all repository implementations.
    
    This abstract class defines the interface that all repository
    implementations should follow. It provides common database
    operations with appropriate error handling.
    
    Generic type T represents the entity model being managed.
    """
    
    def __init__(self, connection_config: Dict[str, Any]):
        """Initialize the repository with connection configuration.
        
        Args:
            connection_config: Database connection parameters
        """
        self.connection_config = connection_config
        self._connection = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """Retrieve an entity by its ID.
        
        Args:
            id: Entity identifier
            
        Returns:
            Optional[T]: Entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_all(self, filter_params: Optional[Dict[str, Any]] = None) -> List[T]:
        """Retrieve all entities matching the filter.
        
        Args:
            filter_params: Filter criteria
            
        Returns:
            List[T]: List of matching entities
        """
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> Optional[T]:
        """Create a new entity in the database.
        
        Args:
            entity: Entity to create
            
        Returns:
            Optional[T]: Created entity with ID, None if failed
        """
        pass
    
    @abstractmethod
    async def update(self, id: str, entity: T) -> Optional[T]:
        """Update an existing entity.
        
        Args:
            id: Entity identifier
            entity: Updated entity data
            
        Returns:
            Optional[T]: Updated entity, None if failed
        """
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an entity by its ID.
        
        Args:
            id: Entity identifier
            
        Returns:
            bool: True if deleted, False otherwise
        """
        pass
    
    async def handle_db_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle database errors in a consistent way.
        
        Args:
            error: The exception that occurred
            operation: Name of the database operation that failed
            
        Returns:
            Dict[str, Any]: Error information
        """
        error_info = {
            "repository": self.__class__.__name__,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        logger.error(f"Database error: {error_info}")
        return error_info 