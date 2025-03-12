"""
In-memory repository implementation for testing.
"""
import logging
from typing import Any, Dict, Generic, List, Optional, TypeVar

from .base_repository import BaseRepository

T = TypeVar('T')
logger = logging.getLogger(__name__)

class InMemoryRepository(BaseRepository[T], Generic[T]):
    """In-memory repository implementation.
    
    This repository stores entities in memory, primarily for
    testing and development purposes.
    """
    
    def __init__(self, connection_config: Dict[str, Any] = None):
        """Initialize the repository with an empty store.
        
        Args:
            connection_config: Not used for in-memory repository
        """
        super().__init__(connection_config or {})
        self._store: Dict[str, T] = {}
        self._is_connected = False
    
    async def connect(self) -> bool:
        """Simulate connecting to a database.
        
        Returns:
            bool: Always returns True
        """
        self._is_connected = True
        logger.info("Connected to in-memory repository")
        return True
    
    async def disconnect(self) -> None:
        """Simulate disconnecting from a database."""
        self._is_connected = False
        logger.info("Disconnected from in-memory repository")
    
    async def get_by_id(self, id: str) -> Optional[T]:
        """Retrieve an entity by its ID.
        
        Args:
            id: Entity identifier
            
        Returns:
            Optional[T]: Entity if found, None otherwise
        """
        self._check_connection()
        return self._store.get(id)
    
    async def get_all(self, filter_params: Optional[Dict[str, Any]] = None) -> List[T]:
        """Retrieve all entities matching the filter.
        
        Args:
            filter_params: Filter criteria
            
        Returns:
            List[T]: List of matching entities
        """
        self._check_connection()
        
        if not filter_params:
            return list(self._store.values())
            
        # Simple filtering logic (can be expanded)
        result = []
        for entity in self._store.values():
            match = True
            for key, value in filter_params.items():
                if hasattr(entity, key) and getattr(entity, key) != value:
                    match = False
                    break
            if match:
                result.append(entity)
                
        return result
    
    async def create(self, entity: T) -> Optional[T]:
        """Create a new entity in the repository.
        
        Args:
            entity: Entity to create
            
        Returns:
            Optional[T]: Created entity
        """
        self._check_connection()
        
        if not hasattr(entity, 'id'):
            logger.error("Entity must have an 'id' attribute")
            return None
            
        entity_id = getattr(entity, 'id')
        self._store[entity_id] = entity
        return entity
    
    async def update(self, id: str, entity: T) -> Optional[T]:
        """Update an existing entity.
        
        Args:
            id: Entity identifier
            entity: Updated entity data
            
        Returns:
            Optional[T]: Updated entity, None if not found
        """
        self._check_connection()
        
        if id not in self._store:
            logger.warning(f"Entity with ID {id} not found")
            return None
            
        self._store[id] = entity
        return entity
    
    async def delete(self, id: str) -> bool:
        """Delete an entity by its ID.
        
        Args:
            id: Entity identifier
            
        Returns:
            bool: True if deleted, False if not found
        """
        self._check_connection()
        
        if id not in self._store:
            logger.warning(f"Entity with ID {id} not found")
            return False
            
        del self._store[id]
        return True
    
    def _check_connection(self) -> None:
        """Check if the repository is connected.
        
        Raises:
            RuntimeError: If not connected
        """
        if not self._is_connected:
            raise RuntimeError("Repository is not connected")
    
    def _validate_config(self) -> None:
        """Validate the repository configuration.
        
        Always valid for in-memory repository.
        """
        pass 