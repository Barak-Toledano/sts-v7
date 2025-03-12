"""
Data models for persistence and business logic.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

@dataclass
class Conversation:
    """Model representing a conversation with the AI assistant."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create a model from a dictionary."""
        start_time = datetime.fromisoformat(data["start_time"]) if isinstance(data["start_time"], str) else data["start_time"]
        end_time = datetime.fromisoformat(data["end_time"]) if data.get("end_time") and isinstance(data["end_time"], str) else data.get("end_time")
        
        return cls(
            id=data["id"],
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            start_time=start_time,
            end_time=end_time,
            metadata=data.get("metadata", {}),
            is_active=data.get("is_active", True)
        )


@dataclass
class Message:
    """Model representing a message in a conversation."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    content: str = ""
    role: str = ""  # 'user', 'assistant', or 'system'
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a model from a dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        
        return cls(
            id=data["id"],
            conversation_id=data["conversation_id"],
            content=data["content"],
            role=data["role"],
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )


@dataclass
class User:
    """Model representing a user of the system."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "preferences": self.preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create a model from a dictionary."""
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        
        return cls(
            id=data["id"],
            name=data["name"],
            email=data.get("email"),
            created_at=created_at,
            preferences=data.get("preferences", {})
        ) 