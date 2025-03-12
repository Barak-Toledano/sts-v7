"""
Conversation state management for the OpenAI Realtime Assistant.

This module provides classes and utilities for managing conversation state,
including history, context, and metadata.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class MessageRole(Enum):
    """Possible roles for conversation messages."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class MessageContent:
    """Content for a conversation message."""
    
    text: Optional[str] = None
    audio_duration: Optional[float] = None
    interrupted: bool = False


@dataclass
class Message:
    """Represents a message in a conversation."""
    
    id: str
    role: MessageRole
    content: MessageContent
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a dictionary representation."""
        result = asdict(self)
        result["role"] = self.role.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from a dictionary representation."""
        role = data.get("role", "user")
        role_enum = MessageRole.USER
        
        if role == "assistant":
            role_enum = MessageRole.ASSISTANT
        elif role == "system":
            role_enum = MessageRole.SYSTEM
        
        content_data = data.get("content", {})
        if isinstance(content_data, str):
            content = MessageContent(text=content_data)
        else:
            content = MessageContent(
                text=content_data.get("text"),
                audio_duration=content_data.get("audio_duration"),
                interrupted=content_data.get("interrupted", False)
            )
        
        return cls(
            id=data.get("id", ""),
            role=role_enum,
            content=content,
            created_at=data.get("created_at", time.time())
        )


@dataclass
class ConversationContext:
    """Context for a conversation."""
    
    assistant_id: str
    session_id: str
    thread_id: Optional[str] = None
    run_id: Optional[str] = None
    instructions: Optional[str] = None
    temperature: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to a dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create a context from a dictionary representation."""
        return cls(
            assistant_id=data.get("assistant_id", ""),
            session_id=data.get("session_id", ""),
            thread_id=data.get("thread_id"),
            run_id=data.get("run_id"),
            instructions=data.get("instructions"),
            temperature=data.get("temperature", 1.0),
            metadata=data.get("metadata", {})
        )


class ConversationHistory:
    """
    Manages conversation history and state.
    
    This class provides methods for adding, retrieving, and persisting
    conversation messages and context.
    """
    
    def __init__(self, context: ConversationContext):
        """
        Initialize conversation history.
        
        Args:
            context: Context for the conversation
        """
        self.context = context
        self.messages: List[Message] = []
        self.metadata: Dict[str, Any] = {}
        self.created_at = time.time()
        self.updated_at = self.created_at
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.updated_at = time.time()
        
        logger.debug(f"Added {message.role.value} message to conversation history")
    
    def add_user_message(self, text: Optional[str] = None, audio_duration: Optional[float] = None) -> Message:
        """
        Add a user message to the conversation history.
        
        Args:
            text: Optional text content
            audio_duration: Optional audio duration in seconds
            
        Returns:
            Message: The created message
        """
        message_id = f"msg_{len(self.messages)}_{int(time.time())}"
        content = MessageContent(text=text, audio_duration=audio_duration)
        
        message = Message(
            id=message_id,
            role=MessageRole.USER,
            content=content
        )
        
        self.add_message(message)
        return message
    
    def add_assistant_message(
        self, 
        text: Optional[str] = None, 
        audio_duration: Optional[float] = None,
        interrupted: bool = False
    ) -> Message:
        """
        Add an assistant message to the conversation history.
        
        Args:
            text: Optional text content
            audio_duration: Optional audio duration in seconds
            interrupted: Whether the message was interrupted
            
        Returns:
            Message: The created message
        """
        message_id = f"msg_{len(self.messages)}_{int(time.time())}"
        content = MessageContent(
            text=text, 
            audio_duration=audio_duration,
            interrupted=interrupted
        )
        
        message = Message(
            id=message_id,
            role=MessageRole.ASSISTANT,
            content=content
        )
        
        self.add_message(message)
        return message
    
    def add_system_message(self, text: str) -> Message:
        """
        Add a system message to the conversation history.
        
        Args:
            text: Text content
            
        Returns:
            Message: The created message
        """
        message_id = f"msg_{len(self.messages)}_{int(time.time())}"
        content = MessageContent(text=text)
        
        message = Message(
            id=message_id,
            role=MessageRole.SYSTEM,
            content=content
        )
        
        self.add_message(message)
        return message
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """
        Get a message by ID.
        
        Args:
            message_id: ID of the message to get
            
        Returns:
            Optional[Message]: The message, or None if not found
        """
        for message in self.messages:
            if message.id == message_id:
                return message
        return None
    
    def update_message(self, message_id: str, **kwargs) -> bool:
        """
        Update a message by ID.
        
        Args:
            message_id: ID of the message to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if the message was updated
        """
        for i, message in enumerate(self.messages):
            if message.id == message_id:
                # Update content fields
                if "text" in kwargs or "audio_duration" in kwargs or "interrupted" in kwargs:
                    content_updates = {}
                    if "text" in kwargs:
                        content_updates["text"] = kwargs.pop("text")
                    if "audio_duration" in kwargs:
                        content_updates["audio_duration"] = kwargs.pop("audio_duration")
                    if "interrupted" in kwargs:
                        content_updates["interrupted"] = kwargs.pop("interrupted")
                    
                    # Create new content with updates
                    message.content = MessageContent(
                        text=content_updates.get("text", message.content.text),
                        audio_duration=content_updates.get("audio_duration", message.content.audio_duration),
                        interrupted=content_updates.get("interrupted", message.content.interrupted)
                    )
                
                # Update other fields
                for key, value in kwargs.items():
                    if hasattr(message, key):
                        setattr(message, key, value)
                
                self.updated_at = time.time()
                return True
        
        return False
    
    def get_last_message(self) -> Optional[Message]:
        """
        Get the last message in the conversation.
        
        Returns:
            Optional[Message]: The last message, or None if there are no messages
        """
        if not self.messages:
            return None
        return self.messages[-1]
    
    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """
        Get all messages with a specific role.
        
        Args:
            role: Role to filter by
            
        Returns:
            List[Message]: Messages with the specified role
        """
        return [message for message in self.messages if message.role == role]
    
    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self.messages = []
        self.updated_at = time.time()
        logger.debug("Cleared conversation history")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation history to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the conversation history
        """
        return {
            "context": self.context.to_dict(),
            "messages": [message.to_dict() for message in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_json(self) -> str:
        """
        Convert the conversation history to a JSON string.
        
        Returns:
            str: JSON representation of the conversation history
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationHistory':
        """
        Create a conversation history from a dictionary representation.
        
        Args:
            data: Dictionary representation of a conversation history
            
        Returns:
            ConversationHistory: The created conversation history
        """
        context = ConversationContext.from_dict(data.get("context", {}))
        history = cls(context)
        
        # Add messages
        for message_data in data.get("messages", []):
            message = Message.from_dict(message_data)
            history.messages.append(message)
        
        # Set metadata and timestamps
        history.metadata = data.get("metadata", {})
        history.created_at = data.get("created_at", time.time())
        history.updated_at = data.get("updated_at", time.time())
        
        return history
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationHistory':
        """
        Create a conversation history from a JSON string.
        
        Args:
            json_str: JSON representation of a conversation history
            
        Returns:
            ConversationHistory: The created conversation history
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse conversation history JSON: {e}")
            raise ValueError(f"Invalid conversation history JSON: {e}")


def format_timestamp(timestamp: float) -> str:
    """
    Format a Unix timestamp as a human-readable string.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        str: Formatted timestamp string
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S") 