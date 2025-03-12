"""
Token management utilities for OpenAI API.

This module provides functions for estimating token counts,
truncating content to fit within token limits, and managing
token usage across different contexts.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config.logging_config import get_logger
from src.utils.error_handling import AppError, ErrorSeverity

logger = get_logger(__name__)

# Default token limits for different models
# These are approximate and should be updated if OpenAI changes limits
DEFAULT_TOKEN_LIMITS = {
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
}

# Approximation: average tokens per word for English text
AVG_TOKENS_PER_WORD = 1.3


def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a given text.
    
    This is a rough estimate based on the average number of tokens per word.
    For more accurate counts, use a proper tokenizer.
    
    Args:
        text: The text to estimate token count for
        
    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0
    
    # Split on whitespace to count words
    words = text.split()
    word_count = len(words)
    
    # Apply the approximation factor
    estimated_tokens = int(word_count * AVG_TOKENS_PER_WORD)
    
    # Ensure at least 1 token for non-empty text
    return max(1, estimated_tokens)


def truncate_to_token_limit(
    text: str, 
    max_tokens: int,
    preserve_start: bool = True,
    preserve_end: bool = False
) -> str:
    """
    Truncate text to approximately fit within a token limit.
    
    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens to allow
        preserve_start: Whether to preserve the start (True) or end (False) of the text
        preserve_end: Whether to also preserve the end of the text (creates a "window" effect)
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    # Quick check - if already within limit, return as is
    estimated_tokens = estimate_token_count(text)
    if estimated_tokens <= max_tokens:
        return text
    
    # Calculate approximately how many words we can keep
    approx_word_limit = int(max_tokens / AVG_TOKENS_PER_WORD)
    
    # Split into words
    words = text.split()
    total_words = len(words)
    
    # If we can't preserve anything meaningful, return empty string
    if approx_word_limit < 3:
        logger.warning(f"Token limit {max_tokens} too small to preserve meaningful content")
        return ""
    
    # Handle different preservation strategies
    if preserve_start and preserve_end:
        # Keep half from start, half from end (with small overlap allowance)
        half_limit = approx_word_limit // 2
        start_portion = words[:half_limit]
        end_portion = words[-half_limit:]
        
        result = " ".join(start_portion) + " [...] " + " ".join(end_portion)
    
    elif preserve_start:
        # Keep from the beginning
        result = " ".join(words[:approx_word_limit]) + " [...]"
    
    else:
        # Keep from the end
        result = "[...] " + " ".join(words[-approx_word_limit:])
    
    # Verify we haven't exceeded the token limit after truncation
    if estimate_token_count(result) > max_tokens:
        # Reduce further if necessary
        reduction_factor = 0.9  # 10% reduction
        return truncate_to_token_limit(
            result, 
            int(max_tokens * reduction_factor),
            preserve_start,
            preserve_end
        )
    
    return result


def truncate_conversation_to_fit(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    preserve_system_message: bool = True,
    preserve_recent_messages: int = 3
) -> List[Dict[str, Any]]:
    """
    Truncate a conversation to fit within a token limit.
    
    This function tries to preserve system messages and recent messages,
    removing older messages in the middle of the conversation if needed.
    
    Args:
        messages: List of message objects with 'role' and 'content' keys
        max_tokens: Maximum tokens to allow
        preserve_system_message: Whether to preserve the system message
        preserve_recent_messages: Number of recent messages to preserve
        
    Returns:
        Truncated list of messages
    """
    if not messages:
        return []
    
    # First, estimate total tokens
    total_tokens = sum(estimate_token_count(msg.get("content", "")) for msg in messages)
    
    # If already within limit, return as is
    if total_tokens <= max_tokens:
        return messages
    
    logger.info(f"Conversation exceeds token limit: {total_tokens} > {max_tokens}. Truncating...")
    
    # Make a copy to avoid modifying the original
    truncated_messages = messages.copy()
    
    # Find system message if present
    system_messages = [i for i, msg in enumerate(truncated_messages) 
                      if msg.get("role") == "system"]
    
    # Strategy: preserve system message and most recent messages,
    # then remove/truncate older messages
    
    # First, identify messages we want to preserve
    preserve_indices = set()
    
    # Preserve system message if requested
    if preserve_system_message and system_messages:
        preserve_indices.update(system_messages)
    
    # Preserve recent messages
    preserve_indices.update(range(max(0, len(truncated_messages) - preserve_recent_messages),
                               len(truncated_messages)))
    
    # Estimate tokens for preserved messages
    preserved_tokens = sum(estimate_token_count(truncated_messages[i].get("content", "")) 
                        for i in preserve_indices)
    
    # If preserved messages already exceed limit, we need to truncate them
    if preserved_tokens > max_tokens:
        logger.warning(f"Even preserved messages exceed token limit. Truncating content.")
        
        # Sort indices in reverse order (to avoid changing positions as we truncate)
        for idx in sorted(preserve_indices, reverse=True):
            msg = truncated_messages[idx]
            content = msg.get("content", "")
            
            # Skip empty content
            if not content:
                continue
            
            # Calculate how many tokens we need to save
            current_tokens = estimate_token_count(content)
            target_tokens = max(1, current_tokens - (preserved_tokens - max_tokens))
            
            # Truncate this message
            is_system = msg.get("role") == "system"
            is_recent = idx >= len(truncated_messages) - 2
            
            truncated_content = truncate_to_token_limit(
                content,
                target_tokens,
                preserve_start=is_system or not is_recent,  # Preserve start for system messages
                preserve_end=is_recent,  # Preserve end for most recent messages
            )
            
            # Update message
            msg["content"] = truncated_content
            preserved_tokens = preserved_tokens - current_tokens + estimate_token_count(truncated_content)
            
            # If we're now within limit, stop truncating
            if preserved_tokens <= max_tokens:
                break
    
        # Return only the preserved messages that we've now truncated
        return [truncated_messages[i] for i in sorted(preserve_indices)]
    
    # If preserved messages are within limit, remove older messages
    # that aren't in the preserve set
    remaining_tokens = max_tokens - preserved_tokens
    candidate_indices = [i for i in range(len(truncated_messages)) if i not in preserve_indices]
    candidate_indices.sort(reverse=True)  # Start with more recent messages
    
    # Collect messages we can include without exceeding the limit
    include_indices = set()
    
    for idx in candidate_indices:
        msg = truncated_messages[idx]
        token_count = estimate_token_count(msg.get("content", ""))
        
        if token_count <= remaining_tokens:
            include_indices.add(idx)
            remaining_tokens -= token_count
    
    # Combine preserved and included indices and sort them to maintain order
    final_indices = sorted(preserve_indices | include_indices)
    
    # Return the final truncated conversation
    return [truncated_messages[i] for i in final_indices]


def get_token_limit_for_model(model_name: str) -> int:
    """
    Get the token limit for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Token limit for the model, or a default limit if unknown
    """
    # Remove any version or variant info to match base model
    base_model = model_name.split('-')[0].lower()
    
    # Try to match the exact model name
    if model_name in DEFAULT_TOKEN_LIMITS:
        return DEFAULT_TOKEN_LIMITS[model_name]
    
    # Try to match based on base model
    for model, limit in DEFAULT_TOKEN_LIMITS.items():
        if model.startswith(base_model):
            return limit
    
    # Default fallback
    logger.warning(f"Unknown model: {model_name}. Using conservative token limit.")
    return 4096  # Conservative default


class TokenTracker:
    """
    Utility class to track token usage across multiple requests.
    
    This helps manage token usage within budget limits and provides
    insights into token consumption patterns.
    """
    
    def __init__(self, token_limit: Optional[int] = None, model_name: Optional[str] = None):
        """
        Initialize the token tracker.
        
        Args:
            token_limit: Maximum tokens to allow (default: None = no limit)
            model_name: Name of the model to derive token limit from (if token_limit not provided)
        """
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        
        # Set token limit
        if token_limit is not None:
            self.token_limit = token_limit
        elif model_name is not None:
            self.token_limit = get_token_limit_for_model(model_name)
        else:
            self.token_limit = None
    
    def add_usage(self, prompt_tokens: int, completion_tokens: int = 0) -> None:
        """
        Add token usage from an API request.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.request_count += 1
        
        logger.debug(f"Token usage: +{prompt_tokens} prompt, +{completion_tokens} completion")
    
    def reset(self) -> None:
        """Reset all counters to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of token usage.
        
        Returns:
            Dictionary with token usage statistics
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "token_limit": self.token_limit,
            "remaining_tokens": None if self.token_limit is None else max(0, self.token_limit - self.total_tokens),
            "average_tokens_per_request": 0 if self.request_count == 0 else self.total_tokens / self.request_count
        }
    
    def will_exceed_limit(self, estimated_new_tokens: int) -> bool:
        """
        Check if adding more tokens would exceed the limit.
        
        Args:
            estimated_new_tokens: Estimated number of tokens to add
            
        Returns:
            True if adding these tokens would exceed the limit, False otherwise
        """
        if self.token_limit is None:
            return False
        
        return (self.total_tokens + estimated_new_tokens) > self.token_limit
    
    def save_to_file(self, file_path: Union[str, Path]) -> bool:
        """
        Save token usage statistics to a file.
        
        Args:
            file_path: Path to save the statistics
            
        Returns:
            True if saving was successful
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        try:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get usage summary
            data = self.get_usage_summary()
            
            # Add timestamp
            import time
            data["timestamp"] = time.time()
            
            # Save as JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved token usage statistics to {file_path}")
            return True
            
        except Exception as e:
            error = AppError(
                f"Failed to save token usage statistics: {str(e)}",
                severity=ErrorSeverity.WARNING,
                cause=e,
            )
            logger.error(str(error))
            return False 