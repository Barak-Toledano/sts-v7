# src/utils/logging_utils.py
import logging
import os
import sys
import uuid
import time
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

def setup_logger(name: str, level: str = None, session_id: str = None) -> logging.Logger:
    """
    Set up and configure logger with session-based logging
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        session_id: Optional session ID for session-specific logs
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # If logger is already configured, return it
    if logger.handlers:
        return logger
    
    # Determine log level
    if level:
        log_level = getattr(logging, level.upper(), logging.INFO)
    else:
        log_level = logging.INFO
    
    logger.setLevel(log_level)
    
    # Create log directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create session directory for session-specific logs
    if not session_id:
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    session_dir = log_dir / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    # 1. Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(log_level)
    
    # 2. Component-specific log file
    component_file = session_dir / f"{name}.log"
    component_handler = RotatingFileHandler(
        component_file, 
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    component_handler.setFormatter(detailed_formatter)
    component_handler.setLevel(log_level)
    
    # 3. Combined session log file
    session_file = session_dir / "session.log"
    session_handler = RotatingFileHandler(
        session_file, 
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    session_handler.setFormatter(detailed_formatter)
    session_handler.setLevel(log_level)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(component_handler)
    logger.addHandler(session_handler)
    
    # Store session ID in logger for reference
    logger.session_id = session_id
    
    return logger

def get_session_id(logger: logging.Logger) -> str:
    """
    Get the session ID from a logger
    
    Args:
        logger: Logger to get session ID from
        
    Returns:
        str: Session ID
    """
    return getattr(logger, "session_id", None)