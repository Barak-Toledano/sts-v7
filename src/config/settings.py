"""
Application settings and configuration.

This module provides a centralized configuration management system using Pydantic.
It loads settings from environment variables, .env files, or falls back to defaults.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
LOG_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"

# Ensure required directories exist
for directory in [LOG_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True)


class ApiSettings(BaseModel):
    """OpenAI API configuration settings."""
    
    api_key: str = Field(
        default="",
        description="OpenAI API key for authentication"
    )
    
    @validator("api_key")
    def api_key_must_not_be_empty(cls, v):
        """Validate that API key is provided."""
        if not v:
            logging.warning("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
        return v


class AudioSettings(BaseModel):
    """Audio configuration settings."""
    
    input_device: Optional[int] = Field(
        default=None,
        description="Index of audio input device (None for system default)"
    )
    
    output_device: Optional[int] = Field(
        default=None,
        description="Index of audio output device (None for system default)"
    )
    
    sample_rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz (required by OpenAI)"
    )
    
    channels: int = Field(
        default=1,
        description="Number of audio channels (1 for mono, 2 for stereo)"
    )
    
    chunk_size: int = Field(
        default=4096,
        description="Audio chunk size in bytes"
    )
    
    vad_threshold: float = Field(
        default=0.85,
        description="Voice activity detection threshold (0.0-1.0)"
    )
    
    silence_duration_ms: int = Field(
        default=1200,
        description="Silence duration in ms to consider speech ended"
    )
    
    prefix_padding_ms: int = Field(
        default=500,
        description="Padding before speech in ms"
    )


class LoggingSettings(BaseModel):
    """Logging configuration settings."""
    
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    file_enabled: bool = Field(
        default=True,
        description="Whether to write logs to a file"
    )
    
    console_enabled: bool = Field(
        default=True,
        description="Whether to write logs to console"
    )
    
    @validator("level")
    def validate_log_level(cls, v):
        """Validate that log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class Settings(BaseModel):
    """Main application settings."""
    
    # Application info
    app_name: str = Field(
        default="OpenAI Realtime Assistant",
        description="Application name"
    )
    
    app_version: str = Field(
        default="0.1.0",
        description="Application version"
    )
    
    # Sub-configurations
    api: ApiSettings = Field(default_factory=lambda: ApiSettings(
        api_key=os.environ.get("OPENAI_API_KEY", "")
    ))
    
    audio: AudioSettings = Field(default_factory=lambda: AudioSettings(
        input_device=_parse_optional_int(os.environ.get("AUDIO_INPUT_DEVICE")),
        output_device=_parse_optional_int(os.environ.get("AUDIO_OUTPUT_DEVICE")),
        sample_rate=int(os.environ.get("AUDIO_SAMPLE_RATE", "24000")),
        channels=int(os.environ.get("AUDIO_CHANNELS", "1")),
        chunk_size=int(os.environ.get("AUDIO_CHUNK_SIZE", "4096")),
        vad_threshold=float(os.environ.get("VAD_THRESHOLD", "0.85")),
        silence_duration_ms=int(os.environ.get("SILENCE_DURATION_MS", "1200")),
        prefix_padding_ms=int(os.environ.get("PREFIX_PADDING_MS", "500"))
    ))
    
    logging: LoggingSettings = Field(default_factory=lambda: LoggingSettings(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format=os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        file_enabled=_parse_bool(os.environ.get("LOG_FILE_ENABLED", "True")),
        console_enabled=_parse_bool(os.environ.get("LOG_CONSOLE_ENABLED", "True"))
    ))
    
    # Paths
    root_dir: Path = ROOT_DIR
    logs_dir: Path = LOG_DIR
    data_dir: Path = DATA_DIR
    
    # Runtime configs
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    def __init__(self, **data: Any):
        """Initialize settings and create any required directories."""
        super().__init__(**data)
        
        # Allow debug mode override from environment
        self.debug_mode = _parse_bool(os.environ.get("DEBUG_MODE", str(self.debug_mode)))
        
        # Additional initialization (if any)
        self._create_required_directories()
    
    def _create_required_directories(self) -> None:
        """Create any required application directories."""
        # Logs directory (with session subdirectory)
        session_log_dir = self.logs_dir / "sessions"
        session_log_dir.mkdir(exist_ok=True)
        
        # Data directory with subdirectories
        recordings_dir = self.data_dir / "recordings"
        recordings_dir.mkdir(exist_ok=True)
        
        # Ensure directories are writable
        for directory in [self.logs_dir, self.data_dir]:
            if not os.access(directory, os.W_OK):
                logging.warning(f"Directory {directory} is not writable")
    
    def get_session_log_path(self, session_id: str) -> Path:
        """Get path for session-specific log file."""
        return self.logs_dir / "sessions" / f"{session_id}.log"


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    """Parse string to optional int."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_bool(value: str) -> bool:
    """Parse string to boolean."""
    return value.lower() in ("true", "1", "t", "yes", "y") 