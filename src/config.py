import os
from pydantic_settings import BaseSettings  # Changed from pydantic import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # OpenAI API settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_REALTIME_MODEL: str = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    
    # Voice settings
    VOICE: str = os.getenv("VOICE", "alloy")
    
    # WebSocket settings
    WEBSOCKET_URL: str = "wss://api.openai.com/v1/realtime"
    
    # Audio settings
    AUDIO_FORMAT: str = os.getenv("AUDIO_FORMAT", "pcm16")
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "24000"))  # 24kHz sample rate (required by OpenAI)
    CHANNELS: int = int(os.getenv("CHANNELS", "1"))         # Mono audio
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "4096"))    # Default buffer size
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    
    # Server settings
    PORT: int = int(os.getenv("PORT", "3000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", '["*"]')
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields in the settings

settings = Settings()