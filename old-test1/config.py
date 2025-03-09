import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ✅ OpenAI API Key (loaded securely from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ OpenAI WebSocket URL (kept static for stability)
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

# ✅ Audio Configuration
AUDIO_FORMAT = "pcm16"
SAMPLE_RATE = 24000  # 24kHz sample rate (required by OpenAI)
CHANNELS = 1         # Mono audio
CHUNK_SIZE = 4096    # Default buffer size

# ✅ Voice Activity Detection (VAD) Settings
VAD_CONFIG = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 200,
    "create_response": True,
    "interrupt_response": True
}

# ✅ Logging Configuration
LOG_FILE = "logs.txt"

# ✅ WebSocket Headers
HEADERS = [
    ("Authorization", f"Bearer {OPENAI_API_KEY}"),
    ("openai-beta", "realtime=v1")
]

# ✅ Voice Settings (Choose OpenAI's voice model)
DEFAULT_VOICE = "alloy"

# ✅ General Instructions for AI
DEFAULT_INSTRUCTIONS = "Always respond in English, regardless of detected language."
