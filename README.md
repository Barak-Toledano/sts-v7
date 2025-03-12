# OpenAI Realtime Assistant

A fully-featured voice interface for the OpenAI Realtime API, enabling natural voice conversations with AI assistants.

## Features

- **Real-time Voice Conversations**: Engage in natural, back-and-forth voice conversations with OpenAI's assistants
- **Seamless Experience**: Low-latency voice processing with incremental responses
- **Interrupt Capability**: Interrupt the assistant mid-response, just like in a natural conversation
- **Native Speech Processing**: Direct audio processing by the model for optimal performance and low latency
- **Parallel Transcription**: Optional Whisper-based transcription for UI display and logging
- **CLI Interface**: Simple command-line interface with status indicators and customizable display options
- **Robust Architecture**: Clean, modular design with proper error handling and logging
- **Event-Driven Design**: Flexible event system for responsive interactions
- **Configurability**: Customizable settings for API credentials, audio devices, and more

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key with access to the Realtime API
- PyAudio (for audio input/output)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/openai-realtime-assistant.git
   cd openai-realtime-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a `.env` file in the project root with the following:
   ```
   OPENAI_API_KEY=your_api_key_here
   OPENAI_ASSISTANT_ID=your_assistant_id_here
   ```

## Usage

### Basic Usage

Run the assistant:

```bash
python -m src --assistant-id your-assistant-id
```

This will start the application in conversation mode, listening for your voice input.

### Command-line Options

```bash
python -m src --help
```

This will display all available command-line options:

- `--assistant-id`: ID of the OpenAI assistant to use (required)
- `--instructions`: Custom instructions for the assistant
- `--temperature`: Temperature parameter for generation (0.0-2.0)
- `--input-device`: Input device index for audio
- `--output-device`: Output device index for audio
- `--save-recordings`: Save audio recordings to disk
- `--debug`: Enable debug mode

### In-app Commands

During a conversation, you can use the following commands:

- `/quit`: Exit the application
- `/help`: Display help information
- `/restart`: Restart the conversation
- `/pause`: Pause listening
- `/resume`: Resume listening  
- `/interrupt`: Interrupt the assistant
- `/status`: Toggle status display
- `/timestamps`: Toggle timestamps
- `/compact`: Toggle compact mode

## Project Structure

The project follows a clean, modular architecture:

```
src/
├── __init__.py           # Package initialization
├── __main__.py           # Entry point
├── application.py        # Main application class
├── config/               # Configuration management
│   ├── __init__.py       # Config package initialization
│   ├── settings.py       # Application settings using Pydantic
│   └── logging_config.py # Logging configuration
├── domain/               # Business logic
│   ├── audio/            # Audio domain
│   │   └── manager.py    # Audio management
│   └── conversation/     # Conversation domain
│       ├── manager.py    # Conversation orchestration
│       └── state.py      # Conversation state management
├── events/               # Event system
│   └── event_interface.py # Event definitions and event bus
├── presentation/         # User interfaces
│   └── cli.py            # Command-line interface
├── services/             # External services
│   ├── api_client.py     # OpenAI API client
│   └── audio_service.py  # Audio recording and playback
└── utils/                # Utility modules
    ├── async_helpers.py  # Async utilities
    ├── audio_utilities.py # Audio processing utilities
    ├── error_handling.py # Error handling
    ├── token_management.py # Token tracking and management
    └── transcription.py  # Speech transcription utilities
```

## Configuration

The application uses a hierarchical configuration system:

1. Default values defined in `src/config/settings.py`
2. Environment variables (from `.env` file or system environment)
3. Command-line arguments

### Available Settings

- `api.key`: OpenAI API key
- `api.base_url`: Base URL for the OpenAI API
- `api.timeout`: Request timeout in seconds
- `audio.input_device`: Audio input device index
- `audio.output_device`: Audio output device index
- `audio.sample_rate`: Audio sample rate in Hz
- `audio.channels`: Number of audio channels
- `audio.chunk_size`: Audio chunk size
- `audio.vad_threshold`: Voice activity detection threshold
- `logging.level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `logging.file_path`: Path to log file
- `debug_mode`: Enable debug mode

## Transcription Handling

The application uses a dual-path approach to handle speech:

1. **Primary Path**: Direct audio processing by the Realtime API model
   - Raw audio is sent directly to the model for processing
   - This provides the lowest latency and best performance

2. **Secondary Path**: Optional Whisper-based transcriptions via the Realtime API
   - Enabled using the `input_audio_transcription` configuration
   - Transcripts are received via the `conversation.item.input_audio_transcription.completed` event
   - The application processes these events using dedicated handlers in the `transcription.py` module
   - Emits `USER_TRANSCRIPTION_COMPLETED` events with formatted transcripts
   - Useful for UI display, logging, and debugging purposes

This dual approach provides optimal performance while still offering human-readable transcripts when needed.

### Enabling Transcription

Transcription is enabled by default when using the OpenAI Realtime API client. You can control this behavior:

```python
# Enable transcription (default)
client.connect(assistant_id="your_assistant_id", enable_transcription=True)

# Disable transcription
client.connect(assistant_id="your_assistant_id", enable_transcription=False)
```

When enabled, the system will:
1. Configure the session with Whisper transcription
2. Process transcription events as they arrive
3. Emit formatted transcription events through the event bus
4. Provide logging for debugging purposes

## Development

### Running Tests

```bash
pytest
```

### Building Documentation

```bash
cd docs
make html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for providing the Realtime API
- PyAudio for audio processing capabilities
- The Python community for excellent libraries and tools

## Voice Commands

The application supports the following voice commands:

- **"Goodbye"** or **"Good bye"**: Exits the application gracefully, with a farewell message.

## Event Handling System

The application uses a comprehensive event system to handle communication between components:

1. **Event Bus**: Central pub/sub mechanism that routes events between components
2. **Event Types**: Structured event definitions for consistent handling
3. **Event Handlers**: Component-specific handlers for processing events

### Realtime API Event Handler

The application includes a dedicated Realtime API event handler (`src/services/realtime_event_handler.py`) that:

- Centralizes handling of all OpenAI Realtime API events
- Provides a comprehensive registry of all valid server and client events
- Maps API events to application-specific events
- Ensures consistent event processing across the application
- Validates events against the API specification

### Custom Events

The application extends the standard OpenAI Realtime API with several custom client events:

- **audio**: Direct audio data transmission (simplifies the standard input_audio_buffer.append)
- **interrupt**: Allows interrupting the assistant mid-response
- **heartbeat**: Keeps the WebSocket connection alive

These custom events are clearly marked in the code and logged when used to maintain transparency about extensions to the standard API. 