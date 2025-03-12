# OpenAI Realtime API Assistant

This repository contains a Python implementation of a voice assistant using OpenAI's Realtime API. It supports bidirectional audio communication, custom instructions, and function calling capabilities.

## Architecture Overview

The application follows a clean architecture pattern with clear separation of concerns:

### Layers

1. **Service Layer** (`src/services/`)
   - Handles communication with external APIs and services
   - Provides abstractions over third-party dependencies
   - `OpenAIService` manages all interactions with OpenAI APIs

2. **Domain Layer** (`src/domain/`)
   - Contains core business logic independent of external services
   - `ConversationManager` manages conversation state and flow
   - `AudioManager` handles audio processing and streaming

3. **Data Layer** (`src/data/`)
   - Manages data persistence and retrieval
   - Models define entities and their relationships
   - Repositories provide data access interfaces

4. **Application Layer** (`app.py`)
   - Orchestrates the application components
   - Handles user interaction and configuration
   - Provides command-line interface

### Key Components

- **BaseService**: Abstract interface for external service integrations
- **BaseRepository**: Generic interface for data persistence
- **Domain Models**: Type-safe data models with serialization
- **Event-Driven Architecture**: Components communicate via events

## Key Configuration Options

### Audio Flow Control

The system provides several parameters to control the conversational flow and audio sensitivity:

#### VAD (Voice Activity Detection) Settings

Located in `src/conversation.py` in the `configure_session` method:

```python
session_config["turn_detection"] = {
    "type": "server_vad",
    "threshold": 0.85,  # Higher number means less sensitive (0-1 range)
    "prefix_padding_ms": 500,  # Padding before speech
    "silence_duration_ms": 1200,  # Wait time for silence to ensure user is done speaking
    "create_response": auto_response,
    "interrupt_response": True  # Allow interruptions
}
```

- **threshold**: Controls sensitivity to background noise (0.85 = less sensitive, 0.6 = more sensitive)
- **silence_duration_ms**: How long to wait after detecting silence before concluding the user has finished speaking
- **prefix_padding_ms**: How much audio to include before speech is detected

#### Speech Detection Timing

Located in `src/audio_handler.py` in the `speech_stopped_handler` function:

```python
# Add a delay before resuming AI audio to accommodate slow speakers or brief pauses
async def delayed_resume():
    # Wait for additional time to ensure the user is really done speaking
    await asyncio.sleep(0.8)  # 800ms extra delay before AI starts speaking again
    
    # Check if we're still in a state where resuming makes sense
    if self.output_stream and not self.output_stream.is_active():
        logger.info("Delay period ended - resuming AI audio output")
        await self.resume_playback()
```

- The 800ms delay prevents the AI from responding too quickly after the user stops speaking
- This helps with slower speakers or people who pause briefly while thinking

### AI Instructions

Instructions are provided to the AI in two main ways:

1. **System Instructions**:
   - Defined in `src/system_instructions.py`
   - Used when configuring a new session through `configure_session`
   - Example: `APPOINTMENT_SCHEDULER` contains the full instructions for the appointment scheduling assistant

2. **Runtime Instructions**:
   - Can be passed to each response request using the `request_response` method:
   ```python
   await conversation.request_response(
       modalities=["text", "audio"],
       instructions="Additional specific instructions for this response"
   )
   ```

### Configuring Who Speaks First

#### AI Speaks First

To make the AI speak first (initiating the conversation), uncomment these lines in the `setup_conversation` function in `app.py`:

```python
# Send initial greeting message
await conversation.send_text_message(
    "Hello! I'm your appointment scheduling assistant. I can help you book appointments for our services including consultations, basic services, and premium services. Please let me know how I can assist you today."
)

# Request first response from the model
await conversation.request_response(modalities=["text", "audio"])
```

#### User Speaks First

To make the AI wait for the user to speak first (default behavior), ensure the above lines remain commented out. The AI will only respond after detecting user speech.

## Fine-Tuning Audio Sensitivity

### For Environments with Background Noise

If the AI responds to background noise too often:
- Increase the `threshold` value in VAD settings (up to 0.95)
- Increase `silence_duration_ms` (up to 1500ms)
- Increase the speech duration threshold in `speech_stopped_handler` (currently 1.0 seconds)

### For More Responsive Experience

If the AI seems slow to respond:
- Decrease the `silence_duration_ms` (down to 800ms)
- Decrease the delay in `delayed_resume` (from 0.8 to 0.5 seconds)
- Slightly lower the `threshold` (down to 0.8, but not too low to avoid triggering on background noise)

## Running the Application

### Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

Run the interactive voice assistant:
```
python app.py
```

Process an audio file:
```
python app.py --audio-file path/to/file.wav
```

## Extending the System

### Adding a New Service

1. Create a new service class in `src/services/` that inherits from `BaseService`
2. Implement required abstract methods (`connect`, `disconnect`, etc.)
3. Add specialized methods for the specific service

### Adding a New Repository

1. Create a new repository class in `src/data/` that inherits from `BaseRepository`
2. Implement CRUD operations for the specific database technology

### Adding a New Function

1. Define the function schema in `src/function_definitions.py`
2. Implement the function handler logic in `app.py`
3. Register the handler in the `run_interactive_session` function

## Debugging

For detailed logging of the communication with the OpenAI API:
- Check the `src/services/openai_service.py` file which logs all API interactions
- Review the logs in the `logs/` directory
- Set `LOG_LEVEL=DEBUG` in your `.env` file for more detailed logs

## Environment Configuration

Essential environment variables (stored in `.env`):
- `OPENAI_API_KEY`: Your OpenAI API key
- `AUDIO_INPUT_DEVICE`: (Optional) Index of the audio input device
- `AUDIO_OUTPUT_DEVICE`: (Optional) Index of the audio output device
- `LOG_LEVEL`: (Optional) Logging level (DEBUG, INFO, WARNING, ERROR) 