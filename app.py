"""
Main application entry point for the OpenAI Realtime API Assistant.
"""
import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple

# Internal imports
from src.services.openai_service import OpenAIService
from src.domain.conversation_manager import ConversationManager
from src.domain.audio_manager import AudioManager
from src.system_instructions import APPOINTMENT_SCHEDULER
from src.function_definitions import APPOINTMENT_TOOLS
from src.data.models import Conversation, Message
from src.data.memory_repository import InMemoryRepository

# Configure logging
logger = logging.getLogger(__name__)

# Global variables for signal handling
shutdown_event = asyncio.Event()

async def handle_text_response(
    delta: str, 
    full_text: str, 
    response_id: str
) -> None:
    """
    Handle text response from the AI.
    
    Args:
        delta: New text chunk
        full_text: Complete text so far
        response_id: Response identifier
    """
    # Just print the delta to avoid repeating the full text
    print(delta, end="", flush=True)

async def handle_transcript(
    delta: str, 
    full_transcript: str, 
    part_id: str, 
    response_id: str
) -> None:
    """
    Handle transcript of audio from the AI.
    
    Args:
        delta: New transcript chunk
        full_transcript: Complete transcript so far
        part_id: Part identifier
        response_id: Response identifier
    """
    # We could display the transcript, but it's not necessary for this demo
    pass

async def handle_function_call(
    function_name: str, 
    arguments: Dict[str, Any], 
    call_id: str, 
    response_id: str,
    conversation_manager: ConversationManager
) -> None:
    """
    Handle function call from the AI.
    
    Args:
        function_name: Name of the function to call
        arguments: Function arguments
        call_id: Call identifier
        response_id: Response identifier
        conversation_manager: Conversation manager instance
    """
    logger.info(f"Function call: {function_name}")
    logger.info(f"Arguments: {arguments}")
    
    # In a real application, you'd implement actual function logic here
    result = {
        "status": "success",
        "message": f"Function {function_name} called successfully"
    }
    
    # Submit the result back to the AI
    await conversation_manager.submit_function_result(call_id, result)

async def handle_response_complete(response: Dict[str, Any]) -> None:
    """
    Handle response completion.
    
    Args:
        response: Response data including usage statistics
    """
    usage = response.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens
    
    print("\n[Response complete]")
    logger.info(f"Usage: {total_tokens} total tokens ({input_tokens} input, {output_tokens} output)")

async def handle_error(error_info: Dict[str, Any]) -> None:
    """
    Handle errors from the AI service.
    
    Args:
        error_info: Error information
    """
    error_type = error_info.get("type", "unknown")
    error_message = error_info.get("message", "Unknown error")
    
    logger.error(f"API Error: {error_type} - {error_message}")
    print(f"\nError: {error_message}")

async def setup_conversation(conversation: ConversationManager) -> None:
    """
    Set up the conversation with initial configuration
    
    Args:
        conversation: Conversation manager
    """
    # Configure the conversation
    success = await conversation.configure_session(
        instructions=APPOINTMENT_SCHEDULER,
        tools=APPOINTMENT_TOOLS,
        voice="alloy",  # Options include: alloy, echo, fable, onyx, nova, shimmer
        vad_enabled=True,
        auto_response=True
    )
    
    if not success:
        logger.error("Failed to configure conversation session")
        return
    
    # Don't send an initial greeting or request a response
    # The AI will only respond after the user speaks
    
    # NOTE: To make the AI speak first (without waiting for user input), uncomment these lines:
    # 
    # # Send initial greeting message
    # await conversation.send_text_message(
    #     "Hello! I'm your appointment scheduling assistant. I can help you book appointments for our services including consultations, basic services, and premium services. Please let me know how I can assist you today."
    # )
    # 
    # # Request first response from the model
    # await conversation.request_response(modalities=["text", "audio"])
    
    logger.info("Session configured successfully")

async def select_audio_devices() -> Tuple[Optional[int], Optional[int]]:
    """
    Allows user to select audio input and output devices
    
    Returns:
        Tuple[Optional[int], Optional[int]]: Selected input and output device indices
    """
    # Use PyAudio temporarily to list devices
    import pyaudio
    p = pyaudio.PyAudio()
    
    input_devices = []
    output_devices = []
    
    # List available devices
    print("\nAvailable Audio Devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        device_name = device_info.get("name", "Unknown Device")
        
        # Check if this is an input or output device
        if device_info.get("maxInputChannels", 0) > 0:
            print(f"  Input {i}: {device_name}")
            input_devices.append(i)
        if device_info.get("maxOutputChannels", 0) > 0:
            print(f"  Output {i}: {device_name}")
            output_devices.append(i)
    
    # Clean up PyAudio instance
    p.terminate()
    
    # Get default device indices from environment or use None
    default_input = os.environ.get("AUDIO_INPUT_DEVICE")
    default_output = os.environ.get("AUDIO_OUTPUT_DEVICE")
    
    if default_input:
        try:
            default_input = int(default_input)
            if default_input not in input_devices:
                default_input = None
        except ValueError:
            default_input = None
    
    if default_output:
        try:
            default_output = int(default_output)
            if default_output not in output_devices:
                default_output = None
        except ValueError:
            default_output = None
    
    # Interactive selection
    input_index = None
    output_index = None
    
    # Select input device
    if not input_devices:
        print("No input devices found!")
    elif default_input is not None:
        input_index = default_input
        print(f"Using default input device (index {input_index})")
    else:
        # Ask user to select
        valid_selection = False
        while not valid_selection:
            try:
                selection = input("\nSelect input device index (or press Enter for default): ").strip()
                if not selection:
                    # Use default
                    valid_selection = True
                else:
                    idx = int(selection)
                    if idx in input_devices:
                        input_index = idx
                        valid_selection = True
                    else:
                        print("Invalid device index. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Select output device
    if not output_devices:
        print("No output devices found!")
    elif default_output is not None:
        output_index = default_output
        print(f"Using default output device (index {output_index})")
    else:
        # Ask user to select
        valid_selection = False
        while not valid_selection:
            try:
                selection = input("\nSelect output device index (or press Enter for default): ").strip()
                if not selection:
                    # Use default
                    valid_selection = True
                else:
                    idx = int(selection)
                    if idx in output_devices:
                        output_index = idx
                        valid_selection = True
                    else:
                        print("Invalid device index. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    return input_index, output_index

async def run_interactive_session(config: Dict[str, Any]) -> None:
    """
    Run an interactive voice conversation session.
    
    Args:
        config: Application configuration
    """
    api_key = config.get("api_key")
    
    if not api_key:
        logger.error("API key not found. Please set OPENAI_API_KEY in your environment or .env file.")
        return
    
    # Select audio devices
    input_device, output_device = await select_audio_devices()
    
    # Initialize repositories (using in-memory for this example)
    conversation_repo = InMemoryRepository[Conversation]()
    message_repo = InMemoryRepository[Message]()
    
    # Connect to repositories
    await conversation_repo.connect()
    await message_repo.connect()
    
    # Create service instances
    openai_service = OpenAIService({"api_key": api_key})
    
    # Initialize managers
    conversation_manager = ConversationManager(openai_service)
    audio_manager = AudioManager(
        openai_service,
        input_device_index=input_device,
        output_device_index=output_device
    )
    
    # Connect to the OpenAI API
    if not await openai_service.connect():
        logger.error("Failed to connect to OpenAI API")
        return
    
    # Create a new session
    session_id = await openai_service.create_session()
    if not session_id:
        logger.error("Failed to create session")
        return
    
    # Register event handlers
    conversation_manager.register_event_handler("text_delta", handle_text_response)
    conversation_manager.register_event_handler("transcript_delta", handle_transcript)
    conversation_manager.register_event_handler("function_call", handle_function_call)
    conversation_manager.register_event_handler("response_complete", handle_response_complete)
    conversation_manager.register_event_handler("error", handle_error)
    
    # Define a function handler closure to pass conversation_manager
    async def bound_function_call_handler(function_name, arguments, call_id, response_id):
        await handle_function_call(
            function_name, 
            arguments, 
            call_id, 
            response_id, 
            conversation_manager
        )
    
    # Use the bound function handler instead
    conversation_manager.register_event_handler("function_call", bound_function_call_handler)
    
    # Configure the session
    await setup_conversation(conversation_manager)
    
    # Start audio streaming
    print("\nStarting audio streaming session...")
    print("Speak to interact with the assistant (press Ctrl+C to exit)...")
    
    await audio_manager.start_recording()
    await audio_manager.start_playback()
    
    # Wait for shutdown signal
    await shutdown_event.wait()
    
    # Clean up resources
    await audio_manager.cleanup()
    await conversation_manager.cleanup()
    await openai_service.disconnect()
    
    # Close repository connections
    await conversation_repo.disconnect()
    await message_repo.disconnect()
    
    print("Session ended by user.")

def handle_signals() -> None:
    """Register signal handlers for graceful shutdown."""
    
    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()
    
    # Handle Ctrl+C on Windows/Unix and SIGTERM
    try:
        if sys.platform == "win32":
            # Windows doesn't support SIGTERM
            signal.signal(signal.SIGINT, lambda sig, frame: signal_handler())
        else:
            # Unix-like systems
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        # Fallback for environments that don't support add_signal_handler
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler())
        # Try to register SIGTERM if available
        try:
            signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler())
        except AttributeError:
            pass

def validate_audio_file(file_path: str) -> bool:
    """
    Validate that an audio file exists and is in WAV format.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        bool: True if the file is valid
    """
    # Check file existence
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
    
    # Check file extension
    if not file_path.lower().endswith('.wav'):
        logger.error(f"File must be a WAV file: {file_path}")
        return False
    
    # Check if file is readable
    try:
        with open(file_path, 'rb') as _:
            pass
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return False
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        logger.error("File is empty")
        return False
    
    # Basic WAV file check
    try:
        import wave
        with wave.open(file_path, 'rb') as wf:
            # Ensure it's PCM format and has at least one frame
            if wf.getnframes() == 0:
                logger.error("WAV file has no frames")
                return False
    except Exception as e:
        logger.error(f"Not a valid WAV file: {e}")
        return False
    
    return True

async def process_audio_file(file_path: str, config: Dict[str, Any]) -> None:
    """
    Process an audio file and get a response from the AI.
    
    Args:
        file_path: Path to the audio file
        config: Application configuration
    """
    api_key = config.get("api_key")
    
    if not api_key:
        logger.error("API key not found. Please set OPENAI_API_KEY in your environment or .env file.")
        return
    
    if not validate_audio_file(file_path):
        return
    
    print(f"Processing audio file: {file_path}")
    
    # Initialize services and managers
    openai_service = OpenAIService({"api_key": api_key})
    conversation_manager = ConversationManager(openai_service)
    
    # Connect to the OpenAI API
    if not await openai_service.connect():
        logger.error("Failed to connect to OpenAI API")
        return
    
    # Create a new session
    session_id = await openai_service.create_session()
    if not session_id:
        logger.error("Failed to create session")
        return
    
    # Register event handlers
    conversation_manager.register_event_handler("text_delta", handle_text_response)
    conversation_manager.register_event_handler("response_complete", handle_response_complete)
    conversation_manager.register_event_handler("error", handle_error)
    
    # Define a function handler closure to pass conversation_manager
    async def bound_function_call_handler(function_name, arguments, call_id, response_id):
        await handle_function_call(
            function_name, 
            arguments, 
            call_id, 
            response_id, 
            conversation_manager
        )
    
    # Register function call handler
    conversation_manager.register_event_handler("function_call", bound_function_call_handler)
    
    # Configure the session
    await setup_conversation(conversation_manager)
    
    # Send the audio file
    audio_manager = AudioManager(openai_service)
    file_sent = await audio_manager.send_audio_file(file_path)
    
    if not file_sent:
        logger.error("Failed to send audio file")
        await openai_service.disconnect()
        return
    
    # Request a response
    response_id = await conversation_manager.request_response()
    if not response_id:
        logger.error("Failed to request response")
        await openai_service.disconnect()
        return
    
    # Wait for response to complete
    completed = await conversation_manager.wait_for_all_responses()
    
    if not completed:
        logger.warning("Response timed out")
    
    # Clean up
    await conversation_manager.cleanup()
    await openai_service.disconnect()

def load_config_file(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from .env file and/or environment variables.
    
    Args:
        config_path: Optional path to .env file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Load .env file if specified or default
    if config_path and os.path.exists(config_path):
        load_dotenv(config_path)
    else:
        # Try default locations
        if os.path.exists(".env"):
            load_dotenv(".env")
        elif os.path.exists("../.env"):
            load_dotenv("../.env")
    
    # Create config dictionary
    config = {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "log_level": os.environ.get("LOG_LEVEL", "INFO").upper(),
        "audio_input_device": os.environ.get("AUDIO_INPUT_DEVICE", None),
        "audio_output_device": os.environ.get("AUDIO_OUTPUT_DEVICE", None)
    }
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Add timestamp to log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["log_file"] = os.path.join(log_dir, f"assistant_{timestamp}.log")
    
    return config

def configure_logging(log_level: str) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"assistant_{timestamp}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set lower level for some verbose loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

def main():
    """
    Main application entry point.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OpenAI Realtime API Assistant")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--audio-file", help="Path to WAV file for processing")
    parser.add_argument("--log-level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_file(args.config)
    
    # Override log level from command line if specified
    if args.log_level:
        config["log_level"] = args.log_level
    
    # Configure logging
    configure_logging(config["log_level"])
    
    logger.info("Starting OpenAI Realtime Assistant")
    
    try:
        # Set up asyncio event loop
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Register signal handlers
        handle_signals()
        
        # Run appropriate mode
        if args.audio_file:
            # Process audio file mode
            asyncio.run(process_audio_file(args.audio_file, config))
        else:
            # Interactive session mode
            asyncio.run(run_interactive_session(config))
        
        logger.info("Session ended")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 