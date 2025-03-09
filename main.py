# In main.py, add these imports at the top
import asyncio
import os
import logging
import argparse
import signal
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv
import wave

from src.utils.logging_utils import setup_logger
from src.config import settings
from src.realtime_client import RealtimeClient
from src.audio_handler import AudioHandler
from src.conversation import ConversationManager

# Set up logging
logger = setup_logger("main")

# Global variables for handling graceful shutdown
shutdown_event = asyncio.Event()
reconnect_attempt = 0
MAX_RECONNECT_ATTEMPTS = 5

async def handle_text_response(
    delta: str, 
    full_text: str, 
    response_id: str
) -> None:
    """
    Handler for text response events
    
    Args:
        delta: New text segment
        full_text: Complete text so far
        response_id: Response ID
    """
    # Print just the delta for a streaming effect
    print(delta, end="", flush=True)

async def handle_transcript(
    delta: str, 
    full_transcript: str, 
    part_id: str, 
    response_id: str
) -> None:
    """
    Handler for audio transcript events
    
    Args:
        delta: New transcript segment
        full_transcript: Complete transcript so far
        part_id: Audio part ID
        response_id: Response ID
    """
    # We could print the transcript, but for a cleaner UI, we'll skip it
    # Uncomment if you want to see the transcript
    # print(f"\nTranscript: {delta}", end="", flush=True)
    pass

async def handle_function_call(
    function_name: str, 
    arguments: Dict[str, Any], 
    call_id: str, 
    response_id: str,
    conversation_manager: ConversationManager
) -> None:
    """
    Handler for function call events
    
    Args:
        function_name: Name of the function
        arguments: Function arguments
        call_id: Call ID
        response_id: Response ID
        conversation_manager: Conversation manager for sending results
    """
    print(f"\n[Function Call] {function_name}({json.dumps(arguments, indent=2)})")
    
    # Sample implementation of function call handling
    try:
        result = {}
        
        if function_name == "check_availability":
            # In a real implementation, this would query a database
            date = arguments.get("date", "")
            service_type = arguments.get("service_type", "")
            
            logger.info(f"Checking availability for date: {date}, service: {service_type}")
            
            # Mock data - in real app would come from database/API
            result = {
                "available_slots": [
                    {"time": "09:00", "duration": 30},
                    {"time": "10:00", "duration": 60},
                    {"time": "14:00", "duration": 120}
                ],
                "date": date,
                "service_type": service_type
            }
            
        elif function_name == "schedule_appointment":
            # In a real implementation, this would write to a database
            date = arguments.get("date", "")
            time = arguments.get("time", "")
            name = arguments.get("name", "")
            service_type = arguments.get("service_type", "")
            
            logger.info(f"Scheduling appointment for {name} on {date} at {time} for {service_type}")
            
            # Mock confirmation - in real app would confirm with database/API
            result = {
                "confirmation_id": f"APPT-{int(time.time())}",
                "status": "confirmed",
                "appointment_details": {
                    "date": date,
                    "time": time,
                    "name": name,
                    "service_type": service_type
                }
            }
        else:
            logger.warning(f"Unknown function: {function_name}")
            result = {"error": f"Unknown function: {function_name}"}
        
        # Send the result back to the conversation
        await conversation_manager.submit_function_result(call_id, result)
        
    except Exception as e:
        error_msg = f"Error executing function {function_name}: {str(e)}"
        logger.error(error_msg)
        # Send error back to the conversation
        await conversation_manager.submit_function_result(call_id, {"error": error_msg})

async def handle_response_complete(response: Dict[str, Any]) -> None:
    """
    Handler for response complete events
    
    Args:
        response: Complete response data
    """
    print("\n[Response complete]")
    
    # Log usage statistics if available
    usage = response.get("usage")
    if usage:
        total_tokens = usage.get("total_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        logger.info(f"Usage: {total_tokens} total tokens ({input_tokens} input, {output_tokens} output)")

async def handle_error(error_info: Dict[str, Any]) -> None:
    """
    Handler for error events
    
    Args:
        error_info: Error information
    """
    message = error_info.get("message", "Unknown error")
    exception = error_info.get("exception")
    error_data = error_info.get("error_data")
    
    error_str = f"[Error] {message}"
    if exception:
        error_str += f" - Exception: {type(exception).__name__}: {str(exception)}"
    
    print(f"\n{error_str}")
    
    # Check if this is a connection error that requires reconnection
    if error_data and "connection" in message.lower():
        logger.warning("Connection error detected, attempting reconnection")
        # Here we could trigger reconnection logic

async def setup_conversation(conversation: ConversationManager) -> None:
    """
    Set up the conversation with initial configuration
    
    Args:
        conversation: Conversation manager
    """
    # Example tool definition for appointment scheduling
    tools = [
        {
            "type": "function",
            "name": "check_availability",
            "description": "Check available appointment slots",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string", 
                        "description": "Date to check in YYYY-MM-DD format"
                    },
                    "service_type": {
                        "type": "string",
                        "description": "Type of service needed",
                        "enum": ["Consultation", "Basic service", "Premium service"]
                    }
                },
                "required": ["date"]
            }
        },
        {
            "type": "function",
            "name": "schedule_appointment",
            "description": "Schedule an appointment",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Appointment date in YYYY-MM-DD format"
                    },
                    "time": {
                        "type": "string",
                        "description": "Appointment time in HH:MM format"
                    },
                    "name": {
                        "type": "string",
                        "description": "Customer name"
                    },
                    "service_type": {
                        "type": "string",
                        "description": "Type of service needed",
                        "enum": ["Consultation", "Basic service", "Premium service"]
                    }
                },
                "required": ["date", "time", "name", "service_type"]
            }
        }
    ]
    
    # System instructions for the appointment agent
    instructions = """
    You are an appointment scheduling assistant. Your goal is to help users schedule appointments.
    
    Available services:
    - Consultation (30 minutes)
    - Basic service (1 hour)
    - Premium service (2 hours)
    
    When scheduling appointments:
    1. First determine what service the user needs
    2. Check availability for that service using the check_availability function
    3. Help the user select a time and schedule the appointment with schedule_appointment
    
    Be friendly, professional, and efficient. Always confirm details before finalizing an appointment.
    
    Always speak naturally like a human appointment scheduler would speak.
    """
    
    # Configure the conversation
    success = await conversation.configure_session(
        instructions=instructions,
        tools=tools,
        voice="alloy",  # Options include: alloy, echo, fable, onyx, nova, shimmer
        vad_enabled=True,
        auto_response=True
    )
    
    if not success:
        logger.error("Failed to configure conversation session")
        return
    
    # Send initial message
    initial_prompt = "I'm your appointment scheduling assistant. How can I help you today?"
    await conversation.send_text_message(initial_prompt)
    await conversation.request_response(modalities=["text", "audio"])

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
    print("------------------------")
    
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev.get('name', 'Unknown Device')
        max_input = dev.get('maxInputChannels', 0)
        max_output = dev.get('maxOutputChannels', 0)
        
        if max_input > 0:
            input_devices.append((i, name))
            print(f"Input  {i}: {name}")
            
        if max_output > 0:
            output_devices.append((i, name))
            print(f"Output {i}: {name}")
    
    # Clean up PyAudio
    p.terminate()
    
    # Default to system default devices
    input_device_index = None
    output_device_index = None
    
    # Ask user to select input device
    if input_devices:
        try:
            choice = input("\nSelect input device # (press Enter for default): ")
            if choice.strip():
                input_choice = int(choice)
                if any(idx == input_choice for idx, _ in input_devices):
                    input_device_index = input_choice
                else:
                    print("Invalid selection, using default input device.")
        except ValueError:
            print("Invalid input, using default input device.")
    else:
        print("No input devices found.")
    
    # Ask user to select output device
    if output_devices:
        try:
            choice = input("Select output device # (press Enter for default): ")
            if choice.strip():
                output_choice = int(choice)
                if any(idx == output_choice for idx, _ in output_devices):
                    output_device_index = output_choice
                else:
                    print("Invalid selection, using default output device.")
        except ValueError:
            print("Invalid input, using default output device.")
    else:
        print("No output devices found.")
    
    return input_device_index, output_device_index


# In the run_interactive_session function, improve the interaction flow:
async def run_interactive_session(config: Dict[str, Any]) -> None:
    """
    Run an interactive speech-to-speech session
    
    Args:
        config: Application configuration
    """
    global reconnect_attempt
    
    client = None
    audio = None
    conversation = None
    
    try:
        # Get audio device selection
        input_device, output_device = await select_audio_devices() if config.get("select_devices", True) else (None, None)
        
        # Create the Realtime client
        client = RealtimeClient()
        
        # Set up reconnection logic
        while not client.connected and reconnect_attempt < MAX_RECONNECT_ATTEMPTS:
            # Connect to the API
            logger.info(f"Connecting to OpenAI Realtime API (attempt {reconnect_attempt + 1}/{MAX_RECONNECT_ATTEMPTS})...")
            connected = await client.connect()
            
            if not connected:
                reconnect_attempt += 1
                logger.error(f"Failed to connect to OpenAI Realtime API (attempt {reconnect_attempt}/{MAX_RECONNECT_ATTEMPTS})")
                
                if reconnect_attempt < MAX_RECONNECT_ATTEMPTS:
                    # Exponential backoff: wait longer after each failed attempt
                    backoff = min(30, 2 ** reconnect_attempt)
                    logger.info(f"Retrying in {backoff} seconds...")
                    await asyncio.sleep(backoff)
                else:
                    logger.error("Maximum reconnection attempts reached. Giving up.")
                    return
            else:
                reconnect_attempt = 0  # Reset counter on successful connection
                
        if not client.connected:
            logger.error("Failed to connect to OpenAI Realtime API")
            return
            
        logger.info("Connected to OpenAI Realtime API")
        
        # Create conversation manager
        conversation = ConversationManager(client)
        
        # Create a bound function call handler with access to the conversation manager
        async def bound_function_call_handler(function_name, arguments, call_id, response_id):
            await handle_function_call(function_name, arguments, call_id, response_id, conversation)
        
        # Register event handlers
        conversation.register_event_handler("on_text_response", handle_text_response)
        conversation.register_event_handler("on_transcript", handle_transcript)
        conversation.register_event_handler("on_function_call", bound_function_call_handler)
        conversation.register_event_handler("on_response_complete", handle_response_complete)
        conversation.register_event_handler("on_error", handle_error)
        
        # Set up the conversation
        logger.info("Configuring session...")
        await setup_conversation(conversation)
        logger.info("Session configured successfully")
        
        # Create audio handler with selected devices
        audio = AudioHandler(
            client, 
            input_device_index=input_device,
            output_device_index=output_device
        )
        
        # List available audio devices
        devices = audio.list_audio_devices()
        print(f"Found {len(devices)} audio devices:")
        for i, device in enumerate(devices):
            name = device.get("name", "Unknown")
            inputs = device.get("maxInputChannels", 0)
            outputs = device.get("maxOutputChannels", 0)
            if inputs > 0:
                print(f"  Input {i}: {name}")
            if outputs > 0:
                print(f"  Output {i}: {name}")
        
        # Start real audio streaming
        print("\nStarting audio streaming session...")
        print("Speak to interact with the assistant (press Ctrl+C to exit)...")
        
        # Start audio recording and playback
        await audio.start_recording()
        await audio.start_playback()
        
        # Initial greeting via text to start the conversation
        await conversation.send_text_message("Hello, I need help scheduling an appointment.")
        await conversation.request_response(modalities=["text", "audio"])
        
        # Main loop - keep the session running until interrupted
        while not shutdown_event.is_set():
            await asyncio.sleep(0.1)  # Small sleep to prevent CPU hogging
            
        print("\nSession ended by user.")
        
    except Exception as e:
        logger.error(f"Error in interactive session: {type(e).__name__}: {e}", exc_info=True)
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up resources
        logger.info("Shutting down...")
        if audio:
            await audio.cleanup()
        if conversation:
            await conversation.cleanup()
        if client and client.connected:
            await client.disconnect()
        
        logger.info("Session ended")

def handle_signals() -> None:
    """Set up signal handlers for graceful shutdown"""
    # Import signal module at the top of the function for Windows compatibility
    import signal
    
    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()
    
    # Handle Ctrl+C (SIGINT)
    try:
        asyncio.get_event_loop().add_signal_handler(
            signal.SIGINT, 
            signal_handler
        )
        
        # Handle SIGTERM
        asyncio.get_event_loop().add_signal_handler(
            signal.SIGTERM, 
            signal_handler
        )
    except NotImplementedError:
        # For Windows compatibility where add_signal_handler is not supported
        signal.signal(signal.SIGINT, lambda s, f: signal_handler())
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler())

def validate_audio_file(file_path: str) -> bool:
    """
    Validate that an audio file is in a compatible format
    
    Args:
        file_path: Path to audio file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        # Check if it's a WAV file
        if not file_path.lower().endswith('.wav'):
            logger.error(f"File must be a WAV file: {file_path}")
            return False
            
        # Open and check WAV file properties
        with wave.open(file_path, 'rb') as wf:
            # Check channels
            channels = wf.getnchannels()
            if channels != settings.CHANNELS:
                logger.warning(f"File has {channels} channels, but API expects {settings.CHANNELS}")
                
            # Check sample rate
            sample_rate = wf.getframerate()
            if sample_rate != settings.SAMPLE_RATE:
                logger.warning(f"File sample rate is {sample_rate}Hz, but API expects {settings.SAMPLE_RATE}Hz")
                
            # Check bit depth (16-bit PCM)
            sample_width = wf.getsampwidth()
            if sample_width != 2:  # 2 bytes = 16 bits
                logger.warning(f"File has {sample_width * 8}-bit samples, but API expects 16-bit PCM")
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating audio file: {e}")
        return False

async def process_audio_file(file_path: str, config: Dict[str, Any]) -> None:
    """
    Process an audio file and get a response
    
    Args:
        file_path: Path to the audio file
        config: Application configuration
    """
    client = None
    audio = None
    conversation = None
    
    try:
        # Validate the audio file
        if not validate_audio_file(file_path):
            print(f"Error: Invalid audio file: {file_path}")
            return
            
        # Create and connect the client
        client = RealtimeClient()
        connected = await client.connect()
        
        if not connected:
            logger.error("Failed to connect to OpenAI Realtime API")
            print("Error: Failed to connect to OpenAI Realtime API")
            return
            
        # Create conversation manager
        conversation = ConversationManager(client)
        
        # Create a bound function call handler with access to the conversation manager
        async def bound_function_call_handler(function_name, arguments, call_id, response_id):
            await handle_function_call(function_name, arguments, call_id, response_id, conversation)
        
        # Register event handlers
        conversation.register_event_handler("on_text_response", handle_text_response)
        conversation.register_event_handler("on_function_call", bound_function_call_handler)
        conversation.register_event_handler("on_response_complete", handle_response_complete)
        conversation.register_event_handler("on_error", handle_error)
        
        # Configure session
        await setup_conversation(conversation)
        
        # Create audio handler
        audio = AudioHandler(client)
        
        # Process the file
        logger.info(f"Processing audio file: {file_path}")
        success = await audio.send_audio_file(file_path)
        
        if not success:
            logger.error("Failed to process audio file")
            print("Error: Failed to process audio file")
            return
            
        # Generate output filename
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"response_{file_name}")
        
        # Save response to file
        logger.info(f"Saving response to: {output_path}")
        success = await audio.save_output_to_file(output_path)
        
        if not success:
            logger.error("Failed to save response audio")
            print("Error: Failed to save response audio")
        else:
            print(f"\nResponse saved to: {output_path}")
        
        # Wait for all responses to complete
        await conversation.wait_for_all_responses(timeout=60.0)
        
    except Exception as e:
        logger.error(f"Error processing audio file: {type(e).__name__}: {e}")
        print(f"Error processing audio file: {e}")
    finally:
        # Clean up resources
        if audio:
            await audio.cleanup()
        if conversation:
            await conversation.cleanup()
        if client and client.connected:
            await client.disconnect()

def load_config_file(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Default configuration
    config = {
        "log_level": "INFO",
        "select_devices": True,
        "output_dir": "output",
        "model": "gpt-4o-realtime-preview-2024-12-17",
        "voice": "alloy"
    }
    
    # If config path is provided, load and merge with defaults
    if config_path:
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            print(f"Warning: Failed to load config file: {e}")
    
    return config

def configure_logging(log_level: str) -> None:
    """
    Configure global logging settings
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    try:
        level = getattr(logging, log_level.upper())
    except AttributeError:
        print(f"Invalid log level: {log_level}, using INFO")
        level = logging.INFO
        
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )

# In main.py, modify the logging setup to use session ID

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-time Speech-to-Speech System")
    parser.add_argument("--file", help="Process an audio file instead of interactive mode")
    parser.add_argument("--config", help="Path to configuration file (JSON)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="INFO", help="Set logging level")
    parser.add_argument("--output-dir", default="output", help="Directory for output files")
    parser.add_argument("--no-device-select", action="store_true", 
                        help="Skip audio device selection in interactive mode")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load config file if provided, otherwise use defaults
    config = load_config_file(args.config)
    
    # Override config with command line arguments
    if args.log_level:
        config["log_level"] = args.log_level
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.no_device_select:
        config["select_devices"] = False
    
    # Create a session ID for this run
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Set up logger with session ID
    logger = setup_logger("main", config["log_level"], session_id)
    logger.info(f"Starting new session: {session_id}")
    
    # Configure other loggers with the same session ID
    for module in ["realtime_client", "audio_handler", "conversation"]:
        setup_logger(module, config["log_level"], session_id)
    
    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Set up signal handlers
    handle_signals()
    
    # Run the appropriate mode
    if args.file:
        if os.path.exists(args.file):
            asyncio.run(process_audio_file(args.file, config))
        else:
            print(f"Error: File not found: {args.file}")
    else:
        asyncio.run(run_interactive_session(config))

if __name__ == "__main__":
    main()