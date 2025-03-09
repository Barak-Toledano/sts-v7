# audio_test.py
import asyncio
import os
import signal  # Make sure signal is imported at the top level
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.realtime_client import RealtimeClient
from src.audio_handler import AudioHandler
from src.utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger("audio_test")

# Global shutdown event
shutdown_event = asyncio.Event()


def handle_signals():
    """Set up signal handlers for graceful shutdown"""
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


async def audio_test():
    """Test audio recording and playback with the Realtime API"""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    client = None
    audio = None
    
    try:
        # Create and connect client
        print("Connecting to OpenAI Realtime API...")
        client = RealtimeClient()
        connected = await client.connect()
        
        if not connected:
            print(f"Failed to connect: {client.connection_error}")
            return
        
        print("Connected successfully!")
        
        # Register callback for audio transcript
        async def transcript_handler(event_data):
            transcript = event_data.get("delta", "")
            if transcript:
                print(f"Transcript: {transcript}", end="", flush=True)
        
        client.register_event_callback("response.audio_transcript.delta", transcript_handler)
        
        # Create audio handler
        audio = AudioHandler(client)
        
        # List available devices and let user select
        devices = audio.list_audio_devices()
        
        print("\nAvailable input devices:")
        input_devices = []
        for i, device in enumerate(devices):
            if device.get("maxInputChannels", 0) > 0:
                input_devices.append((i, device.get("name")))
                print(f"  {len(input_devices)-1}: {device.get('name')}")
        
        print("\nAvailable output devices:")
        output_devices = []
        for i, device in enumerate(devices):
            if device.get("maxOutputChannels", 0) > 0:
                output_devices.append((i, device.get("name")))
                print(f"  {len(output_devices)-1}: {device.get('name')}")
        
        # Let user select devices
        input_idx = 0
        output_idx = 0
        
        try:
            choice = input("\nSelect input device (or press Enter for default): ")
            if choice.strip():
                input_idx = int(choice)
                if input_idx < 0 or input_idx >= len(input_devices):
                    print("Invalid selection, using default input device.")
                    input_idx = 0
        except ValueError:
            print("Invalid input, using default input device.")
        
        try:
            choice = input("Select output device (or press Enter for default): ")
            if choice.strip():
                output_idx = int(choice)
                if output_idx < 0 or output_idx >= len(output_devices):
                    print("Invalid selection, using default output device.")
                    output_idx = 0
        except ValueError:
            print("Invalid input, using default output device.")
        
        # Get actual device indices
        input_device_idx = input_devices[input_idx][0] if input_devices else None
        output_device_idx = output_devices[output_idx][0] if output_devices else None
        
        # Create audio handler with selected devices
        audio = AudioHandler(
            client,
            input_device_index=input_device_idx,
            output_device_index=output_device_idx
        )
        
        # Configure session
        await client.update_session({
            "instructions": "You are a helpful assistant for scheduling appointments. Keep your responses brief and clear.",
            "voice": "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
        })
        
        print("\nSession configured. Starting audio recording...")
        
        # Start recording
        recording_started = await audio.start_recording()
        
        if not recording_started:
            print("Failed to start recording. Check your microphone settings.")
            return
        
        print("\n🎤 Recording active! Speak into your microphone.")
        print("The assistant will respond when you pause speaking.")
        print("Press Ctrl+C to exit.")
        
        # Wait until shutdown
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Error in audio test: {e}")
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up resources
        print("\nCleaning up resources...")
        if audio:
            await audio.cleanup()
        if client and client.connected:
            await client.disconnect()
        print("Done!")

if __name__ == "__main__":
    print("Starting audio test with the OpenAI Realtime API...")
    handle_signals()
    asyncio.run(audio_test())