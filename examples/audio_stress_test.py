# examples/audio_stress_test.py
import os
import sys
import asyncio
import signal
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.realtime_client import RealtimeClient
from src.audio_handler import AudioHandler
from src.utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger("audio_stress_test")

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

async def connection_monitor(client, interval=5.0):
    """Monitor connection status and attempt reconnection if needed"""
    while not shutdown_event.is_set():
        if not client.connected:
            logger.warning("Connection monitor detected disconnection, attempting to reconnect...")
            connected = await client.connect()
            if connected:
                logger.info("Connection monitor successfully reconnected")
            else:
                logger.error("Connection monitor failed to reconnect")
        
        # Wait for the next check
        await asyncio.sleep(interval)

async def main():
    """Run the audio stress test"""
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
        
        # Start connection monitor
        monitor_task = asyncio.create_task(connection_monitor(client))
        
        # Register callbacks
        transcript_text = ""
        
        async def transcript_handler(event_data):
            nonlocal transcript_text
            delta = event_data.get("delta", "")
            if delta:
                transcript_text += delta
                print(f"\rTranscript: {transcript_text}", end="", flush=True)
        
        client.register_event_callback("response.audio_transcript.delta", transcript_handler)
        
        # Create audio handler with default devices
        audio = AudioHandler(client)
        
        # Configure session
        await client.update_session({
            "instructions": "You are a helpful assistant for scheduling appointments. Keep your responses brief and clear.",
            "voice": "alloy"
        })
        
        print("\nSession configured. Starting audio recording...")
        
        # Start recording
        await audio.start_recording()
        
        print("\n🎤 Recording active! Speak into your microphone.")
        print("Connection will be tested for stability.")
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
    print("Starting audio stress test...")
    handle_signals()
    asyncio.run(main())