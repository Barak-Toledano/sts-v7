# demo_fixed.py - A simpler demo to test Realtime API
import asyncio
import os
import json
import logging
from dotenv import load_dotenv

from src.realtime_client import RealtimeClient
from src.utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger("demo_fixed")

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Create client and connect
    print("Connecting to OpenAI Realtime API...")
    client = RealtimeClient()
    connected = await client.connect()
    
    if not connected:
        print(f"Failed to connect: {client.connection_error}")
        return
    
    print("Connected successfully!")
    
    # Create callback for text deltas
    async def handle_text_delta(event_data):
        delta = event_data.get("delta", "")
        if delta:
            print(delta, end="", flush=True)
    
    # Register callback
    client.register_event_callback("response.text.delta", handle_text_delta)
    
    try:
        # Update session configuration
        await client.update_session({"instructions": "You are a helpful assistant."})
        print("\nSession updated.")
        
        # Send a message
        print("\nSending message...")
        message = {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "I need to schedule an appointment for next Tuesday"
                }
            ]
        }
        await client.create_conversation_item(message)
        
        # Request a response
        print("\nWaiting for response:\n")
        await client.create_response(modalities=["text"])
        
        # Wait for response to complete
        await asyncio.sleep(10)  # Give it time to respond
        
        # Send a follow-up message
        print("\n\nSending follow-up message...")
        follow_up = {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "I'd like a basic service at 10:30 AM. My name is John Smith."
                }
            ]
        }
        await client.create_conversation_item(follow_up)
        
        # Request another response
        print("\nWaiting for response:\n")
        await client.create_response(modalities=["text"])
        
        # Wait longer for the final response
        await asyncio.sleep(10)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Disconnect
        print("\n\nDisconnecting...")
        await client.disconnect()
        print("Disconnected!")

if __name__ == "__main__":
    print("Starting simplified demo...")
    asyncio.run(main())