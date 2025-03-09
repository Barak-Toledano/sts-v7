# simple_test.py
import asyncio
import json
import os
import logging
import websockets
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simple_test.log')
    ]
)
logger = logging.getLogger("simple_test")

async def simple_openai_realtime_test():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    # API endpoint
    model = "gpt-4o-realtime-preview-2024-12-17"
    url = f"wss://api.openai.com/v1/realtime?model={model}"
    
    # Headers for authentication
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1"
    }
    
    logger.info(f"Connecting to {url}")
    
    try:
        # Connect to WebSocket
        async with websockets.connect(url, extra_headers=headers) as websocket:
            logger.info("Connected to WebSocket successfully")
            
            # First, let's listen for any initial messages
            logger.info("Waiting for initial messages...")
            try:
                # Set a timeout for initial messages
                initial_message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                logger.info(f"Received initial message: {initial_message}")
                
                # Try to parse as JSON
                try:
                    initial_data = json.loads(initial_message)
                    logger.info(f"Parsed initial message: {json.dumps(initial_data, indent=2)}")
                except json.JSONDecodeError:
                    logger.warning("Initial message is not valid JSON")
                    
            except asyncio.TimeoutError:
                logger.info("No initial messages received within timeout")
            
            # Send session.update event
            session_update = {
                "type": "session.update",
                "session": {
                    "instructions": "You are a helpful assistant."
                }
            }
            
            logger.info(f"Sending session.update: {json.dumps(session_update)}")
            await websocket.send(json.dumps(session_update))
            
            # Wait for response to session.update
            try:
                update_response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                logger.info(f"Received session.update response: {update_response}")
                
                # Try to parse as JSON
                try:
                    update_data = json.loads(update_response)
                    logger.info(f"Parsed session.update response: {json.dumps(update_data, indent=2)}")
                except json.JSONDecodeError:
                    logger.warning("Session update response is not valid JSON")
                    
            except asyncio.TimeoutError:
                logger.info("No session.update response received within timeout")
            
            # Create a simple text message
            message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Hello, can you hear me?"
                        }
                    ]
                }
            }
            
            logger.info(f"Sending message: {json.dumps(message)}")
            await websocket.send(json.dumps(message))
            
            # Wait for message response
            try:
                message_response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                logger.info(f"Received message response: {message_response}")
                
                # Try to parse as JSON
                try:
                    message_data = json.loads(message_response)
                    logger.info(f"Parsed message response: {json.dumps(message_data, indent=2)}")
                except json.JSONDecodeError:
                    logger.warning("Message response is not valid JSON")
            except asyncio.TimeoutError:
                logger.info("No message response received within timeout")
            
            # Request a model response
            response_request = {
                "type": "response.create",
                "response": {
                    "modalities": ["text"]
                }
            }
            
            logger.info(f"Requesting response: {json.dumps(response_request)}")
            await websocket.send(json.dumps(response_request))
            
            # Listen for any responses for a longer period
            logger.info("Listening for model responses...")
            try:
                # Wait for multiple responses
                start_time = asyncio.get_event_loop().time()
                end_time = start_time + 10.0  # Listen for up to 10 seconds
                
                while asyncio.get_event_loop().time() < end_time:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        logger.info(f"Received response: {response}")
                        
                        # Try to parse as JSON
                        try:
                            response_data = json.loads(response)
                            logger.info(f"Parsed response: {json.dumps(response_data, indent=2)}")
                            
                            # If this is a text delta, print it
                            if response_data.get("type") == "response.text.delta":
                                delta = response_data.get("delta", "")
                                print(delta, end="", flush=True)
                                
                        except json.JSONDecodeError:
                            logger.warning("Response is not valid JSON")
                            
                    except asyncio.TimeoutError:
                        # Just continue the loop
                        pass
                    
            except Exception as e:
                logger.error(f"Error during response listening: {e}")
            
            logger.info("Finished listening for responses")
            
    except Exception as e:
        logger.error(f"Error connecting to WebSocket: {e}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    print("Starting simple OpenAI Realtime API test...")
    asyncio.run(simple_openai_realtime_test())
    print("\nTest completed. Check simple_test.log for detailed logs.")