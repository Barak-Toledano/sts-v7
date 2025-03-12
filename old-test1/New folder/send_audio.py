import asyncio
import websockets
import json
import os
import base64
import logging
from dotenv import load_dotenv  # Load API key from .env

# Load .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI Realtime WebSocket URL
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

# Path to the PCM16 audio file
PCM16_AUDIO_FILE = "converted_audio.raw"

# ✅ Set up logging to a file
LOG_FILE = "logs.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

async def send_audio():
    extra_headers = [
        ("Authorization", f"Bearer {OPENAI_API_KEY}"),
        ("openai-beta", "realtime=v1")  # Required Beta Header
    ]

    async with websockets.connect(OPENAI_REALTIME_URL, extra_headers=extra_headers) as ws:
        logging.info("✅ Connected to OpenAI Realtime API")

        # Step 1: Wait for OpenAI's initial session response
        session_response = await ws.recv()
        logging.info(f"🔹 OpenAI Response: {session_response}")

        # Step 2: Read and send the PCM16 audio file in chunks
        chunk_size = 4096  # Adjust if needed
        speech_detected = False  # ✅ Track if OpenAI has started/stopped speech

        with open(PCM16_AUDIO_FILE, "rb") as f:
            while not speech_detected:
                # ✅ BEFORE sending, check for OpenAI response
                try:
                    server_response = await asyncio.wait_for(ws.recv(), timeout=0.1)  # Fast response check
                    logging.info(f"🔹 OpenAI Response (Before Sending): {server_response}")

                    response_data = json.loads(server_response)
                    
                    # ✅ STOP if OpenAI detected speech started
                    if response_data.get("type") in ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"]:
                        logging.info("🛑 Speech detected. Stopping audio send BEFORE sending the next chunk.")
                        speech_detected = True
                        break  # Stop sending audio immediately

                except asyncio.TimeoutError:
                    pass  # No response yet, continue

                # ✅ Read the next chunk but DO NOT send it if speech was detected
                audio_chunk = f.read(chunk_size)
                if not audio_chunk or speech_detected:
                    break  # Stop if there is no more audio to send or speech was detected

                encoded_audio = base64.b64encode(audio_chunk).decode("utf-8")  # Convert to Base64

                audio_event = {
                    "type": "input_audio_buffer.append",
                    "audio": encoded_audio  # Send Base64 encoded audio
                }

                await ws.send(json.dumps(audio_event))
                logging.info("📤 Sent audio chunk...")

        logging.info("✅ Finished sending audio!")

        # Step 3: Commit the audio buffer immediately after speech detection
        commit_event = {
            "type": "input_audio_buffer.commit"
        }
        await ws.send(json.dumps(commit_event))
        logging.info("📤 Sent commit request...")

        # Step 4: Listen for OpenAI responses
        while True:
            try:
                response = await ws.recv()
                logging.info(f"🔹 OpenAI Response: {response}")

                # Stop listening if OpenAI sends `response.stop`
                response_data = json.loads(response)
                if response_data.get("type") == "response.stop":
                    logging.info("🛑 OpenAI finished responding.")
                    break

            except websockets.exceptions.ConnectionClosed:
                logging.error("🔴 Connection closed by OpenAI.")
                break

asyncio.run(send_audio())
