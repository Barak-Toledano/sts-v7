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

# File paths
INPUT_AUDIO_FILE = "converted_audio.raw"
OUTPUT_AUDIO_FILE = "output_audio.raw"

# ✅ Set up logging to a file
LOG_FILE = "logs.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

async def send_audio_and_request_response():
    extra_headers = [
        ("Authorization", f"Bearer {OPENAI_API_KEY}"),
        ("openai-beta", "realtime=v1")  # Required Beta Header
    ]

    async with websockets.connect(OPENAI_REALTIME_URL, extra_headers=extra_headers) as ws:
        logging.info("✅ Connected to OpenAI Realtime API")

        # Step 1: Wait for OpenAI's initial session response
        session_response = await ws.recv()
        logging.info(f"🔹 OpenAI Response: {session_response}")

        # Step 1.5: Update the session with English language enforcement
        update_session_event = {
            "type": "session.update",
            "session": {
                "instructions": "Always respond in English, regardless of detected language.",
                "modalities": ["audio", "text"],
                "voice": "alloy"  # Ensure the AI responds in speech
            }
        }
        await ws.send(json.dumps(update_session_event))
        logging.info("📤 Sent session.update request to enforce English.")

        # ✅ Wait for OpenAI's confirmation
        session_update_response = await ws.recv()
        logging.info(f"🔹 OpenAI Response (Session Update): {session_update_response}")

        # Step 2: Send audio
        chunk_size = 4096
        speech_detected = False

        with open(INPUT_AUDIO_FILE, "rb") as f:
            while not speech_detected:
                try:
                    server_response = await asyncio.wait_for(ws.recv(), timeout=0.05)  # ✅ Reduced timeout for better responsiveness
                    logging.info(f"🔹 OpenAI Response (Before Sending): {server_response}")

                    response_data = json.loads(server_response)
                    if response_data.get("type") in ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"]:
                        logging.info("🛑 Speech detected. Stopping audio send BEFORE sending the next chunk.")
                        speech_detected = True
                        break

                except asyncio.TimeoutError:
                    pass  

                audio_chunk = f.read(chunk_size)
                if not audio_chunk or speech_detected:
                    break  

                encoded_audio = base64.b64encode(audio_chunk).decode("utf-8")  
                audio_event = {"type": "input_audio_buffer.append", "audio": encoded_audio}
                await ws.send(json.dumps(audio_event))
                logging.info("📤 Sent audio chunk...")

        logging.info("✅ Finished sending audio!")

        # Step 3: Commit the audio buffer
        commit_event = {"type": "input_audio_buffer.commit"}
        await ws.send(json.dumps(commit_event))
        logging.info("📤 Sent commit request...")

        # Step 4: Explicitly request a response from OpenAI
        response_request = {
            "type": "response.create"
        }
        await ws.send(json.dumps(response_request))
        logging.info("📤 Sent response.create request...")

        # Step 5: Receive OpenAI's response and save it
        with open(OUTPUT_AUDIO_FILE, "wb") as output_file:
            response_received = False  # Track if we get any response

            while True:
                try:
                    response = await ws.recv()
                    logging.info(f"🔹 OpenAI Response: {response}")

                    response_data = json.loads(response)

                    if response_data.get("type") == "response.audio.delta":
                        response_received = True  # ✅ We got AI audio
                        audio_chunk = base64.b64decode(response_data["delta"])
                        output_file.write(audio_chunk)
                        logging.info(f"💾 Saved AI audio chunk ({len(audio_chunk)} bytes)")

                    elif response_data.get("type") == "response.stop":
                        logging.info("🛑 OpenAI finished responding.")
                        break  

                except websockets.exceptions.ConnectionClosed:
                    logging.error("🔴 Connection closed by OpenAI.")
                    break

            if not response_received:
                logging.error("❌ No `response.audio.delta` received from OpenAI!")

        # ✅ Explicitly close file after writing
        output_file.close()
        logging.info("✅ AI Response saved to file!")

asyncio.run(send_audio_and_request_response())
