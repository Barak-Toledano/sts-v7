import pyaudio

# OpenAI PCM16 format
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1              # Mono
RATE = 24000              # 24kHz sample rate
CHUNK_SIZE = 4096         # Buffer size

RAW_AUDIO_FILE = "output_audio.raw"  # Path to AI-generated audio

def play_raw_audio():
    audio = pyaudio.PyAudio()

    # Open a stream for playback
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

    # Read and play audio in chunks
    with open(RAW_AUDIO_FILE, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            stream.write(chunk)

    # Cleanup
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    print("🔊 Playing AI-generated speech...")
    play_raw_audio()
    print("✅ Playback finished.")
