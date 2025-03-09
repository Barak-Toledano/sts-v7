from pydub import AudioSegment

# Load MP3 file
mp3_path = "sample.mp3"  # Replace with your MP3 file
audio = AudioSegment.from_mp3(mp3_path)

# Convert to PCM16 (16-bit, 24kHz, mono, little-endian)
audio = audio.set_channels(1)  # Mono
audio = audio.set_frame_rate(24000)  # 24kHz
audio = audio.set_sample_width(2)  # 16-bit (2 bytes per sample)

# Save as a raw PCM file (no header)
pcm_path = "converted_audio.raw"
audio.export(pcm_path, format="raw")

print(f"✅ Converted {mp3_path} to {pcm_path} in PCM16 format.")
