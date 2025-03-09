from pydub import AudioSegment

# Input and output file paths
input_raw_file = "output_audio.raw"  # OpenAI's PCM16 raw audio
output_mp3_file = "output_audio.mp3"  # MP3 output

# Load raw PCM16 file (24kHz, mono, little-endian)
audio = AudioSegment.from_raw(
    input_raw_file,
    sample_width=2,  # 16-bit PCM = 2 bytes per sample
    frame_rate=24000,  # OpenAI uses 24kHz
    channels=1  # Mono
)

# Export to MP3
audio.export(output_mp3_file, format="mp3", bitrate="192k")

print(f"✅ Converted {input_raw_file} to {output_mp3_file}")
