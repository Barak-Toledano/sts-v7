# src/utils/audio_utils.py
import base64
import struct
import numpy as np
import wave
import io
import os
import logging
from typing import Tuple, Union, Optional, BinaryIO, List, Dict, Any

# Set up logger
logger = logging.getLogger("audio_utils")

def float_to_pcm16(float_audio: np.ndarray) -> bytes:
    """
    Convert floating point audio data to 16-bit PCM
    
    Args:
        float_audio: Floating point audio data in range [-1.0, 1.0]
        
    Returns:
        bytes: 16-bit PCM audio data
    """
    # Ensure input is within [-1.0, 1.0] range
    float_audio = np.clip(float_audio, -1.0, 1.0)
    
    # Convert to 16-bit PCM
    pcm_data = (float_audio * 32767.0).astype(np.int16)
    
    # Convert to bytes (little-endian)
    return pcm_data.tobytes()

def pcm16_to_float(pcm_data: bytes) -> np.ndarray:
    """
    Convert 16-bit PCM audio data to floating point
    
    Args:
        pcm_data: 16-bit PCM audio data
        
    Returns:
        np.ndarray: Floating point audio data in range [-1.0, 1.0]
    """
    # Convert bytes to int16 array
    int_data = np.frombuffer(pcm_data, dtype=np.int16)
    
    # Convert to float in range [-1.0, 1.0]
    return int_data.astype(np.float32) / 32767.0

def base64_encode_audio(audio_data: Union[bytes, np.ndarray]) -> str:
    """
    Encode audio data as Base64 string
    
    Args:
        audio_data: Audio data as bytes or numpy array
        
    Returns:
        str: Base64-encoded string
    """
    if isinstance(audio_data, np.ndarray):
        # Convert float array to PCM16 bytes
        audio_bytes = float_to_pcm16(audio_data)
    else:
        # Already bytes
        audio_bytes = audio_data
    
    # Encode as Base64
    return base64.b64encode(audio_bytes).decode('ascii')

def base64_decode_audio(base64_str: str) -> bytes:
    """
    Decode Base64 string to audio bytes
    
    Args:
        base64_str: Base64-encoded string
        
    Returns:
        bytes: Audio data
    """
    return base64.b64decode(base64_str)

def read_wav_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Read a WAV file and return the audio data and sample rate
    
    Args:
        file_path: Path to WAV file
        
    Returns:
        Tuple[np.ndarray, int]: Audio data as float array and sample rate
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            # Get file properties
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read all frames
            pcm_data = wf.readframes(n_frames)
            
            # Convert to float
            float_data = pcm16_to_float(pcm_data)
            
            # Reshape for multi-channel
            if channels > 1:
                float_data = float_data.reshape(-1, channels)
            
            return float_data, sample_rate
            
    except Exception as e:
        logger.error(f"Error reading WAV file {file_path}: {e}")
        raise

def write_wav_file(file_path: str, audio_data: Union[bytes, np.ndarray], 
                 sample_rate: int, channels: int = 1) -> bool:
    """
    Write audio data to a WAV file
    
    Args:
        file_path: Path to save WAV file
        audio_data: Audio data as bytes or numpy array
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if isinstance(audio_data, np.ndarray):
            # Convert float array to PCM16 bytes
            audio_bytes = float_to_pcm16(audio_data)
        else:
            # Already bytes
            audio_bytes = audio_data
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        
        return True
    
    except Exception as e:
        logger.error(f"Error writing WAV file {file_path}: {e}")
        return False

def validate_wav_file(file_path: str, expected_rate: Optional[int] = None, 
                    expected_channels: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a WAV file against expected parameters
    
    Args:
        file_path: Path to WAV file
        expected_rate: Expected sample rate (or None to skip check)
        expected_channels: Expected number of channels (or None to skip check)
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, file_info)
    """
    try:
        if not os.path.exists(file_path):
            return False, {"error": "File not found"}
            
        if not file_path.lower().endswith('.wav'):
            return False, {"error": "Not a WAV file"}
        
        with wave.open(file_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            file_info = {
                "channels": channels,
                "sample_width": sample_width,
                "sample_rate": sample_rate,
                "n_frames": n_frames,
                "duration": n_frames / sample_rate
            }
            
            # Validate parameters if provided
            if expected_rate and sample_rate != expected_rate:
                file_info["warning"] = f"Sample rate mismatch: expected {expected_rate}, got {sample_rate}"
                
            if expected_channels and channels != expected_channels:
                file_info["warning"] = f"Channel count mismatch: expected {expected_channels}, got {channels}"
                
            # Check for 16-bit PCM format
            if sample_width != 2:
                file_info["warning"] = f"Sample width is {sample_width * 8}-bit, expected 16-bit"
            
            return True, file_info
            
    except Exception as e:
        logger.error(f"Error validating WAV file {file_path}: {e}")
        return False, {"error": str(e)}

def chunk_audio(audio_data: bytes, chunk_size: int) -> List[bytes]:
    """
    Split audio data into chunks of specified size
    
    Args:
        audio_data: Audio data as bytes
        chunk_size: Size of each chunk in bytes
        
    Returns:
        List[bytes]: List of audio chunks
    """
    return [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]