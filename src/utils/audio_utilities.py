"""
Audio utility functions for the OpenAI Realtime Assistant.

This module provides helper functions for working with audio data,
including format conversion, validation, and signal processing.
"""

import io
import logging
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from src.config.logging_config import get_logger
from src.utils.error_handling import AudioError, safe_execute

logger = get_logger(__name__)


def validate_audio_format(
    sample_rate: int,
    channels: int,
    sample_width: int
) -> bool:
    """
    Validate that audio format is supported by OpenAI's API.
    
    Args:
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Sample width in bytes
        
    Returns:
        bool: True if format is supported
    """
    # Check sample rate (OpenAI requires 24000 Hz)
    if sample_rate != 24000:
        logger.warning(f"Sample rate {sample_rate} Hz may not be supported. OpenAI requires 24000 Hz.")
        return False
    
    # Check channels (should be mono)
    if channels != 1:
        logger.warning(f"Channel count {channels} may not be supported. OpenAI requires mono audio.")
        return False
    
    # Check sample width (should be 16-bit)
    if sample_width != 2:  # 2 bytes = 16 bits
        logger.warning(f"Sample width {sample_width * 8} bits may not be supported. OpenAI requires 16-bit PCM.")
        return False
    
    return True


def convert_to_pcm16(
    audio_data: Union[bytes, np.ndarray],
    source_sample_rate: Optional[int] = None,
    target_sample_rate: int = 24000
) -> bytes:
    """
    Convert audio data to 16-bit PCM format at the required sample rate.
    
    Args:
        audio_data: Audio data as bytes or numpy array
        source_sample_rate: Original sample rate (if resampling is needed)
        target_sample_rate: Target sample rate (default: 24000 Hz for OpenAI)
        
    Returns:
        bytes: Converted audio data in 16-bit PCM format
        
    Raises:
        AudioError: If conversion fails
    """
    try:
        # Handle different input types
        if isinstance(audio_data, bytes):
            # If already bytes, check if it needs conversion
            if source_sample_rate is None or source_sample_rate == target_sample_rate:
                # No conversion needed
                return audio_data
            else:
                # Convert bytes to numpy array for resampling
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Convert to float for resampling
                audio_float = audio_array.astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            # If numpy array, normalize if needed
            if np.issubdtype(audio_data.dtype, np.floating):
                # Already float, just ensure it's normalized to [-1, 1]
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_float = audio_data / np.max(np.abs(audio_data))
                else:
                    audio_float = audio_data.copy()
            else:
                # Convert integer types to float
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_float = audio_data.astype(np.float32) / 2147483648.0
                elif audio_data.dtype == np.uint8:
                    audio_float = (audio_data.astype(np.float32) - 128) / 128.0
                else:
                    raise AudioError(f"Unsupported input array dtype: {audio_data.dtype}")
        else:
            raise AudioError(f"Unsupported audio data type: {type(audio_data)}")
        
        # Resample if needed
        if source_sample_rate is not None and source_sample_rate != target_sample_rate:
            try:
                import samplerate
                
                # Calculate ratio
                ratio = target_sample_rate / source_sample_rate
                
                # Resample
                resampled = samplerate.resample(audio_float, ratio, "sinc_best")
                audio_float = resampled
                
                logger.debug(f"Resampled audio from {source_sample_rate}Hz to {target_sample_rate}Hz")
            except ImportError:
                # Fallback to simple resampling if samplerate is not available
                logger.warning("samplerate library not found, using simple resampling")
                
                # Simple resampling by linear interpolation
                original_length = len(audio_float)
                new_length = int(original_length * target_sample_rate / source_sample_rate)
                indices = np.linspace(0, original_length - 1, new_length)
                audio_float = np.interp(indices, np.arange(original_length), audio_float)
        
        # Convert back to int16
        audio_int16 = (audio_float * 32767).astype(np.int16)
        
        # Convert to bytes
        return audio_int16.tobytes()
    
    except Exception as e:
        raise AudioError(f"Failed to convert audio to PCM16: {str(e)}", cause=e)


def read_wav_file(file_path: Union[str, Path]) -> Tuple[bytes, int, int, int]:
    """
    Read a WAV file and return its data and properties.
    
    Args:
        file_path: Path to the WAV file
        
    Returns:
        Tuple[bytes, int, int, int]: Audio data, sample rate, channels, sample width
        
    Raises:
        AudioError: If reading the file fails
    """
    try:
        with wave.open(str(file_path), 'rb') as wf:
            # Get audio properties
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            
            # Read audio data
            audio_data = wf.readframes(wf.getnframes())
            
            logger.debug(f"Read WAV file: {file_path}, {sample_rate}Hz, {channels} channels, {sample_width*8}-bit")
            
            return audio_data, sample_rate, channels, sample_width
    except Exception as e:
        raise AudioError(f"Failed to read WAV file {file_path}: {str(e)}", cause=e)


def write_wav_file(
    file_path: Union[str, Path],
    audio_data: Union[bytes, np.ndarray],
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2
) -> None:
    """
    Write audio data to a WAV file.
    
    Args:
        file_path: Path to save the WAV file
        audio_data: Audio data as bytes or numpy array
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Sample width in bytes
        
    Raises:
        AudioError: If writing the file fails
    """
    try:
        # Convert numpy array to bytes if needed
        if isinstance(audio_data, np.ndarray):
            if np.issubdtype(audio_data.dtype, np.floating):
                # Convert float to int16
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            elif audio_data.dtype == np.int16:
                # Already int16, just convert to bytes
                audio_bytes = audio_data.tobytes()
            else:
                # Convert to int16 first
                audio_bytes = audio_data.astype(np.int16).tobytes()
        else:
            # Already bytes
            audio_bytes = audio_data
        
        # Write WAV file
        with wave.open(str(file_path), 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        
        logger.debug(f"Wrote WAV file: {file_path}, {sample_rate}Hz, {channels} channels, {sample_width*8}-bit")
    except Exception as e:
        raise AudioError(f"Failed to write WAV file {file_path}: {str(e)}", cause=e)


def get_audio_duration(
    audio_data: Union[bytes, np.ndarray],
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2
) -> float:
    """
    Calculate the duration of audio data in seconds.
    
    Args:
        audio_data: Audio data as bytes or numpy array
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Sample width in bytes
        
    Returns:
        float: Duration in seconds
    """
    try:
        # Get number of frames
        if isinstance(audio_data, bytes):
            # For bytes, calculate frames from byte count
            num_frames = len(audio_data) // (channels * sample_width)
        else:
            # For numpy array, use array length
            num_frames = len(audio_data)
            if len(audio_data.shape) > 1 and audio_data.shape[1] == channels:
                # Multi-channel array
                num_frames = audio_data.shape[0]
        
        # Calculate duration
        duration = num_frames / sample_rate
        return duration
    except Exception as e:
        logger.error(f"Failed to calculate audio duration: {str(e)}")
        return 0.0


def detect_silence(
    audio_data: Union[bytes, np.ndarray],
    threshold: float = 0.05,
    min_silence_duration: float = 0.3,
    sample_rate: int = 24000
) -> bool:
    """
    Detect if audio data is mostly silence.
    
    Args:
        audio_data: Audio data as bytes or numpy array
        threshold: Amplitude threshold for silence (0.0-1.0)
        min_silence_duration: Minimum duration in seconds to consider as silence
        sample_rate: Sample rate in Hz
        
    Returns:
        bool: True if audio is mostly silence
    """
    try:
        # Convert to numpy array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Normalize to [-1, 1]
            audio_norm = audio_array.astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_norm = audio_data
            else:
                # Convert to float
                if audio_data.dtype == np.int16:
                    audio_norm = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_norm = audio_data.astype(np.float32) / 2147483648.0
                elif audio_data.dtype == np.uint8:
                    audio_norm = (audio_data.astype(np.float32) - 128) / 128.0
                else:
                    raise AudioError(f"Unsupported input array dtype: {audio_data.dtype}")
        else:
            raise AudioError(f"Unsupported audio data type: {type(audio_data)}")
        
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(np.square(audio_norm)))
        
        # Check if RMS is below threshold
        if rms < threshold:
            # Audio is quiet, but check duration
            duration = len(audio_norm) / sample_rate
            return duration >= min_silence_duration
        
        return False
    except Exception as e:
        logger.error(f"Failed to detect silence: {str(e)}")
        return False


def split_audio_chunks(
    audio_data: bytes,
    chunk_size: int,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2
) -> list:
    """
    Split audio data into chunks of roughly equal duration.
    
    Args:
        audio_data: Audio data as bytes
        chunk_size: Size of each chunk in frames
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Sample width in bytes
        
    Returns:
        list: List of audio data chunks
    """
    try:
        # Calculate bytes per frame
        bytes_per_frame = channels * sample_width
        
        # Calculate chunk size in bytes
        chunk_bytes = chunk_size * bytes_per_frame
        
        # Split into chunks
        chunks = []
        for i in range(0, len(audio_data), chunk_bytes):
            chunk = audio_data[i:i + chunk_bytes]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        return chunks
    except Exception as e:
        logger.error(f"Failed to split audio into chunks: {str(e)}")
        return [audio_data]  # Return original as single chunk 