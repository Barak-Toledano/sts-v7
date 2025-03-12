"""
Audio service for the OpenAI Realtime Assistant.

This module handles audio recording from microphone and playback to speakers,
including voice activity detection and audio format conversion.
"""

import asyncio
import collections
import wave
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pyaudio

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import (
    EventType,
    UserSpeechEvent,
    event_bus,
)
from src.utils.async_helpers import TaskManager, debounce
from src.utils.audio_utilities import (
    convert_to_pcm16,
    detect_silence,
    get_audio_duration,
    validate_audio_format,
)
from src.utils.error_handling import AudioError, safe_execute

logger = get_logger(__name__)


class AudioState(Enum):
    """Possible states for the audio service."""
    
    IDLE = auto()
    RECORDING = auto()
    PLAYING = auto()
    PAUSED = auto()
    ERROR = auto()


class AudioService:
    """
    Service for recording from microphone and playing back audio.
    
    This class handles:
    - Audio device selection and configuration
    - Recording audio from microphone with VAD
    - Converting and buffering audio data
    - Playing audio through speakers
    - Managing audio state and callbacks
    """
    
    def __init__(
        self,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
    ):
        """
        Initialize the audio service.
        
        Args:
            input_device_index: Index of the input device to use (None for default)
            output_device_index: Index of the output device to use (None for default)
        
        Raises:
            AudioError: If audio initialization fails
        """
        # PyAudio instance
        self.py_audio = None
        
        # State tracking
        self.state = AudioState.IDLE
        self.task_manager = TaskManager()
        
        # Streams
        self.input_stream = None
        self.output_stream = None
        
        # Device configuration
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        
        # Audio format
        self.sample_rate = settings.audio.sample_rate
        self.channels = settings.audio.channels
        self.sample_width = settings.audio.sample_width
        self.frames_per_buffer = settings.audio.frames_per_buffer
        
        # Voice Activity Detection settings
        self.vad_threshold = settings.audio.vad_threshold
        self.silence_duration_ms = settings.audio.silence_duration_ms
        self.prefix_padding_ms = settings.audio.prefix_padding_ms
        
        # For tracking speech
        self.is_speech_active = False
        self.silence_frames = 0
        self.speech_frames = collections.deque(maxlen=100)  # ~2.5 seconds at 44.1kHz
        self.speech_buffer = bytearray()
        self.pending_pause_task = None
        
        # Output buffer and state
        self.playback_buffer = collections.deque()
        self.is_playing = False
        self.should_stop_playback = False
        
        # Initialize audio
        self._initialize_audio()
    
    def _initialize_audio(self) -> None:
        """
        Initialize PyAudio and configure audio parameters.
        
        Raises:
            AudioError: If audio initialization fails
        """
        try:
            self.py_audio = pyaudio.PyAudio()
            
            # List available devices
            self._log_available_devices()
            
            # Configure default devices if not specified
            if self.input_device_index is None:
                self.input_device_index = self.py_audio.get_default_input_device_info()['index']
            
            if self.output_device_index is None:
                self.output_device_index = self.py_audio.get_default_output_device_info()['index']
            
            # Log selected devices
            input_device = self.py_audio.get_device_info_by_index(self.input_device_index)
            output_device = self.py_audio.get_device_info_by_index(self.output_device_index)
            
            logger.info(f"Selected input device: {input_device['name']} (index: {self.input_device_index})")
            logger.info(f"Selected output device: {output_device['name']} (index: {self.output_device_index})")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio: {str(e)}")
            raise AudioError(
                f"Failed to initialize audio: {str(e)}",
                cause=e
            )
    
    def _log_available_devices(self) -> None:
        """Log information about available audio devices."""
        if not self.py_audio:
            return
        
        logger.info("Available audio devices:")
        
        # Input devices
        logger.info("Input devices:")
        input_devices = []
        for i in range(self.py_audio.get_device_count()):
            device_info = self.py_audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(f"  {i}: {device_info['name']}")
        
        for device in input_devices:
            logger.info(device)
        
        # Output devices
        logger.info("Output devices:")
        output_devices = []
        for i in range(self.py_audio.get_device_count()):
            device_info = self.py_audio.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                output_devices.append(f"  {i}: {device_info['name']}")
        
        for device in output_devices:
            logger.info(device)
    
    async def start_recording(self) -> None:
        """
        Start recording audio from the microphone with VAD.
        
        This method starts a new input stream if one is not already active
        and sets up callbacks for audio processing.
        
        Raises:
            AudioError: If recording cannot be started
        """
        if self.state == AudioState.RECORDING:
            logger.warning("Already recording")
            return
        
        if not self.py_audio:
            self._initialize_audio()
        
        logger.info("Starting audio recording with VAD")
        
        try:
            # Reset state
            self.is_speech_active = False
            self.silence_frames = 0
            self.speech_frames.clear()
            self.speech_buffer = bytearray()
            
            if self.pending_pause_task:
                self.pending_pause_task.cancel()
                self.pending_pause_task = None
            
            # Create and start the input stream
            self.input_stream = self.py_audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self._audio_callback
            )
            
            self.input_stream.start_stream()
            self.state = AudioState.RECORDING
            logger.info("Audio recording started")
            
        except Exception as e:
            self.state = AudioState.ERROR
            logger.error(f"Failed to start recording: {str(e)}")
            raise AudioError(
                f"Failed to start recording: {str(e)}",
                cause=e
            )
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Process incoming audio data from the microphone.
        
        This is called by PyAudio in a separate thread.
        
        Args:
            in_data: Audio data from microphone
            frame_count: Number of frames
            time_info: Time information
            status: Status flag
            
        Returns:
            tuple: (None, pyaudio.paContinue)
        """
        if status:
            logger.warning(f"Audio input status: {status}")
        
        # Process for VAD
        self._process_audio_for_vad(in_data)
        
        # Continue recording
        return (None, pyaudio.paContinue)
    
    def _process_audio_for_vad(self, audio_data: bytes) -> None:
        """
        Process audio data for voice activity detection.
        
        This determines if the current audio frame contains speech
        and manages the speech state accordingly.
        
        Args:
            audio_data: Raw audio data from microphone
        """
        # Convert to numpy array for processing
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate energy for VAD
        energy = np.sqrt(np.mean(np.square(audio_array.astype(np.float32) / 32768.0)))
        
        # Keep track of recent frames for context
        self.speech_frames.append(audio_data)
        
        # Determine if this is speech
        is_speech = energy > self.vad_threshold
        
        # State machine for speech detection
        if not self.is_speech_active and is_speech:
            # Transition from silence to speech
            self._handle_speech_start()
        elif self.is_speech_active and is_speech:
            # Continue speech
            self._handle_speech_continue(audio_data)
        elif self.is_speech_active and not is_speech:
            # Possible end of speech, increment silence counter
            self.silence_frames += 1
            
            # Calculate silence duration
            silence_duration_ms = (self.silence_frames * self.frames_per_buffer / self.sample_rate) * 1000
            
            if silence_duration_ms >= self.silence_duration_ms:
                # Enough silence to end speech
                self._schedule_delayed_pause()
            else:
                # Still collecting speech
                self._handle_speech_continue(audio_data)
    
    def _handle_speech_start(self) -> None:
        """Handle transition from silence to speech."""
        # Mark speech as active
        self.is_speech_active = True
        self.silence_frames = 0
        
        # Initialize new speech buffer
        self.speech_buffer = bytearray()
        
        # Include prefix padding (previous audio) to catch the start of speech
        prefix_frames = int((self.prefix_padding_ms / 1000) * self.sample_rate / self.frames_per_buffer)
        for i in range(min(prefix_frames, len(self.speech_frames))):
            self.speech_buffer.extend(self.speech_frames[i])
        
        # Emit speech started event
        speech_event = UserSpeechEvent(
            type=EventType.USER_SPEECH_STARTED,
            audio_data=bytes(self.speech_buffer),
            is_final=False
        )
        asyncio.run_coroutine_threadsafe(self._emit_speech_event(speech_event), asyncio.get_event_loop())
        
        logger.debug("Speech started")
    
    def _handle_speech_continue(self, audio_data: bytes) -> None:
        """
        Handle continuation of speech.
        
        Args:
            audio_data: New audio data to add to buffer
        """
        # Reset silence frames
        self.silence_frames = 0
        
        # Add audio to buffer
        self.speech_buffer.extend(audio_data)
        
        # Cancel any pending pause
        if self.pending_pause_task:
            self.pending_pause_task.cancel()
            self.pending_pause_task = None
        
        # Emit speech ongoing event
        speech_event = UserSpeechEvent(
            type=EventType.USER_SPEECH_ONGOING,
            audio_data=audio_data,
            is_final=False
        )
        asyncio.run_coroutine_threadsafe(self._emit_speech_event(speech_event), asyncio.get_event_loop())
    
    def _handle_speech_end(self) -> None:
        """Handle end of speech detection."""
        if not self.is_speech_active:
            return
        
        # Calculate speech duration
        duration = len(self.speech_buffer) / (self.sample_width * self.channels * self.sample_rate)
        
        # Only emit if speech is long enough to be valid
        if duration >= 0.5:  # Minimum 500ms
            logger.debug(f"Speech ended (duration: {duration:.2f}s)")
            
            # Create normalized PCM data
            try:
                audio_data = convert_to_pcm16(
                    self.speech_buffer,
                    source_sample_rate=self.sample_rate,
                    target_sample_rate=24000  # OpenAI requires 24kHz
                )
                
                # Emit speech finished event
                speech_event = UserSpeechEvent(
                    type=EventType.USER_SPEECH_FINISHED,
                    audio_data=audio_data,
                    duration=duration,
                    is_final=True
                )
                asyncio.run_coroutine_threadsafe(self._emit_speech_event(speech_event), asyncio.get_event_loop())
                
            except Exception as e:
                logger.error(f"Error processing speech: {str(e)}")
        else:
            logger.debug(f"Discarded speech (too short: {duration:.2f}s)")
            
            # Emit speech cancelled event for short utterances
            speech_event = UserSpeechEvent(
                type=EventType.USER_SPEECH_CANCELLED,
                duration=duration,
                is_final=True
            )
            asyncio.run_coroutine_threadsafe(self._emit_speech_event(speech_event), asyncio.get_event_loop())
        
        # Reset state
        self.is_speech_active = False
        self.silence_frames = 0
        self.speech_buffer = bytearray()
    
    def _schedule_delayed_pause(self) -> None:
        """
        Schedule a delayed pause to ensure we have truly detected the end of speech.
        
        This helps prevent false ends when there are natural pauses in speech.
        """
        if self.pending_pause_task:
            self.pending_pause_task.cancel()
        
        self.pending_pause_task = asyncio.run_coroutine_threadsafe(
            self._delayed_pause(),
            asyncio.get_event_loop()
        )
    
    async def _delayed_pause(self) -> None:
        """
        Wait a short period to confirm the end of speech.
        
        This coroutine runs after the VAD detects potential end of speech,
        but waits to confirm there's no immediate continuation.
        """
        await asyncio.sleep(0.3)  # Wait 300ms to be sure
        
        # Check speech duration
        speech_duration = len(self.speech_buffer) / (self.sample_width * self.channels * self.sample_rate)
        
        # If very short and no new speech, just cancel
        if speech_duration < 0.5 and self.is_speech_active:
            logger.debug(f"Cancelling pending pause, speech too short ({speech_duration:.2f}s)")
            self._handle_speech_end()
            return
        
        # Otherwise, end the speech if still active
        if self.is_speech_active:
            self._handle_speech_end()
    
    async def _emit_speech_event(self, event: UserSpeechEvent) -> None:
        """
        Emit a speech event to the event bus.
        
        Args:
            event: Speech event to emit
        """
        event_bus.emit(event)
    
    async def stop_recording(self) -> None:
        """
        Stop recording audio from the microphone.
        
        Stops the input stream and cleans up resources.
        """
        if self.state != AudioState.RECORDING:
            logger.warning("Not currently recording")
            return
        
        logger.info("Stopping audio recording")
        
        # Finish any pending speech
        if self.is_speech_active:
            self._handle_speech_end()
        
        # Stop the input stream
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
            except Exception as e:
                logger.error(f"Error closing input stream: {str(e)}")
        
        # Update state
        self.state = AudioState.IDLE
        
        logger.info("Audio recording stopped")
    
    async def play_audio(self, audio_data: bytes) -> None:
        """
        Play audio data through the speakers.
        
        Args:
            audio_data: Audio data to play (16-bit PCM, 24kHz, mono)
            
        Raises:
            AudioError: If playback fails
        """
        # Ensure audio format is compatible
        try:
            # Check if we need to convert from the OpenAI format (24kHz)
            converted_audio = convert_to_pcm16(
                audio_data,
                source_sample_rate=24000,  # OpenAI sends 24kHz
                target_sample_rate=self.sample_rate
            )
            
            # Add to playback buffer
            self.playback_buffer.append(converted_audio)
            
            # Start playback if not already playing
            if not self.is_playing:
                self.task_manager.create_task(self._playback_worker(), "playback")
                
        except Exception as e:
            logger.error(f"Error preparing audio for playback: {str(e)}")
            raise AudioError(
                f"Failed to play audio: {str(e)}",
                cause=e
            )
    
    async def _playback_worker(self) -> None:
        """
        Worker coroutine for playing audio from the buffer.
        
        This coroutine runs until the playback buffer is empty or playback is stopped.
        """
        if self.is_playing:
            return
        
        self.is_playing = True
        self.should_stop_playback = False
        self.state = AudioState.PLAYING
        
        try:
            # Create output stream if needed
            if not self.output_stream:
                self.output_stream = self.py_audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.output_device_index
                )
            
            logger.debug("Starting audio playback")
            
            # Play all chunks in buffer
            while self.playback_buffer and not self.should_stop_playback:
                chunk = self.playback_buffer.popleft()
                
                # Play chunk
                self.output_stream.write(chunk)
                
                # Allow other tasks to run
                await asyncio.sleep(0)
            
            logger.debug("Audio playback complete")
            
        except Exception as e:
            logger.error(f"Error during audio playback: {str(e)}")
        finally:
            self.is_playing = False
            
            # Close output stream if needed
            if self.output_stream and not self.playback_buffer:
                try:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                    self.output_stream = None
                except Exception as e:
                    logger.error(f"Error closing output stream: {str(e)}")
            
            # Update state if no more playback
            if not self.playback_buffer:
                self.state = AudioState.IDLE
            elif self.should_stop_playback:
                self.state = AudioState.PAUSED
    
    async def pause_playback(self) -> None:
        """
        Pause audio playback.
        
        This method sets a flag to stop playback after the current chunk.
        """
        if not self.is_playing:
            logger.warning("Not currently playing audio")
            return
        
        logger.info("Pausing audio playback")
        self.should_stop_playback = True
        self.state = AudioState.PAUSED
    
    async def resume_playback(self) -> None:
        """
        Resume paused audio playback.
        
        This method resumes playback from where it was paused.
        """
        if self.state != AudioState.PAUSED or not self.playback_buffer:
            logger.warning("No paused playback to resume")
            return
        
        logger.info("Resuming audio playback")
        
        # Start playback worker
        self.task_manager.create_task(self._playback_worker(), "playback")
    
    async def stop_playback(self) -> None:
        """
        Stop audio playback and clear the buffer.
        
        This method stops playback and discards any remaining audio data.
        """
        logger.info("Stopping audio playback")
        
        # Stop current playback
        self.should_stop_playback = True
        
        # Clear buffer
        self.playback_buffer.clear()
        
        # Close output stream if open
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
            except Exception as e:
                logger.error(f"Error closing output stream: {str(e)}")
        
        # Update state
        self.state = AudioState.IDLE
        
        logger.info("Audio playback stopped")
    
    async def cleanup(self) -> None:
        """
        Clean up all resources used by the audio service.
        
        This method should be called when the application is shutting down.
        """
        logger.info("Cleaning up audio resources")
        
        # Cancel all tasks
        self.task_manager.cancel_all()
        
        # Stop recording and playback
        await self.stop_recording()
        await self.stop_playback()
        
        # Terminate PyAudio
        if self.py_audio:
            try:
                self.py_audio.terminate()
                self.py_audio = None
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {str(e)}")
        
        logger.info("Audio resources cleaned up") 