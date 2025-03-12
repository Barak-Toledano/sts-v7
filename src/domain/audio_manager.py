"""
Domain logic for managing audio input and output.
"""
import asyncio
import logging
import numpy as np
import pyaudio
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

class AudioManager:
    """Manages audio input and output for the application.
    
    This class is responsible for handling audio streams, processing audio data,
    and managing the interaction with the OpenAI service for audio processing.
    """
    
    def __init__(
        self, 
        openai_service: OpenAIService,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        max_queue_size: int = 1000
    ):
        """Initialize the audio manager.
        
        Args:
            openai_service: Service for OpenAI API communication
            input_device_index: Index of the audio input device
            output_device_index: Index of the audio output device
            max_queue_size: Maximum size of audio queues
        """
        self.service = openai_service
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.max_queue_size = max_queue_size
        
        # PyAudio instance
        self._audio = None
        
        # Audio stream objects
        self.input_stream = None
        self.output_stream = None
        
        # State tracking
        self.recording = False
        self.playing = False
        self.paused = False
        self.speech_pause_pending = False
        
        # Audio format parameters
        self.sample_rate = 24000  # Hz, required by OpenAI
        self.chunk_size = 4096
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Queues for audio data
        self.input_queue = asyncio.Queue(maxsize=max_queue_size)
        self.output_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Task tracking
        self.tasks: List[asyncio.Task] = []
        
        # Register event handlers
        self._register_callbacks()
    
    def _register_callbacks(self) -> None:
        """Register handlers for OpenAI API events."""
        # Audio response handler
        self.service.register_event_handler("response.audio.delta", self._handle_audio_delta)
        
        # Speech detection handlers
        self.service.register_event_handler("speech.started", self._handle_speech_started)
        self.service.register_event_handler("speech.stopped", self._handle_speech_stopped)
        
        # Error handler
        self.service.register_event_handler("error", self._handle_error)
    
    async def _handle_audio_delta(self, event_data: Dict[str, Any]) -> None:
        """Handle audio delta event from OpenAI API.
        
        Args:
            event_data: Audio data event
        """
        try:
            audio_data = event_data.get("data", b"")
            if audio_data and self.output_queue.qsize() < self.max_queue_size:
                await self.output_queue.put(audio_data)
                
                # Ensure playback is running when we receive audio
                if not self.playing or self.paused:
                    await self._ensure_playback_running()
        except Exception as e:
            logger.error(f"Error processing audio delta: {e}")
    
    async def _handle_error(self, event_data: Dict[str, Any]) -> None:
        """Handle error event from OpenAI API.
        
        Args:
            event_data: Error event data
        """
        error_type = event_data.get("type", "")
        error_message = event_data.get("message", "")
        error_code = event_data.get("code", "")
        
        logger.error(f"API error in audio manager: {error_type} - {error_message}")
        
        # Handle specific errors that might require action
        if "rate_limit" in error_code.lower() or "too_many_requests" in error_code.lower():
            logger.warning("Rate limit hit, pausing audio input for 2 seconds")
            await self.pause_input(2.0)
    
    async def _handle_speech_started(self, event_data: Dict[str, Any]) -> None:
        """Handle speech started event.
        
        Args:
            event_data: Speech event data
        """
        logger.info("Speech detected: User started speaking")
        
        # Store the timestamp when speech started
        self.last_speech_start_time = time.time()
        
        # If AI is currently speaking, pause the audio output to let the user speak
        if self.output_stream and self.playing:
            logger.info("User started speaking - pausing AI audio output")
            
            # Just mark as paused but don't actually pause the stream yet
            # We'll wait to see if this is a valid speech or just a brief noise
            self.speech_pause_pending = True
            
            # Set a timer to actually pause after a short delay if speech continues
            async def delayed_pause():
                await asyncio.sleep(0.5)  # 500ms delay to confirm this isn't just a brief noise
                if self.speech_pause_pending and time.time() - self.last_speech_start_time >= 0.5:
                    logger.info("Confirmed real speech - pausing AI audio")
                    await self.pause_playback()
                    self.speech_pause_pending = False
            
            # Create and track the task
            pause_task = asyncio.create_task(delayed_pause())
            self.tasks.append(pause_task)
    
    async def _handle_speech_stopped(self, event_data: Dict[str, Any]) -> None:
        """Handle speech stopped event.
        
        Args:
            event_data: Speech event data
        """
        logger.info("Speech detected: User stopped speaking")
        speech_duration = time.time() - getattr(self, 'last_speech_start_time', time.time())
        
        # Cancel pending pause if this was just a brief noise
        if hasattr(self, 'speech_pause_pending') and self.speech_pause_pending and speech_duration < 1.0:
            self.speech_pause_pending = False
            logger.info(f"Ignored brief noise (duration: {speech_duration:.2f}s) - not considered speech")
            return
            
        # Only consider it valid speech if it lasted for a minimum duration
        if speech_duration >= 1.0:  # Threshold for speech detection
            logger.info(f"Valid speech detected (duration: {speech_duration:.2f}s)")
            
            # Add a delay before resuming AI audio to accommodate slow speakers or brief pauses
            async def delayed_resume():
                # Wait for additional time to ensure the user is really done speaking
                await asyncio.sleep(0.8)  # 800ms extra delay before AI starts speaking again
                
                # Check if we're still in a state where resuming makes sense
                if self.output_stream and not self.output_stream.is_active():
                    logger.info("Delay period ended - resuming AI audio output")
                    await self.resume_playback()
            
            # Create and track the resume task
            resume_task = asyncio.create_task(delayed_resume())
            self.tasks.append(resume_task)
        else:
            logger.info(f"Ignored brief noise (duration: {speech_duration:.2f}s) - not considered speech")
            self.speech_pause_pending = False
    
    async def pause_input(self, duration: float = None) -> None:
        """Pause audio input for a specified duration.
        
        Args:
            duration: Duration in seconds to pause, or None for indefinite
        """
        if not self.recording or not self.input_stream:
            return
            
        try:
            # Stop the input stream
            self.input_stream.stop_stream()
            self.paused = True
            logger.info(f"Audio input paused{f' for {duration}s' if duration else ''}")
            
            # If duration is provided, set up auto-resume
            if duration:
                async def unpause_after_delay():
                    await asyncio.sleep(duration)
                    await self.resume_input()
                
                # Start the timer task
                unpause_task = asyncio.create_task(unpause_after_delay())
                self.tasks.append(unpause_task)
        except Exception as e:
            logger.error(f"Error pausing input stream: {e}")
    
    async def resume_input(self) -> None:
        """Resume audio input after pausing."""
        if not self.recording or not self.input_stream or not self.paused:
            return
            
        try:
            self.input_stream.start_stream()
            self.paused = False
            logger.info("Audio input resumed")
        except Exception as e:
            logger.error(f"Error resuming input stream: {e}")
    
    def list_audio_devices(self) -> List[Dict[str, Any]]:
        """List available audio input and output devices.
        
        Returns:
            List[Dict[str, Any]]: List of available devices
        """
        devices = []
        
        if not self._audio:
            self._audio = pyaudio.PyAudio()
        
        for i in range(self._audio.get_device_count()):
            device_info = self._audio.get_device_info_by_index(i)
            if device_info.get("maxInputChannels", 0) > 0:
                device_type = "input"
            elif device_info.get("maxOutputChannels", 0) > 0:
                device_type = "output"
            else:
                continue
                
            devices.append({
                "index": i,
                "name": device_info.get("name", ""),
                "type": device_type,
                "channels": device_info.get("maxInputChannels" if device_type == "input" else "maxOutputChannels", 0),
                "default": (i == self._audio.get_default_input_device_info()["index"] if device_type == "input" else 
                           i == self._audio.get_default_output_device_info()["index"])
            })
        
        return devices
    
    async def start_recording(self) -> bool:
        """Start audio recording from the input device.
        
        Returns:
            bool: True if successful
        """
        if self.recording:
            logger.warning("Recording already in progress")
            return True
            
        try:
            if not self._audio:
                self._audio = pyaudio.PyAudio()
            
            # Open input stream
            self.input_stream = self._audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback
            )
            
            # Start the stream
            self.input_stream.start_stream()
            self.recording = True
            
            # Start the audio processing task
            process_task = asyncio.create_task(self._process_audio_queue())
            self.tasks.append(process_task)
            
            logger.info("Started audio recording")
            return True
        except Exception as e:
            logger.error(f"Error starting audio recording: {e}")
            return False
    
    async def stop_recording(self) -> None:
        """Stop audio recording."""
        if not self.recording or not self.input_stream:
            return
            
        try:
            # Stop and close the input stream
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            self.recording = False
            
            logger.info("Stopped audio recording")
        except Exception as e:
            logger.error(f"Error stopping audio recording: {e}")
    
    async def _ensure_playback_running(self) -> None:
        """Ensure playback is running if needed."""
        if not self.playing and not self.output_queue.empty():
            await self.start_playback()
    
    async def start_playback(self) -> bool:
        """Start audio playback to the output device.
        
        Returns:
            bool: True if successful
        """
        if self.playing and self.output_stream and self.output_stream.is_active():
            return True
            
        try:
            if not self._audio:
                self._audio = pyaudio.PyAudio()
            
            # If we have an existing stream but it's not active, try to start it
            if self.output_stream:
                if not self.output_stream.is_active():
                    try:
                        self.output_stream.start_stream()
                        self.playing = True
                        self.paused = False
                        logger.info("Resumed existing audio playback")
                        return True
                    except Exception as e:
                        logger.error(f"Could not resume existing stream: {e}")
                        # Fall through to creating a new stream
                
                # Close the existing stream if it can't be restarted
                try:
                    self.output_stream.close()
                except Exception:
                    pass
                self.output_stream = None
            
            # Create a new output stream
            self.output_stream = self._audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.playing = True
            self.paused = False
            
            # Start the playback task
            playback_task = asyncio.create_task(self._play_audio_queue())
            self.tasks.append(playback_task)
            
            logger.info("Started audio playback")
            return True
        except Exception as e:
            logger.error(f"Error starting audio playback: {e}")
            return False
    
    async def stop_playback(self) -> None:
        """Stop audio playback."""
        if not self.playing or not self.output_stream:
            return
            
        try:
            # Stop and close the output stream
            if self.output_stream.is_active():
                self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
            self.playing = False
            self.paused = False
            
            logger.info("Stopped audio playback")
        except Exception as e:
            logger.error(f"Error stopping audio playback: {e}")
            
    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input data.
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status: Status flag
            
        Returns:
            tuple: (data, flag)
        """
        # If paused, return empty data
        if self.paused:
            return (in_data, pyaudio.paContinue)
        
        # Add data to the queue if we have room
        if self.input_queue.qsize() < self.max_queue_size:
            try:
                asyncio.create_task(self.input_queue.put(in_data))
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
        else:
            logger.warning("Input queue full, dropping audio frame")
        
        return (in_data, pyaudio.paContinue)
    
    async def _process_audio_queue(self) -> None:
        """Process audio data from the input queue."""
        try:
            while self.recording:
                # Wait for data in the queue
                try:
                    audio_data = await asyncio.wait_for(self.input_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # No data within timeout, just continue
                    continue
                
                # Send audio data to OpenAI
                if not await self.service.send_audio(audio_data):
                    logger.warning("Failed to send audio to OpenAI")
                
                self.input_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Audio processing task cancelled")
        except Exception as e:
            logger.error(f"Error in audio processing task: {e}")
    
    async def _play_audio_queue(self) -> None:
        """Play audio data from the output queue."""
        try:
            while self.playing:
                # Only process if we have a valid stream and aren't paused
                if self.output_stream and self.output_stream.is_active() and not self.paused:
                    try:
                        # Get audio data with a short timeout
                        audio_data = await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
                        
                        # Convert to pcm16 if needed
                        try:
                            pcm_data = await self.convert_to_pcm16(audio_data)
                            self.output_stream.write(pcm_data)
                        except Exception as e:
                            logger.error(f"Error writing to audio stream: {e}")
                        
                        self.output_queue.task_done()
                    except asyncio.TimeoutError:
                        # No data within timeout, just continue
                        await asyncio.sleep(0.01)
                else:
                    # Stream not active, wait a bit before checking again
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Audio playback task cancelled")
        except Exception as e:
            logger.error(f"Error in audio playback task: {e}")
            
            # Try to restart playback
            await asyncio.sleep(0.5)
            if self.playing:
                try:
                    await self.stop_playback()
                    await asyncio.sleep(0.5)
                    await self.start_playback()
                except Exception as restart_error:
                    logger.error(f"Failed to restart playback: {restart_error}")
    
    async def send_audio_file(self, file_path: str) -> bool:
        """Send an audio file to the OpenAI Realtime API.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            bool: True if successful
        """
        try:
            import wave
            
            # Open the wave file
            with wave.open(file_path, 'rb') as wf:
                # Check if format is compatible
                if wf.getnchannels() != self.channels or wf.getsampwidth() != 2:
                    logger.error(f"Incompatible audio format: channels={wf.getnchannels()}, "
                                f"sample_width={wf.getsampwidth()}")
                    return False
                
                # Create temporary sample rate converter if needed
                if wf.getframerate() != self.sample_rate:
                    logger.warning(f"Audio file sample rate ({wf.getframerate()} Hz) "
                                  f"differs from required rate ({self.sample_rate} Hz). "
                                  f"Converting on-the-fly.")
                
                # Read and process the entire file
                chunk_size = self.chunk_size
                data = wf.readframes(chunk_size)
                
                while data:
                    # Send to OpenAI
                    if not await self.service.send_audio(data):
                        logger.error("Failed to send audio chunk to OpenAI")
                        return False
                    
                    # Read next chunk
                    data = wf.readframes(chunk_size)
                
                logger.info(f"Successfully sent audio file: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Error sending audio file: {e}")
            return False
    
    async def save_output_to_file(self, file_path: str, duration: float = None) -> bool:
        """Save audio output to a WAV file.
        
        Args:
            file_path: Output file path
            duration: Recording duration in seconds, or None for unlimited
            
        Returns:
            bool: True if successful
        """
        try:
            import wave
            
            # Create a wave file
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                
                # Set up capture
                audio_buffer = bytearray()
                start_time = time.time()
                
                # Create handler for audio deltas
                async def capture_audio_delta(event_data):
                    nonlocal audio_buffer
                    audio_data = event_data.get("data", b"")
                    if audio_data:
                        audio_buffer.extend(audio_data)
                
                # Register handler
                handler_id = self.service.register_event_handler("response.audio.delta", capture_audio_delta)
                
                try:
                    # Wait for specified duration or until interrupted
                    if duration:
                        await asyncio.sleep(duration)
                    else:
                        # Wait until manually interrupted
                        while True:
                            await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    # Capture stopped
                    pass
                finally:
                    # Unregister handler
                    self.service.unregister_event_handler("response.audio.delta", capture_audio_delta)
                    
                    # Write the captured audio to file
                    if audio_buffer:
                        wf.writeframes(bytes(audio_buffer))
                
                logger.info(f"Saved {len(audio_buffer)} bytes of audio to {file_path}")
                return True
        except Exception as e:
            logger.error(f"Error saving audio output: {e}")
            return False
    
    async def convert_to_pcm16(self, audio_data: Union[bytes, np.ndarray]) -> bytes:
        """Convert audio data to PCM16 format.
        
        Args:
            audio_data: Audio data to convert
            
        Returns:
            bytes: Converted audio data
        """
        # If already bytes, assume it's in the correct format
        if isinstance(audio_data, bytes):
            return audio_data
            
        # If numpy array, convert to PCM16
        if isinstance(audio_data, np.ndarray):
            # Scale to int16 range if float
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
            # Convert to bytes
            return audio_data.tobytes()
            
        # Unsupported type
        raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
    
    async def cleanup(self) -> None:
        """Clean up resources used by the audio manager."""
        # Stop recording and playback
        await self.stop_recording()
        await self.stop_playback()
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clean up PyAudio
        if self._audio:
            self._audio.terminate()
            self._audio = None
            
        logger.info("Audio handler resources cleaned up")
    
    async def pause_playback(self) -> None:
        """Pause audio playback."""
        if not self.playing or not self.output_stream or self.paused:
            return
            
        try:
            self.output_stream.stop_stream()
            self.paused = True
            logger.info("Audio playback paused")
        except Exception as e:
            logger.error(f"Error pausing output stream: {e}")
    
    async def resume_playback(self) -> None:
        """Resume audio playback after pausing."""
        if not self.playing or not self.output_stream:
            return
            
        try:
            # Check if the stream exists and is not active
            if self.output_stream and not self.output_stream.is_active():
                # Ensure the stream is actually able to be started
                if hasattr(self.output_stream, '_is_stopped') and self.output_stream._is_stopped:
                    self.output_stream.start_stream()
                    logger.info("Audio playback resumed")
                else:
                    # If we can't resume, recreate the stream
                    logger.info("Stream not in resumable state, recreating...")
                    await self.stop_playback()
                    await asyncio.sleep(0.1)  # Brief delay to ensure cleanup
                    await self.start_playback()
        except Exception as e:
            logger.error(f"Error resuming output stream: {e}")
            # If resuming fails, try to start fresh with a delay to avoid rapid cycling
            await self.stop_playback()
            await asyncio.sleep(0.2)  # Slightly longer delay before restart
            await self.start_playback() 