import asyncio
import base64
import numpy as np
import logging
import queue
import threading
import pyaudio
import wave
import time
import struct
from typing import Dict, Any, Optional, Callable, List, Tuple, Union, BinaryIO

from src.utils.logging_utils import setup_logger
from src.config import settings
from src.realtime_client import RealtimeClient

# Set up logger
logger = setup_logger("audio_handler")

class AudioHandler:
    """
    Handles audio input/output for the OpenAI Realtime API
    Manages microphone input, speaker output, and audio processing
    """
    
    def __init__(
        self, 
        realtime_client: RealtimeClient, 
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        max_queue_size: int = 1000  # Prevent unbounded memory growth
    ):
        """
        Initialize the audio handler
        
        Args:
            realtime_client: The connected Realtime API client
            input_device_index: PyAudio device index for input (None for default)
            output_device_index: PyAudio device index for output (None for default)
            max_queue_size: Maximum number of chunks to keep in queues
        """
        self.client = realtime_client
        self.pyaudio = pyaudio.PyAudio()
        
        # Audio configuration
        self.format = pyaudio.paInt16  # 16-bit PCM (matches AUDIO_FORMAT=pcm16)
        self.channels = settings.CHANNELS
        self.sample_rate = settings.SAMPLE_RATE
        self.chunk_size = settings.CHUNK_SIZE
        self.max_queue_size = max_queue_size
        
        # Device selection
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        
        # Streams
        self.input_stream = None
        self.output_stream = None
        
        # Audio buffers
        self.input_queue = asyncio.Queue(maxsize=max_queue_size)
        self.output_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Processing flags
        self.recording = False
        self.playing = False
        self.is_input_paused = False
        
        # Tasks
        self.processing_task = None
        self.playing_task = None
        self.tasks = []
        
        # For synchronizing with PyAudio callbacks
        self._callback_queue = queue.Queue(maxsize=max_queue_size)
        
        # For VAD (voice activity detection)
        self.speech_active = False
        self.last_speech_start_time = 0.0
        self.speech_pause_pending = False
        
        # Register event callbacks
        self._register_callbacks()
    
    def _register_callbacks(self) -> None:
        """Register callbacks for Realtime API events"""
        # Ensure we don't register duplicates by removing any existing handlers first
        for event_type in ["response.audio.delta", "error", 
                        "input_audio_buffer.speech_started", 
                        "input_audio_buffer.speech_stopped"]:
            if event_type in self.client.event_callbacks:
                self.client.event_callbacks[event_type] = []

        # Create wrapper functions that handle async callbacks properly
        async def audio_delta_handler(event_data):
            try:
                audio_base64 = event_data.get("delta", "")
                if not audio_base64:
                    return
                    
                audio_bytes = base64.b64decode(audio_base64)
                
                # Add more detailed logging
                logger.debug(f"Received audio delta: {len(audio_bytes)} bytes")
                
                await self.output_queue.put(audio_bytes)
                
                # If playback is not already running, start it
                if not self.playing and not self.output_queue.empty():
                    await self._ensure_playback_running()
            except Exception as e:
                logger.error(f"Error handling audio delta: {e}")
                
        async def error_handler(event_data):
            error_message = event_data.get("message", "Unknown error")
            error_code = event_data.get("code", "unknown")
            logger.error(f"Realtime API error: {error_code} - {error_message}")
            
            # Handle specific errors that might require action
            if "rate_limit" in error_code.lower() or "too_many_requests" in error_code.lower():
                logger.warning("Rate limit hit, pausing audio input for 2 seconds")
                self.pause_input(2.0)
        
        async def speech_started_handler(event_data):
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
                    await asyncio.sleep(0.5)  # Increased delay to 500ms to confirm this isn't just a brief noise
                    if self.speech_pause_pending and time.time() - self.last_speech_start_time >= 0.5:
                        logger.info("Confirmed real speech - pausing AI audio")
                        await self.pause_playback()
                        self.speech_pause_pending = False
                
                # Create and track the task
                pause_task = asyncio.create_task(delayed_pause())
                self.tasks.append(pause_task)
        
        async def speech_stopped_handler(event_data):
            logger.info("Speech detected: User stopped speaking")
            speech_duration = time.time() - getattr(self, 'last_speech_start_time', time.time())
            
            # Cancel pending pause if this was just a brief noise
            if hasattr(self, 'speech_pause_pending') and self.speech_pause_pending and speech_duration < 1.0:
                self.speech_pause_pending = False
                logger.info(f"Ignored brief noise (duration: {speech_duration:.2f}s) - not considered speech")
                return
                
            # Only consider it valid speech if it lasted for a minimum duration
            if speech_duration >= 1.0:  # Increased threshold for more reliable speech detection
                logger.info(f"Valid speech detected (duration: {speech_duration:.2f}s)")
                
                # Add a delay before resuming AI audio to accommodate slow speakers or brief pauses
                async def delayed_resume():
                    # Wait for additional time to ensure the user is really done speaking
                    await asyncio.sleep(0.8)  # 800ms extra delay before AI starts speaking again
                    
                    # Check if we're still in a state where resuming makes sense
                    # (user might have started speaking again during this delay)
                    if self.output_stream and not self.output_stream.is_active():
                        logger.info("Delay period ended - resuming AI audio output")
                        await self.resume_playback()
                
                # Create and track the resume task
                resume_task = asyncio.create_task(delayed_resume())
                self.tasks.append(resume_task)
            else:
                logger.info(f"Ignored brief noise (duration: {speech_duration:.2f}s) - not considered speech")
                self.speech_pause_pending = False
        
        # Register the async handlers
        self.client.register_event_callback("response.audio.delta", audio_delta_handler)
        self.client.register_event_callback("error", error_handler)
        self.client.register_event_callback("input_audio_buffer.speech_started", speech_started_handler)
        self.client.register_event_callback("input_audio_buffer.speech_stopped", speech_stopped_handler)
    
    async def pause_input(self, duration: float = None) -> None:
        """
        Temporarily pause audio input processing
        
        Args:
            duration: Duration in seconds to pause, or None for indefinite
        """
        self.is_input_paused = True
        logger.info(f"Audio input paused for {duration if duration else 'indefinite'} seconds")
        
        if duration:
            # Create a task to unpause after the specified duration
            async def unpause_after_delay():
                await asyncio.sleep(duration)
                self.is_input_paused = False
                logger.info("Audio input resumed")
            
            task = asyncio.create_task(unpause_after_delay())
            self.tasks.append(task)
            # Clean up finished tasks
            self.tasks = [t for t in self.tasks if not t.done()]
    
    async def resume_input(self) -> None:
        """Resume audio input processing if paused"""
        self.is_input_paused = False
        logger.info("Audio input resumed")
    
    def list_audio_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio input and output devices
        
        Returns:
            List of dictionaries with device information
        """
        devices = []
        for i in range(self.pyaudio.get_device_count()):
            device_info = self.pyaudio.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": device_info.get("name"),
                "maxInputChannels": device_info.get("maxInputChannels"),
                "maxOutputChannels": device_info.get("maxOutputChannels"),
                "defaultSampleRate": device_info.get("defaultSampleRate")
            })
        return devices
    
    async def start_recording(self) -> bool:
        """
        Start recording audio from the microphone
        
        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if self.recording:
            logger.warning("Recording is already active")
            return True
            
        try:
            # Open audio input stream
            self.input_stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback
            )
            
            self.recording = True
            self.is_input_paused = False
            logger.info("Started audio recording")
            
            # Start the processing task if not already running
            if not self.processing_task or self.processing_task.done():
                self.processing_task = asyncio.create_task(self._process_audio_queue())
                self.tasks.append(self.processing_task)
            
            return True
                
        except Exception as e:
            logger.error(f"Error starting audio recording: {e}")
            self.recording = False
            if self.input_stream:
                try:
                    self.input_stream.close()
                except:
                    pass
                self.input_stream = None
            return False
    
    async def stop_recording(self) -> None:
        """Stop recording audio"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Stop and close the input stream
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
            except Exception as e:
                logger.error(f"Error closing input stream: {e}")
            finally:
                self.input_stream = None
            
        logger.info("Stopped audio recording")
        
        # Cancel processing task
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
    
    async def _ensure_playback_running(self) -> None:
        """Ensure playback is running if there's audio to play"""
        if not self.playing and not self.output_queue.empty():
            await self.start_playback()
    
    async def start_playback(self) -> bool:
        """
        Start playing audio output
        
        Returns:
            bool: True if playback started successfully, False otherwise
        """
        if self.playing:
            return True
            
        try:
            # Open audio output stream
            self.output_stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.playing = True
            logger.info("Started audio playback")
            
            # Start the playback task
            if not self.playing_task or self.playing_task.done():
                self.playing_task = asyncio.create_task(self._play_audio_queue())
                self.tasks.append(self.playing_task)
            
            return True
                
        except Exception as e:
            logger.error(f"Error starting audio playback: {e}")
            self.playing = False
            if self.output_stream:
                try:
                    self.output_stream.close()
                except:
                    pass
                self.output_stream = None
            return False
    
    async def stop_playback(self) -> None:
        """Stop playing audio output"""
        if not self.playing:
            return
            
        self.playing = False
        
        # Stop and close the output stream
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception as e:
                logger.error(f"Error closing output stream: {e}")
            finally:
                self.output_stream = None
            
        logger.info("Stopped audio playback")
        
        # Cancel playback task
        if self.playing_task and not self.playing_task.done():
            self.playing_task.cancel()
            try:
                await self.playing_task
            except asyncio.CancelledError:
                pass
            self.playing_task = None
    
    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for audio input stream
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time info
            status: Status flags
            
        Returns:
            Tuple of (None, paContinue)
        """
        if status:
            logger.warning(f"Audio input status: {status}")
            
        if self.recording and not self.is_input_paused:
            # Add to the thread-safe queue for the async task to process
            try:
                self._callback_queue.put_nowait(in_data)
            except queue.Full:
                # If queue is full, log warning and drop the chunk
                logger.warning("Input queue full, dropping audio chunk")
            
        return (None, pyaudio.paContinue)
    
    async def _process_audio_queue(self) -> None:
        """Process audio chunks from the input queue and send to API"""
        try:
            logger.debug("Started audio processing task")
            
            connection_error_count = 0
            max_connection_errors = 5
            
            while self.recording:
                # Check the PyAudio callback queue for data
                while not self._callback_queue.empty() and not self.is_input_paused:
                    try:
                        # Get data from the thread-safe queue
                        audio_chunk = self._callback_queue.get_nowait()
                        
                        # Add to the async queue for processing
                        try:
                            await self.input_queue.put(audio_chunk)
                        except asyncio.QueueFull:
                            logger.warning("Async input queue full, dropping audio chunk")
                    except queue.Empty:
                        # Queue became empty between our check and get
                        break
                
                # Process chunks from the async queue
                while not self.input_queue.empty() and not self.is_input_paused:
                    # Get audio chunk from queue
                    audio_chunk = await self.input_queue.get()
                    
                    # Send to Realtime API
                    try:
                        await self.client.append_audio(audio_chunk)
                        self.input_queue.task_done()
                        # Reset error count on success
                        connection_error_count = 0
                    except Exception as e:
                        logger.error(f"Error sending audio to API: {e}")
                        
                        if "1000" in str(e) or "connection closed" in str(e).lower():
                            # This is a normal closure, so increment the error count
                            connection_error_count += 1
                            
                            if connection_error_count >= max_connection_errors:
                                logger.warning("Too many connection errors, stopping audio processing")
                                self.is_input_paused = True
                                break
                            
                            # Try to reconnect
                            try:
                                if hasattr(self.client, 'ensure_connected'):
                                    connected = await self.client.ensure_connected()
                                    if not connected:
                                        # Failed to reconnect, pause input
                                        await self.pause_input(2.0)
                            except Exception as conn_err:
                                logger.error(f"Error reconnecting: {conn_err}")
                                await self.pause_input(2.0)
                        
                        # For other errors, pause briefly
                        elif "ConnectionError" in str(type(e)) or "rate" in str(e).lower():
                            await self.pause_input(1.0)
                
                # Short sleep to prevent busy waiting
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logger.debug("Audio processing task cancelled")
        except Exception as e:
            logger.error(f"Error in audio processing task: {e}")
    
    async def _play_audio_queue(self) -> None:
        """Play audio chunks from the output queue"""
        try:
            logger.debug("Started audio playback task")
            
            # Counter for empty queue checks
            empty_checks = 0
            # Counter for audio errors
            consecutive_errors = 0
            max_consecutive_errors = 3
            
            while self.playing:
                # Check if there's data in the queue
                try:
                    # Wait for up to 0.5 seconds for data
                    audio_chunk = await asyncio.wait_for(self.output_queue.get(), 0.5)
                    empty_checks = 0  # Reset counter when we get data
                    consecutive_errors = 0  # Reset error counter when we successfully get data
                    
                    # Play the audio
                    if self.output_stream and self.playing:
                        try:
                            # Only write to stream if it's active (not paused)
                            if self.output_stream.is_active():
                                self.output_stream.write(audio_chunk)
                            else:
                                logger.debug("Skipping audio chunk - stream is paused")
                        except Exception as e:
                            consecutive_errors += 1
                            logger.error(f"Error writing to audio output: {e}")
                            
                            # If we get multiple consecutive errors, restart the output stream
                            if consecutive_errors >= max_consecutive_errors:
                                logger.warning(f"Too many consecutive audio errors ({consecutive_errors}), restarting audio output")
                                await self.stop_playback()
                                await asyncio.sleep(0.2)  # Short delay before restarting
                                await self.start_playback()
                                consecutive_errors = 0
                    
                    # Mark task as done
                    self.output_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No data received within timeout
                    empty_checks += 1
                    
                    # If queue has been empty for a while, do some idle processing
                    if empty_checks > 10:  # 5 seconds of empty checks
                        empty_checks = 0
                        if not self.playing:
                            break
                except Exception as e:
                    logger.error(f"Error in audio playback loop: {e}")
                    await asyncio.sleep(0.1)  # Avoid tight loop on errors
                
        except asyncio.CancelledError:
            logger.debug("Audio playback task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in playback task: {e}")
        finally:
            logger.debug("Audio playback task ended")
    
    async def send_audio_file(self, file_path: str) -> bool:
        """
        Send audio from a file to the Realtime API
        
        Args:
            file_path: Path to the audio file (WAV)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with wave.open(file_path, 'rb') as wf:
                # Check file format compatibility
                if wf.getnchannels() != self.channels:
                    logger.warning(f"File has {wf.getnchannels()} channels, but API expects {self.channels}")
                
                if wf.getframerate() != self.sample_rate:
                    logger.warning(f"File sample rate is {wf.getframerate()}, but API expects {self.sample_rate}")
                
                # Read all frames
                audio_data = wf.readframes(wf.getnframes())
                
                # Send in chunks
                chunk_size = 32768  # 32KB chunks
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    await self.client.append_audio(chunk)
                    # Small delay to prevent flooding
                    await asyncio.sleep(0.01)
                
                # If VAD is disabled, we need to manually commit
                if not self.client.vad_enabled:
                    await self.client.commit_audio_buffer()
                    await self.client.create_response()
                
                return True
                    
        except Exception as e:
            logger.error(f"Error sending audio file: {e}")
            return False
    
    async def save_output_to_file(self, file_path: str, duration: float = None) -> bool:
        """
        Save audio output to a WAV file
        
        Args:
            file_path: Path where to save the WAV file
            duration: Optional recording duration in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create a buffer to store audio data
        audio_buffer = []
        
        # Set up a queue for thread-safe access
        file_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # Define a callback to capture audio deltas
        async def capture_audio_delta(event_data):
            audio_base64 = event_data.get("delta", "")
            if audio_base64:
                audio_bytes = base64.b64decode(audio_base64)
                try:
                    await file_queue.put(audio_bytes)
                except asyncio.QueueFull:
                    logger.warning("File recording queue full, dropping audio chunk")
        
        # Track the callback for proper cleanup
        callback_ref = capture_audio_delta
        
        # Register the callback
        self.client.register_event_callback("response.audio.delta", callback_ref)
        
        try:
            # If duration is provided, record for that time
            start_time = time.time()
            last_audio_time = start_time
            
            while duration is None or (time.time() - start_time) < duration:
                try:
                    # Use a timeout to prevent blocking forever
                    audio_chunk = await asyncio.wait_for(file_queue.get(), 0.5)
                    audio_buffer.append(audio_chunk)
                    file_queue.task_done()
                    last_audio_time = time.time()
                except asyncio.TimeoutError:
                    # No new audio in the last 0.5 seconds
                    # If we've gone 3 seconds without audio and not recording a fixed duration,
                    # assume we're done
                    if duration is None and (time.time() - last_audio_time) > 3.0:
                        break
            
            # Write the captured audio to a WAV file
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio = 2 bytes
                wf.setframerate(self.sample_rate)
                for chunk in audio_buffer:
                    wf.writeframes(chunk)
                
            logger.info(f"Saved audio output to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio to file: {e}")
            return False
        finally:
            # Unregister the callback to avoid memory leaks
            if "response.audio.delta" in self.client.event_callbacks:
                callbacks = self.client.event_callbacks["response.audio.delta"]
                for i, cb in enumerate(callbacks):
                    if cb == callback_ref:
                        callbacks.pop(i)
                        break
    
    async def convert_to_pcm16(self, audio_data: Union[bytes, np.ndarray]) -> bytes:
        """
        Convert audio data to PCM16 format expected by the API
        
        Args:
            audio_data: Audio data as bytes or numpy array
            
        Returns:
            bytes: PCM16 formatted audio bytes
        """
        if isinstance(audio_data, np.ndarray):
            # Ensure values are in [-1.0, 1.0] range
            normalized = np.clip(audio_data, -1.0, 1.0)
            # Convert to 16-bit PCM
            pcm16_data = (normalized * 32767).astype(np.int16).tobytes()
            return pcm16_data
        elif isinstance(audio_data, bytes):
            # Assume it's already in PCM16 format
            return audio_data
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        # Stop processing
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
        self.tasks = []
        
        # Close PyAudio
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
            
        logger.info("Audio handler resources cleaned up")
    
    async def pause_playback(self) -> None:
        """Temporarily pause audio playback without closing the stream"""
        if not self.playing or not self.output_stream:
            return
            
        try:
            # Just pause the stream but keep it open
            if self.output_stream:
                self.output_stream.stop_stream()
                logger.info("Audio playback paused")
        except Exception as e:
            logger.error(f"Error pausing output stream: {e}")
            # If pausing fails, stop completely
            await self.stop_playback()
    
    async def resume_playback(self) -> None:
        """Resume paused audio playback"""
        if not self.output_stream:
            # If we don't have a stream, start a new one
            await self.start_playback()
            return
            
        try:
            # Check if the stream exists and is not active
            if self.output_stream and not self.output_stream.is_active():
                # Ensure the stream is actually able to be started
                if self.output_stream._is_stopped:
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