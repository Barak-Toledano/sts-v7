# tests/test_audio_handler.py
import asyncio
import pytest
import base64
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.audio_handler import AudioHandler
from src.realtime_client import RealtimeClient

@pytest.fixture
def mock_realtime_client():
    """Create a mock Realtime client"""
    client = MagicMock(spec=RealtimeClient)
    client.append_audio = AsyncMock()
    client.commit_audio_buffer = AsyncMock()
    client.clear_audio_buffer = AsyncMock()
    client.vad_enabled = True
    client.is_speech_active = MagicMock(return_value=False)
    
    # Register event callbacks
    callback_registry = {}
    
    def register_callback(event_type, callback):
        if event_type not in callback_registry:
            callback_registry[event_type] = []
        callback_registry[event_type].append(callback)
    
    client.register_event_callback = register_callback
    client.event_callbacks = callback_registry
    
    return client

@pytest.fixture
def mock_pyaudio():
    """Create a mock PyAudio interface"""
    mock = MagicMock()
    
    # Mock stream
    mock_stream = MagicMock()
    mock_stream.start_stream = MagicMock()
    mock_stream.stop_stream = MagicMock()
    mock_stream.close = MagicMock()
    mock_stream.write = MagicMock()
    
    # Mock PyAudio methods
    mock.open = MagicMock(return_value=mock_stream)
    mock.get_device_count = MagicMock(return_value=2)
    mock.get_device_info_by_index = MagicMock(return_value={
        "name": "Test Device",
        "maxInputChannels": 2,
        "maxOutputChannels": 2,
        "defaultSampleRate": 44100
    })
    mock.terminate = MagicMock()
    
    return mock

@pytest.mark.asyncio
async def test_list_audio_devices(mock_realtime_client, mock_pyaudio):
    """Test listing audio devices"""
    with patch('pyaudio.PyAudio', return_value=mock_pyaudio):
        handler = AudioHandler(mock_realtime_client)
        devices = handler.list_audio_devices()
        
        # Check that the devices were listed
        assert len(devices) == 2
        assert "name" in devices[0]
        assert "maxInputChannels" in devices[0]
        
        # Check that PyAudio methods were called
        mock_pyaudio.get_device_count.assert_called_once()
        assert mock_pyaudio.get_device_info_by_index.call_count == 2
        
        # Cleanup
        await handler.cleanup()

@pytest.mark.asyncio
async def test_start_recording(mock_realtime_client, mock_pyaudio):
    """Test starting audio recording"""
    with patch('pyaudio.PyAudio', return_value=mock_pyaudio):
        handler = AudioHandler(mock_realtime_client)
        result = await handler.start_recording()
        
        # Check that recording started successfully
        assert result is True
        assert handler.recording is True
        
        # Check that the stream was opened
        mock_pyaudio.open.assert_called_once()
        
        # Cleanup
        await handler.cleanup()

@pytest.mark.asyncio
async def test_stop_recording(mock_realtime_client, mock_pyaudio):
    """Test stopping audio recording"""
    with patch('pyaudio.PyAudio', return_value=mock_pyaudio):
        handler = AudioHandler(mock_realtime_client)
        
        # First start recording
        await handler.start_recording()
        
        # Then stop it
        await handler.stop_recording()
        
        # Check that recording was stopped
        assert handler.recording is False
        
        # Check that the stream was closed
        mock_stream = mock_pyaudio.open.return_value
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        
        # Cleanup
        await handler.cleanup()

@pytest.mark.asyncio
async def test_handle_audio_delta(mock_realtime_client, mock_pyaudio):
    """Test handling audio delta events"""
    with patch('pyaudio.PyAudio', return_value=mock_pyaudio):
        handler = AudioHandler(mock_realtime_client)
        
        # Mock start_playback to avoid issues
        handler.start_playback = AsyncMock(return_value=True)
        
        # Create audio delta event data
        test_audio = b"test audio data"
        base64_audio = base64.b64encode(test_audio).decode('ascii')
        event_data = {
            "delta": base64_audio
        }
        
        # Get the callback for audio delta events
        callbacks = mock_realtime_client.event_callbacks.get("response.audio.delta", [])
        assert len(callbacks) > 0, "No callback registered for audio delta events"
        
        # Call the callback directly
        await callbacks[0](event_data)
        
        # Check that start_playback was called
        handler.start_playback.assert_called_once()
        
        # Cleanup
        await handler.cleanup()

@pytest.mark.asyncio
async def test_send_audio_file(mock_realtime_client):
    """Test sending audio from a file"""
    # Mock the wave module
    mock_wave_file = MagicMock()
    mock_wave_file.getnchannels.return_value = 1
    mock_wave_file.getframerate.return_value = 24000
    mock_wave_file.getnframes.return_value = 1000
    mock_wave_file.readframes.return_value = b"test audio data"
    
    # Create a context manager mock
    mock_wave_file.__enter__ = MagicMock(return_value=mock_wave_file)
    mock_wave_file.__exit__ = MagicMock(return_value=None)
    
    with patch('pyaudio.PyAudio', return_value=MagicMock()):
        with patch('wave.open', return_value=mock_wave_file):
            handler = AudioHandler(mock_realtime_client)
            result = await handler.send_audio_file("test.wav")
            
            # Check that the file was processed successfully
            assert result is True
            
            # Check that the audio was sent to the client
            mock_realtime_client.append_audio.assert_called()
            
            # Cleanup
            await handler.cleanup()