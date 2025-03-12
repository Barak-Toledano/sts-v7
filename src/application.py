"""
Main application for the OpenAI Realtime Assistant.

This module brings together all components of the application architecture,
coordinating services, domain logic, and user interaction.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from src.config import settings
from src.config.logging_config import get_logger
from src.domain.audio.manager import AudioManager, AudioMode
from src.domain.conversation.manager import ConversationManager, ConversationState
from src.events.event_interface import Event, EventType, event_bus
from src.services.api_client import RealtimeClient
from src.utils.async_helpers import TaskManager, wait_for_event
from src.utils.error_handling import AppError, ErrorSeverity

logger = get_logger(__name__)


class Application:
    """
    Main application class for the OpenAI Realtime Assistant.
    
    This class brings together all components of the application and 
    provides high-level functionality for managing the assistant.
    """
    
    def __init__(
        self,
        assistant_id: str,
        instructions: Optional[str] = None,
        temperature: float = 1.0,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        save_recordings: bool = False,
        debug_mode: Optional[bool] = None,
    ):
        """
        Initialize the application.
        
        Args:
            assistant_id: ID of the OpenAI assistant to use
            instructions: Optional custom instructions for the assistant
            temperature: Temperature parameter for generation (0.0-2.0)
            input_device: Optional audio input device index
            output_device: Optional audio output device index
            save_recordings: Whether to save audio recordings to disk
            debug_mode: Whether to enable debug mode (overrides config)
        """
        # Set debug mode if provided
        if debug_mode is not None:
            settings.debug_mode = debug_mode
        
        # Configure logging based on debug mode
        log_level = "DEBUG" if settings.debug_mode else settings.logging.level
        logging.getLogger().setLevel(getattr(logging, log_level))
        
        # Initialize components
        self.audio_manager = AudioManager(
            input_device=input_device,
            output_device=output_device,
            save_recordings=save_recordings
        )
        
        self.conversation_manager = ConversationManager(
            assistant_id=assistant_id,
            instructions=instructions,
            temperature=temperature,
            input_device=input_device,
            output_device=output_device
        )
        
        # Task management
        self.task_manager = TaskManager("application")
        self.shutdown_event = asyncio.Event()
        
        # Register event handlers
        self._register_event_handlers()
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        logger.info("Application initialized")
    
    async def start(self) -> bool:
        """
        Start the application.
        
        This method starts the conversation session and initializes
        all required components.
        
        Returns:
            bool: True if the application started successfully
        """
        logger.info("Starting application")
        
        try:
            # Start the conversation
            success = await self.conversation_manager.start()
            
            if success:
                logger.info("Application started successfully")
            else:
                logger.error("Failed to start application")
            
            return success
            
        except Exception as e:
            error = AppError(
                f"Failed to start application: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                cause=e
            )
            logger.error(str(error))
            return False
    
    async def stop(self) -> None:
        """Stop the application and clean up resources."""
        logger.info("Stopping application")
        
        try:
            # Stop the conversation
            await self.conversation_manager.stop()
            
            # Clean up audio manager
            await self.audio_manager.cleanup()
            
            # Cancel all tasks
            await self.task_manager.cancel_all()
            
            logger.info("Application stopped")
            
        except Exception as e:
            error = AppError(
                f"Error during application shutdown: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
    
    async def run(self) -> None:
        """
        Run the application until shutdown is requested.
        
        This method starts the application and waits for a shutdown signal.
        """
        success = await self.start()
        
        if not success:
            logger.error("Application failed to start")
            return
        
        # Wait for shutdown signal
        await self.shutdown_event.wait()
        
        # Clean up
        await self.stop()
    
    def shutdown(self) -> None:
        """Request application shutdown."""
        logger.info("Shutdown requested")
        self.shutdown_event.set()
    
    async def request_response(self, instructions: Optional[str] = None) -> bool:
        """
        Request a response from the assistant.
        
        Args:
            instructions: Optional custom instructions for this response
            
        Returns:
            bool: True if the request was successful
        """
        return await self.conversation_manager.request_response(instructions)
    
    async def interrupt(self) -> bool:
        """
        Interrupt the assistant's current response.
        
        Returns:
            bool: True if the interrupt was successful
        """
        return await self.conversation_manager.interrupt()
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for the application."""
        # Error handling
        event_bus.on(EventType.ERROR, self._handle_error)
        
        # Register for shutdown events
        event_bus.on(EventType.SHUTDOWN, self._handle_shutdown_event)
    
    def _handle_error(self, event: Event) -> None:
        """
        Handle error events.
        
        Args:
            event: Error event
        """
        error_data = event.data.get("error", {})
        error_type = error_data.get("type", "unknown")
        error_message = error_data.get("message", "Unknown error")
        
        # For critical errors, initiate shutdown
        if error_type in ("authentication_error", "authorization_error"):
            logger.critical(f"Critical error: {error_type} - {error_message}")
            self.task_manager.create_task(self._delayed_shutdown(), "shutdown_after_error")
    
    async def _delayed_shutdown(self, delay: float = 2.0) -> None:
        """
        Shutdown the application after a delay.
        
        Args:
            delay: Delay in seconds before shutdown
        """
        logger.info(f"Application will shut down in {delay} seconds due to critical error")
        await asyncio.sleep(delay)
        self.shutdown()
    
    def _handle_shutdown_event(self, event: Event) -> None:
        """
        Handle shutdown events from the application.
        
        Args:
            event: The shutdown event with reason details
        """
        reason = event.data.get("reason", "unknown")
        command = event.data.get("command", "")
        transcript = event.data.get("transcript", "")
        
        logger.info(f"Processing shutdown event. Reason: {reason}, Command: {command}")
        if transcript:
            logger.info(f"Shutdown triggered by transcript: '{transcript}'")
        
        # Log this at a higher level to ensure visibility
        if reason == "user_exit_command":
            logger.warning(f"User requested application exit with command: {command}")
            
        # Make sure the event is logged thoroughly before shutting down
        # This helps with debugging shutdown issues
        time.sleep(0.5)  # Brief delay to ensure logging completes
            
        # Trigger application shutdown
        self.shutdown()
    
    def _register_signal_handlers(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        # Only register signal handlers if running in the main thread
        if not self._is_main_thread():
            return
        
        # Register SIGINT and SIGTERM handlers
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
                logger.debug(f"Registered signal handler for {sig}")
            except NotImplementedError:
                # Windows does not support add_signal_handler
                if sys.platform == 'win32':
                    signal.signal(sig, lambda s, f: self._signal_handler())
                    logger.debug(f"Registered Windows signal handler for {sig}")
    
    def _signal_handler(self) -> None:
        """Handle OS signals for graceful shutdown."""
        logger.info("Received shutdown signal")
        self.shutdown()
    
    def _is_main_thread(self) -> bool:
        """Check if current thread is the main thread."""
        import threading
        return threading.current_thread() is threading.main_thread()


async def run_application(
    assistant_id: str,
    instructions: Optional[str] = None,
    temperature: float = 1.0,
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    save_recordings: bool = False,
    debug_mode: Optional[bool] = None,
) -> None:
    """
    Run the application with the given parameters.
    
    Args:
        assistant_id: ID of the OpenAI assistant to use
        instructions: Optional custom instructions for the assistant
        temperature: Temperature parameter for generation (0.0-2.0)
        input_device: Optional audio input device index
        output_device: Optional audio output device index
        save_recordings: Whether to save audio recordings to disk
        debug_mode: Whether to enable debug mode (overrides config)
    """
    app = Application(
        assistant_id=assistant_id,
        instructions=instructions,
        temperature=temperature,
        input_device=input_device,
        output_device=output_device,
        save_recordings=save_recordings,
        debug_mode=debug_mode
    )
    
    await app.run()


def main() -> None:
    """Entry point for running the application from the command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI Realtime Assistant")
    
    parser.add_argument(
        "--assistant-id",
        type=str,
        required=True,
        help="ID of the OpenAI assistant to use"
    )
    
    parser.add_argument(
        "--instructions",
        type=str,
        help="Custom instructions for the assistant"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for generation (0.0-2.0)"
    )
    
    parser.add_argument(
        "--input-device",
        type=int,
        help="Input device index for audio"
    )
    
    parser.add_argument(
        "--output-device",
        type=int,
        help="Output device index for audio"
    )
    
    parser.add_argument(
        "--save-recordings",
        action="store_true",
        help="Save audio recordings to disk"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Run the application
    asyncio.run(
        run_application(
            assistant_id=args.assistant_id,
            instructions=args.instructions,
            temperature=args.temperature,
            input_device=args.input_device,
            output_device=args.output_device,
            save_recordings=args.save_recordings,
            debug_mode=args.debug
        )
    )


if __name__ == "__main__":
    main() 