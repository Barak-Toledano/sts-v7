"""
Command-line interface for the OpenAI Realtime Assistant.

This module provides a CLI interface for interacting with the assistant,
displaying status information, and controlling the conversation.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from src.config.logging_config import get_logger
from src.domain.conversation.manager import ConversationState
from src.events.event_interface import Event, EventType, event_bus

logger = get_logger(__name__)


class CliInterface:
    """
    Command-line interface for the OpenAI Realtime Assistant.
    
    This class provides a terminal-based interface for interacting with
    the assistant, including command processing, status display, and
    conversation visualization.
    """
    
    def __init__(
        self,
        show_timestamps: bool = False,
        compact_mode: bool = False,
        color_output: bool = True,
        show_status: bool = True,
    ):
        """
        Initialize the CLI interface.
        
        Args:
            show_timestamps: Whether to show timestamps for messages
            compact_mode: Whether to use a compact display mode
            color_output: Whether to use colored output
            show_status: Whether to show status information
        """
        self.show_timestamps = show_timestamps
        self.compact_mode = compact_mode
        self.color_output = color_output and self._supports_color()
        self.show_status = show_status
        
        # Display state
        self.conversation_state = ConversationState.IDLE
        self.last_status_update = 0.0
        self.status_update_interval = 0.5  # seconds
        self.transcript_cache = []
        self.current_user_input = ""
        
        # Register event handlers
        self._register_event_handlers()
        
        # Terminal colors
        if self.color_output:
            self.RESET = "\033[0m"
            self.BOLD = "\033[1m"
            self.RED = "\033[31m"
            self.GREEN = "\033[32m"
            self.YELLOW = "\033[33m"
            self.BLUE = "\033[34m"
            self.MAGENTA = "\033[35m"
            self.CYAN = "\033[36m"
            self.GRAY = "\033[90m"
        else:
            self.RESET = ""
            self.BOLD = ""
            self.RED = ""
            self.GREEN = ""
            self.YELLOW = ""
            self.BLUE = ""
            self.MAGENTA = ""
            self.CYAN = ""
            self.GRAY = ""
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for conversation events."""
        # State update events
        event_bus.on(EventType.CONVERSATION_STATE_CHANGED, self._handle_state_changed)
        
        # Speech events
        event_bus.on(EventType.USER_SPEECH_STARTED, self._handle_user_speech_started)
        event_bus.on(EventType.USER_SPEECH_CONTENT, self._handle_user_speech_content)
        event_bus.on(EventType.USER_SPEECH_FINISHED, self._handle_user_speech_finished)
        
        # Assistant events
        event_bus.on(EventType.ASSISTANT_MESSAGE_STARTED, self._handle_assistant_message_started)
        event_bus.on(EventType.ASSISTANT_MESSAGE_CONTENT, self._handle_assistant_message_content)
        event_bus.on(EventType.ASSISTANT_MESSAGE_COMPLETED, self._handle_assistant_message_completed)
        
        # Error events
        event_bus.on(EventType.ERROR, self._handle_error)
    
    def _handle_state_changed(self, event: Event) -> None:
        """
        Handle conversation state change events.
        
        Args:
            event: State change event
        """
        self.conversation_state = event.data.get("state", ConversationState.IDLE)
        self._update_status_display()
    
    def _handle_user_speech_started(self, event: Event) -> None:
        """
        Handle user speech started events.
        
        Args:
            event: User speech started event
        """
        if not self.compact_mode:
            timestamp = self._format_timestamp() if self.show_timestamps else ""
            print(f"\n{timestamp}{self.BOLD}{self.BLUE}You: {self.RESET}", end="", flush=True)
    
    def _handle_user_speech_content(self, event: Event) -> None:
        """
        Handle user speech content events.
        
        Args:
            event: User speech content event
        """
        transcript = event.data.get("text", "")
        if transcript and transcript != self.current_user_input:
            if self.compact_mode:
                # In compact mode, wait until the speech is finished
                self.current_user_input = transcript
            else:
                # Clear the line and print the updated transcript
                print(f"\r{' ' * 100}\r{self.BOLD}{self.BLUE}You: {self.RESET}{transcript}", end="", flush=True)
                self.current_user_input = transcript
    
    def _handle_user_speech_finished(self, event: Event) -> None:
        """
        Handle user speech finished events.
        
        Args:
            event: User speech finished event
        """
        final_transcript = event.data.get("text", "")
        
        if self.compact_mode:
            timestamp = self._format_timestamp() if self.show_timestamps else ""
            print(f"{timestamp}{self.BOLD}{self.BLUE}You: {self.RESET}{final_transcript}")
        else:
            # Add a newline to complete the transcript
            print()
        
        # Reset the current input
        self.current_user_input = ""
    
    def _handle_assistant_message_started(self, event: Event) -> None:
        """
        Handle assistant message started events.
        
        Args:
            event: Assistant message started event
        """
        if not self.compact_mode:
            timestamp = self._format_timestamp() if self.show_timestamps else ""
            print(f"{timestamp}{self.BOLD}{self.GREEN}Assistant: {self.RESET}", end="", flush=True)
        
        # Clear transcript cache
        self.transcript_cache = []
    
    def _handle_assistant_message_content(self, event: Event) -> None:
        """
        Handle assistant message content events.
        
        Args:
            event: Assistant message content event
        """
        content = event.data.get("text", "")
        
        if content:
            if self.compact_mode:
                # In compact mode, accumulate content for later display
                self.transcript_cache.append(content)
            else:
                # In normal mode, print incremental updates
                print(content, end="", flush=True)
    
    def _handle_assistant_message_completed(self, event: Event) -> None:
        """
        Handle assistant message completed events.
        
        Args:
            event: Assistant message completed event
        """
        if self.compact_mode and self.transcript_cache:
            # In compact mode, print the full message at once
            timestamp = self._format_timestamp() if self.show_timestamps else ""
            message = "".join(self.transcript_cache)
            print(f"{timestamp}{self.BOLD}{self.GREEN}Assistant: {self.RESET}{message}")
            self.transcript_cache = []
        else:
            # Add a newline to complete the output
            print()
    
    def _handle_error(self, event: Event) -> None:
        """
        Handle error events.
        
        Args:
            event: Error event
        """
        error_data = event.data.get("error", {})
        error_message = error_data.get("message", "Unknown error")
        error_type = error_data.get("type", "unknown")
        
        timestamp = self._format_timestamp() if self.show_timestamps else ""
        print(f"{timestamp}{self.BOLD}{self.RED}Error ({error_type}): {self.RESET}{error_message}")
    
    def _update_status_display(self) -> None:
        """Update the status display with current state information."""
        if not self.show_status:
            return
        
        # Rate limit status updates
        current_time = time.time()
        if current_time - self.last_status_update < self.status_update_interval:
            return
        
        self.last_status_update = current_time
        
        # Map states to display strings
        state_display = {
            ConversationState.IDLE: f"{self.GRAY}Idle{self.RESET}",
            ConversationState.CONNECTING: f"{self.YELLOW}Connecting...{self.RESET}",
            ConversationState.READY: f"{self.GREEN}Ready{self.RESET}",
            ConversationState.USER_SPEAKING: f"{self.BLUE}Listening...{self.RESET}",
            ConversationState.THINKING: f"{self.YELLOW}Thinking...{self.RESET}",
            ConversationState.ASSISTANT_SPEAKING: f"{self.GREEN}Speaking...{self.RESET}",
            ConversationState.ERROR: f"{self.RED}Error{self.RESET}",
            ConversationState.DISCONNECTED: f"{self.GRAY}Disconnected{self.RESET}",
        }
        
        # Get display string for current state
        status = state_display.get(self.conversation_state, f"{self.GRAY}Unknown{self.RESET}")
        
        # Print status (replacing the current line)
        sys.stdout.write(f"\r{status}{' ' * 20}\r")
        sys.stdout.flush()
    
    def _format_timestamp(self) -> str:
        """
        Format a timestamp for display.
        
        Returns:
            Formatted timestamp string
        """
        current_time = time.strftime("%H:%M:%S")
        return f"{self.GRAY}[{current_time}] {self.RESET}"
    
    def _supports_color(self) -> bool:
        """
        Check if the terminal supports colored output.
        
        Returns:
            True if color is supported
        """
        # Check for NO_COLOR environment variable (https://no-color.org/)
        if os.environ.get("NO_COLOR"):
            return False
        
        # Check for color support in various terminals
        plat = sys.platform
        supported_platform = plat != 'Pocket PC' and (plat != 'win32' or 'ANSICON' in os.environ)
        
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        return supported_platform and is_a_tty
    
    def display_welcome_message(self) -> None:
        """Display a welcome message when the application starts."""
        print(f"\n{self.BOLD}{self.CYAN}=== OpenAI Realtime Assistant ==={self.RESET}")
        print(f"{self.GRAY}Speak naturally or type commands prefixed with '/'{self.RESET}")
        print(f"{self.GRAY}Available commands: /quit, /help, /restart, /pause, /resume{self.RESET}\n")
    
    def display_help(self) -> None:
        """Display help information."""
        print(f"\n{self.BOLD}Available Commands:{self.RESET}")
        print(f"  {self.BOLD}/quit{self.RESET} - Exit the application")
        print(f"  {self.BOLD}/help{self.RESET} - Display this help message")
        print(f"  {self.BOLD}/restart{self.RESET} - Restart the conversation")
        print(f"  {self.BOLD}/pause{self.RESET} - Pause listening")
        print(f"  {self.BOLD}/resume{self.RESET} - Resume listening")
        print(f"  {self.BOLD}/interrupt{self.RESET} - Interrupt the assistant")
        print(f"  {self.BOLD}/status{self.RESET} - Toggle status display")
        print(f"  {self.BOLD}/timestamps{self.RESET} - Toggle timestamps")
        print(f"  {self.BOLD}/compact{self.RESET} - Toggle compact mode")
        print()
    
    def parse_command(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Parse user input for commands.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (is_command, command, args)
        """
        text = text.strip()
        
        # Check if this is a command (starts with /)
        if not text.startswith('/'):
            return False, None, None
        
        # Split into command and arguments
        parts = text.split(' ', 1)
        command = parts[0][1:].lower()  # Remove the leading /
        args = parts[1] if len(parts) > 1 else None
        
        return True, command, args
    
    def process_command(
        self, 
        command: str, 
        args: Optional[str] = None, 
        command_callbacks: Optional[Dict[str, callable]] = None
    ) -> bool:
        """
        Process a command from the user.
        
        Args:
            command: Command name (without the leading /)
            args: Command arguments
            command_callbacks: Dictionary of command callbacks
            
        Returns:
            True if application should continue, False if it should exit
        """
        # Handle internal commands
        if command == "help":
            self.display_help()
            return True
        
        elif command == "status":
            self.show_status = not self.show_status
            print(f"Status display: {'on' if self.show_status else 'off'}")
            return True
        
        elif command == "timestamps":
            self.show_timestamps = not self.show_timestamps
            print(f"Timestamps: {'on' if self.show_timestamps else 'off'}")
            return True
        
        elif command == "compact":
            self.compact_mode = not self.compact_mode
            print(f"Compact mode: {'on' if self.compact_mode else 'off'}")
            return True
        
        # Handle external commands via callbacks
        if command_callbacks and command in command_callbacks:
            command_callbacks[command](args)
            return True
        
        # Handle quit command
        if command in ["quit", "exit", "bye"]:
            print("Exiting application...")
            return False
        
        # Unknown command
        print(f"{self.YELLOW}Unknown command: /{command}{self.RESET}")
        print(f"Type {self.BOLD}/help{self.RESET} for a list of commands")
        return True
    
    async def input_loop(
        self, 
        command_callbacks: Optional[Dict[str, callable]] = None
    ) -> None:
        """
        Run the input loop to process user commands from the terminal.
        
        Args:
            command_callbacks: Dictionary of command callbacks for external commands
        """
        # Display welcome message
        self.display_welcome_message()
        
        # Create an event for signaling shutdown
        shutdown_event = asyncio.Event()
        
        # Function to handle terminal input
        async def handle_input():
            try:
                while not shutdown_event.is_set():
                    # Get input in a non-blocking way
                    try:
                        line = await asyncio.to_thread(input)
                    except EOFError:
                        # Handle CTRL+D
                        shutdown_event.set()
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this is a command
                    is_command, command, args = self.parse_command(line)
                    
                    if is_command:
                        # Process command
                        should_continue = self.process_command(command, args, command_callbacks)
                        if not should_continue:
                            shutdown_event.set()
                            break
                    else:
                        # Regular text input - emit as text command
                        if command_callbacks and "text_input" in command_callbacks:
                            command_callbacks["text_input"](line)
            except Exception as e:
                logger.error(f"Error in input loop: {str(e)}")
                shutdown_event.set()
        
        # Start the input handling task
        input_task = asyncio.create_task(handle_input())
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Cancel input task if it's still running
        if not input_task.done():
            input_task.cancel()
            try:
                await input_task
            except asyncio.CancelledError:
                pass


def print_cli_header() -> None:
    """Print application header with version information."""
    header = """
╭──────────────────────────────────────────────╮
│ OpenAI Realtime Assistant                     │
│ Version: 0.1.0                                │
│ Realtime API - Voice Conversation Interface   │
╰──────────────────────────────────────────────╯
"""
    print(header) 