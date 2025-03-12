"""
Configuration package for the OpenAI Realtime Assistant.

This package contains modules for managing application settings,
environment variables, and logging configuration.
"""

from src.config.settings import Settings

# Export settings singleton for app-wide use
settings = Settings()

__all__ = ["settings"] 