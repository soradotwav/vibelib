class VibeLibError(Exception):
    """Base exception for VibeLib operations."""

class ConfigurationError(VibeLibError):
    """Configuration-related errors."""

class APIError(VibeLibError):
    """API communication errors."""

class ParseError(VibeLibError):
    """Response parsing errors."""

class ValidationError(VibeLibError):
    """Input validation errors."""
