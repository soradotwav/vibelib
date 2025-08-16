from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass(frozen=True)
class Config:
    """Configuration for VibeLib operations."""
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"  # Fixed: was "chatgpt-4o-mini"
    timeout: float = 30.0
    max_retries: int = 3
    temperature: float = 0.0

    def __post_init__(self) -> None:
        # Validate that we can resolve an API key
        resolved = self.resolved_api_key
        if not resolved or resolved.strip() == "":
            raise ValueError("API key required")

    @property
    def resolved_api_key(self) -> str:
        """Get the API key, preferring direct value over environment variable."""
        if self.api_key:
            return self.api_key
        env_key = os.getenv('OPENAI_API_KEY', '')
        return env_key

    def __repr__(self) -> str:
        """Custom repr that hides the API key for security."""
        api_key_display = "***" if self.api_key else None
        return (
            f"Config("
            f"api_key={api_key_display}, "
            f"model='{self.model}', "
            f"timeout={self.timeout}, "
            f"max_retries={self.max_retries}, "
            f"temperature={self.temperature}"
            f")"
        )
