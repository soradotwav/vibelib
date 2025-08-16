"""
VibeLib: AI-powered computational operations

Copyright (C) 2024 Your Name

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from typing import List, Optional, Union

from .client import Client
from .config import Config
from .sort import SortingService, Sortable
from .exceptions import (
    VibeLibError,
    ConfigurationError,
    APIError,
    ParseError,
    ValidationError
)

__version__ = "1.0.1"
__all__ = [
    "sort",
    "Config",
    "Client",
    "SortingService",
    "VibeLibError",
    "ConfigurationError",
    "APIError",
    "ParseError",
    "ValidationError"
]

_default_service: Optional[SortingService] = None
_current_api_key: Optional[str] = None

def _get_service(api_key: Optional[str] = None) -> SortingService:
    """Get or create sorting service instance."""
    global _default_service, _current_api_key

    effective_key = api_key
    if effective_key is None:
        import os
        effective_key = os.getenv('OPENAI_API_KEY')

    if _default_service is None or _current_api_key != effective_key:
        config = Config(api_key=api_key)
        client = Client(config)
        _default_service = SortingService(client)
        _current_api_key = effective_key

    return _default_service

def sort(items: List[Union[int, float, str]],
         api_key: Optional[str] = None) -> List[Union[int, float, str]]:
    """
    Sort an array using AI processing.

    Args:
        items: Array to sort
        api_key: Optional API key override

    Returns:
        Sorted array

    Raises:
        ValidationError: Invalid input
        APIError: Service communication failure
        ParseError: Response parsing failure
    """
    service = _get_service(api_key)
    return service.sort(items)
