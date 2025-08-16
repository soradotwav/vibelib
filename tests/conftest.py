import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from vibelib.config import Config
from vibelib.client import Client
from vibelib.sort import SortingService

@pytest.fixture
def sample_config():
    return Config(api_key="test-key-12345")

@pytest.fixture
def minimal_config():
    return Config(api_key="minimal-key")

@pytest.fixture
def custom_config():
    return Config(
        api_key="custom-key",
        model="gpt-4",
        timeout=60.0,
        max_retries=5,
        temperature=0.5
    )

@pytest.fixture
def mock_openai_client():
    with patch('vibelib.client.OpenAI') as mock:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"response": [1, 2, 3]}'
        mock.return_value.chat.completions.create.return_value = mock_response
        yield mock

@pytest.fixture
def mock_client():
    client = Mock(spec=Client)
    client.request.return_value = '{"response": [1, 2, 3]}'
    return client

@pytest.fixture
def sorting_service(mock_client):
    return SortingService(mock_client)

@pytest.fixture
def real_client(sample_config, mock_openai_client):
    return Client(sample_config)

@pytest.fixture
def real_service(real_client):
    return SortingService(real_client)

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables and global state before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)
    # Reset the global service
    import vibelib
    vibelib._default_service = None
