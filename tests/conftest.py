"""
Comprehensive test fixtures and configuration for VibeLib.

Provides enterprise-grade test setup with proper isolation,
mocking, and state management across all test scenarios.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Test constants
TEST_API_KEY = "test-api-key-12345"
CUSTOM_API_KEY = "custom-test-key"
ENV_API_KEY = "env-test-key"

# Mock response templates
MOCK_JSON_RESPONSE = '{"response": [1, 2, 3]}'
MOCK_STRING_RESPONSE = '{"response": "test_result"}'
MOCK_NUMBER_RESPONSE = '{"response": 42}'


@pytest.fixture(scope="function", autouse=True)
def reset_global_state():
    """
    Reset all global state before and after each test.

    Ensures complete test isolation by clearing environment variables,
    global service instances, and any cached state.
    """
    # Store original environment
    original_env = os.environ.copy()

    # Reset VibeLib global state
    import vibelib

    # Clear all global service variables
    global_attrs = [
        '_services', '_current_api_key', '_sortingservice', '_basicservice',
        '_stringservice', '_listservice', '_default_service'
    ]

    original_values = {}
    for attr in global_attrs:
        if hasattr(vibelib, attr):
            original_values[attr] = getattr(vibelib, attr)
            if attr == '_services':
                setattr(vibelib, attr, {})
            else:
                setattr(vibelib, attr, None)

    yield

    # Restore original state
    os.environ.clear()
    os.environ.update(original_env)

    # Restore global attributes
    for attr, value in original_values.items():
        if hasattr(vibelib, attr):
            setattr(vibelib, attr, value)


@pytest.fixture
def sample_config():
    """Standard test configuration with minimal required parameters."""
    from vibelib.config import Config
    return Config(api_key=TEST_API_KEY)


@pytest.fixture
def minimal_config():
    """Minimal configuration for basic testing scenarios."""
    from vibelib.config import Config
    return Config(api_key="minimal-key")


@pytest.fixture
def custom_config():
    """Custom configuration with all parameters specified."""
    from vibelib.config import Config
    return Config(
        api_key=CUSTOM_API_KEY,
        model="gpt-4",
        timeout=60.0,
        max_retries=5,
        temperature=0.7
    )


@pytest.fixture
def high_performance_config():
    """Configuration optimized for performance testing."""
    from vibelib.config import Config
    return Config(
        api_key=TEST_API_KEY,
        model="gpt-4o-mini",
        timeout=120.0,
        max_retries=1,
        temperature=0.1
    )


@pytest.fixture
def mock_openai_client():
    """
    Mock OpenAI client with realistic response structure.

    Provides a complete mock of the OpenAI client including
    proper response structure and configurable behavior.
    """
    with patch('vibelib.client.OpenAI') as mock_openai:
        # Create properly structured mock response
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_choice = MagicMock()

        # Default response content
        mock_message.content = MOCK_JSON_RESPONSE
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Configure the mock
        mock_client_instance = mock_openai.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        yield mock_openai


@pytest.fixture
def mock_client():
    """Mock VibeLib client for service testing."""
    from vibelib.client import Client
    client = Mock(spec=Client)
    client.request.return_value = MOCK_JSON_RESPONSE
    return client


@pytest.fixture
def real_client(sample_config, mock_openai_client):
    """Real client instance with mocked OpenAI backend."""
    from vibelib.client import Client
    return Client(sample_config)


@pytest.fixture
def isolated_openai_mock():
    """Completely isolated OpenAI mock for specific test scenarios."""
    with patch('vibelib.client.OpenAI') as mock:
        mock_instance = mock.return_value
        mock_response = MagicMock()
        mock_response.choices[0].message.content = MOCK_JSON_RESPONSE
        mock_instance.chat.completions.create.return_value = mock_response
        yield mock


# Service fixtures
@pytest.fixture
def sorting_service(mock_client):
    """SortingService with mocked client."""
    from vibelib.operations.sorting import SortingService
    return SortingService(mock_client)


@pytest.fixture
def basic_service(mock_client):
    """BasicService with mocked client."""
    from vibelib.operations.basic import BasicService
    return BasicService(mock_client)


@pytest.fixture
def string_service(mock_client):
    """StringService with mocked client."""
    from vibelib.operations.strings import StringService
    return StringService(mock_client)


@pytest.fixture
def list_service(mock_client):
    """ListService with mocked client."""
    from vibelib.operations.lists import ListService
    return ListService(mock_client)


# Integration test fixtures
@pytest.fixture
def integration_service(sample_config, mock_openai_client):
    """Service with full integration chain for end-to-end testing."""
    from vibelib.client import Client
    from vibelib.operations.sorting import SortingService
    client = Client(sample_config)
    return SortingService(client)


# Environment fixtures
@pytest.fixture
def clean_environment():
    """Clean environment with no OpenAI-related variables."""
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def env_with_api_key():
    """Environment with OPENAI_API_KEY set."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': ENV_API_KEY}):
        yield


# Data fixtures for parameterized tests
@pytest.fixture
def test_data_sets():
    """Comprehensive test data for various scenarios."""
    return {
        'integers': [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
        'floats': [3.14, 1.41, 2.71, 0.57],
        'strings': ['cherry', 'apple', 'banana', 'date'],
        'mixed_numbers': [3, 1.5, 4, 2.7, 5],
        'unicode_strings': ['café', 'naïve', 'résumé', 'piñata'],
        'empty': [],
        'single': [42],
        'duplicates': [1, 1, 2, 2, 3, 3],
        'negative_numbers': [-3, -1, 0, 1, 3],
        'large_numbers': [1e6, 1e7, 1e8],
        'special_strings': ['', ' ', '\\t', '\\n', 'hello world']
    }


@pytest.fixture
def performance_data():
    """Large datasets for performance testing."""
    return {
        'small': list(range(10)),
        'medium': list(range(100)),
        'large': list(range(1000)),
        'very_large': list(range(5000)),
        'reverse_sorted': list(range(1000, 0, -1)),
        'random_floats': [i + 0.5 for i in range(500)]
    }


# Error simulation fixtures
@pytest.fixture
def api_error_scenarios():
    """Various API error scenarios for testing."""
    return {
        'network_error': ConnectionError("Network connection failed"),
        'timeout_error': TimeoutError("Request timed out"),
        'auth_error': ValueError("Invalid API key"),
        'rate_limit': Exception("Rate limit exceeded"),
        'server_error': Exception("Internal server error"),
        'empty_response': None,
        'malformed_json': 'this is not json',
        'wrong_json_format': '{"data": [1, 2, 3]}',
        'missing_response_field': '{"result": [1, 2, 3]}'
    }


# Utility fixtures
@pytest.fixture
def capture_logs():
    """Capture and provide access to log messages during tests."""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)

    # Add handler to all VibeLib loggers
    loggers = [
        logging.getLogger('vibelib'),
        logging.getLogger('vibelib.client'),
        logging.getLogger('vibelib.operations'),
    ]

    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    yield log_capture

    # Cleanup
    for logger in loggers:
        logger.removeHandler(handler)


@pytest.fixture
def response_templates():
    """Template responses for different operation types."""
    return {
        'sort_integers': '{"response": [1, 2, 3, 4, 5]}',
        'sort_strings': '{"response": ["a", "b", "c"]}',
        'max_result': '{"response": 42}',
        'min_result': '{"response": 1}',
        'sum_result': '{"response": 100}',
        'abs_result': '{"response": 5}',
        'upper_result': '{"response": "HELLO"}',
        'lower_result': '{"response": "hello"}',
        'split_result': '{"response": ["a", "b", "c"]}',
        'join_result': '{"response": "a,b,c"}',
        'strip_result': '{"response": "hello"}',
        'replace_result': '{"response": "hexxo"}',
        'count_result': '{"response": 2}',
        'index_result': '{"response": 1}',
        'reverse_result': '{"response": [3, 2, 1]}'
    }


# Configuration validation fixtures
@pytest.fixture
def invalid_config_scenarios():
    """Invalid configuration scenarios for testing."""
    return {
        'negative_temperature': {'temperature': -0.1},
        'high_temperature': {'temperature': 1.1},
        'zero_retries': {'max_retries': 0},
        'negative_retries': {'max_retries': -1},
        'zero_timeout': {'timeout': 0},
        'negative_timeout': {'timeout': -10},
        'empty_api_key': {'api_key': ''},
        'whitespace_api_key': {'api_key': '   '}
    }
