"""
Comprehensive tests for the OpenAI API client wrapper.

Tests all client functionality including initialization, requests,
error handling, retry logic, and integration scenarios.
"""

import pytest
from unittest.mock import patch, call, MagicMock

from vibelib.client import Client
from vibelib.config import Config
from vibelib.exceptions import APIError, ConfigurationError


class TestClientInitialization:
    """Test client initialization scenarios."""

    def test_successful_initialization(self, sample_config, mock_openai_client):
        """Test successful client initialization."""
        client = Client(sample_config)

        mock_openai_client.assert_called_once_with(api_key="test-api-key-12345")
        assert client._config == sample_config

    def test_initialization_with_custom_config(self, custom_config, mock_openai_client):
        """Test initialization with custom configuration."""
        client = Client(custom_config)

        mock_openai_client.assert_called_once_with(api_key="custom-test-key")
        assert client._config == custom_config

    def test_initialization_failure_generic_error(self, sample_config):
        """Test initialization failure with generic error."""
        with patch('vibelib.client.OpenAI', side_effect=Exception("OpenAI SDK error")):
            with pytest.raises(ConfigurationError, match="Client initialization failed: OpenAI SDK error"):
                Client(sample_config)

    def test_initialization_failure_invalid_key(self, sample_config):
        """Test initialization failure with invalid API key."""
        with patch('vibelib.client.OpenAI', side_effect=ValueError("Invalid API key")):
            with pytest.raises(ConfigurationError, match="Client initialization failed: Invalid API key"):
                Client(sample_config)

    def test_initialization_preserves_config_reference(self, sample_config, mock_openai_client):
        """Test that client preserves reference to original config."""
        client = Client(sample_config)

        assert client._config is sample_config
        assert client._config.api_key == sample_config.api_key


class TestBasicRequests:
    """Test basic request functionality."""

    def test_successful_request_user_prompt_only(self, sample_config, mock_openai_client):
        """Test successful request with only user prompt."""
        client = Client(sample_config)

        response = client.request("test prompt")

        assert response == '{"response": [1, 2, 3]}'
        mock_openai_client.return_value.chat.completions.create.assert_called_once()

    def test_successful_request_with_system_prompt(self, sample_config, mock_openai_client):
        """Test successful request with both system and user prompts."""
        client = Client(sample_config)

        response = client.request("user prompt", "system prompt")

        assert response == '{"response": [1, 2, 3]}'

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == 'system prompt'
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'user prompt'

    def test_request_none_system_prompt(self, sample_config, mock_openai_client):
        """Test request with None system prompt."""
        client = Client(sample_config)

        client.request("user prompt", None)

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'user prompt'

    def test_request_empty_system_prompt(self, sample_config, mock_openai_client):
        """Test request with empty string system prompt."""
        client = Client(sample_config)

        client.request("user prompt", "")

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        assert len(messages) == 1  # Empty string should not add system message

    def test_request_whitespace_system_prompt(self, sample_config, mock_openai_client):
        """Test request with whitespace-only system prompt."""
        client = Client(sample_config)

        client.request("user prompt", "   \n\t  ")

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        # Whitespace-only system prompt should still be included
        assert len(messages) == 2
        assert messages[0]['content'] == "   \n\t  "


class TestRequestParameters:
    """Test request parameter handling."""

    def test_request_parameters_default_config(self, sample_config, mock_openai_client):
        """Test request parameters with default configuration."""
        client = Client(sample_config)

        client.request("test")

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'gpt-4o-mini'
        assert call_args.kwargs['temperature'] == 0.3
        assert call_args.kwargs['timeout'] == 30.0

    def test_request_parameters_custom_config(self, custom_config, mock_openai_client):
        """Test request parameters with custom configuration."""
        client = Client(custom_config)

        client.request("test")

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'gpt-4'
        assert call_args.kwargs['temperature'] == 0.7
        assert call_args.kwargs['timeout'] == 60.0

    def test_request_message_structure(self, sample_config, mock_openai_client):
        """Test that request messages are properly structured."""
        client = Client(sample_config)

        client.request("user message", "system instruction")

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        # Verify message structure
        assert isinstance(messages, list)
        assert all(isinstance(msg, dict) for msg in messages)
        assert all('role' in msg and 'content' in msg for msg in messages)


class TestResponseHandling:
    """Test response handling scenarios."""

    def test_response_content_stripping(self, sample_config, mock_openai_client):
        """Test that response content is properly stripped."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = "  \n  content  \n  "
        client = Client(sample_config)

        response = client.request("test")

        assert response == "content"

    def test_empty_response_content_raises_error(self, sample_config, mock_openai_client):
        """Test that None response content raises appropriate error."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = None
        client = Client(sample_config)

        with pytest.raises(APIError, match="Received empty response from API"):
            client.request("test")

    def test_empty_string_response_raises_error(self, sample_config, mock_openai_client):
        """Test that empty string response raises appropriate error."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = ""
        client = Client(sample_config)

        with pytest.raises(APIError, match="Received empty response from API"):
            client.request("test")

    def test_whitespace_only_response_raises_error(self, sample_config, mock_openai_client):
        """Test that whitespace-only response raises appropriate error."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = "   \n\t  "
        client = Client(sample_config)

        with pytest.raises(APIError, match="Received empty response from API"):
            client.request("test")

    @pytest.mark.parametrize("content", [
        "valid response",
        "response with\nnewlines",
        "response with special chars !@#$%",
        "unicode response café naïve",
        '{"json": "response"}',
        "very " + "long " * 100 + "response"
    ])
    def test_valid_response_content(self, sample_config, mock_openai_client, content):
        """Test various valid response content scenarios."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = content
        client = Client(sample_config)

        response = client.request("test")

        assert response == content


class TestRetryLogic:
    """Test retry logic and error handling."""

    @patch('time.sleep')
    def test_retry_success_on_second_attempt(self, mock_sleep, sample_config, mock_openai_client):
        """Test successful retry on second attempt."""
        # First call fails, second succeeds
        mock_openai_client.return_value.chat.completions.create.side_effect = [
            Exception("Network error"),
            mock_openai_client.return_value.chat.completions.create.return_value
        ]

        client = Client(sample_config)
        response = client.request("test")

        assert response == '{"response": [1, 2, 3]}'
        assert mock_openai_client.return_value.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1

    @patch('time.sleep')
    def test_retry_success_on_third_attempt(self, mock_sleep, sample_config, mock_openai_client):
        """Test successful retry on third attempt."""
        mock_response = mock_openai_client.return_value.chat.completions.create.return_value
        mock_openai_client.return_value.chat.completions.create.side_effect = [
            Exception("Network error"),
            Exception("Rate limit"),
            mock_response
        ]

        client = Client(sample_config)
        response = client.request("test")

        assert response == '{"response": [1, 2, 3]}'
        assert mock_openai_client.return_value.chat.completions.create.call_count == 3
        expected_calls = [call(1), call(2)]  # 2^0=1, 2^1=2
        mock_sleep.assert_has_calls(expected_calls)

    @patch('time.sleep')
    def test_retry_exhaustion_default_config(self, mock_sleep, sample_config, mock_openai_client):
        """Test retry exhaustion with default configuration."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("Persistent error")

        client = Client(sample_config)

        with pytest.raises(APIError, match="Request failed after 3 attempts"):
            client.request("test")

        assert mock_openai_client.return_value.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries, not after last

    @patch('time.sleep')
    def test_retry_exhaustion_custom_config(self, mock_sleep, mock_openai_client):
        """Test retry exhaustion with custom retry count."""
        config = Config(api_key="test", max_retries=5)
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("Persistent error")

        client = Client(config)

        with pytest.raises(APIError, match="Request failed after 5 attempts"):
            client.request("test")

        assert mock_openai_client.return_value.chat.completions.create.call_count == 5
        assert mock_sleep.call_count == 4

    @patch('time.sleep')
    def test_exponential_backoff_pattern(self, mock_sleep, sample_config, mock_openai_client):
        """Test exponential backoff timing pattern."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("Error")

        client = Client(sample_config)

        with pytest.raises(APIError):
            client.request("test")

        # Check exponential backoff: 2^0=1, 2^1=2
        expected_calls = [call(1), call(2)]
        mock_sleep.assert_has_calls(expected_calls)

    @patch('time.sleep')
    def test_different_exception_types_retried(self, mock_sleep, sample_config, mock_openai_client):
        """Test that different exception types are properly retried."""
        exceptions = [
            ConnectionError("Connection failed"),
            TimeoutError("Request timed out"),
            ValueError("Invalid response")
        ]
        mock_openai_client.return_value.chat.completions.create.side_effect = exceptions

        client = Client(sample_config)

        with pytest.raises(APIError) as exc_info:
            client.request("test")

        assert "Invalid response" in str(exc_info.value)
        assert mock_openai_client.return_value.chat.completions.create.call_count == 3

    def test_no_retry_on_first_success(self, sample_config, mock_openai_client):
        """Test that no retry occurs on first success."""
        client = Client(sample_config)

        with patch('time.sleep') as mock_sleep:
            response = client.request("test")

        assert response == '{"response": [1, 2, 3]}'
        mock_sleep.assert_not_called()
        assert mock_openai_client.return_value.chat.completions.create.call_count == 1

    @patch('time.sleep')
    def test_error_chaining_in_final_exception(self, mock_sleep, sample_config, mock_openai_client):
        """Test that original exception is chained in final APIError."""
        original_error = ConnectionError("Network failed")
        mock_openai_client.return_value.chat.completions.create.side_effect = original_error

        client = Client(sample_config)

        with pytest.raises(APIError) as exc_info:
            client.request("test")

        assert exc_info.value.__cause__ == original_error

    @patch('time.sleep')
    def test_progressive_backoff_timing(self, mock_sleep, mock_openai_client):
        """Test that backoff timing increases progressively."""
        config = Config(api_key="test", max_retries=6)
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("Error")

        client = Client(config)

        with pytest.raises(APIError):
            client.request("test")

        # Check progressive backoff: 1, 2, 4, 8, 16
        expected_calls = [call(2**i) for i in range(5)]
        mock_sleep.assert_has_calls(expected_calls)


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""

    def test_malformed_response_structure(self, sample_config, mock_openai_client):
        """Test handling of malformed response structure."""
        # Simulate malformed response without choices
        mock_response = MagicMock()
        mock_response.choices = []
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response

        client = Client(sample_config)

        with pytest.raises(APIError):
            client.request("test")

    def test_response_without_message(self, sample_config, mock_openai_client):
        """Test handling of response without message content."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = None
        mock_response.choices = [mock_choice]
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response

        client = Client(sample_config)

        with pytest.raises(APIError):
            client.request("test")

    @pytest.mark.parametrize("error_type", [
        ConnectionError,
        TimeoutError,
        ValueError,
        RuntimeError,
        Exception
    ])
    def test_various_openai_exceptions(self, sample_config, mock_openai_client, error_type):
        """Test handling of various OpenAI exception types."""
        mock_openai_client.return_value.chat.completions.create.side_effect = error_type("Test error")

        client = Client(sample_config)

        with pytest.raises(APIError):
            client.request("test")


class TestConcurrency:
    """Test concurrent usage scenarios."""

    def test_multiple_clients_isolation(self, mock_openai_client):
        """Test that multiple clients work independently."""
        config1 = Config(api_key="key1")
        config2 = Config(api_key="key2")

        client1 = Client(config1)
        client2 = Client(config2)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = "response"

        response1 = client1.request("test1")
        response2 = client2.request("test2")

        assert response1 == "response"
        assert response2 == "response"
        assert mock_openai_client.return_value.chat.completions.create.call_count == 2

    def test_concurrent_requests_same_client(self, sample_config, mock_openai_client):
        """Test concurrent requests on the same client."""
        client = Client(sample_config)

        # Simulate concurrent calls (in reality would be threaded)
        responses = []
        for i in range(5):
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f"response{i}"
            response = client.request(f"test{i}")
            responses.append(response)

        assert len(responses) == 5
        assert all(f"response{i}" == responses[i] for i in range(5))


class TestLoggingAndDebugging:
    """Test logging and debugging functionality."""

    def test_debug_logging_on_success(self, sample_config, mock_openai_client, capture_logs):
        """Test that successful requests generate appropriate logs."""
        client = Client(sample_config)

        client.request("test prompt")

        log_contents = capture_logs.getvalue()
        assert "Making API request (attempt 1/3)" in log_contents
        assert "API request successful" in log_contents

    @patch('time.sleep')
    def test_debug_logging_on_retry(self, mock_sleep, sample_config, mock_openai_client, capture_logs):
        """Test that retries generate appropriate logs."""
        mock_openai_client.return_value.chat.completions.create.side_effect = [
            Exception("Network error"),
            mock_openai_client.return_value.chat.completions.create.return_value
        ]

        client = Client(sample_config)
        client.request("test")

        log_contents = capture_logs.getvalue()
        assert "API request attempt 1 failed" in log_contents
        assert "Retrying in 1 seconds" in log_contents
        assert "Making API request (attempt 2/3)" in log_contents
