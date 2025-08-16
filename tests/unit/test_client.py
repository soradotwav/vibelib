import pytest
from unittest.mock import patch, call
from vibelib.client import Client
from vibelib.config import Config
from vibelib.exceptions import APIError, ConfigurationError

class TestClient:

    def test_client_initialization_success(self, sample_config, mock_openai_client):
        client = Client(sample_config)
        mock_openai_client.assert_called_once_with(api_key="test-key-12345")

    def test_client_initialization_with_custom_config(self, custom_config, mock_openai_client):
        client = Client(custom_config)
        mock_openai_client.assert_called_once_with(api_key="custom-key")

    def test_client_initialization_failure(self, sample_config):
        with patch('vibelib.client.OpenAI', side_effect=Exception("OpenAI SDK error")):
            with pytest.raises(ConfigurationError, match="Client initialization failed"):
                Client(sample_config)

    def test_client_initialization_failure_with_invalid_key(self, sample_config):
        with patch('vibelib.client.OpenAI', side_effect=ValueError("Invalid API key")):
            with pytest.raises(ConfigurationError, match="Client initialization failed"):
                Client(sample_config)

    def test_successful_request_user_only(self, sample_config, mock_openai_client):
        client = Client(sample_config)

        response = client.request("test prompt")

        assert response == '{"response": [1, 2, 3]}'
        mock_openai_client.return_value.chat.completions.create.assert_called_once()

    def test_successful_request_with_system_prompt(self, sample_config, mock_openai_client):
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

    def test_request_with_none_system_prompt(self, sample_config, mock_openai_client):
        client = Client(sample_config)

        client.request("user prompt", None)

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        assert len(messages) == 1
        assert messages[0]['role'] == 'user'

    def test_request_with_empty_system_prompt(self, sample_config, mock_openai_client):
        client = Client(sample_config)

        client.request("user prompt", "")

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        assert len(messages) == 1  # Empty string should not add system message

    def test_request_parameters_default_config(self, sample_config, mock_openai_client):
        client = Client(sample_config)

        client.request("test")

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'gpt-4o-mini'
        assert call_args.kwargs['temperature'] == 0.0
        assert call_args.kwargs['timeout'] == 30.0

    def test_request_parameters_custom_config(self, custom_config, mock_openai_client):
        client = Client(custom_config)

        client.request("test")

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'gpt-4'
        assert call_args.kwargs['temperature'] == 0.5
        assert call_args.kwargs['timeout'] == 60.0

    def test_empty_response_content_raises_error(self, sample_config, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = None
        client = Client(sample_config)

        with pytest.raises(APIError, match="Empty response"):
            client.request("test")

    def test_empty_string_response_raises_error(self, sample_config, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = ""
        client = Client(sample_config)

        with pytest.raises(APIError, match="Empty response"):
            client.request("test")

    def test_whitespace_response_raises_error(self, sample_config, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = "   \n  "
        client = Client(sample_config)

        with pytest.raises(APIError, match="Empty response"):
            client.request("test")

    @patch('time.sleep')
    def test_retry_logic_success_on_second_attempt(self, mock_sleep, sample_config, mock_openai_client):
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
    def test_retry_logic_success_on_third_attempt(self, mock_sleep, sample_config, mock_openai_client):
        # First two calls fail, third succeeds
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
    def test_retry_logic_exhausted_default_retries(self, mock_sleep, sample_config, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("Persistent error")

        client = Client(sample_config)

        with pytest.raises(APIError, match="Request failed"):
            client.request("test")

        assert mock_openai_client.return_value.chat.completions.create.call_count == 3  # default max_retries
        assert mock_sleep.call_count == 2  # Sleep between retries, not after last

    @patch('time.sleep')
    def test_retry_logic_exhausted_custom_retries(self, mock_sleep, mock_openai_client):
        config = Config(api_key="test", max_retries=5)
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("Persistent error")

        client = Client(config)

        with pytest.raises(APIError, match="Request failed"):
            client.request("test")

        assert mock_openai_client.return_value.chat.completions.create.call_count == 5
        assert mock_sleep.call_count == 4

    @patch('time.sleep')
    def test_exponential_backoff_pattern(self, mock_sleep, sample_config, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("Error")

        client = Client(sample_config)

        with pytest.raises(APIError):
            client.request("test")

        # Check exponential backoff: 2^0=1, 2^1=2
        expected_calls = [call(1), call(2)]
        mock_sleep.assert_has_calls(expected_calls)

    @patch('time.sleep')
    def test_different_exception_types_retried(self, mock_sleep, sample_config, mock_openai_client):
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

    def test_no_retry_on_success(self, sample_config, mock_openai_client):
        client = Client(sample_config)

        with patch('time.sleep') as mock_sleep:
            response = client.request("test")

        assert response == '{"response": [1, 2, 3]}'
        mock_sleep.assert_not_called()
        assert mock_openai_client.return_value.chat.completions.create.call_count == 1
