import pytest
from unittest.mock import patch, MagicMock
from vibelib.config import Config
from vibelib.client import Client
from vibelib.sort import SortingService
from vibelib.exceptions import APIError, ParseError, ConfigurationError, ValidationError


class TestSortingServiceIntegration:

    @pytest.fixture
    def integration_service(self, mock_openai_client):
        config = Config(api_key="test-key")
        client = Client(config)
        return SortingService(client)

    def test_full_integration_success(self, mock_openai_client, integration_service):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 3, 4]}'

        result = integration_service.sort([4, 1, 3])

        assert result == [1, 3, 4]
        mock_openai_client.return_value.chat.completions.create.assert_called_once()

    def test_config_to_client_to_service_chain(self, mock_openai_client):
        # Test that config values propagate through the entire chain
        config = Config(
            api_key="chain-test-key",
            model="gpt-4",
            temperature=0.7,
            timeout=45.0,
            max_retries=2
        )

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2]}'

        client = Client(config)
        service = SortingService(client)

        result = service.sort([2, 1])

        # Verify config propagated correctly
        call_kwargs = mock_openai_client.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs['model'] == 'gpt-4'
        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['timeout'] == 45.0
        assert result == [1, 2]

    def test_api_error_propagation(self, mock_openai_client, integration_service):
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("API down")

        with pytest.raises(APIError, match="Request failed"):
            integration_service.sort([1, 2, 3])

    def test_malformed_json_response_handling(self, mock_openai_client, integration_service):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2'

        with pytest.raises(ParseError, match="Invalid JSON response"):
            integration_service.sort([1, 2, 3])

    def test_wrong_json_format_handling(self, mock_openai_client, integration_service):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"data": [1, 2, 3]}'

        with pytest.raises(ParseError, match="Response must contain 'response' field"):
            integration_service.sort([1, 2, 3])

    @patch('time.sleep')
    def test_retry_behavior_integration(self, mock_sleep, mock_openai_client, integration_service):
        # First two calls fail, third succeeds
        responses = [
            Exception("Network error"),
            Exception("Rate limit"),
            MagicMock()
        ]
        responses[2].choices[0].message.content = '{"response": [1, 2, 3]}'

        mock_openai_client.return_value.chat.completions.create.side_effect = responses

        result = integration_service.sort([3, 1, 2])

        assert result == [1, 2, 3]
        assert mock_openai_client.return_value.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2  # Two retries

    def test_empty_response_error_propagation(self, mock_openai_client, integration_service):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = None

        with pytest.raises(APIError, match="Empty response"):
            integration_service.sort([1, 2, 3])

    def test_validation_bypasses_client(self, mock_openai_client):
        # Test that validation errors don't reach the client
        config = Config(api_key="test")
        client = Client(config)
        service = SortingService(client)

        with pytest.raises(ValidationError):
            service.sort("not a list")

        # Client should never be called
        mock_openai_client.return_value.chat.completions.create.assert_not_called()

    def test_client_initialization_error_propagation(self, mock_openai_client):
        config = Config(api_key="test")

        with patch('vibelib.client.OpenAI', side_effect=Exception("SDK Error")):
            with pytest.raises(ConfigurationError):
                Client(config)

    def test_different_data_types_integration(self, mock_openai_client):
        config = Config(api_key="test")
        client = Client(config)
        service = SortingService(client)

        test_cases = [
            ([3, 1, 2], '{"response": [1, 2, 3]}'),
            ([3.5, 1.1, 2.2], '{"response": [1.1, 2.2, 3.5]}'),
            (["c", "a", "b"], '{"response": ["a", "b", "c"]}'),
        ]

        for input_data, ai_response in test_cases:
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = ai_response
            result = service.sort(input_data)
            import json
            expected = json.loads(ai_response)["response"]
            assert result == expected

    def test_service_state_isolation(self, mock_openai_client):
        # Test that multiple service instances don't interfere
        config1 = Config(api_key="key1", model="gpt-3.5-turbo")
        config2 = Config(api_key="key2", model="gpt-4")

        client1 = Client(config1)
        client2 = Client(config2)

        service1 = SortingService(client1)
        service2 = SortingService(client2)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2]}'

        service1.sort([2, 1])
        service2.sort([2, 1])

        # Both should work independently
        assert mock_openai_client.return_value.chat.completions.create.call_count == 2

    def test_json_with_extra_fields_integration(self, mock_openai_client, integration_service):
        # Test that extra fields in JSON response are ignored
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3], "confidence": 0.95, "model": "gpt-4o-mini"}'

        result = integration_service.sort([3, 1, 2])

        assert result == [1, 2, 3]
