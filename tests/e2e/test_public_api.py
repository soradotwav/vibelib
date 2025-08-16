import pytest
import os
from unittest.mock import patch, MagicMock
import vibelib
from vibelib.exceptions import ValidationError, APIError, ConfigurationError, ParseError


class TestPublicAPI:

    @pytest.fixture(autouse=True)
    def reset_global_service(self):
        """Reset global service before each test."""
        vibelib._default_service = None
        yield
        vibelib._default_service = None

    def test_sort_function_with_direct_api_key(self, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        result = vibelib.sort([3, 1, 2], api_key="test-key")

        assert result == [1, 2, 3]
        mock_openai_client.assert_called_with(api_key="test-key")

    def test_sort_function_with_env_var(self, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            result = vibelib.sort([3, 1, 2])

        assert result == [1, 2, 3]
        mock_openai_client.assert_called_with(api_key="env-key")

    def test_sort_function_no_api_key_raises_error(self, mock_openai_client):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                vibelib.sort([1, 2, 3])

    @pytest.mark.parametrize("input_data,ai_response,expected", [
        ([3, 1, 2], '{"response": [1, 2, 3]}', [1, 2, 3]),
        ([3.5, 1.1, 2.2], '{"response": [1.1, 2.2, 3.5]}', [1.1, 2.2, 3.5]),
        (["c", "a", "b"], '{"response": ["a", "b", "c"]}', ["a", "b", "c"]),
        ([1], '{"response": [1]}', [1]),
        ([], '{"response": []}', []),
    ])
    def test_sort_function_different_types(self, mock_openai_client, input_data, ai_response, expected):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = ai_response

        result = vibelib.sort(input_data, api_key="test-key")

        assert result == expected

    def test_sort_function_validation_error(self):
        with pytest.raises(ValidationError, match="Expected list"):
            vibelib.sort("not a list", api_key="test-key")

    def test_sort_function_api_error_propagation(self, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(APIError, match="Request failed"):
            vibelib.sort([1, 2, 3], api_key="test-key")

    def test_sort_function_json_parse_error(self, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"wrong": [1, 2, 3]}'

        with pytest.raises(ParseError, match="Invalid JSON response"):
            vibelib.sort([1, 2, 3], api_key="test-key")

    def test_sort_function_configuration_error(self):
        with patch('vibelib.client.OpenAI', side_effect=Exception("Invalid API key")):
            with pytest.raises(ConfigurationError):
                vibelib.sort([1, 2, 3], api_key="invalid-key")

    def test_global_service_creation_and_reuse(self, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1]}'

        # First call creates service
        vibelib.sort([1], api_key="test-key")
        first_service = vibelib._default_service
        assert first_service is not None

        # Second call with same key reuses service
        vibelib.sort([1], api_key="test-key")
        second_service = vibelib._default_service

        assert first_service is second_service

    def test_global_service_recreation_with_different_key(self, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1]}'

        vibelib.sort([1], api_key="key1")
        first_service = vibelib._default_service

        vibelib.sort([1], api_key="key2")
        second_service = vibelib._default_service

        assert first_service is not second_service

    def test_global_service_with_env_var_then_direct_key(self, mock_openai_client):
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1]}'

        # First call with env var
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            vibelib.sort([1])
            env_service = vibelib._default_service

        # Second call with direct key should create new service
        vibelib.sort([1], api_key="direct-key")
        direct_service = vibelib._default_service

        assert env_service is not direct_service

    def test_empty_array_optimization(self, mock_openai_client):
        result = vibelib.sort([], api_key="test-key")

        assert result == []
        # Should not make API call for empty array
        mock_openai_client.return_value.chat.completions.create.assert_not_called()

    def test_single_element_optimization(self, mock_openai_client):
        result = vibelib.sort([42], api_key="test-key")

        assert result == [42]
        # Should not make API call for single element
        mock_openai_client.return_value.chat.completions.create.assert_not_called()

    @pytest.mark.parametrize("large_input_size", [100, 500, 1000])
    def test_sort_function_large_inputs(self, mock_openai_client, large_input_size):
        large_input = list(range(large_input_size, 0, -1))
        expected_output = list(range(1, large_input_size + 1))
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {expected_output}}}'

        result = vibelib.sort(large_input, api_key="test-key")

        assert len(result) == large_input_size
        assert result == expected_output

    def test_sort_function_too_large_input(self):
        large_array = list(range(10001))

        with pytest.raises(ValidationError, match="Array too large"):
            vibelib.sort(large_array, api_key="test-key")

    def test_module_level_imports(self):
        # Test that all expected items are available at module level
        assert hasattr(vibelib, 'sort')
        assert hasattr(vibelib, 'Config')
        assert hasattr(vibelib, '__version__')

        # Test that internal items are not exposed
        assert hasattr(vibelib, '_get_service')
        assert hasattr(vibelib, '_default_service')

    def test_version_attribute(self):
        assert hasattr(vibelib, '__version__')
        assert isinstance(vibelib.__version__, str)
        assert vibelib.__version__ == "1.0.0"

    def test_concurrent_calls_different_keys(self, mock_openai_client):
        # Simulate concurrent calls with different API keys
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2]}'

        result1 = vibelib.sort([2, 1], api_key="key1")
        result2 = vibelib.sort([2, 1], api_key="key2")

        assert result1 == [1, 2]
        assert result2 == [1, 2]
        assert mock_openai_client.return_value.chat.completions.create.call_count == 2

    def test_error_handling_preserves_global_state(self, mock_openai_client):
        # Test that errors don't corrupt global service state
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2]}'

        # Successful call
        vibelib.sort([2, 1], api_key="test-key")
        service_after_success = vibelib._default_service

        # Failed call due to validation
        with pytest.raises(ValidationError):
            vibelib.sort("invalid", api_key="test-key")

        # Service should be unchanged
        assert vibelib._default_service is service_after_success

        # Subsequent successful call should work
        result = vibelib.sort([2, 1], api_key="test-key")
        assert result == [1, 2]

    @patch('time.sleep')
    def test_end_to_end_retry_behavior(self, mock_sleep, mock_openai_client):
        # Test retry behavior through the entire public API
        responses = [
            Exception("Network error"),
            MagicMock()
        ]
        responses[1].choices[0].message.content = '{"response": [1, 2, 3]}'
        mock_openai_client.return_value.chat.completions.create.side_effect = responses

        result = vibelib.sort([3, 1, 2], api_key="test-key")

        assert result == [1, 2, 3]
        assert mock_openai_client.return_value.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1)

    def test_json_response_with_extra_fields(self, mock_openai_client):
        # Test that extra fields in JSON are handled gracefully
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3], "confidence": 0.95, "reasoning": "sorted numerically"}'

        result = vibelib.sort([3, 1, 2], api_key="test-key")

        assert result == [1, 2, 3]
