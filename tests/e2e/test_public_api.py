"""
Comprehensive tests for the public API of VibeLib.

Tests all public functions, global service management, error handling,
and real-world usage patterns through the main module interface.
"""

import pytest
from unittest.mock import patch, MagicMock

import vibelib
from vibelib.exceptions import ValidationError, APIError, ConfigurationError, ParseError


class TestPublicAPIBasics:
    """Test basic public API functionality."""

    def test_module_exports_all_expected_functions(self):
        """Test that all expected functions are exported by the module."""
        expected_functions = [
            'sort', 'max', 'min', 'sum', 'abs',
            'upper', 'lower', 'split', 'join', 'strip', 'replace',
            'count', 'index', 'reverse'
        ]

        for func_name in expected_functions:
            assert hasattr(vibelib, func_name), f"Function {func_name} not exported"
            assert callable(getattr(vibelib, func_name)), f"{func_name} is not callable"

    def test_module_exports_classes_and_exceptions(self):
        """Test that classes and exceptions are properly exported."""
        expected_classes = ['Config', 'Client']
        expected_exceptions = [
            'VibeLibError', 'ConfigurationError', 'APIError',
            'ParseError', 'ValidationError'
        ]

        for class_name in expected_classes:
            assert hasattr(vibelib, class_name), f"Class {class_name} not exported"

        for exc_name in expected_exceptions:
            assert hasattr(vibelib, exc_name), f"Exception {exc_name} not exported"

    def test_version_attribute_exists(self):
        """Test that version attribute exists and is properly formatted."""
        assert hasattr(vibelib, '__version__')
        assert isinstance(vibelib.__version__, str)
        assert vibelib.__version__ == "1.0.5"

    def test_all_attribute_completeness(self):
        """Test that __all__ contains all expected exports."""
        expected_all = [
            'sort', 'max', 'min', 'sum', 'abs',
            'upper', 'lower', 'split', 'join', 'strip', 'replace',
            'count', 'index', 'reverse',
            'Config', 'Client',
            'VibeLibError', 'ConfigurationError', 'APIError', 'ParseError', 'ValidationError'
        ]

        assert hasattr(vibelib, '__all__')
        assert set(vibelib.__all__) == set(expected_all)


class TestSortingAPI:
    """Test the public sorting API."""

    def test_sort_with_direct_api_key(self, mock_openai_client):
        """Test sort function with directly provided API key."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        result = vibelib.sort([3, 1, 2], api_key="direct-test-key")

        assert result == [1, 2, 3]
        mock_openai_client.assert_called_with(api_key="direct-test-key")

    def test_sort_with_environment_variable(self, mock_openai_client, env_with_api_key):
        """Test sort function using environment variable for API key."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        result = vibelib.sort([3, 1, 2])

        assert result == [1, 2, 3]
        mock_openai_client.assert_called_with(api_key="env-test-key")

    def test_sort_no_api_key_raises_error(self, mock_openai_client, clean_environment):
        """Test that sort without API key raises appropriate error."""
        with pytest.raises(ValueError, match="API key required"):
            vibelib.sort([1, 2, 3])

    @pytest.mark.parametrize("input_data,ai_response,expected", [
        ([3, 1, 2], '{"response": [1, 2, 3]}', [1, 2, 3]),
        ([3.5, 1.1, 2.2], '{"response": [1.1, 2.2, 3.5]}', [1.1, 2.2, 3.5]),
        (["c", "a", "b"], '{"response": ["a", "b", "c"]}', ["a", "b", "c"]),
        ([True, False], '{"response": [false, true]}', [False, True]),
        ([1], '{"response": [1]}', [1]),
        ([], '{"response": []}', []),
    ])
    def test_sort_with_different_data_types(self, mock_openai_client, input_data, ai_response, expected):
        """Test sort function with various data types."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = ai_response

        result = vibelib.sort(input_data, api_key="test-key")

        assert result == expected

    def test_sort_empty_array_optimization(self, mock_openai_client):
        """Test that empty arrays are optimized and don't make API calls."""
        result = vibelib.sort([], api_key="test-key")

        assert result == []
        mock_openai_client.return_value.chat.completions.create.assert_not_called()

    def test_sort_single_element_optimization(self, mock_openai_client):
        """Test that single elements are optimized and don't make API calls."""
        result = vibelib.sort([42], api_key="test-key")

        assert result == [42]
        mock_openai_client.return_value.chat.completions.create.assert_not_called()

    def test_sort_oversized_array_raises_validation_error(self):
        """Test that oversized arrays raise validation errors."""
        oversized_array = list(range(10001))

        with pytest.raises(ValidationError, match="Input too large"):
            vibelib.sort(oversized_array, api_key="test-key")


class TestBasicOperationsAPI:
    """Test the public basic operations API."""

    @pytest.mark.parametrize("func_name,input_data,ai_response,expected", [
        ("max", ([1, 5, 3],), '{"response": 5}', 5),
        ("max", ([1.1, 5.5, 3.3],), '{"response": 5.5}', 5.5),
        ("min", ([1, 5, 3],), '{"response": 1}', 1),
        ("min", ([1.1, 5.5, 3.3],), '{"response": 1.1}', 1.1),
        ("sum", ([1, 2, 3],), '{"response": 6}', 6),
        ("sum", ([1.1, 2.2, 3.3],), '{"response": 6.6}', 6.6),
        ("abs", (5,), '{"response": 5}', 5),
        ("abs", (-5,), '{"response": 5}', 5),
        ("abs", (-3.14,), '{"response": 3.14}', 3.14),
    ])
    def test_basic_operations_functions(self, mock_openai_client, func_name, input_data, ai_response, expected):
        """Test all basic operations functions through public API."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = ai_response

        func = getattr(vibelib, func_name)
        result = func(*input_data, api_key="test-key")

        assert result == expected

    def test_basic_operations_with_environment_key(self, mock_openai_client, env_with_api_key):
        """Test basic operations using environment API key."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 5}'

        result = vibelib.max([1, 5, 3])

        assert result == 5
        mock_openai_client.assert_called_with(api_key="env-test-key")

    def test_basic_operations_type_preservation(self, mock_openai_client):
        """Test that basic operations preserve type information in prompts."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 5}'

        vibelib.max([1, 5, 3], api_key="test-key")

        # Check that the prompt includes type preservation instructions
        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), "")

        assert "Preserve the original number type" in system_prompt


class TestStringOperationsAPI:
    """Test the public string operations API."""

    @pytest.mark.parametrize("func_name,input_data,ai_response,expected", [
        ("upper", ("hello",), '{"response": "HELLO"}', "HELLO"),
        ("lower", ("HELLO",), '{"response": "hello"}', "hello"),
        ("split", ("a,b,c", ","), '{"response": ["a", "b", "c"]}', ["a", "b", "c"]),
        ("join", (["a", "b", "c"], ","), '{"response": "a,b,c"}', "a,b,c"),
        ("strip", ("  hello  ",), '{"response": "hello"}', "hello"),
        ("replace", ("hello", "l", "x"), '{"response": "hexxo"}', "hexxo"),
    ])
    def test_string_operations_functions(self, mock_openai_client, func_name, input_data, ai_response, expected):
        """Test all string operations functions through public API."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = ai_response

        func = getattr(vibelib, func_name)
        result = func(*input_data, api_key="test-key")

        assert result == expected

    def test_string_operations_unicode_support(self, mock_openai_client):
        """Test string operations with Unicode characters."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": "CAFÉ NAÏVE"}'

        result = vibelib.upper("café naïve", api_key="test-key")

        assert result == "CAFÉ NAÏVE"

    def test_string_operations_empty_string_handling(self, mock_openai_client):
        """Test string operations with empty strings."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": ""}'

        result = vibelib.upper("", api_key="test-key")

        assert result == ""

    def test_join_with_mixed_types(self, mock_openai_client):
        """Test join operation with mixed data types."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": "1,hello,2.5"}'

        result = vibelib.join([1, "hello", 2.5], ",", api_key="test-key")

        assert result == "1,hello,2.5"


class TestListOperationsAPI:
    """Test the public list operations API."""

    @pytest.mark.parametrize("func_name,input_data,ai_response,expected", [
        ("count", ([1, 2, 1, 3], 1), '{"response": 2}', 2),
        ("count", (["a", "b", "a"], "a"), '{"response": 2}', 2),
        ("index", ([1, 2, 3], 2), '{"response": 1}', 1),
        ("index", (["a", "b", "c"], "c"), '{"response": 2}', 2),
        ("reverse", ([1, 2, 3],), '{"response": [3, 2, 1]}', [3, 2, 1]),
        ("reverse", (["a", "b", "c"],), '{"response": ["c", "b", "a"]}', ["c", "b", "a"]),
    ])
    def test_list_operations_functions(self, mock_openai_client, func_name, input_data, ai_response, expected):
        """Test all list operations functions through public API."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = ai_response

        func = getattr(vibelib, func_name)
        result = func(*input_data, api_key="test-key")

        assert result == expected

    def test_list_operations_with_mixed_types(self, mock_openai_client):
        """Test list operations with mixed data types."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 1}'

        result = vibelib.count([1, "hello", 2.5, "hello"], "hello", api_key="test-key")

        assert result == 1

    def test_list_operations_empty_list(self, mock_openai_client):
        """Test list operations with empty lists."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": []}'

        result = vibelib.reverse([], api_key="test-key")

        assert result == []


class TestGlobalServiceManagement:
    """Test global service caching and management."""

    def test_service_creation_and_caching(self, mock_openai_client):
        """Test that services are created and cached correctly."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # First call should create service
        vibelib.sort([3, 1, 2], api_key="cache-test-key")

        # Check that service was cached
        assert 'sortingservice' in vibelib._services
        first_service = vibelib._services['sortingservice']

        # Second call should reuse service
        vibelib.sort([2, 3, 1], api_key="cache-test-key")
        second_service = vibelib._services['sortingservice']

        assert first_service is second_service
        assert mock_openai_client.return_value.chat.completions.create.call_count == 2

    def test_service_recreation_with_different_api_key(self, mock_openai_client):
        """Test that services are recreated when API key changes."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # First call with key1
        vibelib.sort([3, 1, 2], api_key="key1")
        first_service = vibelib._services.get('sortingservice')

        # Second call with key2 should recreate service
        vibelib.sort([3, 1, 2], api_key="key2")
        second_service = vibelib._services.get('sortingservice')

        assert first_service is not second_service

    def test_service_isolation_between_operation_types(self, mock_openai_client):
        """Test that different operation types have isolated services."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # Call different operation types
        vibelib.sort([3, 1, 2], api_key="test-key")

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 5}'
        vibelib.max([1, 5, 3], api_key="test-key")

        # Should have different services cached
        assert 'sortingservice' in vibelib._services
        assert 'basicservice' in vibelib._services
        assert vibelib._services['sortingservice'] is not vibelib._services['basicservice']

    def test_global_state_reset_with_api_key_change(self, mock_openai_client, env_with_api_key):
        """Test that global state is properly reset when API key changes."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # First call with environment key
        vibelib.sort([3, 1, 2])
        original_key = vibelib._current_api_key

        # Second call with direct key should reset state
        vibelib.sort([3, 1, 2], api_key="direct-key")
        new_key = vibelib._current_api_key

        assert original_key != new_key
        assert new_key == "direct-key"

    def test_service_state_persistence_across_calls(self, mock_openai_client):
        """Test that service state persists across multiple calls."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        api_key = "persistence-test-key"

        # Multiple calls should use same service
        for i in range(5):
            vibelib.sort([3, 1, 2], api_key=api_key)

        assert mock_openai_client.return_value.chat.completions.create.call_count == 5
        assert vibelib._current_api_key == api_key


class TestErrorHandlingThroughPublicAPI:
    """Test error handling through the public API."""

    def test_configuration_error_propagation(self, clean_environment):
        """Test that configuration errors propagate through public API."""
        with pytest.raises(ValueError, match="API key required"):
            vibelib.sort([1, 2, 3])

    def test_api_error_propagation(self, mock_openai_client):
        """Test that API errors propagate through public API."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(APIError, match="Request failed"):
            vibelib.sort([1, 2, 3], api_key="test-key")

    def test_parse_error_propagation(self, mock_openai_client):
        """Test that parse errors propagate through public API."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = 'invalid json'

        with pytest.raises(ParseError, match="Invalid JSON response"):
            vibelib.sort([1, 2, 3], api_key="test-key")

    def test_validation_error_propagation(self):
        """Test that validation errors propagate through public API."""
        oversized_array = list(range(10001))

        with pytest.raises(ValidationError, match="Input too large"):
            vibelib.sort(oversized_array, api_key="test-key")

    def test_client_initialization_error_propagation(self):
        """Test that client initialization errors propagate through public API."""
        with patch('vibelib.client.OpenAI', side_effect=Exception("OpenAI SDK Error")):
            with pytest.raises(ConfigurationError):
                vibelib.sort([1, 2, 3], api_key="test-key")


class TestRetryBehaviorThroughPublicAPI:
    """Test retry behavior through the public API."""

    @patch('time.sleep')
    def test_successful_retry_through_public_api(self, mock_sleep, mock_openai_client):
        """Test successful retry behavior through public API."""
        # First call fails, second succeeds
        mock_success = MagicMock()
        mock_success.choices[0].message.content = '{"response": [1, 2, 3]}'

        mock_openai_client.return_value.chat.completions.create.side_effect = [
            Exception("Network error"),
            mock_success
        ]

        result = vibelib.sort([3, 1, 2], api_key="retry-test-key")

        assert result == [1, 2, 3]
        assert mock_openai_client.return_value.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1)

    @patch('time.sleep')
    def test_retry_exhaustion_through_public_api(self, mock_sleep, mock_openai_client):
        """Test retry exhaustion through public API."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("Persistent error")

        with pytest.raises(APIError, match="Request failed after 3 attempts"):
            vibelib.sort([1, 2, 3], api_key="retry-exhaustion-test-key")

        assert mock_openai_client.return_value.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2


class TestPerformanceAndScalability:
    """Test performance and scalability through public API."""

    @pytest.mark.parametrize("size", [10, 100, 500, 1000])
    def test_various_input_sizes(self, mock_openai_client, size):
        """Test public API with various input sizes."""
        test_array = list(reversed(range(size)))
        expected_sorted = list(range(size))
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {expected_sorted}}}'

        result = vibelib.sort(test_array, api_key="size-test-key")

        assert result == expected_sorted

    def test_large_input_performance(self, mock_openai_client, performance_data):
        """Test public API performance with large inputs."""
        large_array = performance_data['large']  # 1000 items
        expected_sorted = sorted(large_array)
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {expected_sorted}}}'

        result = vibelib.sort(large_array, api_key="performance-test-key")

        assert result == expected_sorted

    def test_multiple_concurrent_operations(self, mock_openai_client):
        """Test multiple concurrent operations through public API."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # Simulate concurrent operations (in reality would be threaded)
        results = []
        for i in range(10):
            result = vibelib.sort([3, 1, 2], api_key="concurrent-test-key")
            results.append(result)

        assert all(result == [1, 2, 3] for result in results)
        assert mock_openai_client.return_value.chat.completions.create.call_count == 10


class TestRealWorldUsagePatterns:
    """Test real-world usage patterns through public API."""

    def test_mixed_operation_workflow(self, mock_openai_client):
        """Test workflow using multiple different operations."""
        # Setup different responses for different operations
        responses = {
            'sort': '{"response": [1, 2, 3, 4, 5]}',
            'max': '{"response": 5}',
            'min': '{"response": 1}',
            'sum': '{"response": 15}',
            'upper': '{"response": "HELLO"}',
            'count': '{"response": 2}'
        }

        data = [3, 1, 4, 5, 2]
        api_key = "workflow-test-key"

        # Sort the data
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = responses['sort']
        sorted_data = vibelib.sort(data, api_key=api_key)

        # Find max
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = responses['max']
        max_val = vibelib.max(sorted_data, api_key=api_key)

        # Find min
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = responses['min']
        min_val = vibelib.min(sorted_data, api_key=api_key)

        # Calculate sum
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = responses['sum']
        total = vibelib.sum(sorted_data, api_key=api_key)

        assert sorted_data == [1, 2, 3, 4, 5]
        assert max_val == 5
        assert min_val == 1
        assert total == 15

    def test_error_recovery_in_workflow(self, mock_openai_client):
        """Test error recovery in multi-operation workflow."""
        api_key = "error-recovery-test-key"

        # First operation succeeds
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'
        sorted_data = vibelib.sort([3, 1, 2], api_key=api_key)

        # Second operation fails then succeeds
        mock_success = MagicMock()
        mock_success.choices[0].message.content = '{"response": 3}'
        mock_openai_client.return_value.chat.completions.create.side_effect = [
            Exception("Temporary failure"),
            mock_success
        ]

        with patch('time.sleep'):  # Speed up test
            max_val = vibelib.max(sorted_data, api_key=api_key)

        assert sorted_data == [1, 2, 3]
        assert max_val == 3

    def test_api_key_switching_mid_workflow(self, mock_openai_client):
        """Test switching API keys in the middle of a workflow."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # Start with one API key
        result1 = vibelib.sort([3, 1, 2], api_key="key1")

        # Switch to different API key
        result2 = vibelib.sort([3, 1, 2], api_key="key2")

        # Both should work
        assert result1 == [1, 2, 3]
        assert result2 == [1, 2, 3]
        assert mock_openai_client.call_count == 2  # Two different clients created

    def test_environment_variable_fallback_usage(self, mock_openai_client, env_with_api_key):
        """Test typical usage with environment variable fallback."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # Use without explicit API key (should use environment)
        result = vibelib.sort([3, 1, 2])

        assert result == [1, 2, 3]
        mock_openai_client.assert_called_with(api_key="env-test-key")

    def test_high_volume_usage_pattern(self, mock_openai_client):
        """Test high-volume usage pattern."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        api_key = "high-volume-test-key"
        results = []

        # Simulate high volume of operations
        for i in range(50):
            result = vibelib.sort([3, 1, 2], api_key=api_key)
            results.append(result)

        # All should succeed
        assert all(result == [1, 2, 3] for result in results)
        assert mock_openai_client.return_value.chat.completions.create.call_count == 50

        # Service should have been reused (not recreated each time)
        assert len(vibelib._services) > 0


class TestPublicAPIDocumentation:
    """Test that public API functions have proper documentation."""

    def test_all_functions_have_docstrings(self):
        """Test that all public API functions have docstrings."""
        public_functions = [
            'sort', 'max', 'min', 'sum', 'abs',
            'upper', 'lower', 'split', 'join', 'strip', 'replace',
            'count', 'index', 'reverse'
        ]

        for func_name in public_functions:
            func = getattr(vibelib, func_name)
            assert func.__doc__ is not None, f"Function {func_name} missing docstring"
            assert len(func.__doc__.strip()) > 0, f"Function {func_name} has empty docstring"

    def test_docstring_quality(self):
        """Test that docstrings contain essential information."""
        func = vibelib.sort
        docstring = func.__doc__

        # Should contain description, args, returns, and raises sections
        assert "Sort input using AI-powered analysis" in docstring
        assert "Args:" in docstring
        assert "Returns:" in docstring
        assert "Raises:" in docstring

    def test_function_annotations(self):
        """Test that functions have proper type annotations."""
        import inspect

        # Check a few key functions
        sort_sig = inspect.signature(vibelib.sort)
        assert 'items' in sort_sig.parameters
        assert 'api_key' in sort_sig.parameters

        max_sig = inspect.signature(vibelib.max)
        assert 'items' in max_sig.parameters
        assert 'api_key' in max_sig.parameters


class TestPublicAPIBackwardCompatibility:
    """Test backward compatibility of public API."""

    def test_optional_api_key_parameter_position(self, mock_openai_client, env_with_api_key):
        """Test that api_key parameter is consistently optional and last."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # Should work without api_key
        result1 = vibelib.sort([3, 1, 2])

        # Should work with api_key as keyword argument
        result2 = vibelib.sort([3, 1, 2], api_key="test-key")

        assert result1 == [1, 2, 3]
        assert result2 == [1, 2, 3]

    def test_function_signatures_consistency(self):
        """Test that function signatures are consistent across operations."""
        import inspect

        # All functions should have api_key as optional last parameter
        functions_to_check = [
            vibelib.sort, vibelib.max, vibelib.min, vibelib.sum, vibelib.abs,
            vibelib.upper, vibelib.lower, vibelib.strip
        ]

        for func in functions_to_check:
            sig = inspect.signature(func)
            params = list(sig.parameters.values())

            # Last parameter should be api_key with default None
            last_param = params[-1]
            assert last_param.name == 'api_key'
            assert last_param.default is None

    def test_return_value_consistency(self, mock_openai_client):
        """Test that return values are consistent with expectations."""
        # Sorting operations return lists
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'
        result = vibelib.sort([3, 1, 2], api_key="test-key")
        assert isinstance(result, list)

        # Numeric operations return numbers
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 5}'
        result = vibelib.max([1, 5, 3], api_key="test-key")
        assert isinstance(result, (int, float))

        # String operations return strings
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": "HELLO"}'
        result = vibelib.upper("hello", api_key="test-key")
        assert isinstance(result, str)
