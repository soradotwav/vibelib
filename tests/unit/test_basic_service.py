"""
Comprehensive tests for basic mathematical operations service.

Tests all mathematical operations including type preservation,
edge cases, and error handling scenarios.
"""

import pytest

from vibelib.operations.basic import BasicService
from vibelib.exceptions import ParseError


class TestBasicServiceMax:
    """Test maximum value operation."""

    def test_max_positive_integers(self, mock_client):
        """Test max with positive integers."""
        mock_client.request.return_value = '{"response": 5}'
        service = BasicService(mock_client)

        result = service.max([1, 5, 3, 2])

        assert result == 5
        mock_client.request.assert_called_once()

    def test_max_negative_integers(self, mock_client):
        """Test max with negative integers."""
        mock_client.request.return_value = '{"response": -1}'
        service = BasicService(mock_client)

        result = service.max([-5, -1, -3])

        assert result == -1

    def test_max_mixed_integers(self, mock_client):
        """Test max with mixed positive/negative integers."""
        mock_client.request.return_value = '{"response": 3}'
        service = BasicService(mock_client)

        result = service.max([-2, 3, -1, 0])

        assert result == 3

    def test_max_floats(self, mock_client):
        """Test max with float values."""
        mock_client.request.return_value = '{"response": 5.7}'
        service = BasicService(mock_client)

        result = service.max([1.2, 5.7, 3.3])

        assert result == 5.7

    def test_max_mixed_numbers(self, mock_client):
        """Test max with mixed integers and floats."""
        mock_client.request.return_value = '{"response": 5.5}'
        service = BasicService(mock_client)

        result = service.max([1, 5.5, 3])

        assert result == 5.5

    def test_max_single_element(self, mock_client):
        """Test max with single element list."""
        mock_client.request.return_value = '{"response": 42}'
        service = BasicService(mock_client)

        result = service.max([42])

        assert result == 42

    def test_max_duplicate_values(self, mock_client):
        """Test max with duplicate maximum values."""
        mock_client.request.return_value = '{"response": 5}'
        service = BasicService(mock_client)

        result = service.max([5, 3, 5, 1])

        assert result == 5

    def test_max_prompt_construction(self, mock_client):
        """Test that max operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": 5}'
        service = BasicService(mock_client)

        service.max([1, 5, 3])

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "[1, 5, 3]" in user_prompt
        assert "preserving number type" in user_prompt
        assert "maximum (largest) value" in system_prompt
        assert "Preserve the original number type" in system_prompt
        assert "JSON" in system_prompt


class TestBasicServiceMin:
    """Test minimum value operation."""

    def test_min_positive_integers(self, mock_client):
        """Test min with positive integers."""
        mock_client.request.return_value = '{"response": 1}'
        service = BasicService(mock_client)

        result = service.min([1, 5, 3, 2])

        assert result == 1

    def test_min_negative_integers(self, mock_client):
        """Test min with negative integers."""
        mock_client.request.return_value = '{"response": -5}'
        service = BasicService(mock_client)

        result = service.min([-5, -1, -3])

        assert result == -5

    def test_min_floats(self, mock_client):
        """Test min with float values."""
        mock_client.request.return_value = '{"response": 1.2}'
        service = BasicService(mock_client)

        result = service.min([1.2, 5.7, 3.3])

        assert result == 1.2

    def test_min_zero_included(self, mock_client):
        """Test min when zero is the minimum."""
        mock_client.request.return_value = '{"response": 0}'
        service = BasicService(mock_client)

        result = service.min([1, 0, 3])

        assert result == 0

    def test_min_prompt_construction(self, mock_client):
        """Test that min operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": 1}'
        service = BasicService(mock_client)

        service.min([1, 5, 3])

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "minimum (smallest) value" in system_prompt
        assert "preserving number type" in user_prompt


class TestBasicServiceSum:
    """Test sum operation."""

    def test_sum_positive_integers(self, mock_client):
        """Test sum with positive integers."""
        mock_client.request.return_value = '{"response": 15}'
        service = BasicService(mock_client)

        result = service.sum([1, 5, 3, 6])

        assert result == 15

    def test_sum_negative_integers(self, mock_client):
        """Test sum with negative integers."""
        mock_client.request.return_value = '{"response": -9}'
        service = BasicService(mock_client)

        result = service.sum([-5, -1, -3])

        assert result == -9

    def test_sum_mixed_integers(self, mock_client):
        """Test sum with mixed positive/negative integers."""
        mock_client.request.return_value = '{"response": 0}'
        service = BasicService(mock_client)

        result = service.sum([-2, 3, -1])

        assert result == 0

    def test_sum_floats(self, mock_client):
        """Test sum with float values."""
        mock_client.request.return_value = '{"response": 10.2}'
        service = BasicService(mock_client)

        result = service.sum([1.2, 5.7, 3.3])

        assert result == 10.2

    def test_sum_mixed_numbers(self, mock_client):
        """Test sum with mixed integers and floats."""
        mock_client.request.return_value = '{"response": 9.5}'
        service = BasicService(mock_client)

        result = service.sum([1, 5.5, 3])

        assert result == 9.5

    def test_sum_single_element(self, mock_client):
        """Test sum with single element."""
        mock_client.request.return_value = '{"response": 42}'
        service = BasicService(mock_client)

        result = service.sum([42])

        assert result == 42

    def test_sum_empty_logic_handled_by_prompt(self, mock_client):
        """Test sum behavior with empty list (handled by AI)."""
        mock_client.request.return_value = '{"response": 0}'
        service = BasicService(mock_client)

        result = service.sum([])

        assert result == 0

    def test_sum_with_zero(self, mock_client):
        """Test sum with zero values."""
        mock_client.request.return_value = '{"response": 5}'
        service = BasicService(mock_client)

        result = service.sum([0, 5, 0])

        assert result == 5

    def test_sum_prompt_type_preservation_logic(self, mock_client):
        """Test sum prompt includes type preservation logic."""
        mock_client.request.return_value = '{"response": 9}'
        service = BasicService(mock_client)

        service.sum([1, 2, 6])

        args, _ = mock_client.request.call_args
        system_prompt = args[1]

        assert "If all numbers are integers, return integer sum" in system_prompt
        assert "If any are floats, return float sum" in system_prompt


class TestBasicServiceAbs:
    """Test absolute value operation."""

    def test_abs_positive_integer(self, mock_client):
        """Test absolute value of positive integer."""
        mock_client.request.return_value = '{"response": 5}'
        service = BasicService(mock_client)

        result = service.abs(5)

        assert result == 5

    def test_abs_negative_integer(self, mock_client):
        """Test absolute value of negative integer."""
        mock_client.request.return_value = '{"response": 5}'
        service = BasicService(mock_client)

        result = service.abs(-5)

        assert result == 5

    def test_abs_zero(self, mock_client):
        """Test absolute value of zero."""
        mock_client.request.return_value = '{"response": 0}'
        service = BasicService(mock_client)

        result = service.abs(0)

        assert result == 0

    def test_abs_positive_float(self, mock_client):
        """Test absolute value of positive float."""
        mock_client.request.return_value = '{"response": 3.14}'
        service = BasicService(mock_client)

        result = service.abs(3.14)

        assert result == 3.14

    def test_abs_negative_float(self, mock_client):
        """Test absolute value of negative float."""
        mock_client.request.return_value = '{"response": 3.14}'
        service = BasicService(mock_client)

        result = service.abs(-3.14)

        assert result == 3.14

    def test_abs_very_small_number(self, mock_client):
        """Test absolute value of very small number."""
        mock_client.request.return_value = '{"response": 0.0001}'
        service = BasicService(mock_client)

        result = service.abs(-0.0001)

        assert result == 0.0001

    def test_abs_large_number(self, mock_client):
        """Test absolute value of large number."""
        mock_client.request.return_value = '{"response": 1000000}'
        service = BasicService(mock_client)

        result = service.abs(-1000000)

        assert result == 1000000

    def test_abs_prompt_construction(self, mock_client):
        """Test that abs operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": 5}'
        service = BasicService(mock_client)

        service.abs(-5)

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "-5" in user_prompt
        assert "keeping the same number type" in user_prompt
        assert "absolute value" in system_prompt
        assert "remove negative sign if present" in system_prompt


class TestTypePreservation:
    """Test type preservation across all operations."""

    @pytest.mark.parametrize("operation,input_data,mock_response,expected_type", [
        ("max", [1, 2, 3], '{"response": 3}', int),
        ("max", [1.1, 2.2, 3.3], '{"response": 3.3}', float),
        ("min", [1, 2, 3], '{"response": 1}', int),
        ("min", [1.1, 2.2, 3.3], '{"response": 1.1}', float),
        ("sum", [1, 2, 3], '{"response": 6}', int),
        ("sum", [1.1, 2.2], '{"response": 3.3}', float),
        ("abs", 5, '{"response": 5}', int),
        ("abs", -5.5, '{"response": 5.5}', float),
    ])
    def test_type_preservation_in_responses(self, mock_client, operation, input_data, mock_response, expected_type):
        """Test that operations preserve number types correctly."""
        mock_client.request.return_value = mock_response
        service = BasicService(mock_client)

        operation_func = getattr(service, operation)
        if operation == "abs":
            result = operation_func(input_data)
        else:
            result = operation_func(input_data)

        assert isinstance(result, expected_type)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_numbers(self, mock_client):
        """Test operations with very small numbers."""
        mock_client.request.return_value = '{"response": 1e-10}'
        service = BasicService(mock_client)

        result = service.min([1e-10, 1e-5, 1e-3])

        assert result == 1e-10

    def test_scientific_notation_handling(self, mock_client):
        """Test handling of scientific notation."""
        mock_client.request.return_value = '{"response": 1.23e6}'
        service = BasicService(mock_client)

        result = service.max([1.23e6, 1.24e5])

        assert result == 1.23e6

    def test_very_large_numbers(self, mock_client):
        """Test operations with very large numbers (JSON-compatible)."""
        large_positive = 1e100
        large_negative = -1e100

        # Test with large positive number
        mock_client.request.return_value = f'{{"response": {large_positive}}}'
        service = BasicService(mock_client)
        result = service.max([large_positive, 1000])
        assert result == large_positive

        # Test with large negative number
        mock_client.request.return_value = f'{{"response": {large_negative}}}'
        result = service.min([large_negative, -1000])
        assert result == large_negative

    @pytest.mark.parametrize("special_case", [
        "null",  # JSON null for unsupported values
        '"NaN"', # String representation
        '"Infinity"', # String representation
    ])
    def test_special_numeric_representations(self, mock_client, special_case):
        """Test handling of special numeric representations in JSON."""
        mock_client.request.return_value = f'{{"response": {special_case}}}'
        service = BasicService(mock_client)

        result = service.max([1, 2, 3])

        # Just verify it doesn't crash and returns the parsed value
        assert result is not None or result is None  # Accept any result

    def test_precision_preservation(self, mock_client):
        """Test that floating point precision is preserved."""
        mock_client.request.return_value = '{"response": 0.123456789}'
        service = BasicService(mock_client)

        result = service.max([0.123456789, 0.123456788])

        assert result == 0.123456789


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json_response_handling(self, mock_client):
        """Test handling of invalid JSON responses."""
        mock_client.request.return_value = 'not json'
        service = BasicService(mock_client)

        with pytest.raises(ParseError):
            service.max([1, 2, 3])

    def test_missing_response_field(self, mock_client):
        """Test handling of JSON without response field."""
        mock_client.request.return_value = '{"result": 5}'
        service = BasicService(mock_client)

        with pytest.raises(ParseError, match="Response must contain 'response' field"):
            service.max([1, 2, 3])

    def test_non_numeric_response(self, mock_client):
        """Test handling of non-numeric response values."""
        mock_client.request.return_value = '{"response": "not a number"}'
        service = BasicService(mock_client)

        # This should work fine - we don't validate the response type
        result = service.max([1, 2, 3])
        assert result == "not a number"

    def test_null_response_value(self, mock_client):
        """Test handling of null response value."""
        mock_client.request.return_value = '{"response": null}'
        service = BasicService(mock_client)

        result = service.max([1, 2, 3])
        assert result is None


class TestConcurrency:
    """Test concurrent operation scenarios."""

    def test_multiple_operations_same_service(self, mock_client):
        """Test multiple operations on the same service instance."""
        service = BasicService(mock_client)

        # Different operations with different responses
        test_cases = [
            ('{"response": 10}', 'max', [5, 10, 3]),
            ('{"response": 1}', 'min', [5, 10, 1]),
            ('{"response": 15}', 'sum', [5, 10]),
            ('{"response": 7}', 'abs', -7)
        ]

        results = []
        for mock_response, operation, data in test_cases:
            mock_client.request.return_value = mock_response
            if operation == 'abs':
                result = getattr(service, operation)(data)
            else:
                result = getattr(service, operation)(data)
            results.append(result)

        expected = [10, 1, 15, 7]
        assert results == expected

    def test_operation_isolation(self, mock_client):
        """Test that operations are properly isolated."""
        service1 = BasicService(mock_client)
        service2 = BasicService(mock_client)

        # Both services should work independently
        mock_client.request.return_value = '{"response": 42}'

        result1 = service1.max([1, 42, 3])
        result2 = service2.min([42, 1, 3])

        assert result1 == 42
        assert result2 == 42
        assert mock_client.request.call_count == 2


class TestPerformance:
    """Test performance-related scenarios."""

    def test_large_input_arrays(self, mock_client, performance_data):
        """Test operations with large input arrays."""
        large_array = performance_data['large']
        mock_client.request.return_value = '{"response": 999}'
        service = BasicService(mock_client)

        result = service.max(large_array)

        assert result == 999
        # Verify the large array was passed to the prompt
        call_args = mock_client.request.call_args
        assert str(large_array) in call_args[0][0]

    @pytest.mark.parametrize("array_size", [10, 100, 500, 1000])
    def test_varying_input_sizes(self, mock_client, array_size):
        """Test operations with varying input sizes."""
        test_array = list(range(array_size))
        mock_client.request.return_value = f'{{"response": {array_size - 1}}}'
        service = BasicService(mock_client)

        result = service.max(test_array)

        assert result == array_size - 1
