"""
Comprehensive tests for list manipulation operations service.

Tests all list operations including counting, indexing, reversing,
and edge cases with various data types.
"""
import json

import pytest

from vibelib.operations.lists import ListService
from vibelib.exceptions import ParseError


class TestListServiceCount:
    """Test counting operation."""

    def test_count_single_occurrence(self, mock_client):
        """Test counting single occurrence of a value."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.count([1, 2, 3], 2)

        assert result == 1
        mock_client.request.assert_called_once()

    def test_count_multiple_occurrences(self, mock_client):
        """Test counting multiple occurrences of a value."""
        mock_client.request.return_value = '{"response": 3}'
        service = ListService(mock_client)

        result = service.count([1, 2, 1, 3, 1], 1)

        assert result == 3

    def test_count_zero_occurrences(self, mock_client):
        """Test counting when value is not found."""
        mock_client.request.return_value = '{"response": 0}'
        service = ListService(mock_client)

        result = service.count([1, 2, 3], 4)

        assert result == 0

    def test_count_string_values(self, mock_client):
        """Test counting string values."""
        mock_client.request.return_value = '{"response": 2}'
        service = ListService(mock_client)

        result = service.count(["apple", "banana", "apple", "cherry"], "apple")

        assert result == 2

    def test_count_mixed_types(self, mock_client):
        """Test counting in list with mixed data types."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.count([1, "hello", 2.5, "hello", True], "hello")

        assert result == 1

    def test_count_float_values(self, mock_client):
        """Test counting float values."""
        mock_client.request.return_value = '{"response": 2}'
        service = ListService(mock_client)

        result = service.count([1.5, 2.7, 1.5, 3.9], 1.5)

        assert result == 2

    def test_count_boolean_values(self, mock_client):
        """Test counting boolean values."""
        mock_client.request.return_value = '{"response": 3}'
        service = ListService(mock_client)

        result = service.count([True, False, True, True, False], True)

        assert result == 3

    def test_count_none_values(self, mock_client):
        """Test counting None values."""
        mock_client.request.return_value = '{"response": 2}'
        service = ListService(mock_client)

        result = service.count([1, None, 2, None, 3], None)

        assert result == 2

    def test_count_empty_list(self, mock_client):
        """Test counting in empty list."""
        mock_client.request.return_value = '{"response": 0}'
        service = ListService(mock_client)

        result = service.count([], 1)

        assert result == 0

    def test_count_unicode_strings(self, mock_client):
        """Test counting Unicode strings."""
        mock_client.request.return_value = '{"response": 2}'
        service = ListService(mock_client)

        result = service.count(["café", "naïve", "café", "résumé"], "café")

        assert result == 2

    def test_count_nested_structures(self, mock_client):
        """Test counting nested structures like lists."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.count([[1, 2], [3, 4], [1, 2], [5, 6]], [1, 2])

        assert result == 1

    def test_count_prompt_construction(self, mock_client):
        """Test that count operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": 2}'
        service = ListService(mock_client)

        service.count([1, 2, 1, 3], 1)

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "Count occurrences of 1 in list: [1, 2, 1, 3]" in user_prompt
        assert "Count ALL exact matches" in system_prompt
        assert "precise equality comparison" in system_prompt


class TestListServiceIndex:
    """Test index finding operation."""

    def test_index_found_at_start(self, mock_client):
        """Test finding index at the start of list."""
        mock_client.request.return_value = '{"response": 0}'
        service = ListService(mock_client)

        result = service.index([5, 2, 3], 5)

        assert result == 0

    def test_index_found_in_middle(self, mock_client):
        """Test finding index in the middle of list."""
        mock_client.request.return_value = '{"response": 2}'
        service = ListService(mock_client)

        result = service.index([1, 2, 3, 4], 3)

        assert result == 2

    def test_index_found_at_end(self, mock_client):
        """Test finding index at the end of list."""
        mock_client.request.return_value = '{"response": 3}'
        service = ListService(mock_client)

        result = service.index([1, 2, 3, 4], 4)

        assert result == 3

    def test_index_first_occurrence(self, mock_client):
        """Test finding first occurrence when value appears multiple times."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.index([1, 2, 3, 2, 5], 2)

        assert result == 1

    def test_index_string_values(self, mock_client):
        """Test finding index of string values."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.index(["apple", "banana", "cherry"], "banana")

        assert result == 1

    def test_index_mixed_types(self, mock_client):
        """Test finding index in list with mixed types."""
        mock_client.request.return_value = '{"response": 2}'
        service = ListService(mock_client)

        result = service.index([1, "hello", 2.5, True], 2.5)

        assert result == 2

    def test_index_float_values(self, mock_client):
        """Test finding index of float values."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.index([1.1, 2.2, 3.3], 2.2)

        assert result == 1

    def test_index_boolean_values(self, mock_client):
        """Test finding index of boolean values."""
        mock_client.request.return_value = '{"response": 0}'
        service = ListService(mock_client)

        result = service.index([True, False, True], True)

        assert result == 0

    def test_index_none_value(self, mock_client):
        """Test finding index of None value."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.index([1, None, 2], None)

        assert result == 1

    def test_index_unicode_strings(self, mock_client):
        """Test finding index of Unicode strings."""
        mock_client.request.return_value = '{"response": 2}'
        service = ListService(mock_client)

        result = service.index(["hello", "world", "café"], "café")

        assert result == 2

    def test_index_nested_structures(self, mock_client):
        """Test finding index of nested structures."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.index([[1, 2], [3, 4], [5, 6]], [3, 4])

        assert result == 1

    def test_index_prompt_construction(self, mock_client):
        """Test that index operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        service.index([1, 2, 3], 2)

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "Find the index of value 2 in list: [1, 2, 3]" in user_prompt
        assert "first occurrence" in system_prompt
        assert "Index counting starts at 0" in system_prompt


class TestListServiceReverse:
    """Test list reversal operation."""

    def test_reverse_integers(self, mock_client):
        """Test reversing list of integers."""
        mock_client.request.return_value = '{"response": [3, 2, 1]}'
        service = ListService(mock_client)

        result = service.reverse([1, 2, 3])

        assert result == [3, 2, 1]

    def test_reverse_strings(self, mock_client):
        """Test reversing list of strings."""
        mock_client.request.return_value = '{"response": ["cherry", "banana", "apple"]}'
        service = ListService(mock_client)

        result = service.reverse(["apple", "banana", "cherry"])

        assert result == ["cherry", "banana", "apple"]

    def test_reverse_mixed_types(self, mock_client):
        """Test reversing list with mixed data types."""
        mock_client.request.return_value = '{"response": [true, "hello", 2.5, 1]}'
        service = ListService(mock_client)

        result = service.reverse([1, 2.5, "hello", True])

        assert result == [True, "hello", 2.5, 1]

    def test_reverse_floats(self, mock_client):
        """Test reversing list of floats."""
        mock_client.request.return_value = '{"response": [3.3, 2.2, 1.1]}'
        service = ListService(mock_client)

        result = service.reverse([1.1, 2.2, 3.3])

        assert result == [3.3, 2.2, 1.1]

    def test_reverse_single_element(self, mock_client):
        """Test reversing single-element list."""
        mock_client.request.return_value = '{"response": [42]}'
        service = ListService(mock_client)

        result = service.reverse([42])

        assert result == [42]

    def test_reverse_empty_list(self, mock_client):
        """Test reversing empty list."""
        mock_client.request.return_value = '{"response": []}'
        service = ListService(mock_client)

        result = service.reverse([])

        assert result == []

    def test_reverse_duplicates(self, mock_client):
        """Test reversing list with duplicate values."""
        mock_client.request.return_value = '{"response": [1, 2, 1, 2, 1]}'
        service = ListService(mock_client)

        result = service.reverse([1, 2, 1, 2, 1])

        assert result == [1, 2, 1, 2, 1]

    def test_reverse_boolean_values(self, mock_client):
        """Test reversing list with boolean values."""
        mock_client.request.return_value = '{"response": [false, true, false]}'
        service = ListService(mock_client)

        result = service.reverse([False, True, False])

        assert result == [False, True, False]

    def test_reverse_none_values(self, mock_client):
        """Test reversing list with None values."""
        mock_client.request.return_value = '{"response": [3, null, 2, null, 1]}'
        service = ListService(mock_client)

        result = service.reverse([1, None, 2, None, 3])

        assert result == [3, None, 2, None, 1]

    def test_reverse_unicode_strings(self, mock_client):
        """Test reversing list with Unicode strings."""
        mock_client.request.return_value = '{"response": ["résumé", "naïve", "café"]}'
        service = ListService(mock_client)

        result = service.reverse(["café", "naïve", "résumé"])

        assert result == ["résumé", "naïve", "café"]

    def test_reverse_nested_structures(self, mock_client):
        """Test reversing list with nested structures."""
        mock_client.request.return_value = '{"response": [[5, 6], [3, 4], [1, 2]]}'
        service = ListService(mock_client)

        result = service.reverse([[1, 2], [3, 4], [5, 6]])

        assert result == [[5, 6], [3, 4], [1, 2]]

    def test_reverse_large_list(self, mock_client, performance_data):
        """Test reversing large list."""
        large_list = performance_data['medium']  # 100 items
        reversed_list = list(reversed(large_list))
        mock_client.request.return_value = f'{{"response": {reversed_list}}}'
        service = ListService(mock_client)

        result = service.reverse(large_list)

        assert result == reversed_list

    def test_reverse_prompt_construction(self, mock_client):
        """Test that reverse operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": [3, 2, 1]}'
        service = ListService(mock_client)

        service.reverse([1, 2, 3])

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "Reverse the order of elements in list: [1, 2, 3]" in user_prompt
        assert "reverse order" in system_prompt
        assert "last element becomes first" in system_prompt
        assert "Preserve all original data types" in system_prompt


class TestDataTypePreservation:
    """Test data type preservation across operations."""

    @pytest.mark.parametrize("test_data,expected_type", [
        ([1, 2, 3, 2], int),  # integers
        ([1.1, 2.2, 3.3], float),  # floats
        (["a", "b", "c"], str),  # strings
        ([True, False, True], bool),  # booleans
    ])
    def test_count_preserves_data_context(self, mock_client, test_data, expected_type):
        """Test that count operation handles different data types properly."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        # Use first element as search value
        search_value = test_data[0]
        result = service.count(test_data, search_value)

        assert isinstance(result, int)  # Count always returns int

        # Verify the data was passed correctly to the prompt
        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        assert str(test_data) in user_prompt
        assert str(search_value) in user_prompt

    @pytest.mark.parametrize("test_data", [
        [1, 2, 3],
        [1.1, 2.2, 3.3],
        ["a", "b", "c"],
        [True, False, True],
        [[1, 2], [3, 4]],
        [{"key": "value"}, {"key2": "value2"}]
    ])
    def test_reverse_preserves_data_types(self, mock_client, test_data):
        """Test that reverse operation preserves all data types."""
        expected_reversed = list(reversed(test_data))
        mock_client.request.return_value = json.dumps({"response": expected_reversed})
        service = ListService(mock_client)

        result = service.reverse(test_data)

        assert result == expected_reversed

        # Verify type preservation instruction in prompt
        args, _ = mock_client.request.call_args
        system_prompt = args[1]
        assert "Preserve all original data types exactly" in system_prompt


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_lists(self, mock_client, performance_data):
        """Test operations with very large lists."""
        large_list = performance_data['large']  # 1000 items
        mock_client.request.return_value = '{"response": 5}'
        service = ListService(mock_client)

        result = service.count(large_list, 500)

        assert result == 5

        # Verify large list was handled
        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        assert str(large_list) in user_prompt

    def test_deeply_nested_structures(self, mock_client):
        """Test operations with deeply nested structures."""
        nested_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        search_value = [[1, 2], [3, 4]]
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.count(nested_list, search_value)

        assert result == 1

    def test_edge_case_numeric_values(self, mock_client):
        """Test operations with edge case numeric values."""
        # Use realistic edge cases that are JSON-compatible
        edge_values = [
            1e100,    # Very large positive
            -1e100,   # Very large negative
            1e-100,   # Very small positive
            -1e-100,  # Very small negative
            0,        # Zero
            -0.0      # Negative zero
        ]
        reversed_values = list(reversed(edge_values))

        mock_client.request.return_value = json.dumps({"response": reversed_values})
        service = ListService(mock_client)

        result = service.reverse(edge_values)

        assert len(result) == len(edge_values)
        assert result == reversed_values

    def test_extremely_long_strings(self, mock_client):
        """Test operations with extremely long strings."""
        long_string = "a" * 10000
        test_list = ["short", long_string, "medium"]
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        result = service.count(test_list, long_string)

        assert result == 1

    @pytest.mark.parametrize("empty_like_value", [[], "", 0, False, None])
    def test_empty_like_values(self, mock_client, empty_like_value):
        """Test operations with various empty-like values."""
        test_list = [1, empty_like_value, 2, empty_like_value, 3]
        mock_client.request.return_value = '{"response": 2}'
        service = ListService(mock_client)

        result = service.count(test_list, empty_like_value)

        assert result == 2


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json_response(self, mock_client):
        """Test handling of invalid JSON responses."""
        mock_client.request.return_value = 'not json'
        service = ListService(mock_client)

        with pytest.raises(ParseError):
            service.count([1, 2, 3], 1)

    def test_missing_response_field(self, mock_client):
        """Test handling of JSON without response field."""
        mock_client.request.return_value = '{"result": 2}'
        service = ListService(mock_client)

        with pytest.raises(ParseError, match="Response must contain 'response' field"):
            service.count([1, 2, 3], 1)

    def test_unexpected_response_type(self, mock_client):
        """Test handling of unexpected response types."""
        mock_client.request.return_value = '{"response": "not a number"}'
        service = ListService(mock_client)

        # Should work fine - we don't validate response type
        result = service.count([1, 2, 3], 1)
        assert result == "not a number"

    def test_null_response_value(self, mock_client):
        """Test handling of null response value."""
        mock_client.request.return_value = '{"response": null}'
        service = ListService(mock_client)

        result = service.count([1, 2, 3], 1)
        assert result is None


class TestConcurrency:
    """Test concurrent operation scenarios."""

    def test_multiple_operations_same_service(self, mock_client):
        """Test multiple operations on the same service instance."""
        service = ListService(mock_client)

        test_cases = [
            ('count', ([1, 2, 1, 3], 1), '{"response": 2}', 2),
            ('index', ([1, 2, 3], 2), '{"response": 1}', 1),
            ('reverse', ([1, 2, 3],), '{"response": [3, 2, 1]}', [3, 2, 1]),
        ]

        results = []
        for operation, args, mock_response, expected in test_cases:
            mock_client.request.return_value = mock_response
            result = getattr(service, operation)(*args)
            results.append(result)

        expected_results = [case[3] for case in test_cases]
        assert results == expected_results

    def test_service_isolation(self, mock_client):
        """Test that multiple service instances are properly isolated."""
        service1 = ListService(mock_client)
        service2 = ListService(mock_client)

        mock_client.request.return_value = '{"response": 2}'

        result1 = service1.count([1, 2, 1], 1)
        result2 = service2.count([3, 2, 3], 3)

        assert result1 == 2
        assert result2 == 2
        assert mock_client.request.call_count == 2


class TestPromptValidation:
    """Test that prompts are constructed correctly for AI processing."""

    def test_count_prompt_includes_exact_matching_instruction(self, mock_client):
        """Test that count prompts emphasize exact matching."""
        mock_client.request.return_value = '{"response": 1}'
        service = ListService(mock_client)

        service.count([1, "1", 1.0], 1)

        args, _ = mock_client.request.call_args
        system_prompt = args[1]
        assert "Count ALL exact matches" in system_prompt
        assert "precise equality comparison" in system_prompt

    def test_index_prompt_specifies_zero_based_indexing(self, mock_client):
        """Test that index prompts specify zero-based indexing."""
        mock_client.request.return_value = '{"response": 0}'
        service = ListService(mock_client)

        service.index(["first", "second"], "first")

        args, _ = mock_client.request.call_args
        system_prompt = args[1]
        assert "Index counting starts at 0" in system_prompt
        assert "first occurrence" in system_prompt

    def test_reverse_prompt_emphasizes_type_preservation(self, mock_client):
        """Test that reverse prompts emphasize type preservation."""
        mock_client.request.return_value = '{"response": [3, 2, 1]}'
        service = ListService(mock_client)

        service.reverse([1, 2, 3])

        args, _ = mock_client.request.call_args
        system_prompt = args[1]
        assert "Preserve all original data types exactly" in system_prompt
        assert "numbers stay numbers, strings stay strings" in system_prompt
