"""
Comprehensive tests for AI-powered sorting operations.

Tests all sorting scenarios including various data types, edge cases,
validation, and creative AI interpretation capabilities.
"""
import json

import pytest

from vibelib.operations.sorting import SortingService
from vibelib.exceptions import ValidationError, ParseError


class TestBasicSorting:
    """Test basic sorting functionality."""

    def test_sort_integers_ascending(self, mock_client):
        """Test sorting integers in ascending order."""
        mock_client.request.return_value = '{"response": [1, 2, 3, 4, 5]}'
        service = SortingService(mock_client)

        result = service.sort([3, 1, 4, 5, 2])

        assert result == [1, 2, 3, 4, 5]
        mock_client.request.assert_called_once()

    def test_sort_integers_with_duplicates(self, mock_client):
        """Test sorting integers with duplicate values."""
        mock_client.request.return_value = '{"response": [1, 1, 2, 2, 3]}'
        service = SortingService(mock_client)

        result = service.sort([2, 1, 3, 1, 2])

        assert result == [1, 1, 2, 2, 3]

    def test_sort_floats(self, mock_client):
        """Test sorting floating point numbers."""
        mock_client.request.return_value = '{"response": [0.5, 1.5, 2.5, 3.5]}'
        service = SortingService(mock_client)

        result = service.sort([2.5, 0.5, 3.5, 1.5])

        assert result == [0.5, 1.5, 2.5, 3.5]

    def test_sort_mixed_numbers(self, mock_client):
        """Test sorting mixed integers and floats."""
        mock_client.request.return_value = '{"response": [1, 1.5, 2, 2.5, 3]}'
        service = SortingService(mock_client)

        result = service.sort([2.5, 1, 3, 1.5, 2])

        assert result == [1, 1.5, 2, 2.5, 3]

    def test_sort_negative_numbers(self, mock_client):
        """Test sorting negative numbers."""
        mock_client.request.return_value = '{"response": [-3, -1, 0, 1, 3]}'
        service = SortingService(mock_client)

        result = service.sort([1, -1, 3, -3, 0])

        assert result == [-3, -1, 0, 1, 3]

    def test_sort_strings_alphabetically(self, mock_client):
        """Test sorting strings alphabetically."""
        mock_client.request.return_value = '{"response": ["apple", "banana", "cherry", "date"]}'
        service = SortingService(mock_client)

        result = service.sort(["cherry", "apple", "date", "banana"])

        assert result == ["apple", "banana", "cherry", "date"]

    def test_sort_strings_case_sensitive(self, mock_client):
        """Test sorting strings with mixed case."""
        mock_client.request.return_value = '{"response": ["Apple", "banana", "Cherry"]}'
        service = SortingService(mock_client)

        result = service.sort(["banana", "Apple", "Cherry"])

        assert result == ["Apple", "banana", "Cherry"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_sort_empty_array_returns_empty(self, mock_client):
        """Test that empty array returns empty without API call."""
        service = SortingService(mock_client)

        result = service.sort([])

        assert result == []
        mock_client.request.assert_not_called()

    def test_sort_single_element_returns_same(self, mock_client):
        """Test that single element returns same without API call."""
        service = SortingService(mock_client)

        result = service.sort([42])

        assert result == [42]
        mock_client.request.assert_not_called()

    def test_sort_already_sorted_array(self, mock_client):
        """Test sorting already sorted array."""
        mock_client.request.return_value = '{"response": [1, 2, 3, 4, 5]}'
        service = SortingService(mock_client)

        result = service.sort([1, 2, 3, 4, 5])

        assert result == [1, 2, 3, 4, 5]

    def test_sort_reverse_sorted_array(self, mock_client):
        """Test sorting reverse-sorted array."""
        mock_client.request.return_value = '{"response": [1, 2, 3, 4, 5]}'
        service = SortingService(mock_client)

        result = service.sort([5, 4, 3, 2, 1])

        assert result == [1, 2, 3, 4, 5]

    def test_sort_all_same_elements(self, mock_client):
        """Test sorting array with all identical elements."""
        mock_client.request.return_value = '{"response": [5, 5, 5, 5, 5]}'
        service = SortingService(mock_client)

        result = service.sort([5, 5, 5, 5, 5])

        assert result == [5, 5, 5, 5, 5]

    def test_sort_none_values_included(self, mock_client):
        """Test sorting with None values."""
        mock_client.request.return_value = '{"response": [null, 1, 2, 3]}'
        service = SortingService(mock_client)

        result = service.sort([2, None, 1, 3])

        assert result == [None, 1, 2, 3]

    def test_sort_boolean_values(self, mock_client):
        """Test sorting boolean values."""
        mock_client.request.return_value = '{"response": [false, false, true, true]}'
        service = SortingService(mock_client)

        result = service.sort([True, False, True, False])

        assert result == [False, False, True, True]


class TestDataTypes:
    """Test sorting with various data types."""

    def test_sort_unicode_strings(self, mock_client):
        """Test sorting Unicode strings."""
        mock_client.request.return_value = '{"response": ["café", "naïve", "résumé"]}'
        service = SortingService(mock_client)

        result = service.sort(["résumé", "café", "naïve"])

        assert result == ["café", "naïve", "résumé"]

    def test_sort_very_large_numbers(self, mock_client):
        """Test sorting very large numbers."""
        mock_client.request.return_value = '{"response": [1000000, 9999999999, 10000000000]}'
        service = SortingService(mock_client)

        result = service.sort([9999999999, 1000000, 10000000000])

        assert result == [1000000, 9999999999, 10000000000]

    def test_sort_scientific_notation(self, mock_client):
        """Test sorting numbers in scientific notation."""
        mock_client.request.return_value = '{"response": [1e-5, 1e-3, 1e6]}'
        service = SortingService(mock_client)

        result = service.sort([1e6, 1e-3, 1e-5])

        assert result == [1e-5, 1e-3, 1e6]

    def test_sort_floating_point_precision(self, mock_client):
        """Test sorting with floating point precision edge cases."""
        mock_client.request.return_value = '{"response": [0.1, 0.2, 0.3]}'
        service = SortingService(mock_client)

        result = service.sort([0.3, 0.1, 0.2])

        assert result == [0.1, 0.2, 0.3]

    def test_sort_with_special_value_handling(self, mock_client):
        """Test sorting when AI handles special values as strings."""
        # AI might convert special values to string representations
        mock_client.request.return_value = json.dumps({"response": ["-Infinity", "-1", "0", "1", "Infinity"]})
        service = SortingService(mock_client)

        # Input has special values, AI returns string representations
        result = service.sort([1, float('-inf'), float('inf'), 0, -1])

        assert len(result) == 5
        assert result == ["-Infinity", "-1", "0", "1", "Infinity"]

    def test_sort_mixed_string_types(self, mock_client):
        """Test sorting mixed string types."""
        mock_client.request.return_value = '{"response": ["", " ", "a", "hello"]}'
        service = SortingService(mock_client)

        result = service.sort(["hello", "", "a", " "])

        assert result == ["", " ", "a", "hello"]


class TestCreativeSorting:
    """Test AI's creative interpretation of sorting."""

    def test_sort_mixed_types_creative_interpretation(self, mock_client):
        """Test AI's creative sorting of mixed types."""
        mock_client.request.return_value = '{"response": [1, 2.5, "hello", true]}'
        service = SortingService(mock_client)

        result = service.sort([True, "hello", 1, 2.5])

        # AI might sort by type, value, or other creative criteria
        assert len(result) == 4
        assert set(result) == {1, 2.5, "hello", True}

    def test_sort_semantic_string_sorting(self, mock_client):
        """Test AI's semantic understanding of string sorting."""
        mock_client.request.return_value = '{"response": ["small", "medium", "large", "extra large"]}'
        service = SortingService(mock_client)

        result = service.sort(["large", "small", "extra large", "medium"])

        assert result == ["small", "medium", "large", "extra large"]

    def test_sort_contextual_interpretation(self, mock_client):
        """Test AI's contextual interpretation of sorting."""
        mock_client.request.return_value = '{"response": ["Monday", "Tuesday", "Wednesday"]}'
        service = SortingService(mock_client)

        result = service.sort(["Wednesday", "Monday", "Tuesday"])

        assert result == ["Monday", "Tuesday", "Wednesday"]

    def test_sort_creative_numeric_patterns(self, mock_client):
        """Test AI's creative interpretation of numeric patterns."""
        mock_client.request.return_value = '{"response": [1, 1, 2, 3, 5, 8]}'  # Fibonacci
        service = SortingService(mock_client)

        result = service.sort([8, 1, 5, 2, 3, 1])

        # AI might recognize Fibonacci sequence
        assert len(result) == 6


class TestInputTypes:
    """Test different input types and containers."""

    def test_sort_tuple_input(self, mock_client):
        """Test sorting tuple input (converted to list)."""
        mock_client.request.return_value = '{"response": [1, 2, 3, 4]}'
        service = SortingService(mock_client)

        result = service.sort((3, 1, 4, 2))

        assert result == [1, 2, 3, 4]
        assert isinstance(result, list)

    def test_sort_set_like_input(self, mock_client):
        """Test sorting with set-like properties."""
        mock_client.request.return_value = '{"response": [1, 2, 3]}'
        service = SortingService(mock_client)

        # Set would remove duplicates, but we pass it as list to sort
        result = service.sort(list({3, 1, 2, 1}))

        assert result == [1, 2, 3]

    def test_sort_string_input(self, mock_client):
        """Test sorting string input (characters)."""
        # "hello" as list is ['h', 'e', 'l', 'l', 'o'] (5 chars)
        # Sorted should be ['e', 'h', 'l', 'l', 'o'] (5 chars)
        mock_client.request.return_value = json.dumps({"response": ["e", "h", "l", "l", "o"]})
        service = SortingService(mock_client)

        result = service.sort(list("hello"))  # Convert to list of chars

        assert len(result) == 5  # Should match input length
        assert result == ["e", "h", "l", "l", "o"]

    def test_sort_generator_input(self, mock_client):
        """Test sorting generator input."""
        mock_client.request.return_value = '{"response": [0, 1, 4, 9, 16]}'
        service = SortingService(mock_client)

        # Convert generator to list for sorting
        gen_data = list(i*i for i in range(5))  # [0, 1, 4, 9, 16]
        result = service.sort(gen_data)

        assert result == [0, 1, 4, 9, 16]


class TestPerformance:
    """Test performance-related scenarios."""

    def test_sort_large_array_within_limit(self, mock_client, performance_data):
        """Test sorting large array within size limit."""
        large_array = performance_data['large']  # 1000 items
        sorted_array = sorted(large_array)
        mock_client.request.return_value = f'{{"response": {sorted_array}}}'
        service = SortingService(mock_client)

        result = service.sort(large_array)

        assert result == sorted_array

    def test_sort_maximum_allowed_size(self, mock_client):
        """Test sorting array at maximum allowed size."""
        max_size_array = list(range(10000))  # Exactly at limit
        expected_sorted = list(range(10000))
        mock_client.request.return_value = f'{{"response": {expected_sorted}}}'
        service = SortingService(mock_client)

        result = service.sort(max_size_array)

        assert len(result) == 10000
        mock_client.request.assert_called_once()

    def test_sort_array_too_large_raises_validation_error(self, mock_client):
        """Test that oversized array raises ValidationError."""
        oversized_array = list(range(10001))  # One over limit
        service = SortingService(mock_client)

        with pytest.raises(ValidationError, match="Input too large \\(10001 items\\). Maximum allowed: 10,000 items."):
            service.sort(oversized_array)

        mock_client.request.assert_not_called()

    @pytest.mark.parametrize("size", [10, 50, 100, 500, 1000])
    def test_sort_various_sizes(self, mock_client, size):
        """Test sorting arrays of various sizes."""
        test_array = list(reversed(range(size)))  # [size-1, size-2, ..., 0]
        expected_sorted = list(range(size))
        mock_client.request.return_value = f'{{"response": {expected_sorted}}}'
        service = SortingService(mock_client)

        result = service.sort(test_array)

        assert result == expected_sorted


class TestValidation:
    """Test input validation scenarios."""

    def test_sort_none_input(self, mock_client):
        """Test sorting None input."""
        service = SortingService(mock_client)

        result = service.sort(None)

        assert result == []
        mock_client.request.assert_not_called()

    def test_sort_false_input(self, mock_client):
        """Test sorting False input (falsy but has length)."""
        service = SortingService(mock_client)

        result = service.sort(False)

        assert result == []
        mock_client.request.assert_not_called()

    def test_sort_zero_input(self, mock_client):
        """Test sorting zero input."""
        service = SortingService(mock_client)

        result = service.sort(0)

        assert result == []
        mock_client.request.assert_not_called()

    def test_sort_with_length_property(self, mock_client):
        """Test sorting object with __len__ method."""
        class MockListLike:
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __iter__(self):
                return iter(self.data)

        mock_obj = MockListLike([3, 1, 2])
        mock_client.request.return_value = '{"response": [1, 2, 3]}'
        service = SortingService(mock_client)

        result = service.sort(mock_obj)

        assert result == [1, 2, 3]


class TestPromptConstruction:
    """Test AI prompt construction and instructions."""

    def test_sort_prompt_includes_creativity_instruction(self, mock_client):
        """Test that sort prompts encourage AI creativity."""
        mock_client.request.return_value = '{"response": [1, 2, 3]}'
        service = SortingService(mock_client)

        service.sort([3, 1, 2])

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "Intelligently sort this data: [3, 1, 2]" in user_prompt
        assert "You are an advanced sorting service" in system_prompt
        assert "Be creative but logical" in system_prompt

    def test_sort_prompt_includes_type_preservation(self, mock_client):
        """Test that sort prompts include type preservation instructions."""
        mock_client.request.return_value = '{"response": [1, 2, 3]}'
        service = SortingService(mock_client)

        service.sort([3, 1, 2])

        args, _ = mock_client.request.call_args
        system_prompt = args[1]

        assert "Preserve all original data types exactly" in system_prompt
        assert "numbers stay numbers, strings stay strings" in system_prompt

    def test_sort_prompt_specifies_json_format(self, mock_client):
        """Test that sort prompts specify JSON response format."""
        mock_client.request.return_value = '{"response": [1, 2, 3]}'
        service = SortingService(mock_client)

        service.sort([3, 1, 2])

        args, _ = mock_client.request.call_args
        system_prompt = args[1]

        assert 'Always return JSON in format: {"response": [your_sorted_result]}' in system_prompt

    @pytest.mark.parametrize("input_data,expected_in_prompt", [
        ([1, 2, 3], "[1, 2, 3]"),
        (["a", "b", "c"], "['a', 'b', 'c']"),
        ([1.1, 2.2], "[1.1, 2.2]"),
        ([True, False], "[True, False]"),
    ])
    def test_sort_prompt_includes_input_data(self, mock_client, input_data, expected_in_prompt):
        """Test that sort prompts include the actual input data."""
        mock_client.request.return_value = '{"response": []}'
        service = SortingService(mock_client)

        service.sort(input_data)

        args, _ = mock_client.request.call_args
        user_prompt = args[0]

        assert expected_in_prompt in user_prompt


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json_response_raises_parse_error(self, mock_client):
        """Test that invalid JSON raises ParseError."""
        mock_client.request.return_value = 'this is not json'
        service = SortingService(mock_client)

        with pytest.raises(ParseError, match="Invalid JSON response"):
            service.sort([1, 2, 3])

    def test_malformed_json_response_raises_parse_error(self, mock_client):
        """Test that malformed JSON raises ParseError."""
        mock_client.request.return_value = '{"response": [1, 2'  # Missing closing bracket
        service = SortingService(mock_client)

        with pytest.raises(ParseError, match="Invalid JSON response"):
            service.sort([1, 2, 3])

    def test_missing_response_field_raises_parse_error(self, mock_client):
        """Test that missing response field raises ParseError."""
        mock_client.request.return_value = '{"data": [1, 2, 3]}'
        service = SortingService(mock_client)

        with pytest.raises(ParseError, match="Response must contain 'response' field"):
            service.sort([1, 2, 3])

    def test_non_object_json_response_raises_parse_error(self, mock_client):
        """Test that non-object JSON raises ParseError."""
        mock_client.request.return_value = '"just a string"'
        service = SortingService(mock_client)

        with pytest.raises(ParseError, match="Response must be a JSON object"):
            service.sort([1, 2, 3])

    def test_parse_error_includes_raw_response(self, mock_client):
        """Test that ParseError includes raw response for debugging."""
        bad_response = "this is definitely not json at all"
        mock_client.request.return_value = bad_response
        service = SortingService(mock_client)

        with pytest.raises(ParseError) as exc_info:
            service.sort([1, 2, 3])

        # Should include first 200 chars of bad response
        assert bad_response in str(exc_info.value)

    def test_json_with_extra_fields_ignored(self, mock_client):
        """Test that extra fields in JSON response are ignored gracefully."""
        mock_client.request.return_value = '{"response": [1, 2, 3], "confidence": 0.99, "model": "gpt-4"}'
        service = SortingService(mock_client)

        result = service.sort([3, 1, 2])

        assert result == [1, 2, 3]  # Extra fields ignored


class TestConcurrency:
    """Test concurrent operation scenarios."""

    def test_multiple_sorts_same_service(self, mock_client):
        """Test multiple sort operations on the same service instance."""
        service = SortingService(mock_client)

        test_cases = [
            ([3, 1, 2], '{"response": [1, 2, 3]}', [1, 2, 3]),
            (["c", "a", "b"], '{"response": ["a", "b", "c"]}', ["a", "b", "c"]),
            ([2.2, 1.1], '{"response": [1.1, 2.2]}', [1.1, 2.2]),
        ]

        results = []
        for input_data, mock_response, expected in test_cases:
            mock_client.request.return_value = mock_response
            result = service.sort(input_data)
            results.append(result)

        expected_results = [case[2] for case in test_cases]
        assert results == expected_results

    def test_service_isolation(self, mock_client):
        """Test that multiple service instances are properly isolated."""
        service1 = SortingService(mock_client)
        service2 = SortingService(mock_client)

        mock_client.request.return_value = '{"response": [1, 2, 3]}'

        result1 = service1.sort([3, 1, 2])
        result2 = service2.sort([2, 3, 1])

        assert result1 == [1, 2, 3]
        assert result2 == [1, 2, 3]
        assert mock_client.request.call_count == 2
