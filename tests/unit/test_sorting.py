import pytest
from vibelib.sort import SortingService
from vibelib.exceptions import ValidationError, ParseError

class TestSortingService:

    def test_sort_integers(self, mock_client):
        mock_client.request.return_value = '{"response": [1, 2, 3]}'
        service = SortingService(mock_client)

        result = service.sort([3, 1, 2])

        assert result == [1, 2, 3]
        mock_client.request.assert_called_once()

    def test_sort_empty_array(self, mock_client):
        service = SortingService(mock_client)

        result = service.sort([])

        assert result == []
        mock_client.request.assert_not_called()

    def test_sort_single_element(self, mock_client):
        service = SortingService(mock_client)

        result = service.sort([42])

        assert result == [42]
        mock_client.request.assert_not_called()

    def test_sort_floats(self, mock_client):
        mock_client.request.return_value = '{"response": [0.5, 1.5, 2.5]}'
        service = SortingService(mock_client)

        result = service.sort([2.5, 0.5, 1.5])

        assert result == [0.5, 1.5, 2.5]

    def test_sort_strings(self, mock_client):
        mock_client.request.return_value = '{"response": ["a", "b", "c"]}'
        service = SortingService(mock_client)

        result = service.sort(["c", "a", "b"])

        assert result == ["a", "b", "c"]

    def test_sort_mixed_numbers(self, mock_client):
        mock_client.request.return_value = '{"response": [1, 1.5, 2, 2.5]}'
        service = SortingService(mock_client)

        result = service.sort([2.5, 1, 2, 1.5])

        assert result == [1, 1.5, 2, 2.5]

    def test_sort_duplicate_elements(self, mock_client):
        mock_client.request.return_value = '{"response": [1, 1, 2, 2, 3]}'
        service = SortingService(mock_client)

        result = service.sort([2, 1, 3, 1, 2])

        assert result == [1, 1, 2, 2, 3]

    def test_sort_already_sorted(self, mock_client):
        mock_client.request.return_value = '{"response": [1, 2, 3, 4]}'
        service = SortingService(mock_client)

        result = service.sort([1, 2, 3, 4])

        assert result == [1, 2, 3, 4]

    def test_sort_reverse_sorted(self, mock_client):
        mock_client.request.return_value = '{"response": [1, 2, 3, 4]}'
        service = SortingService(mock_client)

        result = service.sort([4, 3, 2, 1])

        assert result == [1, 2, 3, 4]

    def test_sort_large_array(self, mock_client):
        large_input = list(range(100, 0, -1))  # [100, 99, 98, ..., 1]
        expected_output = list(range(1, 101))   # [1, 2, 3, ..., 100]
        mock_client.request.return_value = f'{{"response": {expected_output}}}'
        service = SortingService(mock_client)

        result = service.sort(large_input)

        assert result == expected_output

    @pytest.mark.parametrize("invalid_input", [
        "not a list",
        123,
        None,
        {"key": "value"},
        set([1, 2, 3]),
    ])
    def test_invalid_input_type(self, mock_client, invalid_input):
        service = SortingService(mock_client)

        with pytest.raises(ValidationError, match="Expected list"):
            service.sort(invalid_input)

    def test_tuple_input_converted_to_list(self, mock_client):
        mock_client.request.return_value = '{"response": [1, 2, 3]}'
        service = SortingService(mock_client)

        result = service.sort((3, 1, 2))

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_array_too_large(self, mock_client):
        service = SortingService(mock_client)
        large_array = list(range(10001))

        with pytest.raises(ValidationError, match="Array too large"):
            service.sort(large_array)

    def test_array_exactly_max_size(self, mock_client):
        max_size_array = [1] * 10000
        mock_client.request.return_value = f'{{"response": {max_size_array}}}'
        service = SortingService(mock_client)

        # Should not raise validation error
        service.sort(max_size_array)
        mock_client.request.assert_called_once()

    @pytest.mark.parametrize("invalid_item", [
        [1, 2, None],
        [1, 2, {"key": "value"}],
        [1, 2, [1, 2, 3]],
        [1, 2, object()],
        [1, 2, complex(1, 2)],
    ])
    def test_invalid_item_types(self, mock_client, invalid_item):
        service = SortingService(mock_client)

        with pytest.raises(ValidationError, match="Invalid type at index"):
            service.sort(invalid_item)

    def test_validation_error_reports_correct_index(self, mock_client):
        service = SortingService(mock_client)

        with pytest.raises(ValidationError, match="Invalid type at index 2"):
            service.sort([1, 2, None, 4])

    def test_client_prompt_construction_integers(self, mock_client):
        mock_client.request.return_value = '{"response": [1, 2, 3]}'
        service = SortingService(mock_client)

        service.sort([3, 1, 2])

        args, kwargs = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "Sort this array: [3, 1, 2]" in user_prompt
        assert '{"response": [sorted_array]}' in user_prompt
        assert "You are a sorting service" in system_prompt
        assert "valid JSON" in system_prompt

    def test_client_prompt_construction_strings(self, mock_client):
        mock_client.request.return_value = '{"response": ["a", "b", "c"]}'
        service = SortingService(mock_client)

        service.sort(["c", "a", "b"])

        args, kwargs = mock_client.request.call_args
        user_prompt = args[0]

        assert "Sort this array: ['c', 'a', 'b']" in user_prompt

    # JSON parsing error tests
    @pytest.mark.parametrize("bad_response", [
        "not json at all",
        '{"wrong_field": [1, 2, 3]}',  # Missing 'response' field
        '{"response": "not a list"}',   # Response field not a list
        '[1, 2, 3]',  # Array instead of object
        '{"response": null}',  # Null response
        '',  # Empty response
        '{"response":}',  # Invalid JSON syntax
        'undefined',
        'true',
        '42',
        '{"array": [1,2,3]}',  # Wrong field name
    ])
    def test_json_parse_error_handling(self, mock_client, bad_response):
        mock_client.request.return_value = bad_response
        service = SortingService(mock_client)

        with pytest.raises(ParseError, match="Invalid JSON response"):
            service.sort([1, 2, 3])

    def test_valid_json_with_extra_fields_ignored(self, mock_client):
        # AI might add extra fields - we should ignore them gracefully
        mock_client.request.return_value = '{"response": [1, 2, 3], "confidence": 0.99, "model": "gpt-4o-mini"}'
        service = SortingService(mock_client)

        result = service.sort([3, 1, 2])

        assert result == [1, 2, 3]  # Extra fields ignored

    def test_json_with_nested_arrays_in_response(self, mock_client):
        # Test that we can handle the response field containing what we expect
        mock_client.request.return_value = '{"response": [1, 2.5, "hello"]}'
        service = SortingService(mock_client)

        result = service.sort([2.5, 1, "hello"])

        assert result == [1, 2.5, "hello"]

    def test_parse_error_includes_raw_response(self, mock_client):
        bad_response = "this is not valid json"
        mock_client.request.return_value = bad_response
        service = SortingService(mock_client)

        with pytest.raises(ParseError) as exc_info:
            service.sort([1, 2, 3])

        # Should include the bad response in error message for debugging
        assert "Invalid JSON response" in str(exc_info.value)
        assert bad_response in str(exc_info.value)

    def test_unicode_strings_handling(self, mock_client):
        mock_client.request.return_value = '{"response": ["café", "naïve", "résumé"]}'
        service = SortingService(mock_client)

        result = service.sort(["résumé", "café", "naïve"])

        assert result == ["café", "naïve", "résumé"]

    def test_negative_numbers(self, mock_client):
        mock_client.request.return_value = '{"response": [-3, -1, 0, 1, 3]}'
        service = SortingService(mock_client)

        result = service.sort([1, -1, 3, -3, 0])

        assert result == [-3, -1, 0, 1, 3]

    def test_floating_point_precision(self, mock_client):
        mock_client.request.return_value = '{"response": [0.1, 0.2, 0.3]}'
        service = SortingService(mock_client)

        result = service.sort([0.3, 0.1, 0.2])

        assert result == [0.1, 0.2, 0.3]

    def test_very_long_strings(self, mock_client):
        long_string = "a" * 1000
        mock_client.request.return_value = f'{{"response": ["{long_string}"]}}'
        service = SortingService(mock_client)

        result = service.sort([long_string])

        assert result == [long_string]

    def test_json_response_structure_validation(self, mock_client):
        # Test that we properly validate JSON structure
        mock_client.request.return_value = '{"response": {"not": "a list"}}'
        service = SortingService(mock_client)

        with pytest.raises(ParseError, match="Response field must be a list"):
            service.sort([1, 2, 3])

    def test_missing_response_field_error(self, mock_client):
        mock_client.request.return_value = '{"data": [1, 2, 3]}'
        service = SortingService(mock_client)

        with pytest.raises(ParseError, match="Response must contain 'response' field"):
            service.sort([1, 2, 3])

    def test_non_object_json_response(self, mock_client):
        mock_client.request.return_value = '"just a string"'
        service = SortingService(mock_client)

        with pytest.raises(ParseError, match="Response must be a JSON object"):
            service.sort([1, 2, 3])
