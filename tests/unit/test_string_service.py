"""
Comprehensive tests for string manipulation operations service.

Tests all string operations including edge cases, Unicode handling,
and error scenarios with proper prompt validation.
"""
import json

import pytest

from vibelib.operations.strings import StringService
from vibelib.exceptions import ParseError


class TestStringServiceUpper:
    """Test uppercase conversion operation."""

    def test_upper_basic_string(self, mock_client):
        """Test basic uppercase conversion."""
        mock_client.request.return_value = '{"response": "HELLO"}'
        service = StringService(mock_client)

        result = service.upper("hello")

        assert result == "HELLO"
        mock_client.request.assert_called_once()

    def test_upper_mixed_case(self, mock_client):
        """Test uppercase conversion with mixed case input."""
        mock_client.request.return_value = '{"response": "HELLO WORLD"}'
        service = StringService(mock_client)

        result = service.upper("Hello World")

        assert result == "HELLO WORLD"

    def test_upper_already_uppercase(self, mock_client):
        """Test uppercase conversion on already uppercase string."""
        mock_client.request.return_value = '{"response": "ALREADY UPPER"}'
        service = StringService(mock_client)

        result = service.upper("ALREADY UPPER")

        assert result == "ALREADY UPPER"

    def test_upper_with_numbers_and_symbols(self, mock_client):
        """Test uppercase conversion with numbers and symbols."""
        mock_client.request.return_value = '{"response": "HELLO123!@#"}'
        service = StringService(mock_client)

        result = service.upper("hello123!@#")

        assert result == "HELLO123!@#"

    def test_upper_empty_string(self, mock_client):
        """Test uppercase conversion with empty string."""
        mock_client.request.return_value = '{"response": ""}'
        service = StringService(mock_client)

        result = service.upper("")

        assert result == ""

    def test_upper_whitespace_only(self, mock_client):
        """Test uppercase conversion with whitespace-only string."""
        mock_client.request.return_value = '{"response": "   "}'
        service = StringService(mock_client)

        result = service.upper("   ")

        assert result == "   "

    def test_upper_unicode_characters(self, mock_client):
        """Test uppercase conversion with Unicode characters."""
        mock_client.request.return_value = '{"response": "CAFÉ NAÏVE"}'
        service = StringService(mock_client)

        result = service.upper("café naïve")

        assert result == "CAFÉ NAÏVE"

    def test_upper_prompt_construction(self, mock_client):
        """Test that upper operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": "HELLO"}'
        service = StringService(mock_client)

        service.upper("hello")

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "hello" in user_prompt
        assert "Convert this string to uppercase" in user_prompt
        assert "fully uppercase format" in system_prompt
        assert "JSON" in system_prompt


class TestStringServiceLower:
    """Test lowercase conversion operation."""

    def test_lower_basic_string(self, mock_client):
        """Test basic lowercase conversion."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        result = service.lower("HELLO")

        assert result == "hello"

    def test_lower_mixed_case(self, mock_client):
        """Test lowercase conversion with mixed case input."""
        mock_client.request.return_value = '{"response": "hello world"}'
        service = StringService(mock_client)

        result = service.lower("Hello World")

        assert result == "hello world"

    def test_lower_already_lowercase(self, mock_client):
        """Test lowercase conversion on already lowercase string."""
        mock_client.request.return_value = '{"response": "already lower"}'
        service = StringService(mock_client)

        result = service.lower("already lower")

        assert result == "already lower"

    def test_lower_with_numbers_and_symbols(self, mock_client):
        """Test lowercase conversion with numbers and symbols."""
        mock_client.request.return_value = '{"response": "hello123!@#"}'
        service = StringService(mock_client)

        result = service.lower("HELLO123!@#")

        assert result == "hello123!@#"

    def test_lower_unicode_characters(self, mock_client):
        """Test lowercase conversion with Unicode characters."""
        mock_client.request.return_value = '{"response": "café naïve"}'
        service = StringService(mock_client)

        result = service.lower("CAFÉ NAÏVE")

        assert result == "café naïve"

    def test_lower_prompt_construction(self, mock_client):
        """Test that lower operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        service.lower("HELLO")

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "HELLO" in user_prompt
        assert "Convert this string to lowercase" in user_prompt
        assert "fully lowercase format" in system_prompt


class TestStringServiceSplit:
    """Test string splitting operation."""

    def test_split_basic_comma_separator(self, mock_client):
        """Test basic string splitting with comma separator."""
        mock_client.request.return_value = '{"response": ["a", "b", "c"]}'
        service = StringService(mock_client)

        result = service.split("a,b,c", ",")

        assert result == ["a", "b", "c"]

    def test_split_space_separator(self, mock_client):
        """Test string splitting with space separator."""
        mock_client.request.return_value = '{"response": ["hello", "world", "test"]}'
        service = StringService(mock_client)

        result = service.split("hello world test", " ")

        assert result == ["hello", "world", "test"]

    def test_split_multi_character_separator(self, mock_client):
        """Test string splitting with multi-character separator."""
        mock_client.request.return_value = '{"response": ["part1", "part2", "part3"]}'
        service = StringService(mock_client)

        result = service.split("part1::part2::part3", "::")

        assert result == ["part1", "part2", "part3"]

    def test_split_no_separator_found(self, mock_client):
        """Test string splitting when separator is not found."""
        mock_client.request.return_value = '{"response": ["hello world"]}'
        service = StringService(mock_client)

        result = service.split("hello world", ",")

        assert result == ["hello world"]

    def test_split_empty_string(self, mock_client):
        """Test splitting an empty string."""
        mock_client.request.return_value = '{"response": [""]}'
        service = StringService(mock_client)

        result = service.split("", ",")

        assert result == [""]

    def test_split_separator_at_start(self, mock_client):
        """Test splitting with separator at the start."""
        mock_client.request.return_value = '{"response": ["", "a", "b"]}'
        service = StringService(mock_client)

        result = service.split(",a,b", ",")

        assert result == ["", "a", "b"]

    def test_split_separator_at_end(self, mock_client):
        """Test splitting with separator at the end."""
        mock_client.request.return_value = '{"response": ["a", "b", ""]}'
        service = StringService(mock_client)

        result = service.split("a,b,", ",")

        assert result == ["a", "b", ""]

    def test_split_consecutive_separators(self, mock_client):
        """Test splitting with consecutive separators."""
        mock_client.request.return_value = '{"response": ["a", "", "", "b"]}'
        service = StringService(mock_client)

        result = service.split("a,,,b", ",")

        assert result == ["a", "", "", "b"]

    def test_split_unicode_content_and_separator(self, mock_client):
        """Test splitting with Unicode content and separator."""
        mock_client.request.return_value = '{"response": ["café", "naïve", "résumé"]}'
        service = StringService(mock_client)

        result = service.split("café•naïve•résumé", "•")

        assert result == ["café", "naïve", "résumé"]

    def test_split_prompt_construction(self, mock_client):
        """Test that split operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": ["a", "b", "c"]}'
        service = StringService(mock_client)

        service.split("a,b,c", ",")

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "a,b,c" in user_prompt
        assert "separator ','" in user_prompt
        assert "Split the given string at every occurrence" in system_prompt  # Fixed capitalization
        assert 'array of strings' in system_prompt


class TestStringServiceJoin:
    """Test string joining operation."""

    def test_join_string_list(self, mock_client):
        """Test joining a list of strings."""
        mock_client.request.return_value = '{"response": "a,b,c"}'
        service = StringService(mock_client)

        result = service.join(["a", "b", "c"], ",")

        assert result == "a,b,c"

    def test_join_mixed_types(self, mock_client):
        """Test joining a list with mixed types."""
        mock_client.request.return_value = '{"response": "1,hello,2.5"}'
        service = StringService(mock_client)

        result = service.join([1, "hello", 2.5], ",")

        assert result == "1,hello,2.5"

    def test_join_integers(self, mock_client):
        """Test joining a list of integers."""
        mock_client.request.return_value = '{"response": "1-2-3"}'
        service = StringService(mock_client)

        result = service.join([1, 2, 3], "-")

        assert result == "1-2-3"

    def test_join_floats(self, mock_client):
        """Test joining a list of floats."""
        mock_client.request.return_value = '{"response": "1.1|2.2|3.3"}'
        service = StringService(mock_client)

        result = service.join([1.1, 2.2, 3.3], "|")

        assert result == "1.1|2.2|3.3"

    def test_join_empty_list(self, mock_client):
        """Test joining an empty list."""
        mock_client.request.return_value = '{"response": ""}'
        service = StringService(mock_client)

        result = service.join([], ",")

        assert result == ""

    def test_join_single_item(self, mock_client):
        """Test joining a list with single item."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        result = service.join(["hello"], ",")

        assert result == "hello"

    def test_join_empty_strings(self, mock_client):
        """Test joining a list containing empty strings."""
        mock_client.request.return_value = '{"response": "a,,b"}'
        service = StringService(mock_client)

        result = service.join(["a", "", "b"], ",")

        assert result == "a,,b"

    def test_join_multi_character_separator(self, mock_client):
        """Test joining with multi-character separator."""
        mock_client.request.return_value = '{"response": "a::b::c"}'
        service = StringService(mock_client)

        result = service.join(["a", "b", "c"], "::")

        assert result == "a::b::c"

    def test_join_unicode_content(self, mock_client):
        """Test joining with Unicode content."""
        mock_client.request.return_value = '{"response": "café•naïve•résumé"}'
        service = StringService(mock_client)

        result = service.join(["café", "naïve", "résumé"], "•")

        assert result == "café•naïve•résumé"

    def test_join_prompt_construction(self, mock_client):
        """Test that join operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": "a,b,c"}'
        service = StringService(mock_client)

        service.join(["a", "b", "c"], ",")

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "['a', 'b', 'c']" in user_prompt
        assert "separator ','" in user_prompt
        assert "Join all elements" in system_prompt  # Fixed capitalization
        assert "Convert non-string elements to strings first" in system_prompt


class TestStringServiceStrip:
    """Test string stripping operation."""

    def test_strip_leading_whitespace(self, mock_client):
        """Test stripping leading whitespace."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        result = service.strip("   hello")

        assert result == "hello"

    def test_strip_trailing_whitespace(self, mock_client):
        """Test stripping trailing whitespace."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        result = service.strip("hello   ")

        assert result == "hello"

    def test_strip_both_ends_whitespace(self, mock_client):
        """Test stripping whitespace from both ends."""
        mock_client.request.return_value = '{"response": "hello world"}'
        service = StringService(mock_client)

        result = service.strip("  hello world  ")

        assert result == "hello world"

    def test_strip_mixed_whitespace_types(self, mock_client):
        """Test stripping mixed whitespace types (spaces, tabs, newlines)."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        result = service.strip(" \t\nhello\n\t ")

        assert result == "hello"

    def test_strip_no_whitespace(self, mock_client):
        """Test stripping string with no whitespace."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        result = service.strip("hello")

        assert result == "hello"

    def test_strip_whitespace_only(self, mock_client):
        """Test stripping string with only whitespace."""
        mock_client.request.return_value = '{"response": ""}'
        service = StringService(mock_client)

        result = service.strip("   \t\n  ")

        assert result == ""

    def test_strip_empty_string(self, mock_client):
        """Test stripping empty string."""
        mock_client.request.return_value = '{"response": ""}'
        service = StringService(mock_client)

        result = service.strip("")

        assert result == ""

    def test_strip_preserves_internal_whitespace(self, mock_client):
        """Test that strip preserves internal whitespace."""
        mock_client.request.return_value = '{"response": "hello world test"}'
        service = StringService(mock_client)

        result = service.strip("  hello world test  ")

        assert result == "hello world test"

    def test_strip_unicode_content(self, mock_client):
        """Test stripping with Unicode content."""
        mock_client.request.return_value = '{"response": "café naïve"}'
        service = StringService(mock_client)

        result = service.strip("  café naïve  ")

        assert result == "café naïve"

    def test_strip_prompt_construction(self, mock_client):
        """Test that strip operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        service.strip("  hello  ")

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "'  hello  '" in user_prompt
        assert "Strip whitespace from the beginning and end" in user_prompt
        assert "Remove all whitespace characters" in system_prompt
        assert "Leave whitespace in the middle untouched" in system_prompt


class TestStringServiceReplace:
    """Test string replacement operation."""

    def test_replace_single_occurrence(self, mock_client):
        """Test replacing single occurrence of substring."""
        mock_client.request.return_value = '{"response": "hexlo"}'
        service = StringService(mock_client)

        result = service.replace("hello", "l", "x")

        assert result == "hexlo"

    def test_replace_multiple_occurrences(self, mock_client):
        """Test replacing multiple occurrences of substring."""
        mock_client.request.return_value = '{"response": "hexxo"}'
        service = StringService(mock_client)

        result = service.replace("hello", "l", "x")

        assert result == "hexxo"

    def test_replace_no_occurrences(self, mock_client):
        """Test replacing when substring is not found."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        result = service.replace("hello", "z", "x")

        assert result == "hello"

    def test_replace_entire_string(self, mock_client):
        """Test replacing the entire string."""
        mock_client.request.return_value = '{"response": "world"}'
        service = StringService(mock_client)

        result = service.replace("hello", "hello", "world")

        assert result == "world"

    def test_replace_with_empty_string(self, mock_client):
        """Test replacing with empty string (deletion)."""
        mock_client.request.return_value = '{"response": "heo"}'
        service = StringService(mock_client)

        result = service.replace("hello", "ll", "")

        assert result == "heo"

    def test_replace_empty_substring(self, mock_client):
        """Test replacing empty substring."""
        mock_client.request.return_value = '{"response": "hello"}'
        service = StringService(mock_client)

        result = service.replace("hello", "", "x")

        assert result == "hello"

    def test_replace_overlapping_patterns(self, mock_client):
        """Test replacing with overlapping patterns."""
        mock_client.request.return_value = '{"response": "abxcd"}'
        service = StringService(mock_client)

        result = service.replace("ababab", "ab", "x")

        assert result == "abxcd"

    def test_replace_multi_character_substrings(self, mock_client):
        """Test replacing multi-character substrings."""
        mock_client.request.return_value = '{"response": "hello python world"}'
        service = StringService(mock_client)

        result = service.replace("hello java world", "java", "python")

        assert result == "hello python world"

    def test_replace_case_sensitive(self, mock_client):
        """Test that replacement is case sensitive."""
        mock_client.request.return_value = '{"response": "Hello world"}'
        service = StringService(mock_client)

        result = service.replace("Hello world", "hello", "hi")

        assert result == "Hello world"

    def test_replace_unicode_content(self, mock_client):
        """Test replacing with Unicode content."""
        mock_client.request.return_value = '{"response": "café and naïve"}'
        service = StringService(mock_client)

        result = service.replace("coffee and naive", "coffee", "café")

        assert result == "café and naïve"

    def test_replace_special_characters(self, mock_client):
        """Test replacing special characters."""
        mock_client.request.return_value = '{"response": "hello@world"}'
        service = StringService(mock_client)

        result = service.replace("hello world", " ", "@")

        assert result == "hello@world"

    def test_replace_prompt_construction(self, mock_client):
        """Test that replace operation constructs proper prompts."""
        mock_client.request.return_value = '{"response": "hexxo"}'
        service = StringService(mock_client)

        service.replace("hello", "l", "x")

        args, _ = mock_client.request.call_args
        user_prompt = args[0]
        system_prompt = args[1]

        assert "'hello'" in user_prompt
        assert "replace all 'l' with 'x'" in user_prompt  # Fixed to match actual prompt
        assert "replace ALL occurrences" in system_prompt


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_string_operations(self, mock_client):
        """Test operations with very long strings."""
        long_string = "a" * 10000
        mock_client.request.return_value = f'{{"response": "{long_string.upper()}"}}'
        service = StringService(mock_client)

        result = service.upper(long_string)

        assert result == long_string.upper()

    def test_string_with_special_json_characters(self, mock_client):
        """Test strings containing JSON special characters."""
        test_string = 'Hello "world" with \\ and / characters'
        mock_client.request.return_value = '{"response": "HELLO \\"WORLD\\" WITH \\\\ AND / CHARACTERS"}'
        service = StringService(mock_client)

        result = service.upper(test_string)

        assert result == 'HELLO "WORLD" WITH \\ AND / CHARACTERS'

    def test_string_with_newlines_and_tabs(self, mock_client):
        """Test strings containing newlines and tabs."""
        test_string = "hello\nworld\ttest"
        mock_client.request.return_value = '{"response": "HELLO\\nWORLD\\tTEST"}'
        service = StringService(mock_client)

        result = service.upper(test_string)

        assert result == "HELLO\nWORLD\tTEST"

    @pytest.mark.parametrize("operation,input_data,mock_response", [
        ("upper", "", '{"response": ""}'),
        ("lower", "", '{"response": ""}'),
        ("split", ("", ","), '{"response": [""]}'),
        ("join", ([], ","), '{"response": ""}'),
        ("strip", "", '{"response": ""}'),
        ("replace", ("", "a", "b"), '{"response": ""}'),
    ])
    def test_empty_string_handling(self, mock_client, operation, input_data, mock_response):
        """Test all operations with empty string inputs."""
        mock_client.request.return_value = mock_response
        service = StringService(mock_client)

        operation_func = getattr(service, operation)
        if isinstance(input_data, tuple):
            result = operation_func(*input_data)
        else:
            result = operation_func(input_data)

        # All should handle empty strings gracefully
        assert result is not None


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json_response_handling(self, mock_client):
        """Test handling of invalid JSON responses."""
        mock_client.request.return_value = 'not json'
        service = StringService(mock_client)

        with pytest.raises(ParseError):
            service.upper("hello")

    def test_missing_response_field(self, mock_client):
        """Test handling of JSON without response field."""
        mock_client.request.return_value = '{"result": "HELLO"}'
        service = StringService(mock_client)

        with pytest.raises(ParseError, match="Response must contain 'response' field"):
            service.upper("hello")

    def test_non_string_response(self, mock_client):
        """Test handling of non-string response values."""
        mock_client.request.return_value = '{"response": 123}'
        service = StringService(mock_client)

        # Should work fine - we don't validate the response type
        result = service.upper("hello")
        assert result == 123

    def test_null_response_value(self, mock_client):
        """Test handling of null response value."""
        mock_client.request.return_value = '{"response": null}'
        service = StringService(mock_client)

        result = service.upper("hello")
        assert result is None


class TestTypeHandling:
    """Test handling of different input types."""

    @pytest.mark.parametrize("input_value", [
        "normal string",
        "",
        " ",
        "string with numbers 123",
        "string with symbols !@#$%",
        "café naïve résumé",  # Unicode
        "line1\nline2\ttab",  # Whitespace characters
        "very " + "long " * 100 + "string",  # Long string
    ])
    def test_various_string_inputs(self, mock_client, input_value):
        """Test various string input types."""
        # Use json.dumps to properly escape control characters
        expected_upper = input_value.upper()
        mock_client.request.return_value = json.dumps({"response": expected_upper})
        service = StringService(mock_client)

        result = service.upper(input_value)

        assert result == expected_upper


class TestConcurrency:
    """Test concurrent operation scenarios."""

    def test_multiple_operations_same_service(self, mock_client):
        """Test multiple operations on the same service instance."""
        service = StringService(mock_client)

        test_cases = [
            ('upper', 'hello', '{"response": "HELLO"}', "HELLO"),
            ('lower', 'WORLD', '{"response": "world"}', "world"),
            ('strip', '  test  ', '{"response": "test"}', "test"),
        ]

        results = []
        for operation, input_data, mock_response, expected in test_cases:
            mock_client.request.return_value = mock_response
            result = getattr(service, operation)(input_data)
            results.append(result)

        expected_results = [case[3] for case in test_cases]
        assert results == expected_results

    def test_service_isolation(self, mock_client):
        """Test that multiple service instances are properly isolated."""
        service1 = StringService(mock_client)
        service2 = StringService(mock_client)

        mock_client.request.return_value = '{"response": "TEST"}'

        result1 = service1.upper("test")
        result2 = service2.upper("test")

        assert result1 == "TEST"
        assert result2 == "TEST"
        assert mock_client.request.call_count == 2
