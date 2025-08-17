"""
Comprehensive tests for BaseService functionality.

Tests the core JSON parsing, request handling, and error management
that all service classes inherit from BaseService.
"""

import json
import pytest

from vibelib.operations.base import BaseService
from vibelib.exceptions import ParseError


class TestBaseServiceInitialization:
    """Test BaseService initialization and setup."""

    def test_base_service_initialization(self, mock_client):
        """Test that BaseService initializes correctly with client."""
        service = BaseService(mock_client)

        assert service._client is mock_client

    def test_base_service_is_abstract(self, mock_client):
        """Test that BaseService can be instantiated (it's not fully abstract)."""
        # BaseService is not fully abstract, it just provides base functionality
        service = BaseService(mock_client)
        assert isinstance(service, BaseService)


class TestRequestJsonMethod:
    """Test the _request_json method."""

    def test_request_json_calls_client_and_parses(self, mock_client):
        """Test that _request_json calls client.request and parses response."""
        mock_client.request.return_value = '{"response": "test_value"}'
        service = BaseService(mock_client)

        result = service._request_json("user prompt", "system prompt")

        assert result == "test_value"
        mock_client.request.assert_called_once_with("user prompt", "system prompt")

    def test_request_json_passes_prompts_correctly(self, mock_client):
        """Test that prompts are passed correctly to client."""
        mock_client.request.return_value = '{"response": "test"}'
        service = BaseService(mock_client)

        service._request_json("test user prompt", "test system prompt")

        mock_client.request.assert_called_once_with("test user prompt", "test system prompt")

    def test_request_json_handles_client_exceptions(self, mock_client):
        """Test that client exceptions are propagated correctly."""
        mock_client.request.side_effect = Exception("Client error")
        service = BaseService(mock_client)

        with pytest.raises(Exception, match="Client error"):
            service._request_json("user", "system")


class TestJsonResponseParsing:
    """Test JSON response parsing functionality."""

    def test_parse_valid_json_response(self, mock_client):
        """Test parsing valid JSON response."""
        service = BaseService(mock_client)

        result = service._parse_json_response('{"response": "test_value"}')

        assert result == "test_value"

    def test_parse_json_with_different_data_types(self, mock_client):
        """Test parsing JSON with various data types."""
        service = BaseService(mock_client)

        test_cases = [
            ('{"response": "string"}', "string"),
            ('{"response": 42}', 42),
            ('{"response": 3.14}', 3.14),
            ('{"response": true}', True),
            ('{"response": false}', False),
            ('{"response": null}', None),
            ('{"response": [1, 2, 3]}', [1, 2, 3]),
            ('{"response": {"key": "value"}}', {"key": "value"}),
        ]

        for json_str, expected in test_cases:
            result = service._parse_json_response(json_str)
            assert result == expected

    def test_parse_json_with_extra_fields(self, mock_client):
        """Test that extra fields in JSON are ignored."""
        service = BaseService(mock_client)

        json_with_extras = '''
        {
            "response": "target_value",
            "confidence": 0.95,
            "model": "gpt-4",
            "timestamp": "2024-01-01"
        }
        '''

        result = service._parse_json_response(json_with_extras)

        assert result == "target_value"

    def test_parse_json_with_whitespace(self, mock_client):
        """Test parsing JSON with extra whitespace."""
        service = BaseService(mock_client)

        json_with_whitespace = '''
        
        {"response": "test_value"}
        
        '''

        result = service._parse_json_response(json_with_whitespace)

        assert result == "test_value"


class TestMarkdownCodeBlockHandling:
    """Test handling of markdown code blocks in responses."""

    def test_parse_json_in_markdown_code_block(self, mock_client):
        """Test parsing JSON wrapped in markdown code block."""
        service = BaseService(mock_client)

        markdown_response = '''```json
        {"response": "test_value"}
        ```'''

        result = service._parse_json_response(markdown_response)

        assert result == "test_value"

    def test_parse_json_in_generic_code_block(self, mock_client):
        """Test parsing JSON wrapped in generic code block."""
        service = BaseService(mock_client)

        code_block_response = '''```
        {"response": "test_value"}
        ```'''

        result = service._parse_json_response(code_block_response)

        assert result == "test_value"

    def test_parse_json_with_mixed_markdown_and_whitespace(self, mock_client):
        """Test parsing JSON with both markdown and whitespace."""
        service = BaseService(mock_client)

        complex_response = '''
        ```json
        
        {"response": "test_value"}
        
        ```
        '''

        result = service._parse_json_response(complex_response)

        assert result == "test_value"

    def test_parse_plain_json_not_affected_by_markdown_logic(self, mock_client):
        """Test that plain JSON still works with markdown logic."""
        service = BaseService(mock_client)

        plain_json = '{"response": "test_value"}'

        result = service._parse_json_response(plain_json)

        assert result == "test_value"


class TestJsonParsingErrors:
    """Test JSON parsing error scenarios."""

    def test_invalid_json_raises_parse_error(self, mock_client):
        """Test that invalid JSON raises ParseError."""
        service = BaseService(mock_client)

        with pytest.raises(ParseError, match="Invalid JSON response"):
            service._parse_json_response("not json at all")

    def test_malformed_json_raises_parse_error(self, mock_client):
        """Test that malformed JSON raises ParseError."""
        service = BaseService(mock_client)

        with pytest.raises(ParseError, match="Invalid JSON response"):
            service._parse_json_response('{"response": "unclosed string}')

    def test_non_object_json_raises_parse_error(self, mock_client):
        """Test that non-object JSON raises ParseError."""
        service = BaseService(mock_client)

        with pytest.raises(ParseError, match="Response must be a JSON object"):
            service._parse_json_response('"just a string"')

    def test_missing_response_field_raises_parse_error(self, mock_client):
        """Test that missing response field raises ParseError."""
        service = BaseService(mock_client)

        with pytest.raises(ParseError, match="Response must contain 'response' field"):
            service._parse_json_response('{"data": "wrong_field"}')

    def test_parse_error_includes_raw_response(self, mock_client):
        """Test that ParseError includes raw response for debugging."""
        service = BaseService(mock_client)
        bad_response = "this is definitely not json"

        with pytest.raises(ParseError) as exc_info:
            service._parse_json_response(bad_response)

        assert bad_response in str(exc_info.value)

    def test_parse_error_truncates_long_responses(self, mock_client):
        """Test that very long responses are truncated in error messages."""
        service = BaseService(mock_client)
        very_long_bad_response = "bad json " * 100  # 800+ characters

        with pytest.raises(ParseError) as exc_info:
            service._parse_json_response(very_long_bad_response)

        error_message = str(exc_info.value)
        # Should include first 200 chars
        assert very_long_bad_response[:200] in error_message
        # But not the entire response
        assert len(error_message) < len(very_long_bad_response)


class TestComplexJsonStructures:
    """Test parsing of complex JSON structures."""

    def test_parse_nested_objects(self, mock_client):
        """Test parsing deeply nested JSON objects."""
        service = BaseService(mock_client)

        nested_json = '''
        {
            "response": {
                "level1": {
                    "level2": {
                        "level3": "deep_value"
                    }
                }
            }
        }
        '''

        result = service._parse_json_response(nested_json)

        assert result == {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                }
            }
        }

    def test_parse_complex_arrays(self, mock_client):
        """Test parsing complex array structures."""
        service = BaseService(mock_client)

        complex_array_json = '''
        {
            "response": [
                {"name": "item1", "value": 100},
                {"name": "item2", "value": 200, "nested": [1, 2, 3]},
                "simple_string",
                42
            ]
        }
        '''

        result = service._parse_json_response(complex_array_json)

        expected = [
            {"name": "item1", "value": 100},
            {"name": "item2", "value": 200, "nested": [1, 2, 3]},
            "simple_string",
            42
        ]

        assert result == expected

    def test_parse_unicode_content(self, mock_client):
        """Test parsing JSON with Unicode content."""
        service = BaseService(mock_client)

        unicode_json = '{"response": "café naïve résumé 🚀"}'

        result = service._parse_json_response(unicode_json)

        assert result == "café naïve résumé 🚀"

    def test_parse_json_with_escaped_characters(self, mock_client):
        """Test parsing JSON with escaped characters."""
        service = BaseService(mock_client)

        escaped_json = r'{"response": "line1\nline2\ttab\"quote\\"}'

        result = service._parse_json_response(escaped_json)

        assert result == 'line1\nline2\ttab"quote\\'


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_parse_empty_string_response_field(self, mock_client):
        """Test parsing JSON with empty string in response field."""
        service = BaseService(mock_client)

        result = service._parse_json_response('{"response": ""}')

        assert result == ""

    def test_parse_zero_response_field(self, mock_client):
        """Test parsing JSON with zero in response field."""
        service = BaseService(mock_client)

        result = service._parse_json_response('{"response": 0}')

        assert result == 0

    def test_parse_false_response_field(self, mock_client):
        """Test parsing JSON with false in response field."""
        service = BaseService(mock_client)

        result = service._parse_json_response('{"response": false}')

        assert result == False

    def test_parse_empty_array_response_field(self, mock_client):
        """Test parsing JSON with empty array in response field."""
        service = BaseService(mock_client)

        result = service._parse_json_response('{"response": []}')

        assert result == []

    def test_parse_empty_object_response_field(self, mock_client):
        """Test parsing JSON with empty object in response field."""
        service = BaseService(mock_client)

        result = service._parse_json_response('{"response": {}}')

        assert result == {}

    def test_parse_very_large_json_response(self, mock_client):
        """Test parsing very large JSON responses."""
        service = BaseService(mock_client)

        # Create large array
        large_array = list(range(1000))
        large_json = json.dumps({"response": large_array})

        result = service._parse_json_response(large_json)

        assert result == large_array
        assert len(result) == 1000


class TestServiceLogging:
    """Test logging behavior in BaseService."""

    def test_debug_logging_on_successful_parsing(self, mock_client, capture_logs):
        """Test that successful parsing generates appropriate debug logs."""
        service = BaseService(mock_client)

        service._parse_json_response('{"response": "test"}')

        log_contents = capture_logs.getvalue()
        assert "Successfully parsed JSON response" in log_contents

    def test_error_logging_on_parsing_failure(self, mock_client, capture_logs):
        """Test that parsing failures generate appropriate error logs."""
        service = BaseService(mock_client)

        with pytest.raises(ParseError):
            service._parse_json_response("invalid json")

        log_contents = capture_logs.getvalue()
        assert "JSON parsing failed" in log_contents


class TestServiceIntegrationWithClient:
    """Test BaseService integration with client."""

    def test_full_request_json_flow(self, mock_client):
        """Test complete flow from _request_json through parsing."""
        mock_client.request.return_value = '''
        ```json
        {
            "response": {"result": "success", "data": [1, 2, 3]},
            "confidence": 0.95
        }
        ```
        '''

        service = BaseService(mock_client)

        result = service._request_json("test user prompt", "test system prompt")

        expected = {"result": "success", "data": [1, 2, 3]}
        assert result == expected
        mock_client.request.assert_called_once_with("test user prompt", "test system prompt")

    def test_error_propagation_from_client_to_service(self, mock_client):
        """Test that client errors propagate correctly through BaseService."""
        mock_client.request.side_effect = ConnectionError("Network failure")
        service = BaseService(mock_client)

        with pytest.raises(ConnectionError, match="Network failure"):
            service._request_json("test", "test")
