"""
Comprehensive integration tests for all VibeLib services.

Tests the complete integration chain from configuration through
client to all service types, including error propagation and cross-service scenarios.
"""

import json
import pytest
import time
from unittest.mock import patch, MagicMock, call

from vibelib.client import Client
from vibelib.config import Config
from vibelib.operations.sorting import SortingService
from vibelib.operations.basic import BasicService
from vibelib.operations.strings import StringService
from vibelib.operations.lists import ListService
from vibelib.exceptions import APIError, ParseError, ConfigurationError, ValidationError


class TestConfigClientServiceChain:
    """Test integration across the complete config->client->service chain."""

    @pytest.fixture
    def all_services(self, mock_openai_client):
        """Create instances of all service types for testing."""
        config = Config(api_key="integration-test-key")
        client = Client(config)

        return {
            'sorting': SortingService(client),
            'basic': BasicService(client),
            'string': StringService(client),
            'list': ListService(client),
            'client': client,
            'config': config
        }

    def test_config_propagation_to_all_services(self, mock_openai_client):
        """Test that configuration propagates correctly to all service types."""
        config = Config(
            api_key="chain-test-key",
            model="gpt-4",
            temperature=0.7,
            timeout=45.0,
            max_retries=2
        )

        client = Client(config)
        services = {
            'sorting': SortingService(client),
            'basic': BasicService(client),
            'string': StringService(client),
            'list': ListService(client)
        }

        # Test each service type
        test_cases = [
            ('sorting', services['sorting'].sort, [[3, 1, 2]], '{"response": [1, 2, 3]}'),
            ('basic', services['basic'].max, [[1, 5, 3]], '{"response": 5}'),
            ('string', services['string'].upper, ["hello"], '{"response": "HELLO"}'),
            ('list', services['list'].reverse, [[1, 2, 3]], '{"response": [3, 2, 1]}')
        ]

        for service_name, method, args, mock_response in test_cases:
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = mock_response

            result = method(*args)

            # Verify config was used
            call_kwargs = mock_openai_client.return_value.chat.completions.create.call_args.kwargs
            assert call_kwargs['model'] == 'gpt-4'
            assert call_kwargs['temperature'] == 0.7
            assert call_kwargs['timeout'] == 45.0

            # Reset for next test
            mock_openai_client.reset_mock()

    def test_service_isolation_with_same_client(self, all_services, mock_openai_client):
        """Test that different service instances using same client are properly isolated."""
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": "test"}'

        # Use different services with same client
        sorting_result = all_services['sorting'].sort([3, 1, 2])
        basic_result = all_services['basic'].max([1, 5, 3])
        string_result = all_services['string'].upper("hello")
        list_result = all_services['list'].count([1, 2, 1], 1)

        # All should work independently
        assert all_services['sorting'] is not all_services['basic']
        assert all_services['basic'] is not all_services['string']
        assert mock_openai_client.return_value.chat.completions.create.call_count == 4

    def test_multiple_clients_with_different_configs(self, mock_openai_client):
        """Test multiple clients with different configurations."""
        configs = [
            Config(api_key="key1", model="gpt-3.5-turbo", temperature=0.1),
            Config(api_key="key2", model="gpt-4", temperature=0.9),
            Config(api_key="key3", model="gpt-4o-mini", temperature=0.5)
        ]

        clients = [Client(config) for config in configs]
        services = [SortingService(client) for client in clients]

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # Test all services
        results = []
        for service in services:
            result = service.sort([3, 1, 2])
            results.append(result)

        # All should work with their respective configs
        assert all(result == [1, 2, 3] for result in results)
        assert mock_openai_client.return_value.chat.completions.create.call_count == 3


class TestErrorPropagationAcrossServices:
    """Test error propagation through all service types."""

    @pytest.mark.parametrize("service_class,method_name,args", [
        (SortingService, 'sort', [[1, 2, 3]]),
        (BasicService, 'max', [[1, 2, 3]]),
        (StringService, 'upper', ["hello"]),
        (ListService, 'reverse', [[1, 2, 3]])
    ])
    def test_api_error_propagation_all_services(self, mock_openai_client, service_class, method_name, args):
        """Test API error propagation across all service types."""
        config = Config(api_key="error-test-key")
        client = Client(config)
        service = service_class(client)

        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("API Error")

        method = getattr(service, method_name)
        with pytest.raises(APIError, match="Request failed"):
            method(*args)

    @pytest.mark.parametrize("service_class,method_name,args", [
        (SortingService, 'sort', [[1, 2, 3]]),
        (BasicService, 'sum', [[1, 2, 3]]),
        (StringService, 'lower', ["HELLO"]),
        (ListService, 'count', [[1, 2, 1], 1])
    ])
    def test_parse_error_propagation_all_services(self, mock_openai_client, service_class, method_name, args):
        """Test parse error propagation across all service types."""
        config = Config(api_key="parse-error-test-key")
        client = Client(config)
        service = service_class(client)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = 'invalid json'

        method = getattr(service, method_name)
        with pytest.raises(ParseError, match="Invalid JSON response"):
            method(*args)

    def test_validation_error_from_sorting_service(self, mock_openai_client):
        """Test validation error from sorting service with oversized input."""
        config = Config(api_key="validation-test-key")
        client = Client(config)
        service = SortingService(client)

        oversized_array = list(range(10001))

        with pytest.raises(ValidationError, match="Input too large"):
            service.sort(oversized_array)

        # Should not make API call
        mock_openai_client.return_value.chat.completions.create.assert_not_called()


class TestRetryBehaviorIntegration:
    """Test retry behavior across all service types."""

    @patch('time.sleep')
    @pytest.mark.parametrize("service_class,method_name,args,mock_response", [
        (SortingService, 'sort', [[3, 1, 2]], '{"response": [1, 2, 3]}'),
        (BasicService, 'max', [[1, 5, 3]], '{"response": 5}'),
        (StringService, 'upper', ["hello"], '{"response": "HELLO"}'),
        (ListService, 'reverse', [[1, 2, 3]], '{"response": [3, 2, 1]}')
    ])
    def test_retry_behavior_all_services(self, mock_sleep, mock_openai_client,
                                         service_class, method_name, args, mock_response):
        """Test retry behavior works correctly across all service types."""
        config = Config(api_key="retry-test-key", max_retries=3)
        client = Client(config)
        service = service_class(client)

        # First two calls fail, third succeeds
        mock_success = MagicMock()
        mock_success.choices[0].message.content = mock_response

        mock_openai_client.return_value.chat.completions.create.side_effect = [
            Exception("Network error"),
            Exception("Rate limit"),
            mock_success
        ]

        method = getattr(service, method_name)
        result = method(*args)

        assert result is not None
        assert mock_openai_client.return_value.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2

        # Verify exponential backoff
        expected_calls = [call(1), call(2)]
        mock_sleep.assert_has_calls(expected_calls)


class TestResponseHandlingIntegration:
    """Test response handling across all service types."""

    def test_markdown_code_block_handling_all_services(self, mock_openai_client):
        """Test markdown code block handling across all service types."""
        config = Config(api_key="markdown-test-key")
        client = Client(config)

        test_cases = [
            (SortingService, 'sort', [[3, 1, 2]], '```json\n{"response": [1, 2, 3]}\n```', [1, 2, 3]),
            (BasicService, 'min', [[1, 5, 3]], '```json\n{"response": 1}\n```', 1),
            (StringService, 'lower', ["HELLO"], '```json\n{"response": "hello"}\n```', "hello"),
            (ListService, 'index', [[1, 2, 3], 2], '```json\n{"response": 1}\n```', 1)
        ]

        for service_class, method_name, args, mock_response, expected in test_cases:
            service = service_class(client)
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = mock_response

            method = getattr(service, method_name)
            result = method(*args)

            assert result == expected

    def test_json_with_extra_fields_all_services(self, mock_openai_client):
        """Test that extra fields in JSON responses are ignored across all services."""
        config = Config(api_key="extra-fields-test-key")
        client = Client(config)

        test_cases = [
            (SortingService, 'sort', [[3, 1, 2]],
             '{"response": [1, 2, 3], "confidence": 0.95, "model": "gpt-4"}', [1, 2, 3]),
            (BasicService, 'abs', [-5],
             '{"response": 5, "reasoning": "absolute value", "timestamp": "2024-01-01"}', 5),
            (StringService, 'strip', ["  hello  "],
             '{"response": "hello", "original_length": 9, "new_length": 5}', "hello"),
            (ListService, 'count', [[1, 2, 1], 1],
             '{"response": 2, "total_items": 3, "search_value": 1}', 2)
        ]

        for service_class, method_name, args, mock_response, expected in test_cases:
            service = service_class(client)
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = mock_response

            method = getattr(service, method_name)
            result = method(*args)

            assert result == expected


class TestConfigurationVariations:
    """Test various configuration scenarios across all services."""

    def test_high_temperature_creative_responses(self, mock_openai_client):
        """Test high temperature configuration across all services."""
        config = Config(api_key="creative-test-key", temperature=0.9, model="gpt-4")
        client = Client(config)

        test_cases = [
            (SortingService, 'sort', [["large", "small", "medium"]],
             '{"response": ["small", "medium", "large"]}'),
            (StringService, 'upper', ["hello world"],
             '{"response": "HELLO WORLD"}'),
        ]

        for service_class, method_name, args, mock_response in test_cases:
            service = service_class(client)
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = mock_response

            method = getattr(service, method_name)
            result = method(*args)

            # Verify high temperature was used
            call_kwargs = mock_openai_client.return_value.chat.completions.create.call_args.kwargs
            assert call_kwargs['temperature'] == 0.9
            assert call_kwargs['model'] == 'gpt-4'

    def test_low_temperature_consistent_responses(self, mock_openai_client):
        """Test low temperature configuration across all services."""
        config = Config(api_key="consistent-test-key", temperature=0.0)
        client = Client(config)

        service = BasicService(client)
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 5}'

        result = service.max([1, 5, 3])

        call_kwargs = mock_openai_client.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs['temperature'] == 0.0
        assert result == 5

    def test_custom_timeout_across_services(self, mock_openai_client):
        """Test custom timeout configuration across all services."""
        config = Config(api_key="timeout-test-key", timeout=120.0)
        client = Client(config)

        services_and_calls = [
            (SortingService, 'sort', [[1, 2, 3]], '{"response": [1, 2, 3]}'),
            (BasicService, 'sum', [[1, 2, 3]], '{"response": 6}'),
            (StringService, 'join', [["a", "b"], ","], '{"response": "a,b"}'),
            (ListService, 'reverse', [[1, 2]], '{"response": [2, 1]}')
        ]

        for service_class, method_name, args, mock_response in services_and_calls:
            service = service_class(client)
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = mock_response

            method = getattr(service, method_name)
            result = method(*args)

            call_kwargs = mock_openai_client.return_value.chat.completions.create.call_args.kwargs
            assert call_kwargs['timeout'] == 120.0


class TestServiceCachingAndManagement:
    """Test service caching and global state management."""

    def test_service_isolation_across_types(self, mock_openai_client):
        """Test that different service types are properly isolated."""
        config = Config(api_key="isolation-test-key")
        client = Client(config)

        # Create different service types
        sorting_service1 = SortingService(client)
        sorting_service2 = SortingService(client)
        basic_service = BasicService(client)
        string_service = StringService(client)

        # All should be different instances
        assert sorting_service1 is not sorting_service2
        assert sorting_service1 is not basic_service
        assert basic_service is not string_service

        # But they should share the same client
        assert sorting_service1._client is client
        assert basic_service._client is client
        assert string_service._client is client

    def test_service_memory_efficiency(self, mock_openai_client):
        """Test that services don't leak memory or resources."""
        config = Config(api_key="memory-test-key")

        # Create and use many service instances
        services = []
        for i in range(10):
            client = Client(config)
            service = SortingService(client)
            services.append(service)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3]}'

        # Use all services
        results = []
        for service in services:
            result = service.sort([3, 1, 2])
            results.append(result)

        # All should work
        assert all(result == [1, 2, 3] for result in results)
        assert mock_openai_client.return_value.chat.completions.create.call_count == 10


class TestRealWorldScenarios:
    """Test realistic usage scenarios with multiple services."""

    def test_multi_service_workflow(self, mock_openai_client):
        """Test workflow using multiple different service types."""
        config = Config(api_key="workflow-test-key")
        client = Client(config)

        # Create all service types
        sorting_service = SortingService(client)
        basic_service = BasicService(client)
        string_service = StringService(client)
        list_service = ListService(client)

        # Simulate a workflow: sort data, find max, process strings, reverse list
        mock_responses = [
            '{"response": [1, 2, 3, 4, 5]}',  # sorted data
            '{"response": 5}',                # max value
            '{"response": "RESULT: 5"}',      # processed string
            '{"response": [5, 4, 3, 2, 1]}'  # reversed list
        ]

        for i, mock_response in enumerate(mock_responses):
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = mock_response

            if i == 0:
                sorted_data = sorting_service.sort([3, 1, 4, 5, 2])
                assert sorted_data == [1, 2, 3, 4, 5]
            elif i == 1:
                max_value = basic_service.max(sorted_data)
                assert max_value == 5
            elif i == 2:
                result_string = string_service.upper(f"result: {max_value}")
                assert result_string == "RESULT: 5"
            elif i == 3:
                reversed_list = list_service.reverse(sorted_data)
                assert reversed_list == [5, 4, 3, 2, 1]

        # Verify all calls were made
        assert mock_openai_client.return_value.chat.completions.create.call_count == 4

    def test_error_recovery_across_services(self, mock_openai_client):
        """Test error recovery in multi-service workflows."""
        config = Config(api_key="recovery-test-key", max_retries=2)
        client = Client(config)

        sorting_service = SortingService(client)
        basic_service = BasicService(client)

        # First service call fails then succeeds
        mock_success1 = MagicMock()
        mock_success1.choices[0].message.content = '{"response": [1, 2, 3]}'
        mock_success2 = MagicMock()
        mock_success2.choices[0].message.content = '{"response": 3}'

        mock_openai_client.return_value.chat.completions.create.side_effect = [
            Exception("Network error"),
            mock_success1,
            mock_success2
        ]

        with patch('time.sleep'):  # Speed up test
            # First operation: retry then succeed
            result1 = sorting_service.sort([3, 1, 2])
            assert result1 == [1, 2, 3]

            # Second operation: succeed immediately
            result2 = basic_service.max(result1)
            assert result2 == 3

        assert mock_openai_client.return_value.chat.completions.create.call_count == 3

    def test_performance_monitoring_across_services(self, mock_openai_client, performance_data):
        """Test performance characteristics across different service types."""
        config = Config(api_key="performance-test-key", timeout=60.0)
        client = Client(config)

        # Test with different data sizes
        test_cases = [
            (SortingService, 'sort', performance_data['medium'], sorted(performance_data['medium'])),
            (ListService, 'reverse', performance_data['small'], list(reversed(performance_data['small'])))
        ]

        for service_class, method_name, input_data, expected_output in test_cases:
            service = service_class(client)
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": expected_output})

            start_time = time.time()
            method = getattr(service, method_name)
            result = method(input_data)
            end_time = time.time()

            assert result == expected_output
            assert end_time - start_time < 1.0  # Should be fast in tests
