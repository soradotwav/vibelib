"""
End-to-end tests for VibeLib.

Tests complete user workflows from initialization through
complex multi-operation scenarios, simulating real production usage.
"""

import json
import pytest
import time
from unittest.mock import patch

import vibelib
from vibelib.exceptions import APIError, ValidationError


class TestCompleteWorkflows:
    """Test complete user workflows from start to finish."""

    def test_basic_data_processing_workflow(self, mock_openai_client, env_with_api_key):
        """Test a complete data processing workflow using multiple operations."""
        # Simulate user workflow: sort data, find statistics, format results

        # Step 1: Sort unsorted data
        unsorted_data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 1, 2, 3, 3, 4, 5, 5, 6, 9]}'
        sorted_data = vibelib.sort(unsorted_data)

        # Step 2: Find max and min values
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 9}'
        max_value = vibelib.max(sorted_data)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 1}'
        min_value = vibelib.min(sorted_data)

        # Step 3: Calculate sum
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 39}'
        total = vibelib.sum(sorted_data)

        # Step 4: Format results as strings
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": "MAX: 9, MIN: 1, TOTAL: 39"}'
        summary = vibelib.upper(f"max: {max_value}, min: {min_value}, total: {total}")

        # Step 5: Split summary for further processing
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": ["MAX: 9", " MIN: 1", " TOTAL: 39"]}'
        summary_parts = vibelib.split(summary, ",")

        # Verify complete workflow
        assert sorted_data == [1, 1, 2, 3, 3, 4, 5, 5, 6, 9]
        assert max_value == 9
        assert min_value == 1
        assert total == 39
        assert summary == "MAX: 9, MIN: 1, TOTAL: 39"
        assert len(summary_parts) == 3

        # Verify all operations were called
        assert mock_openai_client.return_value.chat.completions.create.call_count == 6

    def test_text_processing_workflow(self, mock_openai_client, env_with_api_key):
        """Test a complete text processing workflow."""
        # Simulate processing user text data
        raw_text_data = ["  Hello World  ", "  PYTHON programming  ", "  Data Science  "]

        # Step 1: Clean up text (strip whitespace)
        cleaned_texts = []
        for text in raw_text_data:
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": text.strip()})
            cleaned = vibelib.strip(text)
            cleaned_texts.append(cleaned)

        # Step 2: Normalize to lowercase
        normalized_texts = []
        for text in cleaned_texts:
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": text.lower()})
            normalized = vibelib.lower(text)
            normalized_texts.append(normalized)

        # Step 3: Sort alphabetically
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": ["data science", "hello world", "python programming"]}'
        sorted_texts = vibelib.sort(normalized_texts)

        # Step 4: Join into single string
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": "data science | hello world | python programming"}'
        final_result = vibelib.join(sorted_texts, " | ")

        # Step 5: Count occurrences of a word
        words = final_result.split()  # Use Python split for test simplicity
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 1}'
        python_count = vibelib.count(words, "python")

        # Verify workflow results
        assert cleaned_texts == ["Hello World", "PYTHON programming", "Data Science"]
        assert normalized_texts == ["hello world", "python programming", "data science"]
        assert sorted_texts == ["data science", "hello world", "python programming"]
        assert final_result == "data science | hello world | python programming"
        assert python_count == 1

    def test_list_manipulation_workflow(self, mock_openai_client, env_with_api_key):
        """Test complex list manipulation workflow."""
        # Simulate working with multiple datasets
        dataset1 = [1, 2, 3, 4, 5]
        dataset2 = [3, 4, 5, 6, 7]

        # Step 1: Reverse both datasets
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [5, 4, 3, 2, 1]}'
        reversed1 = vibelib.reverse(dataset1)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [7, 6, 5, 4, 3]}'
        reversed2 = vibelib.reverse(dataset2)

        # Step 2: Combine and sort
        combined = reversed1 + reversed2  # Python concatenation
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3, 3, 4, 4, 5, 5, 6, 7]}'
        sorted_combined = vibelib.sort(combined)

        # Step 3: Find unique values by counting
        unique_values = []
        for value in set(sorted_combined):  # Use Python set for test efficiency
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {sorted_combined.count(value)}}}'
            count = vibelib.count(sorted_combined, value)
            if count == 1:
                unique_values.append(value)

        # Step 4: Get statistics on unique values
        if unique_values:
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {max(unique_values)}}}'
            max_unique = vibelib.max(unique_values)
        else:
            max_unique = None

        # Verify workflow
        assert reversed1 == [5, 4, 3, 2, 1]
        assert reversed2 == [7, 6, 5, 4, 3]
        assert sorted_combined == [1, 2, 3, 3, 4, 4, 5, 5, 6, 7]
        assert unique_values == [1, 2, 6, 7]  # Values that appear only once
        assert max_unique == 7

    def test_mixed_data_type_workflow(self, mock_openai_client, env_with_api_key):
        """Test workflow handling mixed data types."""
        # Simulate processing mixed data
        mixed_data = [3, "hello", 1.5, "world", 2]

        # Step 1: Sort mixed data (AI should handle creatively)
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1.5, 2, 3, "hello", "world"]}'
        sorted_mixed = vibelib.sort(mixed_data)

        # Step 2: Extract strings and process them
        strings = [item for item in sorted_mixed if isinstance(item, str)]
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": "hello world"}'
        joined_strings = vibelib.join(strings, " ")

        # Step 3: Process the joined string
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": "HELLO WORLD"}'
        upper_strings = vibelib.upper(joined_strings)

        # Step 4: Extract numbers and calculate statistics
        numbers = [item for item in sorted_mixed if isinstance(item, (int, float))]
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 6.5}'
        sum_numbers = vibelib.sum(numbers)

        # Verify mixed type handling
        assert sorted_mixed == [1.5, 2, 3, "hello", "world"]
        assert strings == ["hello", "world"]
        assert joined_strings == "hello world"
        assert upper_strings == "HELLO WORLD"
        assert numbers == [1.5, 2, 3]
        assert sum_numbers == 6.5


class TestErrorRecoveryWorkflows:
    """Test error handling and recovery in complete workflows."""

    @patch('time.sleep')
    def test_workflow_with_transient_errors(self, mock_sleep, mock_openai_client, env_with_api_key):
        """Test workflow that encounters and recovers from transient errors."""
        # Step 1: First operation fails, then succeeds on retry
        mock_success1 = mock_openai_client.return_value.chat.completions.create.return_value
        mock_success1.choices[0].message.content = '{"response": [1, 2, 3, 4, 5]}'

        mock_openai_client.return_value.chat.completions.create.side_effect = [
            Exception("Network timeout"),
            mock_success1
        ]

        sorted_data = vibelib.sort([3, 1, 4, 5, 2])

        # Step 2: Second operation succeeds immediately
        mock_openai_client.return_value.chat.completions.create.side_effect = None  # Reset
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 5}'

        max_value = vibelib.max(sorted_data)

        # Verify recovery worked
        assert sorted_data == [1, 2, 3, 4, 5]
        assert max_value == 5
        assert mock_sleep.call_count >= 1  # At least one retry occurred

    def test_workflow_with_validation_errors(self, mock_openai_client, env_with_api_key):
        """Test workflow handling validation errors gracefully."""
        # Attempt to sort oversized array
        oversized_data = list(range(10001))

        with pytest.raises(ValidationError, match="Input too large"):
            vibelib.sort(oversized_data)

        # Continue workflow with smaller, valid data
        valid_data = [3, 1, 4, 5, 2]
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": [1, 2, 3, 4, 5]}'

        sorted_data = vibelib.sort(valid_data)
        assert sorted_data == [1, 2, 3, 4, 5]

    def test_workflow_with_partial_failures(self, mock_openai_client, env_with_api_key):
        """Test workflow where some operations fail but others succeed."""
        data = [1, 2, 3, 4, 5]

        # First operation succeeds
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 5}'
        max_value = vibelib.max(data)
        assert max_value == 5

        # Second operation fails completely
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("Persistent API error")

        with pytest.raises(APIError, match="Request failed"):
            vibelib.min(data)

        # Reset and continue with different operation
        mock_openai_client.return_value.chat.completions.create.side_effect = None
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 15}'

        sum_value = vibelib.sum(data)
        assert sum_value == 15


class TestPerformanceWorkflows:
    """Test performance characteristics in realistic workflows."""

    def test_high_volume_operations_workflow(self, mock_openai_client, env_with_api_key):
        """Test workflow with high volume of operations."""
        # Simulate processing many small datasets
        datasets = [[i, i+1, i+2] for i in range(20)]  # 20 small datasets

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 6}'

        # Process each dataset
        sums = []
        start_time = time.time()

        for dataset in datasets:
            dataset_sum = vibelib.sum(dataset)
            sums.append(dataset_sum)

        end_time = time.time()

        # Verify results and performance
        assert len(sums) == 20
        assert all(s == 6 for s in sums)  # All sums should be the mocked value
        assert mock_openai_client.return_value.chat.completions.create.call_count == 20
        assert end_time - start_time < 5.0  # Should complete reasonably quickly

    def test_large_data_processing_workflow(self, mock_openai_client, env_with_api_key, performance_data):
        """Test workflow with large datasets."""
        large_dataset = performance_data['large']  # 1000 items

        # Sort large dataset
        expected_sorted = sorted(large_dataset)
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": expected_sorted})

        start_time = time.time()
        sorted_large = vibelib.sort(large_dataset)
        end_time = time.time()

        # Find statistics
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {max(expected_sorted)}}}'
        max_value = vibelib.max(sorted_large)

        # Verify performance and results
        assert sorted_large == expected_sorted
        assert max_value == max(expected_sorted)
        assert end_time - start_time < 2.0  # Should handle large data efficiently

    def test_concurrent_style_operations_workflow(self, mock_openai_client, env_with_api_key):
        """Test workflow simulating concurrent-style operations."""
        # Simulate processing multiple independent tasks
        tasks = [
            ([3, 1, 2], 'sort', '{"response": [1, 2, 3]}'),
            ([1, 5, 3], 'max', '{"response": 5}'),
            ("hello", 'upper', '{"response": "HELLO"}'),
            ([1, 2, 3], 'reverse', '{"response": [3, 2, 1]}')
        ]

        results = []

        for data, operation, mock_response in tasks:
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = mock_response

            if operation == 'sort':
                result = vibelib.sort(data)
            elif operation == 'max':
                result = vibelib.max(data)
            elif operation == 'upper':
                result = vibelib.upper(data)
            elif operation == 'reverse':
                result = vibelib.reverse(data)

            results.append(result)

        # Verify all operations completed
        expected_results = [[1, 2, 3], 5, "HELLO", [3, 2, 1]]
        assert results == expected_results
        assert mock_openai_client.return_value.chat.completions.create.call_count == 4


class TestProductionScenarios:
    """Test scenarios that mirror production usage patterns."""

    def test_data_analysis_pipeline(self, mock_openai_client, env_with_api_key):
        """Test a realistic data analysis pipeline."""
        # Simulate raw data from multiple sources
        source1_data = [23, 45, 12, 67, 34, 89, 56, 78]
        source2_data = [34, 56, 23, 78, 45, 67, 89, 12]

        # Step 1: Combine and sort all data
        combined_data = source1_data + source2_data
        expected_sorted = sorted(combined_data)
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": expected_sorted})
        sorted_data = vibelib.sort(combined_data)

        # Step 2: Calculate key statistics
        actual_max = max(expected_sorted)
        actual_min = min(expected_sorted)
        actual_sum = sum(expected_sorted)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {actual_max}}}'
        max_val = vibelib.max(sorted_data)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {actual_min}}}'
        min_val = vibelib.min(sorted_data)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {actual_sum}}}'
        total = vibelib.sum(sorted_data)

        # Step 3: Generate report
        stats = [f"Max: {max_val}", f"Min: {min_val}", f"Total: {total}"]
        expected_report = f"Max: {actual_max} | Min: {actual_min} | Total: {actual_sum}"
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": "{expected_report}"}}'
        report = vibelib.join(stats, " | ")

        # Verify analysis pipeline
        assert len(sorted_data) == 16
        assert max_val == 89
        assert min_val == 12
        assert total == 808  # Correct sum: 404 + 404 = 808
        assert f"Max: {actual_max}" in report and f"Min: {actual_min}" in report

    def test_content_processing_pipeline(self, mock_openai_client, env_with_api_key):
        """Test a content processing pipeline like a CMS might use."""
        # Simulate user-generated content
        raw_content = [
            "  Hello World!  ",
            "  PYTHON IS awesome  ",
            "  data science rocks  ",
            "  machine LEARNING  "
        ]

        # Step 1: Clean and normalize content
        cleaned_content = []
        for content in raw_content:
            # Strip whitespace
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": content.strip()})
            stripped = vibelib.strip(content)

            # Normalize case
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": stripped.lower()})
            normalized = vibelib.lower(stripped)

            cleaned_content.append(normalized)

        # Step 2: Sort content alphabetically
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": sorted(cleaned_content)})
        sorted_content = vibelib.sort(cleaned_content)

        # Step 3: Create searchable index (count words)
        all_words = " ".join(sorted_content).split()  # Simplified for test
        unique_words = list(set(all_words))

        word_counts = {}
        for word in unique_words[:3]:  # Test first 3 words only for performance
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {all_words.count(word)}}}'
            count = vibelib.count(all_words, word)
            word_counts[word] = count

        # Verify content processing
        assert len(cleaned_content) == 4
        assert all(isinstance(content, str) for content in cleaned_content)
        assert len(sorted_content) == 4
        assert isinstance(word_counts, dict)

    def test_monitoring_and_alerting_scenario(self, mock_openai_client, env_with_api_key):
        """Test a monitoring/alerting scenario with thresholds."""
        # Simulate system metrics
        cpu_usage = [45, 67, 89, 23, 78, 56, 90, 34]
        memory_usage = [34, 56, 78, 45, 89, 67, 23, 90]

        # Find peak usage
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 90}'
        peak_cpu = vibelib.max(cpu_usage)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": 90}'
        peak_memory = vibelib.max(memory_usage)

        # Check for threshold violations (> 85)
        high_cpu_readings = [reading for reading in cpu_usage if reading > 85]
        high_memory_readings = [reading for reading in memory_usage if reading > 85]

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {len(high_cpu_readings)}}}'
        cpu_violations = vibelib.count([1 if x > 85 else 0 for x in cpu_usage], 1)

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {len(high_memory_readings)}}}'
        memory_violations = vibelib.count([1 if x > 85 else 0 for x in memory_usage], 1)

        # Generate alerts
        alerts = []
        if peak_cpu > 85:
            alerts.append(f"HIGH CPU: {peak_cpu}%")
        if peak_memory > 85:
            alerts.append(f"HIGH MEMORY: {peak_memory}%")

        if alerts:
            mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": " | ".join(alerts).upper()})
            alert_message = vibelib.upper(vibelib.join(alerts, " | "))
        else:
            alert_message = "NO ALERTS"

        # Verify monitoring workflow
        assert peak_cpu == 90
        assert peak_memory == 90
        assert cpu_violations == 2  # 89, 90
        assert memory_violations == 2  # 89, 90
        assert "HIGH CPU" in alert_message
        assert "HIGH MEMORY" in alert_message

    def test_batch_processing_workflow(self, mock_openai_client, env_with_api_key):
        """Test batch processing scenario."""
        # Simulate batch job processing multiple files
        file_sizes = [1024, 2048, 512, 4096, 256, 8192, 1536]

        # Sort files by size for processing order
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({"response": sorted(file_sizes)})
        sorted_sizes = vibelib.sort(file_sizes)

        # Calculate total processing requirements
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {sum(sorted_sizes)}}}'
        total_size = vibelib.sum(sorted_sizes)

        # Find largest file for special handling
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = f'{{"response": {max(sorted_sizes)}}}'
        largest_file = vibelib.max(sorted_sizes)

        # Generate processing summary
        summary_parts = [
            f"Files: {len(sorted_sizes)}",
            f"Total: {total_size}KB",
            f"Largest: {largest_file}KB"
        ]

        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = '{"response": "FILES: 7 | TOTAL: 17664KB | LARGEST: 8192KB"}'
        processing_summary = vibelib.upper(vibelib.join(summary_parts, " | "))

        # Verify batch processing
        assert sorted_sizes == [256, 512, 1024, 1536, 2048, 4096, 8192]
        assert total_size == 17664
        assert largest_file == 8192
        assert "FILES: 7" in processing_summary
        assert "TOTAL: 17664KB" in processing_summary
