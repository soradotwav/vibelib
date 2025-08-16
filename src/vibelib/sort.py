import json
from typing import List, TypeVar, Union, Any

from .client import Client
from .exceptions import ParseError, ValidationError

T = TypeVar('T', int, float, str)
Sortable = Union[int, float, str]

class SortingService:
    def __init__(self, client: Client) -> None:
        self._client = client

    def sort(self, items: List[Sortable]) -> List[Sortable]:
        """Sort array using AI processing."""
        validated_items = self._validate_input(items)

        if not validated_items:
            return []

        if len(validated_items) == 1:
            return validated_items

        system_prompt = (
            "You are a sorting service. Always respond with valid JSON in this exact format: "
            '{"response": [sorted_array]}. '
            "The response field must contain the sorted array with preserved data types."
        )

        user_prompt = (
            f"Sort this array: {validated_items}\n\n"
            f"Return JSON in format: "
            f'{{"response": [sorted_array]}}'
        )

        response = self._client.request(user_prompt, system_prompt)
        return self._parse_response(response)

    def _validate_input(self, items: List[Any]) -> List[Any]:
        if not isinstance(items, (list, tuple)):
            raise ValidationError(f"Expected list or tuple, got {type(items)}")

        if len(items) == 0:
            return []

        if len(items) > 10000:
            raise ValidationError("Array too large")

        items_list = list(items)

        valid_types = (int, float, str)
        for i, item in enumerate(items_list):
            if not isinstance(item, valid_types):
                raise ValidationError(
                    f"Invalid type at index {i}: {type(item).__name__}. "
                    f"Only {', '.join(t.__name__ for t in valid_types)} are supported."
                )

        return items_list

    def _parse_response(self, response: str) -> List[Sortable]:
        try:
            parsed = json.loads(response.strip())

            # Validate structure
            if not isinstance(parsed, dict):
                raise ValueError("Response must be a JSON object")

            if "response" not in parsed:
                raise ValueError("Response must contain 'response' field")

            result = parsed["response"]

            if not isinstance(result, list):
                raise ValueError("Response field must be a list")

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise ParseError(f"Invalid JSON response: {e}. Raw response: {response}")
