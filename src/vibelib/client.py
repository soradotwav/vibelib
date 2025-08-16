import time
from typing import Optional
import logging

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .config import Config
from .exceptions import APIError, ConfigurationError

logger = logging.getLogger(__name__)

class Client:
    def __init__(self, config: Config) -> None:
        try:
            self._client = OpenAI(api_key=config.resolved_api_key)
        except Exception as e:
            raise ConfigurationError(f"Client initialization failed: {e}")

        self._config = config

    def request(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                response: ChatCompletion = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    temperature=self._config.temperature,
                    timeout=self._config.timeout
                )

                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise APIError("Empty response")

                return content

            except Exception as e:
                last_error = e
                if attempt < self._config.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise APIError(f"Request failed: {last_error}")
