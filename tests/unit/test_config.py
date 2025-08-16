import os
import pytest
from unittest.mock import patch
from vibelib.config import Config

class TestConfig:

    def test_config_with_direct_api_key(self):
        config = Config(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.resolved_api_key == "test-key"

    def test_config_defaults(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            config = Config()
            assert config.api_key is None
            assert config.resolved_api_key == "env-key"
            assert config.model == "gpt-4o-mini"
            assert config.timeout == 30.0
            assert config.max_retries == 3
            assert config.temperature == 0.0

    def test_config_immutable(self):
        config = Config(api_key="test-key")
        with pytest.raises(AttributeError):
            config.api_key = "new-key"
        with pytest.raises(AttributeError):
            config.model = "new-model"

    def test_config_no_api_key_no_env_raises_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                Config()

    def test_config_empty_env_var_raises_error(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            with pytest.raises(ValueError, match="API key required"):
                Config()

    def test_config_none_api_key_with_env(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            config = Config(api_key=None)
            assert config.resolved_api_key == "env-key"

    def test_resolved_api_key_prefers_direct_over_env(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            config = Config(api_key="direct-key")
            assert config.resolved_api_key == "direct-key"

    def test_resolved_api_key_falls_back_to_env(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            config = Config()
            assert config.resolved_api_key == "env-key"

    @pytest.mark.parametrize("model,expected", [
        ("gpt-4", "gpt-4"),
        ("gpt-3.5-turbo", "gpt-3.5-turbo"),
        ("gpt-4o", "gpt-4o"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("custom-model", "custom-model"),
    ])
    def test_custom_model(self, model, expected):
        config = Config(api_key="test", model=model)
        assert config.model == expected

    @pytest.mark.parametrize("timeout", [1.0, 30.0, 60.0, 120.0])
    def test_custom_timeout(self, timeout):
        config = Config(api_key="test", timeout=timeout)
        assert config.timeout == timeout

    @pytest.mark.parametrize("retries", [1, 3, 5, 10])
    def test_custom_max_retries(self, retries):
        config = Config(api_key="test", max_retries=retries)
        assert config.max_retries == retries

    @pytest.mark.parametrize("temp", [0.0, 0.1, 0.5, 1.0])
    def test_custom_temperature(self, temp):
        config = Config(api_key="test", temperature=temp)
        assert config.temperature == temp

    def test_config_repr_hides_api_key(self):
        config = Config(api_key="secret-key")
        config_str = repr(config)
        assert "secret-key" not in config_str
        assert "api_key" in config_str

    def test_config_with_whitespace_api_key(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': '  \n  '}):
            with pytest.raises(ValueError, match="API key required"):
                Config()
