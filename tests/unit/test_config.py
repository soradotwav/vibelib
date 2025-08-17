"""
Comprehensive tests for VibeLib configuration management.

Tests all configuration scenarios including validation,
environment variable handling, and edge cases.
"""
from dataclasses import FrozenInstanceError
import os
import pytest
from unittest.mock import patch
from vibelib.config import Config


class TestConfig:
    """Test suite for configuration functionality."""

    def test_config_with_direct_api_key(self):
        """Test configuration with directly provided API key."""
        config = Config(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.resolved_api_key == "test-key"

    def test_config_defaults(self, env_with_api_key):
        """Test that configuration uses proper default values."""
        config = Config()

        assert config.api_key is None
        assert config.resolved_api_key == "env-test-key"
        assert config.model == "gpt-4o-mini"
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.temperature == 0.3

    def test_config_immutability(self):
        """Test that configuration objects are immutable."""
        config = Config(api_key="test-key")

        # Test that any field assignment raises FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            config.api_key = "new-key"
        with pytest.raises(FrozenInstanceError):
            config.model = "new-model"
        with pytest.raises(FrozenInstanceError):
            config.temperature = 0.5

    def test_config_no_api_key_raises_error(self, clean_environment):
        """Test that missing API key raises appropriate error."""
        with pytest.raises(ValueError, match="API key required"):
            Config()

    def test_config_empty_env_var_raises_error(self):
        """Test that empty environment variable raises error."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            with pytest.raises(ValueError, match="API key required"):
                Config()

    def test_config_whitespace_env_var_raises_error(self):
        """Test that whitespace-only environment variable raises error."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': '   \n\t  '}):
            with pytest.raises(ValueError, match="API key required"):
                Config()

    def test_resolved_api_key_prefers_direct_over_env(self):
        """Test that direct API key takes precedence over environment variable."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            config = Config(api_key="direct-key")
            assert config.resolved_api_key == "direct-key"

    def test_resolved_api_key_falls_back_to_env(self, env_with_api_key):
        """Test that configuration falls back to environment variable."""
        config = Config()
        assert config.resolved_api_key == "env-test-key"

    def test_none_api_key_with_env(self, env_with_api_key):
        """Test explicit None API key with environment fallback."""
        config = Config(api_key=None)
        assert config.resolved_api_key == "env-test-key"

    @pytest.mark.parametrize("model,expected", [
        ("gpt-4", "gpt-4"),
        ("gpt-3.5-turbo", "gpt-3.5-turbo"),
        ("gpt-4o", "gpt-4o"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("claude-3-sonnet", "claude-3-sonnet"),
        ("custom-model-v1", "custom-model-v1"),
    ])
    def test_custom_model_configuration(self, model, expected):
        """Test various model configurations."""
        config = Config(api_key="test", model=model)
        assert config.model == expected

    @pytest.mark.parametrize("timeout", [0.1, 1.0, 30.0, 60.0, 120.0, 300.0])
    def test_valid_timeout_values(self, timeout):
        """Test valid timeout configurations."""
        config = Config(api_key="test", timeout=timeout)
        assert config.timeout == timeout

    @pytest.mark.parametrize("retries", [1, 3, 5, 10, 50])
    def test_valid_retry_values(self, retries):
        """Test valid retry count configurations."""
        config = Config(api_key="test", max_retries=retries)
        assert config.max_retries == retries

    @pytest.mark.parametrize("temperature", [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    def test_valid_temperature_values(self, temperature):
        """Test valid temperature configurations."""
        config = Config(api_key="test", temperature=temperature)
        assert config.temperature == temperature

    @pytest.mark.parametrize("invalid_temp", [-0.1, -1.0, 1.1, 2.0])
    def test_invalid_temperature_raises_error(self, invalid_temp):
        """Test that invalid temperatures raise validation errors."""
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 1.0"):
            Config(api_key="test", temperature=invalid_temp)

    @pytest.mark.parametrize("invalid_retries", [0, -1, -10])
    def test_invalid_retries_raises_error(self, invalid_retries):
        """Test that invalid retry counts raise validation errors."""
        with pytest.raises(ValueError, match="max_retries must be at least 1"):
            Config(api_key="test", max_retries=invalid_retries)

    @pytest.mark.parametrize("invalid_timeout", [0, -1, -30])
    def test_invalid_timeout_raises_error(self, invalid_timeout):
        """Test that invalid timeouts raise validation errors."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            Config(api_key="test", timeout=invalid_timeout)

    def test_config_repr_security(self):
        """Test that repr output doesn't expose sensitive information."""
        config = Config(api_key="super-secret-api-key")
        config_str = repr(config)

        assert "super-secret-api-key" not in config_str
        assert "***" in config_str
        assert "api_key=***" in config_str

    def test_config_repr_none_api_key(self, env_with_api_key):
        """Test repr with None API key but environment variable set."""
        config = Config()
        config_str = repr(config)

        assert "env-test-key" not in config_str
        assert "api_key=***" in config_str

    def test_config_repr_format(self):
        """Test that repr output contains all expected fields."""
        config = Config(
            api_key="test",
            model="gpt-4",
            timeout=60.0,
            max_retries=5,
            temperature=0.7
        )
        config_str = repr(config)

        assert "Config(" in config_str
        assert "api_key=***" in config_str
        assert "model='gpt-4'" in config_str
        assert "timeout=60.0" in config_str
        assert "max_retries=5" in config_str
        assert "temperature=0.7" in config_str

    def test_config_equality_and_hashing(self):
        """Test configuration equality and hashability."""
        config1 = Config(api_key="test", model="gpt-4", temperature=0.3)
        config2 = Config(api_key="test", model="gpt-4", temperature=0.3)
        config3 = Config(api_key="test", model="gpt-4", temperature=0.5)

        assert config1 == config2
        assert config1 != config3
        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)

    def test_config_with_extreme_values(self):
        """Test configuration with boundary values."""
        config = Config(
            api_key="test",
            timeout=0.001,  # Very small timeout
            max_retries=1,  # Minimum retries
            temperature=0.0  # Minimum temperature
        )

        assert config.timeout == 0.001
        assert config.max_retries == 1
        assert config.temperature == 0.0

    def test_config_with_large_values(self):
        """Test configuration with large values."""
        config = Config(
            api_key="test",
            timeout=3600.0,  # 1 hour
            max_retries=100,  # Many retries
            temperature=1.0  # Maximum temperature
        )

        assert config.timeout == 3600.0
        assert config.max_retries == 100
        assert config.temperature == 1.0

    def test_concurrent_config_creation(self):
        """Test that multiple configs can be created concurrently."""
        configs = []
        for i in range(10):
            config = Config(
                api_key=f"test-key-{i}",
                model=f"model-{i}",
                temperature=i / 10.0
            )
            configs.append(config)

        # Verify all configs are properly isolated
        for i, config in enumerate(configs):
            assert config.api_key == f"test-key-{i}"
            assert config.model == f"model-{i}"
            assert config.temperature == i / 10.0

    def test_config_post_init_validation_order(self):
        """Test that validation happens in the correct order."""
        # Test that API key validation happens first
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                Config(temperature=2.0)  # Invalid temp, but API key error should come first
