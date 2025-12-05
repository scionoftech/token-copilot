"""Pytest configuration and shared fixtures."""
import pytest
from unittest.mock import MagicMock
from token_copilot.core.copilot import TokenCoPilot
from token_copilot.utils import ModelConfig


@pytest.fixture
def mock_copilot():
    """Create a mock TokenCoPilot instance."""
    copilot = MagicMock(spec=TokenCoPilot)
    copilot.budget_limit = 100.00
    copilot.budget_period = "total"
    copilot.on_budget_exceeded = "raise"
    copilot.tracker = MagicMock()
    return copilot


@pytest.fixture
def sample_models():
    """Create sample model configurations for testing."""
    return [
        ModelConfig("gpt-4o-mini", 0.7, 0.15, 0.60, 128000),
        ModelConfig("gpt-4o", 0.9, 5.0, 15.0, 128000),
        ModelConfig("gpt-3.5-turbo", 0.6, 0.50, 1.50, 16000),
    ]


@pytest.fixture
def copilot_basic():
    """Create a basic TokenCoPilot instance."""
    return TokenCoPilot(budget_limit=10.00)


@pytest.fixture
def copilot_with_budget():
    """Create a TokenCoPilot with custom budget settings."""
    return TokenCoPilot(
        budget_limit=100.00,
        budget_period="daily",
        on_budget_exceeded="warn"
    )
