"""Tests for adaptive token operations."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from token_copilot import TokenCoPilot as TokenCoPilotCallback
from token_copilot.adaptive import (
    BudgetTier,
    classify_budget_tier,
    get_tier_description,
    TokenAwareOperations,
    token_aware,
    budget_gate,
    track_efficiency,
    adaptive_context,
    budget_aware_section,
    get_current_callback,
    set_current_callback,
)


class TestBudgetTiers:
    """Test budget tier classification."""

    def test_classify_abundant(self):
        """Test ABUNDANT tier (>75% remaining)."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        tier = classify_budget_tier(callback)
        assert tier == BudgetTier.ABUNDANT

    def test_classify_comfortable(self):
        """Test COMFORTABLE tier (50-75% remaining)."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        # Simulate spending $35 (65% remaining)
        callback.tracker.track_cost(
            model="gpt-4o-mini",
            input_tokens=10000,
            output_tokens=10000,
            cost=35.0,
        )
        tier = classify_budget_tier(callback)
        assert tier == BudgetTier.COMFORTABLE

    def test_classify_moderate(self):
        """Test MODERATE tier (25-50% remaining)."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        # Simulate spending $60 (40% remaining)
        callback.tracker.track_cost(
            model="gpt-4o-mini",
            input_tokens=10000,
            output_tokens=10000,
            cost=60.0,
        )
        tier = classify_budget_tier(callback)
        assert tier == BudgetTier.MODERATE

    def test_classify_low(self):
        """Test LOW tier (10-25% remaining)."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        # Simulate spending $80 (20% remaining)
        callback.tracker.track_cost(
            model="gpt-4o-mini",
            input_tokens=10000,
            output_tokens=10000,
            cost=80.0,
        )
        tier = classify_budget_tier(callback)
        assert tier == BudgetTier.LOW

    def test_classify_critical(self):
        """Test CRITICAL tier (<10% remaining)."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        # Simulate spending $95 (5% remaining)
        callback.tracker.track_cost(
            model="gpt-4o-mini",
            input_tokens=10000,
            output_tokens=10000,
            cost=95.0,
        )
        tier = classify_budget_tier(callback)
        assert tier == BudgetTier.CRITICAL

    def test_no_budget_limit_is_abundant(self):
        """Test that no budget limit means ABUNDANT."""
        callback = TokenCoPilotCallback()  # No limit
        tier = classify_budget_tier(callback)
        assert tier == BudgetTier.ABUNDANT

    def test_get_tier_description(self):
        """Test tier description strings."""
        desc = get_tier_description(BudgetTier.ABUNDANT)
        assert "Abundant" in desc
        assert ">75%" in desc


class TestTokenAwareOperations:
    """Test TokenAwareOperations class."""

    def test_initialization(self):
        """Test TokenAwareOperations initialization."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)
        assert ops.callback is callback
        assert ops.enable_logging is True

    def test_get_current_tier(self):
        """Test getting current tier."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)
        tier = ops.get_current_tier()
        assert tier == BudgetTier.ABUNDANT

    def test_get_tier_info(self):
        """Test getting detailed tier info."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)
        info = ops.get_tier_info()

        assert info['tier'] == BudgetTier.ABUNDANT
        assert info['tier_name'] == 'abundant'
        assert 'description' in info
        assert info['total_cost'] == 0.0
        assert info['budget_limit'] == 100.0
        assert info['remaining'] == 100.0
        assert info['remaining_percentage'] == 100.0

    def test_adaptive_params_generate_abundant(self):
        """Test adaptive params for generate operation at ABUNDANT tier."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)
        params = ops._get_adaptive_params(BudgetTier.ABUNDANT, 'generate')

        assert params['max_tokens'] == 2000
        assert params['temperature'] == 0.7

    def test_adaptive_params_generate_critical(self):
        """Test adaptive params for generate operation at CRITICAL tier."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)
        params = ops._get_adaptive_params(BudgetTier.CRITICAL, 'generate')

        assert params['max_tokens'] == 200
        assert params['temperature'] == 0.1

    def test_adaptive_params_search(self):
        """Test adaptive params for search operation."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)

        abundant_params = ops._get_adaptive_params(BudgetTier.ABUNDANT, 'search')
        assert abundant_params['top_k'] == 10

        critical_params = ops._get_adaptive_params(BudgetTier.CRITICAL, 'search')
        assert critical_params['top_k'] == 1

    def test_adaptive_params_retry(self):
        """Test adaptive params for retry operation."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)

        abundant_params = ops._get_adaptive_params(BudgetTier.ABUNDANT, 'retry')
        assert abundant_params['max_retries'] == 5

        critical_params = ops._get_adaptive_params(BudgetTier.CRITICAL, 'retry')
        assert critical_params['max_retries'] == 0

    def test_generate_with_mock_llm(self):
        """Test generate method with mock LLM."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)

        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="response")

        result = ops.generate(mock_llm, "test prompt")

        assert result == "response"
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args
        assert call_args[0][0] == "test prompt"
        assert 'max_tokens' in call_args[1]
        assert callback in call_args[1]['callbacks']

    def test_generate_user_override(self):
        """Test that user params override adaptive defaults."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)

        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="response")

        # User specifies max_tokens
        result = ops.generate(mock_llm, "test", max_tokens=999)

        call_args = mock_llm.invoke.call_args
        assert call_args[1]['max_tokens'] == 999  # User override

    def test_retry_success_first_attempt(self):
        """Test retry with immediate success."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)

        mock_func = Mock(return_value="success")
        result = ops.retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_success_after_failures(self):
        """Test retry with eventual success."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)

        # Fail twice, then succeed
        mock_func = Mock(side_effect=[Exception("fail1"), Exception("fail2"), "success"])
        result = ops.retry(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_all_failures(self):
        """Test retry exhausting all attempts."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)

        mock_func = Mock(side_effect=Exception("always fails"))

        with pytest.raises(Exception, match="always fails"):
            ops.retry(mock_func, max_retries=2)

        assert mock_func.call_count == 3  # Initial + 2 retries


class TestDecorators:
    """Test adaptive decorators."""

    def test_token_aware_decorator_with_context(self):
        """Test @token_aware decorator with context manager."""
        callback = TokenCoPilotCallback(budget_limit=100.0)

        @token_aware(operation='generate')
        def mock_task(prompt, max_tokens=None):
            return f"generated with max_tokens={max_tokens}"

        with adaptive_context(callback):
            result = mock_task("test prompt")

        # Should have received adaptive max_tokens
        assert "max_tokens=" in result
        assert result != "generated with max_tokens=None"

    def test_token_aware_user_override(self):
        """Test that user params override in decorator."""
        callback = TokenCoPilotCallback(budget_limit=100.0)

        @token_aware(operation='generate')
        def mock_task(prompt, max_tokens=None):
            return max_tokens

        with adaptive_context(callback):
            result = mock_task("test", max_tokens=555)

        assert result == 555  # User override

    def test_token_aware_without_context(self):
        """Test @token_aware without context (should work but warn)."""
        @token_aware(operation='generate')
        def mock_task(prompt):
            return "result"

        # Should run without error even without context
        result = mock_task("test")
        assert result == "result"

    def test_budget_gate_allows_execution(self):
        """Test @budget_gate allows execution when budget sufficient."""
        callback = TokenCoPilotCallback(budget_limit=100.0)

        @budget_gate(min_tier=BudgetTier.MODERATE)
        def gated_task():
            return "executed"

        with adaptive_context(callback):
            result = gated_task()

        assert result == "executed"

    def test_budget_gate_blocks_execution(self):
        """Test @budget_gate blocks execution when budget insufficient."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        # Spend to get to CRITICAL tier
        callback.tracker.track_cost("gpt-4o-mini", 1000, 1000, 95.0)

        @budget_gate(min_tier=BudgetTier.COMFORTABLE, skip_on_insufficient=True)
        def gated_task():
            return "executed"

        with adaptive_context(callback):
            result = gated_task()

        assert result is None  # Skipped

    def test_budget_gate_raises_on_insufficient(self):
        """Test @budget_gate raises exception when configured."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        callback.tracker.track_cost("gpt-4o-mini", 1000, 1000, 95.0)

        @budget_gate(min_tier=BudgetTier.COMFORTABLE, raise_on_insufficient=True)
        def gated_task():
            return "executed"

        with adaptive_context(callback):
            with pytest.raises(RuntimeError, match="Insufficient budget"):
                gated_task()

    def test_track_efficiency_decorator(self):
        """Test @track_efficiency decorator."""
        callback = TokenCoPilotCallback(budget_limit=100.0)

        @track_efficiency(metric_name="test_task")
        def task_with_cost():
            # Simulate LLM call
            callback.tracker.track_cost("gpt-4o-mini", 100, 50, 0.05)
            return "done"

        with adaptive_context(callback):
            result = task_with_cost()

        assert result == "done"
        # Cost should have been tracked
        assert callback.get_total_cost() > 0


class TestContextManagers:
    """Test context managers."""

    def test_adaptive_context(self):
        """Test adaptive_context manager."""
        callback = TokenCoPilotCallback(budget_limit=100.0)

        assert get_current_callback() is None

        with adaptive_context(callback):
            assert get_current_callback() is callback

        assert get_current_callback() is None

    def test_nested_adaptive_contexts(self):
        """Test nested adaptive contexts."""
        callback1 = TokenCoPilotCallback(budget_limit=100.0)
        callback2 = TokenCoPilotCallback(budget_limit=50.0)

        with adaptive_context(callback1):
            assert get_current_callback() is callback1

            with adaptive_context(callback2):
                assert get_current_callback() is callback2

            assert get_current_callback() is callback1

    def test_set_and_get_current_callback(self):
        """Test manual callback setting."""
        callback = TokenCoPilotCallback(budget_limit=100.0)

        set_current_callback(callback)
        assert get_current_callback() is callback

        set_current_callback(None)
        assert get_current_callback() is None

    def test_budget_aware_section(self):
        """Test budget_aware_section context manager."""
        callback = TokenCoPilotCallback(budget_limit=100.0)

        with budget_aware_section(callback, "test_section") as section:
            assert section['name'] == "test_section"
            assert section['initial_cost'] == 0.0

            # Simulate cost
            callback.tracker.track_cost("gpt-4o-mini", 100, 50, 0.05)

        # Section info should be updated
        assert section['cost_delta'] > 0
        assert section['tokens_delta'] > 0

    def test_budget_aware_section_no_logging(self):
        """Test budget_aware_section without logging."""
        callback = TokenCoPilotCallback(budget_limit=100.0)

        with budget_aware_section(callback, "test", log_summary=False) as section:
            callback.tracker.track_cost("gpt-4o-mini", 100, 50, 0.05)

        assert section['cost_delta'] > 0


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test complete adaptive workflow."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)

        # Check initial tier
        assert ops.get_current_tier() == BudgetTier.ABUNDANT

        # Simulate spending
        callback.tracker.track_cost("gpt-4o-mini", 10000, 10000, 70.0)
        assert ops.get_current_tier() == BudgetTier.MODERATE

        # Get tier info
        info = ops.get_tier_info()
        assert info['remaining'] == 30.0
        assert 25 < info['remaining_percentage'] < 50

    def test_decorator_and_operations_together(self):
        """Test using decorators with TokenAwareOperations."""
        callback = TokenCoPilotCallback(budget_limit=100.0)
        ops = TokenAwareOperations(callback)

        @token_aware(operation='generate')
        def decorated_func(prompt, max_tokens=None):
            return max_tokens

        # Use both patterns
        with adaptive_context(callback):
            decorator_result = decorated_func("test")

        # Both should use adaptive params
        assert decorator_result is not None
