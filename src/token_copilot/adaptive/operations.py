"""Token-aware operations that adapt based on budget availability."""

import logging
from typing import Any, Dict, List, Optional, Union
from .tiers import BudgetTier, classify_budget_tier, get_tier_description

try:
    from ..langchain import TokenCoPilotCallback
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    TokenCoPilotCallback = None  # type: ignore

logger = logging.getLogger(__name__)


class TokenAwareOperations:
    """Provides budget-aware operations that automatically adapt based on remaining budget.

    This class wraps a TokenCoPilotCallback and provides intelligent operation methods
    that adjust their parameters based on the current budget tier. User-provided
    parameters always take precedence over adaptive defaults.

    Design: Uses composition pattern - does NOT modify the callback itself.

    Args:
        callback: TokenCoPilotCallback instance for budget tracking
        enable_logging: Whether to log adaptive parameter adjustments (default: True)

    Example:
        >>> callback = TokenCoPilotCallback(budget_limit=100.0)
        >>> adaptive = TokenAwareOperations(callback)
        >>>
        >>> # Automatically adjusts max_tokens and temperature based on budget
        >>> result = adaptive.generate(llm, "Explain quantum computing")
        >>>
        >>> # User params always override
        >>> result = adaptive.generate(llm, "prompt", max_tokens=500)
    """

    def __init__(
        self,
        callback: 'TokenCoPilotCallback',
        enable_logging: bool = True
    ):
        """Initialize token-aware operations.

        Args:
            callback: TokenCoPilotCallback instance for budget tracking
            enable_logging: Whether to log adaptive parameter adjustments
        """
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for TokenAwareOperations. "
                "Install with: pip install langchain"
            )

        self.callback = callback
        self.enable_logging = enable_logging

    def get_current_tier(self) -> BudgetTier:
        """Get current budget tier based on remaining budget.

        Returns:
            BudgetTier enum value indicating current budget status
        """
        return classify_budget_tier(self.callback)

    def get_tier_info(self) -> Dict[str, Any]:
        """Get detailed information about current budget tier.

        Returns:
            Dictionary containing tier, description, and budget statistics
        """
        tier = self.get_current_tier()
        total = self.callback.get_total_cost()
        limit = self.callback.budget_limit
        remaining = limit - total if limit else float('inf')
        percentage = (remaining / limit * 100) if limit else 100.0

        return {
            'tier': tier,
            'tier_name': tier.value,
            'description': get_tier_description(tier),
            'total_cost': total,
            'budget_limit': limit,
            'remaining': remaining,
            'remaining_percentage': percentage
        }

    def _get_adaptive_params(
        self,
        tier: BudgetTier,
        operation: str = 'generate'
    ) -> Dict[str, Any]:
        """Get adaptive parameter defaults based on budget tier.

        Args:
            tier: Current budget tier
            operation: Type of operation ('generate', 'search', etc.)

        Returns:
            Dictionary of recommended parameters for the operation
        """
        if operation == 'generate':
            # Adaptive parameters for text generation
            params_by_tier = {
                BudgetTier.ABUNDANT: {
                    'max_tokens': 2000,
                    'temperature': 0.7,
                    'top_p': 0.95,
                },
                BudgetTier.COMFORTABLE: {
                    'max_tokens': 1500,
                    'temperature': 0.7,
                    'top_p': 0.9,
                },
                BudgetTier.MODERATE: {
                    'max_tokens': 1000,
                    'temperature': 0.5,
                    'top_p': 0.85,
                },
                BudgetTier.LOW: {
                    'max_tokens': 500,
                    'temperature': 0.3,
                    'top_p': 0.8,
                },
                BudgetTier.CRITICAL: {
                    'max_tokens': 200,
                    'temperature': 0.1,
                    'top_p': 0.7,
                },
            }
            return params_by_tier.get(tier, params_by_tier[BudgetTier.COMFORTABLE])

        elif operation == 'search':
            # Adaptive parameters for search/retrieval
            params_by_tier = {
                BudgetTier.ABUNDANT: {
                    'top_k': 10,
                    'similarity_threshold': 0.5,
                },
                BudgetTier.COMFORTABLE: {
                    'top_k': 7,
                    'similarity_threshold': 0.6,
                },
                BudgetTier.MODERATE: {
                    'top_k': 5,
                    'similarity_threshold': 0.7,
                },
                BudgetTier.LOW: {
                    'top_k': 3,
                    'similarity_threshold': 0.75,
                },
                BudgetTier.CRITICAL: {
                    'top_k': 1,
                    'similarity_threshold': 0.8,
                },
            }
            return params_by_tier.get(tier, params_by_tier[BudgetTier.COMFORTABLE])

        elif operation == 'retry':
            # Adaptive parameters for retry logic
            params_by_tier = {
                BudgetTier.ABUNDANT: {
                    'max_retries': 5,
                    'exponential_base': 2.0,
                },
                BudgetTier.COMFORTABLE: {
                    'max_retries': 3,
                    'exponential_base': 1.5,
                },
                BudgetTier.MODERATE: {
                    'max_retries': 2,
                    'exponential_base': 1.5,
                },
                BudgetTier.LOW: {
                    'max_retries': 1,
                    'exponential_base': 1.0,
                },
                BudgetTier.CRITICAL: {
                    'max_retries': 0,
                    'exponential_base': 1.0,
                },
            }
            return params_by_tier.get(tier, params_by_tier[BudgetTier.COMFORTABLE])

        return {}

    def _log_adaptive_params(
        self,
        operation: str,
        tier: BudgetTier,
        params: Dict[str, Any],
        user_overrides: List[str]
    ):
        """Log adaptive parameter selections for transparency.

        Args:
            operation: Operation name
            tier: Current budget tier
            params: Final parameters being used
            user_overrides: List of parameter names overridden by user
        """
        if not self.enable_logging:
            return

        msg_parts = [
            f"Adaptive {operation}:",
            f"tier={tier.value}",
        ]

        for key, value in params.items():
            override_marker = " (user)" if key in user_overrides else ""
            msg_parts.append(f"{key}={value}{override_marker}")

        logger.info(" ".join(msg_parts))

    def generate(
        self,
        llm: Any,
        prompt: Union[str, List],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> Any:
        """Generate text with adaptive parameters based on budget tier.

        User-provided parameters always override adaptive defaults.

        Args:
            llm: Language model instance (LangChain LLM or ChatModel)
            prompt: Input prompt (string or message list)
            max_tokens: Maximum tokens to generate (user override)
            temperature: Sampling temperature (user override)
            top_p: Nucleus sampling parameter (user override)
            **kwargs: Additional parameters passed to the LLM

        Returns:
            LLM response (format depends on the LLM type)

        Example:
            >>> adaptive = TokenAwareOperations(callback)
            >>> result = adaptive.generate(llm, "Explain AI")
            >>> # When budget is low, automatically uses lower max_tokens
        """
        tier = self.get_current_tier()
        adaptive_params = self._get_adaptive_params(tier, 'generate')

        # User params override adaptive defaults
        final_params = {}
        user_overrides = []

        if max_tokens is not None:
            final_params['max_tokens'] = max_tokens
            user_overrides.append('max_tokens')
        elif 'max_tokens' in adaptive_params:
            final_params['max_tokens'] = adaptive_params['max_tokens']

        if temperature is not None:
            final_params['temperature'] = temperature
            user_overrides.append('temperature')
        elif 'temperature' in adaptive_params:
            final_params['temperature'] = adaptive_params['temperature']

        if top_p is not None:
            final_params['top_p'] = top_p
            user_overrides.append('top_p')
        elif 'top_p' in adaptive_params:
            final_params['top_p'] = adaptive_params['top_p']

        # Merge with any additional kwargs
        final_params.update(kwargs)

        # Log adaptive behavior
        self._log_adaptive_params('generate', tier, final_params, user_overrides)

        # Ensure callback is attached
        if 'callbacks' not in final_params:
            final_params['callbacks'] = [self.callback]
        elif self.callback not in final_params['callbacks']:
            final_params['callbacks'].append(self.callback)

        # Call the LLM
        if isinstance(prompt, str):
            return llm.invoke(prompt, **final_params)
        else:
            return llm.invoke(prompt, **final_params)

    def search(
        self,
        retriever: Any,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs: Any
    ) -> List[Any]:
        """Search/retrieve documents with adaptive parameters based on budget tier.

        Args:
            retriever: Document retriever (LangChain retriever or similar)
            query: Search query string
            top_k: Number of results to return (user override)
            similarity_threshold: Minimum similarity score (user override)
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents

        Example:
            >>> adaptive = TokenAwareOperations(callback)
            >>> docs = adaptive.search(retriever, "quantum computing")
            >>> # When budget is low, returns fewer documents
        """
        tier = self.get_current_tier()
        adaptive_params = self._get_adaptive_params(tier, 'search')

        final_params = {}
        user_overrides = []

        if top_k is not None:
            final_params['k'] = top_k
            user_overrides.append('top_k')
        elif 'top_k' in adaptive_params:
            final_params['k'] = adaptive_params['top_k']

        if similarity_threshold is not None:
            final_params['score_threshold'] = similarity_threshold
            user_overrides.append('similarity_threshold')
        elif 'similarity_threshold' in adaptive_params:
            final_params['score_threshold'] = adaptive_params['similarity_threshold']

        final_params.update(kwargs)

        self._log_adaptive_params('search', tier, final_params, user_overrides)

        # Use the retriever
        if hasattr(retriever, 'get_relevant_documents'):
            return retriever.get_relevant_documents(query, **final_params)
        elif hasattr(retriever, 'invoke'):
            return retriever.invoke(query, **final_params)
        else:
            raise ValueError(f"Unsupported retriever type: {type(retriever)}")

    def retry(
        self,
        func: Any,
        *args: Any,
        max_retries: Optional[int] = None,
        exponential_base: Optional[float] = None,
        **kwargs: Any
    ) -> Any:
        """Execute function with adaptive retry logic based on budget tier.

        Args:
            func: Function to execute with retries
            *args: Positional arguments for the function
            max_retries: Maximum retry attempts (user override)
            exponential_base: Base for exponential backoff (user override)
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            Last exception if all retries are exhausted

        Example:
            >>> adaptive = TokenAwareOperations(callback)
            >>> result = adaptive.retry(llm.invoke, "prompt")
            >>> # When budget is low, retries fewer times
        """
        import time

        tier = self.get_current_tier()
        adaptive_params = self._get_adaptive_params(tier, 'retry')

        user_overrides = []

        if max_retries is not None:
            final_max_retries = max_retries
            user_overrides.append('max_retries')
        else:
            final_max_retries = adaptive_params.get('max_retries', 3)

        if exponential_base is not None:
            final_exp_base = exponential_base
            user_overrides.append('exponential_base')
        else:
            final_exp_base = adaptive_params.get('exponential_base', 2.0)

        self._log_adaptive_params(
            'retry',
            tier,
            {'max_retries': final_max_retries, 'exponential_base': final_exp_base},
            user_overrides
        )

        last_exception = None
        for attempt in range(final_max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < final_max_retries:
                    wait_time = final_exp_base ** attempt
                    if self.enable_logging:
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{final_max_retries + 1} "
                            f"failed: {e}. Waiting {wait_time}s..."
                        )
                    time.sleep(wait_time)
                else:
                    if self.enable_logging:
                        logger.error(f"All {final_max_retries + 1} attempts failed")

        if last_exception:
            raise last_exception
