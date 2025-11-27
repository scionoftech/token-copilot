"""Cross-model routing for cost optimization."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import deque
import re


class RoutingStrategy(str, Enum):
    """Routing strategy options."""

    CHEAPEST_FIRST = "cheapest_first"  # Always route to cheapest model
    QUALITY_FIRST = "quality_first"  # Always route to highest quality
    BALANCED = "balanced"  # Balance cost/quality by complexity
    COST_THRESHOLD = "cost_threshold"  # Use cheapest under threshold
    LEARNED = "learned"  # Use historical quality data


@dataclass
class ModelConfig:
    """Model configuration for routing."""

    name: str
    quality_score: float  # 0-1 score
    cost_per_1k_input: float  # USD per 1K input tokens
    cost_per_1k_output: float  # USD per 1K output tokens
    max_tokens: int
    supports_functions: bool = False
    supports_vision: bool = False


@dataclass
class RoutingDecision:
    """Routing decision with rationale."""

    selected_model: str
    reason: str
    estimated_cost: float
    quality_score: float
    alternatives: List[Dict[str, Any]]


class ModelRouter:
    """
    Routes requests to optimal model based on complexity and strategy.

    Categorizes requests as:
    - Simple: <100 chars, basic patterns
    - Medium: Default category
    - Complex: >500 chars, contains "analyze", "debug", "code", etc.

    Routing strategies:
    - CHEAPEST_FIRST: Always use cheapest model
    - QUALITY_FIRST: Always use best model
    - BALANCED: Weight cost/quality by complexity (simple: 70/30, medium: 50/50, complex: 30/70)
    - COST_THRESHOLD: Use cheapest model under cost threshold
    - LEARNED: Use historical quality scores to optimize

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> from token_copilot.routing import ModelConfig, RoutingStrategy
        >>> models = [
        ...     ModelConfig("gpt-4o-mini", 0.7, 0.15, 0.60, 128000),
        ...     ModelConfig("gpt-4o", 0.9, 5.0, 15.0, 128000),
        ... ]
        >>> callback = TokenPilotCallback(
        ...     auto_routing=True,
        ...     routing_models=models,
        ...     routing_strategy=RoutingStrategy.BALANCED
        ... )
        >>> decision = callback.suggest_model("What is Python?", 500)
        >>> print(f"Use {decision.selected_model}: {decision.reason}")
    """

    def __init__(
        self,
        models: List[ModelConfig],
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        quality_threshold: float = 0.7,
        cost_threshold: float = 0.01,
    ):
        """
        Initialize ModelRouter.

        Args:
            models: List of available models
            strategy: Routing strategy to use
            quality_threshold: Minimum quality score required
            cost_threshold: Cost threshold for COST_THRESHOLD strategy
        """
        self.models = {m.name: m for m in models}
        self.strategy = strategy
        self.quality_threshold = quality_threshold
        self.cost_threshold = cost_threshold

        # Quality history for LEARNED strategy (last 100 scores per model)
        self._quality_history: Dict[str, deque] = {
            name: deque(maxlen=100) for name in self.models.keys()
        }

    def route(
        self,
        prompt: str,
        estimated_tokens: int = 1000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route request to optimal model.

        Args:
            prompt: Input prompt text
            estimated_tokens: Estimated total tokens (input + output)
            metadata: Optional metadata (can include required_capabilities)

        Returns:
            RoutingDecision with selected model and rationale

        Example:
            >>> decision = router.route("Explain quantum computing", 2000)
            >>> print(f"Route to {decision.selected_model}")
            >>> print(f"Estimated cost: ${decision.estimated_cost:.4f}")
        """
        metadata = metadata or {}

        # Categorize request complexity
        complexity = self._categorize_complexity(prompt)

        # Filter models by constraints
        available_models = self._filter_models(
            estimated_tokens=estimated_tokens,
            required_capabilities=metadata.get('required_capabilities', {}),
        )

        if not available_models:
            raise ValueError("No models available that meet requirements")

        # Select based on strategy
        if self.strategy == RoutingStrategy.CHEAPEST_FIRST:
            selected = self._select_cheapest(available_models, estimated_tokens)
        elif self.strategy == RoutingStrategy.QUALITY_FIRST:
            selected = self._select_highest_quality(available_models, estimated_tokens)
        elif self.strategy == RoutingStrategy.BALANCED:
            selected = self._select_balanced(available_models, estimated_tokens, complexity)
        elif self.strategy == RoutingStrategy.COST_THRESHOLD:
            selected = self._select_cost_threshold(available_models, estimated_tokens)
        elif self.strategy == RoutingStrategy.LEARNED:
            selected = self._select_learned(available_models, estimated_tokens)
        else:
            selected = self._select_balanced(available_models, estimated_tokens, complexity)

        # Calculate alternatives
        alternatives = []
        for model_name, model in available_models.items():
            if model_name != selected['model']:
                cost = self._estimate_cost(model, estimated_tokens)
                alternatives.append({
                    'model': model_name,
                    'cost': cost,
                    'quality': model.quality_score,
                })

        # Sort alternatives by cost
        alternatives.sort(key=lambda x: x['cost'])

        return RoutingDecision(
            selected_model=selected['model'],
            reason=selected['reason'],
            estimated_cost=selected['cost'],
            quality_score=selected['quality'],
            alternatives=alternatives[:3],  # Top 3 alternatives
        )

    def record_quality(self, model: str, quality_score: float):
        """
        Record quality score for a model (for LEARNED strategy).

        Args:
            model: Model name
            quality_score: Quality score (0-1)

        Example:
            >>> router.record_quality("gpt-4o", 0.95)
        """
        if model in self._quality_history:
            self._quality_history[model].append(quality_score)

    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics per model.

        Returns:
            Dict mapping model name to stats (avg_quality, calls, etc.)

        Example:
            >>> stats = router.get_model_stats()
            >>> for model, data in stats.items():
            ...     print(f"{model}: avg quality = {data['avg_quality']:.2f}")
        """
        stats = {}
        for model_name, history in self._quality_history.items():
            if history:
                avg_quality = sum(history) / len(history)
                stats[model_name] = {
                    'avg_quality': avg_quality,
                    'calls': len(history),
                    'base_quality': self.models[model_name].quality_score,
                }
            else:
                stats[model_name] = {
                    'avg_quality': self.models[model_name].quality_score,
                    'calls': 0,
                    'base_quality': self.models[model_name].quality_score,
                }

        return stats

    def _categorize_complexity(self, prompt: str) -> str:
        """Categorize request complexity: simple, medium, or complex."""
        # Simple: <100 chars, basic patterns
        if len(prompt) < 100:
            # Check for basic question patterns
            simple_patterns = [
                r'^what is ',
                r'^who is ',
                r'^when ',
                r'^where ',
                r'^define ',
                r'^translate ',
            ]
            if any(re.match(pattern, prompt.lower()) for pattern in simple_patterns):
                return 'simple'

        # Complex: >500 chars or contains complex keywords
        if len(prompt) > 500:
            return 'complex'

        complex_keywords = [
            'analyze', 'debug', 'code', 'implement', 'refactor',
            'optimize', 'architecture', 'design', 'algorithm',
            'explain in detail', 'comprehensive', 'thorough'
        ]
        if any(keyword in prompt.lower() for keyword in complex_keywords):
            return 'complex'

        # Default: medium
        return 'medium'

    def _filter_models(
        self,
        estimated_tokens: int,
        required_capabilities: Dict[str, bool],
    ) -> Dict[str, ModelConfig]:
        """Filter models by constraints."""
        filtered = {}

        for name, model in self.models.items():
            # Check token limit
            if estimated_tokens > model.max_tokens:
                continue

            # Check quality threshold
            if model.quality_score < self.quality_threshold:
                continue

            # Check required capabilities
            if required_capabilities.get('functions', False) and not model.supports_functions:
                continue
            if required_capabilities.get('vision', False) and not model.supports_vision:
                continue

            filtered[name] = model

        return filtered

    def _estimate_cost(self, model: ModelConfig, estimated_tokens: int) -> float:
        """Estimate cost for a model (assume 50/50 input/output split)."""
        input_tokens = estimated_tokens * 0.5
        output_tokens = estimated_tokens * 0.5

        input_cost = (input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model.cost_per_1k_output

        return input_cost + output_cost

    def _select_cheapest(
        self,
        available_models: Dict[str, ModelConfig],
        estimated_tokens: int,
    ) -> Dict[str, Any]:
        """Select cheapest model."""
        cheapest_model = None
        cheapest_cost = float('inf')

        for name, model in available_models.items():
            cost = self._estimate_cost(model, estimated_tokens)
            if cost < cheapest_cost:
                cheapest_cost = cost
                cheapest_model = (name, model)

        return {
            'model': cheapest_model[0],
            'cost': cheapest_cost,
            'quality': cheapest_model[1].quality_score,
            'reason': f"Cheapest available model (${cheapest_cost:.4f})",
        }

    def _select_highest_quality(
        self,
        available_models: Dict[str, ModelConfig],
        estimated_tokens: int,
    ) -> Dict[str, Any]:
        """Select highest quality model."""
        best_model = None
        best_quality = 0.0

        for name, model in available_models.items():
            if model.quality_score > best_quality:
                best_quality = model.quality_score
                best_model = (name, model)

        cost = self._estimate_cost(best_model[1], estimated_tokens)

        return {
            'model': best_model[0],
            'cost': cost,
            'quality': best_quality,
            'reason': f"Highest quality model (score: {best_quality:.2f})",
        }

    def _select_balanced(
        self,
        available_models: Dict[str, ModelConfig],
        estimated_tokens: int,
        complexity: str,
    ) -> Dict[str, Any]:
        """Select model balancing cost and quality by complexity."""
        # Weight by complexity: simple (70/30), medium (50/50), complex (30/70)
        weights = {
            'simple': {'cost': 0.7, 'quality': 0.3},
            'medium': {'cost': 0.5, 'quality': 0.5},
            'complex': {'cost': 0.3, 'quality': 0.7},
        }
        weight = weights.get(complexity, weights['medium'])

        # Normalize costs and qualities
        costs = {}
        qualities = {}
        for name, model in available_models.items():
            costs[name] = self._estimate_cost(model, estimated_tokens)
            qualities[name] = model.quality_score

        # Normalize to 0-1 range
        min_cost = min(costs.values())
        max_cost = max(costs.values())
        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0

        min_quality = min(qualities.values())
        max_quality = max(qualities.values())
        quality_range = max_quality - min_quality if max_quality > min_quality else 1.0

        # Calculate scores (lower cost is better, so invert)
        best_model = None
        best_score = -1.0

        for name, model in available_models.items():
            cost_norm = 1.0 - ((costs[name] - min_cost) / cost_range)
            quality_norm = (qualities[name] - min_quality) / quality_range

            score = (weight['cost'] * cost_norm) + (weight['quality'] * quality_norm)

            if score > best_score:
                best_score = score
                best_model = (name, model)

        return {
            'model': best_model[0],
            'cost': costs[best_model[0]],
            'quality': qualities[best_model[0]],
            'reason': f"Balanced selection for {complexity} task (score: {best_score:.2f})",
        }

    def _select_cost_threshold(
        self,
        available_models: Dict[str, ModelConfig],
        estimated_tokens: int,
    ) -> Dict[str, Any]:
        """Select cheapest model under cost threshold, else best quality."""
        # Find cheapest under threshold
        under_threshold = []
        for name, model in available_models.items():
            cost = self._estimate_cost(model, estimated_tokens)
            if cost <= self.cost_threshold:
                under_threshold.append((name, model, cost))

        if under_threshold:
            # Use cheapest under threshold
            under_threshold.sort(key=lambda x: x[2])
            name, model, cost = under_threshold[0]
            return {
                'model': name,
                'cost': cost,
                'quality': model.quality_score,
                'reason': f"Cheapest under threshold ${self.cost_threshold:.4f}",
            }
        else:
            # Use highest quality (threshold exceeded anyway)
            return self._select_highest_quality(available_models, estimated_tokens)

    def _select_learned(
        self,
        available_models: Dict[str, ModelConfig],
        estimated_tokens: int,
    ) -> Dict[str, Any]:
        """Select based on historical quality performance."""
        # Use average historical quality if available
        best_model = None
        best_value = -1.0

        for name, model in available_models.items():
            history = self._quality_history[name]

            if history:
                # Use historical average
                avg_quality = sum(history) / len(history)
            else:
                # Use base quality score
                avg_quality = model.quality_score

            cost = self._estimate_cost(model, estimated_tokens)

            # Value = quality / cost (maximize quality per dollar)
            value = avg_quality / cost if cost > 0 else 0

            if value > best_value:
                best_value = value
                best_model = (name, model, avg_quality, cost)

        return {
            'model': best_model[0],
            'cost': best_model[3],
            'quality': best_model[2],
            'reason': f"Best quality/cost ratio from learned data (value: {best_value:.2f})",
        }
