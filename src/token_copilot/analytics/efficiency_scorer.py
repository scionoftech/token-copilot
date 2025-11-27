"""Efficiency scoring for users, organizations, and models."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import math


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics for an entity."""

    entity_id: str
    entity_type: str
    token_efficiency: float  # 0-1 score
    cost_efficiency: float  # 0-1 score
    quality_estimate: float  # 0-1 score
    overall_score: float  # Weighted average
    total_calls: int
    total_tokens: int
    total_cost: float
    avg_tokens_per_call: float
    avg_cost_per_call: float
    recommendations: List[str]


class EfficiencyScorer:
    """
    Scores entities (users/orgs/models) on efficiency using sigmoid functions.

    Calculates three efficiency scores:
    - Token efficiency: How well tokens are used (vs benchmark of 1000 tokens/request)
    - Cost efficiency: How well budget is managed (vs benchmark of $0.01/request)
    - Quality estimate: Inferred from input:output ratio (ideal ~2.0)

    Overall score is weighted average of all three.

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> callback = TokenPilotCallback()
        >>> # ... make LLM calls ...
        >>> score = callback.get_efficiency_score('user_id', 'user_123')
        >>> print(f"Overall efficiency: {score.overall_score:.2f}")
        >>> print(f"Token efficiency: {score.token_efficiency:.2f}")
        >>> for rec in score.recommendations:
        ...     print(f"  - {rec}")
    """

    def __init__(
        self,
        token_benchmark: float = 1000.0,
        cost_benchmark: float = 0.01,
        ideal_io_ratio: float = 2.0,
    ):
        """
        Initialize EfficiencyScorer.

        Args:
            token_benchmark: Benchmark tokens per request for comparison
            cost_benchmark: Benchmark cost per request in USD
            ideal_io_ratio: Ideal input:output token ratio for quality
        """
        self.token_benchmark = token_benchmark
        self.cost_benchmark = cost_benchmark
        self.ideal_io_ratio = ideal_io_ratio

    def score_entity(
        self,
        df: pd.DataFrame,
        entity_type: str,
        entity_id: str,
    ) -> EfficiencyMetrics:
        """
        Score a specific entity.

        Args:
            df: DataFrame from TokenPilotCallback.to_dataframe()
            entity_type: Entity type ('user_id', 'org_id', 'model')
            entity_id: Entity identifier

        Returns:
            EfficiencyMetrics with all scores and recommendations

        Example:
            >>> metrics = scorer.score_entity(df, 'user_id', 'user_123')
            >>> print(f"Score: {metrics.overall_score:.2f}")
        """
        # Filter to entity
        if entity_type not in df.columns:
            raise ValueError(f"Column '{entity_type}' not found in DataFrame")

        entity_df = df[df[entity_type] == entity_id]

        if entity_df.empty:
            raise ValueError(f"No data found for {entity_type}={entity_id}")

        # Calculate stats
        total_calls = len(entity_df)
        total_tokens = entity_df['total_tokens'].sum()
        total_cost = entity_df['cost'].sum()
        avg_tokens = total_tokens / total_calls
        avg_cost = total_cost / total_calls

        # Calculate efficiency scores
        token_efficiency = self._sigmoid_efficiency(avg_tokens, self.token_benchmark)
        cost_efficiency = self._sigmoid_efficiency(avg_cost, self.cost_benchmark)
        quality_estimate = self._estimate_quality(entity_df)

        # Overall score (weighted average)
        overall_score = (
            0.4 * token_efficiency +
            0.4 * cost_efficiency +
            0.2 * quality_estimate
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            token_efficiency,
            cost_efficiency,
            quality_estimate,
            avg_tokens,
            avg_cost,
        )

        return EfficiencyMetrics(
            entity_id=entity_id,
            entity_type=entity_type,
            token_efficiency=token_efficiency,
            cost_efficiency=cost_efficiency,
            quality_estimate=quality_estimate,
            overall_score=overall_score,
            total_calls=total_calls,
            total_tokens=int(total_tokens),
            total_cost=total_cost,
            avg_tokens_per_call=avg_tokens,
            avg_cost_per_call=avg_cost,
            recommendations=recommendations,
        )

    def score_all(
        self,
        df: pd.DataFrame,
        entity_type: str,
    ) -> Dict[str, EfficiencyMetrics]:
        """
        Score all entities of a given type.

        Args:
            df: DataFrame from TokenPilotCallback.to_dataframe()
            entity_type: Entity type ('user_id', 'org_id', 'model')

        Returns:
            Dict mapping entity_id to EfficiencyMetrics

        Example:
            >>> all_scores = scorer.score_all(df, 'user_id')
            >>> for user_id, metrics in all_scores.items():
            ...     print(f"{user_id}: {metrics.overall_score:.2f}")
        """
        if entity_type not in df.columns:
            raise ValueError(f"Column '{entity_type}' not found in DataFrame")

        scores = {}
        for entity_id in df[entity_type].dropna().unique():
            try:
                scores[str(entity_id)] = self.score_entity(df, entity_type, entity_id)
            except Exception:
                # Skip entities that cause errors
                continue

        return scores

    def get_leaderboard(
        self,
        df: pd.DataFrame,
        entity_type: str,
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Get efficiency leaderboard.

        Args:
            df: DataFrame from TokenPilotCallback.to_dataframe()
            entity_type: Entity type ('user_id', 'org_id', 'model')
            top_n: Number of top performers to return

        Returns:
            List of dicts with rank, entity_id, and scores (sorted by overall_score)

        Example:
            >>> leaderboard = scorer.get_leaderboard(df, 'user_id', top_n=5)
            >>> for entry in leaderboard:
            ...     print(f"{entry['rank']}. {entry['entity_id']}: {entry['overall_score']:.2f}")
        """
        scores = self.score_all(df, entity_type)

        # Sort by overall score
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True,
        )

        # Build leaderboard
        leaderboard = []
        for rank, (entity_id, metrics) in enumerate(sorted_scores[:top_n], start=1):
            leaderboard.append({
                'rank': rank,
                'entity_id': entity_id,
                'overall_score': metrics.overall_score,
                'token_efficiency': metrics.token_efficiency,
                'cost_efficiency': metrics.cost_efficiency,
                'quality_estimate': metrics.quality_estimate,
                'total_calls': metrics.total_calls,
                'total_cost': metrics.total_cost,
            })

        return leaderboard

    def _sigmoid_efficiency(self, value: float, benchmark: float) -> float:
        """
        Calculate efficiency score using sigmoid function.

        Lower values are better (more efficient).
        Score approaches 1.0 when value << benchmark.
        Score approaches 0.0 when value >> benchmark.

        Formula: 1 / (1 + e^((value - benchmark) / benchmark))
        """
        if benchmark == 0:
            return 0.5

        try:
            exponent = (value - benchmark) / benchmark
            # Clamp exponent to prevent overflow
            exponent = max(-50, min(50, exponent))
            return 1.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            # If value much larger than benchmark, return low score
            return 0.01 if value > benchmark else 0.99

    def _estimate_quality(self, df: pd.DataFrame) -> float:
        """
        Estimate output quality from input:output ratio.

        Ideal ratio is ~2.0 (concise inputs, substantive outputs).
        Uses inverted sigmoid to score based on deviation from ideal.
        """
        if len(df) == 0:
            return 0.5

        # Calculate average input:output ratio
        total_input = df['input_tokens'].sum()
        total_output = df['output_tokens'].sum()

        if total_output == 0:
            return 0.5  # No outputs to judge

        avg_ratio = total_input / total_output

        # Score based on deviation from ideal
        # Closer to ideal_io_ratio = higher score
        deviation = abs(avg_ratio - self.ideal_io_ratio)

        # Use inverted sigmoid: high score when deviation is low
        try:
            exponent = (deviation - 1.0) / 1.0
            exponent = max(-50, min(50, exponent))
            return 1.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            return 0.5

    def _generate_recommendations(
        self,
        token_efficiency: float,
        cost_efficiency: float,
        quality_estimate: float,
        avg_tokens: float,
        avg_cost: float,
    ) -> List[str]:
        """Generate personalized recommendations based on scores."""
        recommendations = []

        # Token efficiency
        if token_efficiency < 0.5:
            pct_over = ((avg_tokens / self.token_benchmark) - 1) * 100
            recommendations.append(
                f"Token usage is {pct_over:.0f}% above benchmark. "
                f"Consider reducing context or using prompt compression."
            )
        elif token_efficiency > 0.8:
            recommendations.append(
                "Excellent token efficiency! Keep up the concise prompts."
            )

        # Cost efficiency
        if cost_efficiency < 0.5:
            pct_over = ((avg_cost / self.cost_benchmark) - 1) * 100
            recommendations.append(
                f"Costs are {pct_over:.0f}% above benchmark. "
                f"Consider using cheaper models for simple tasks."
            )
        elif cost_efficiency > 0.8:
            recommendations.append(
                "Great cost efficiency! Your model selection is optimal."
            )

        # Quality estimate
        if quality_estimate < 0.5:
            recommendations.append(
                "Input:output ratio suggests potential quality issues. "
                "Review prompt clarity and expected output length."
            )
        elif quality_estimate > 0.8:
            recommendations.append(
                "Excellent input:output balance! Your prompts are well-optimized."
            )

        # Overall
        overall = (token_efficiency + cost_efficiency + quality_estimate) / 3
        if overall > 0.8:
            recommendations.append(
                "Outstanding overall efficiency! You're in the top tier."
            )
        elif overall < 0.4:
            recommendations.append(
                "Consider reviewing your LLM usage patterns. "
                "Many opportunities for optimization available."
            )

        return recommendations if recommendations else ["Efficiency is within normal range."]
