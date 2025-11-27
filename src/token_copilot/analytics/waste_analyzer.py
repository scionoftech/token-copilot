"""Token waste analysis for cost optimization."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from collections import Counter
import pandas as pd


@dataclass
class WasteCategory:
    """Waste category with cost and recommendations."""

    category: str
    waste_tokens: int
    waste_cost: float
    percentage: float
    instances: int
    recommendation: str
    monthly_savings: float


class WasteAnalyzer:
    """
    Analyzes LLM usage data to detect token waste and generate recommendations.

    Detects three types of waste:
    - Repeated prompts: Same prompt sent multiple times
    - Excessive context: Top 10% of input tokens (estimate 30% removable)
    - Verbose outputs: Top 15% of output tokens (estimate 20% redundancy)

    Example:
        >>> from token_copilot import TokenPilotCallback
        >>> callback = TokenPilotCallback()
        >>> # ... make LLM calls ...
        >>> report = callback.analyze_waste()
        >>> print(f"Total waste: ${report['summary']['total_waste_cost']:.2f}")
        >>> for rec in report['recommendations']:
        ...     print(f"  - {rec}")
    """

    def __init__(self):
        """Initialize WasteAnalyzer."""
        pass

    def analyze(self, df: pd.DataFrame) -> Dict[str, WasteCategory]:
        """
        Analyze DataFrame to detect waste patterns.

        Args:
            df: DataFrame from TokenPilotCallback.to_dataframe()
                Expected columns: timestamp, model, input_tokens, output_tokens,
                total_tokens, cost, user_id, org_id, etc.

        Returns:
            Dict mapping category name to WasteCategory

        Example:
            >>> df = callback.to_dataframe()
            >>> waste = analyzer.analyze(df)
            >>> print(waste['repeated_prompts'].waste_cost)
        """
        if df.empty:
            return {}

        waste_categories = {}

        # Detect repeated prompts
        repeated_waste = self._detect_repeated_prompts(df)
        if repeated_waste:
            waste_categories['repeated_prompts'] = repeated_waste

        # Detect excessive context
        excessive_context = self._detect_excessive_context(df)
        if excessive_context:
            waste_categories['excessive_context'] = excessive_context

        # Detect verbose outputs
        verbose_outputs = self._detect_verbose_outputs(df)
        if verbose_outputs:
            waste_categories['verbose_outputs'] = verbose_outputs

        return waste_categories

    def get_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive waste report with recommendations.

        Args:
            df: DataFrame from TokenPilotCallback.to_dataframe()

        Returns:
            Dict with keys:
                - summary: Overall waste statistics
                - categories: Detailed breakdown by category
                - recommendations: List of actionable recommendations
                - potential_savings: Monthly savings estimates

        Example:
            >>> report = analyzer.get_report(df)
            >>> print(f"Total waste: ${report['summary']['total_waste_cost']:.2f}")
            >>> print(f"Monthly savings: ${report['summary']['monthly_savings']:.2f}")
            >>> for rec in report['recommendations']:
            ...     print(f"  • {rec}")
        """
        if df.empty:
            return {
                'summary': {
                    'total_waste_cost': 0.0,
                    'total_waste_tokens': 0,
                    'waste_percentage': 0.0,
                    'monthly_savings': 0.0,
                },
                'categories': {},
                'recommendations': [],
                'potential_savings': 0.0,
            }

        waste_categories = self.analyze(df)

        # Calculate summary
        total_waste_cost = sum(cat.waste_cost for cat in waste_categories.values())
        total_waste_tokens = sum(cat.waste_tokens for cat in waste_categories.values())
        total_cost = df['cost'].sum()
        waste_percentage = (total_waste_cost / total_cost * 100) if total_cost > 0 else 0.0
        monthly_savings = sum(cat.monthly_savings for cat in waste_categories.values())

        # Generate recommendations
        recommendations = []
        for category in waste_categories.values():
            recommendations.append(category.recommendation)

        return {
            'summary': {
                'total_waste_cost': total_waste_cost,
                'total_waste_tokens': total_waste_tokens,
                'waste_percentage': waste_percentage,
                'monthly_savings': monthly_savings,
                'total_cost': total_cost,
            },
            'categories': {
                name: {
                    'waste_tokens': cat.waste_tokens,
                    'waste_cost': cat.waste_cost,
                    'percentage': cat.percentage,
                    'instances': cat.instances,
                    'recommendation': cat.recommendation,
                    'monthly_savings': cat.monthly_savings,
                }
                for name, cat in waste_categories.items()
            },
            'recommendations': recommendations,
            'potential_savings': monthly_savings,
        }

    def _detect_repeated_prompts(self, df: pd.DataFrame) -> Optional[WasteCategory]:
        """
        Detect repeated system prompts.

        Identifies prompts that appear multiple times and calculates waste
        from the duplicates (all occurrences after the first).
        """
        if 'input_tokens' not in df.columns or len(df) < 2:
            return None

        # Create a hashable representation of each call
        # We'll use input_tokens as a proxy for prompt similarity
        # In a real implementation, you'd compare actual prompts
        token_counts = df['input_tokens'].tolist()

        # Find duplicates
        duplicates = []
        seen = set()
        for i, tokens in enumerate(token_counts):
            if tokens in seen and tokens > 100:  # Only consider substantial prompts
                duplicates.append(i)
            seen.add(tokens)

        if not duplicates:
            return None

        # Calculate waste
        waste_df = df.iloc[duplicates]
        waste_tokens = waste_df['input_tokens'].sum()
        waste_cost = waste_df['cost'].sum()
        instances = len(duplicates)

        # Estimate monthly savings (assume pattern continues)
        days_tracked = (df.index.max() - df.index.min()).total_seconds() / 86400
        days_tracked = max(1, days_tracked)  # Avoid division by zero
        monthly_savings = (waste_cost / days_tracked) * 30

        percentage = (waste_cost / df['cost'].sum() * 100) if df['cost'].sum() > 0 else 0.0

        return WasteCategory(
            category='repeated_prompts',
            waste_tokens=int(waste_tokens),
            waste_cost=waste_cost,
            percentage=percentage,
            instances=instances,
            recommendation=f"Cache system prompts to avoid {instances} repeated calls → Save ${monthly_savings:.2f}/mo",
            monthly_savings=monthly_savings,
        )

    def _detect_excessive_context(self, df: pd.DataFrame) -> Optional[WasteCategory]:
        """
        Detect excessive context windows.

        Identifies calls in the top 10% of input tokens and estimates
        30% of those tokens could be removed.
        """
        if 'input_tokens' not in df.columns or len(df) < 10:
            return None

        # Find top 10% by input tokens
        threshold = df['input_tokens'].quantile(0.90)
        excessive_df = df[df['input_tokens'] >= threshold]

        if len(excessive_df) == 0:
            return None

        # Estimate 30% could be removed
        waste_tokens = int(excessive_df['input_tokens'].sum() * 0.30)

        # Calculate waste cost based on input token cost
        # Approximate by using the proportion of input tokens to total cost
        total_tokens = excessive_df['input_tokens'] + excessive_df['output_tokens']
        input_proportion = excessive_df['input_tokens'] / total_tokens
        waste_cost = (excessive_df['cost'] * input_proportion * 0.30).sum()

        instances = len(excessive_df)

        # Estimate monthly savings
        days_tracked = (df.index.max() - df.index.min()).total_seconds() / 86400
        days_tracked = max(1, days_tracked)
        monthly_savings = (waste_cost / days_tracked) * 30

        percentage = (waste_cost / df['cost'].sum() * 100) if df['cost'].sum() > 0 else 0.0

        return WasteCategory(
            category='excessive_context',
            waste_tokens=waste_tokens,
            waste_cost=waste_cost,
            percentage=percentage,
            instances=instances,
            recommendation=f"Reduce context window in {instances} calls (top 10%) → Save ${monthly_savings:.2f}/mo",
            monthly_savings=monthly_savings,
        )

    def _detect_verbose_outputs(self, df: pd.DataFrame) -> Optional[WasteCategory]:
        """
        Detect verbose outputs.

        Identifies calls in the top 15% of output tokens and estimates
        20% redundancy.
        """
        if 'output_tokens' not in df.columns or len(df) < 10:
            return None

        # Find top 15% by output tokens
        threshold = df['output_tokens'].quantile(0.85)
        verbose_df = df[df['output_tokens'] >= threshold]

        if len(verbose_df) == 0:
            return None

        # Estimate 20% redundancy
        waste_tokens = int(verbose_df['output_tokens'].sum() * 0.20)

        # Calculate waste cost based on output token cost
        total_tokens = verbose_df['input_tokens'] + verbose_df['output_tokens']
        output_proportion = verbose_df['output_tokens'] / total_tokens
        waste_cost = (verbose_df['cost'] * output_proportion * 0.20).sum()

        instances = len(verbose_df)

        # Estimate monthly savings
        days_tracked = (df.index.max() - df.index.min()).total_seconds() / 86400
        days_tracked = max(1, days_tracked)
        monthly_savings = (waste_cost / days_tracked) * 30

        percentage = (waste_cost / df['cost'].sum() * 100) if df['cost'].sum() > 0 else 0.0

        return WasteCategory(
            category='verbose_outputs',
            waste_tokens=waste_tokens,
            waste_cost=waste_cost,
            percentage=percentage,
            instances=instances,
            recommendation=f"Reduce verbosity in prompts for {instances} calls → Save ${monthly_savings:.2f}/mo",
            monthly_savings=monthly_savings,
        )
