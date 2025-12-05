# Adaptive Token Operations Examples

This directory contains examples demonstrating the **Adaptive Token Operations** feature, which automatically adjusts LLM parameters based on remaining budget.

## What is Adaptive Token Operations?

Adaptive operations intelligently optimize your LLM usage based on budget tiers:

- **ABUNDANT** (>75% remaining): Premium settings, maximum quality
- **COMFORTABLE** (50-75%): Balanced approach
- **MODERATE** (25-50%): Start optimizing, reduce token usage
- **LOW** (10-25%): Aggressive optimization
- **CRITICAL** (<10%): Minimal token usage

## Examples

### 1. `basic_adaptive.py`
Basic usage of `TokenAwareOperations` class for adaptive text generation.

```bash
python basic_adaptive.py
```

### 2. `decorator_example.py`
Using `@token_aware` decorator for declarative adaptive behavior.

```bash
python decorator_example.py
```

### 3. `budget_gating.py`
Using `@budget_gate` to prevent expensive operations when budget is low.

```bash
python budget_gating.py
```

### 4. `complete_workflow.py`
Complete end-to-end example with multiple adaptive features.

```bash
python complete_workflow.py
```

## Key Features

### TokenAwareOperations Class

Main class that provides adaptive operations:
- `generate()` - Adaptive text generation
- `search()` - Adaptive document retrieval
- `retry()` - Adaptive retry logic

### Decorators

- `@token_aware` - Make any function budget-aware
- `@budget_gate` - Gate function execution by minimum tier
- `@track_efficiency` - Track efficiency metrics

### Context Managers

- `adaptive_context()` - Set callback for decorators
- `budget_aware_section()` - Track specific code sections

## Installation

Adaptive operations require LangChain:

```bash
pip install langchain langchain-openai
```

## Learn More

See the [Phase 1 Implementation Plan](../../PHASE1_IMPLEMENTATION_PLAN.md) for detailed architecture and design decisions.
