# Planning Wrapper for ITS Algorithms

## Overview

The `PlanningWrapper` is a generic enhancement that adds a planning phase to any inference-time scaling (ITS) algorithm. It allows the model to generate multiple solution approaches before execution, potentially improving performance through diverse strategy exploration.

## Key Features

- **Universal Compatibility**: Works with any ITS algorithm (Self-Consistency, Best-of-N, Particle Filtering, Beam Search)
- **Unified Interface**: Maintains the same `infer()` method signature across all enhanced algorithms
- **Smart Budget Allocation**: Automatically divides computational budget across planned approaches
- **Robust Plan Parsing**: Handles various plan formats with intelligent fallbacks

## Architecture

### Core Components

1. **PlanningWrapper**: Main class that wraps any base ITS algorithm
2. **PlanningPromptTemplate**: Generates prompts that encourage diverse approach planning
3. **PlanParser**: Extracts structured approaches from natural language plans
4. **ApproachPromptTemplate**: Creates approach-specific prompts for execution

### Process Flow

1. **Planning Phase**: Generate plan with 3 distinct approaches (costs 1 from budget)
2. **Approach Parsing**: Extract approaches using regex patterns with fallbacks
3. **Budget Allocation**: Divide remaining budget equally across approaches
4. **Execution**: Run base algorithm for each approach with approach-specific prompts
5. **Selection**: Choose best result based on algorithm-specific scoring

## Usage

### Manual Wrapping
```python
from its_hub.algorithms.planning_wrapper import PlanningWrapper
from its_hub.algorithms import SelfConsistency

base_algorithm = SelfConsistency(extract_fn)
planning_algorithm = PlanningWrapper(base_algorithm)

result = planning_algorithm.infer(lm, prompt, budget=16, return_response_only=False)
```

### Convenience Functions
```python
from its_hub.algorithms.planning_wrapper import (
    create_planning_self_consistency,
    create_planning_particle_filtering, 
    create_planning_best_of_n,
    create_planning_beam_search
)

# Enhanced algorithms
planning_sc = create_planning_self_consistency(extract_fn)
planning_pf = create_planning_particle_filtering(sg, prm)
planning_bon = create_planning_best_of_n(orm)
planning_bs = create_planning_beam_search(sg, prm, beam_width=4)

# Same interface for all
result = planning_sc.infer(lm, prompt, budget=16, return_response_only=False)
```

### Result Object
```python
# Planning-enhanced results include additional information
result = planning_algorithm.infer(lm, prompt, budget=16, return_response_only=False)

print(f"Best answer: {result.the_one}")
print(f"Generated plan: {result.plan}")
print(f"Approaches used: {result.approaches}")
print(f"Best approach: {result.best_approach}")
print(f"Budget allocation: {result.approach_budgets}")
```

## Supported Algorithms

- ✅ **Self-Consistency**: Enhanced with planning via `create_planning_self_consistency()`
- ✅ **Best-of-N**: Enhanced with planning via `create_planning_best_of_n()`
- ✅ **Particle Filtering**: Enhanced with planning via `create_planning_particle_filtering()`
- ✅ **Beam Search**: Enhanced with planning via `create_planning_beam_search()`

## Testing

Run the comprehensive test suite:
```bash
python test_planning_wrapper.py
```

This test validates:
- Planning-enhanced versions of all supported algorithms
- Proper budget allocation across approaches
- Result aggregation and best approach selection
- Fallback handling for plan parsing failures

## Implementation Details

### Plan Generation
The wrapper generates plans using a structured prompt that encourages the model to think of 3 distinct mathematical approaches:

```
APPROACH 1: [Brief description of first method/strategy]
APPROACH 2: [Brief description of second method/strategy] 
APPROACH 3: [Brief description of third method/strategy]
```

### Budget Allocation
- Planning phase uses 1 generation from the total budget
- Remaining budget is divided equally across parsed approaches
- Any remainder is distributed to the first few approaches

### Approach Selection
The wrapper selects the best approach based on algorithm-specific scoring:
- Tries various score attributes (`best_score`, `confidence`, `scores`, etc.)
- Falls back to response length as a proxy for quality
- Returns the approach with the highest score

### Error Handling
- Robust plan parsing with regex patterns and fallbacks
- Generic fallback approaches if parsing fails completely
- Graceful handling of missing score attributes

## Performance Considerations

- **Overhead**: 1 additional generation for planning
- **Benefits**: Potentially better results through diverse approaches
- **Trade-offs**: Lower budgets may suffer from planning overhead, higher budgets benefit more

## Future Enhancements

- Adaptive planning based on problem complexity
- Dynamic budget allocation based on approach confidence
- Cross-approach result fusion techniques
- Problem-specific approach templates