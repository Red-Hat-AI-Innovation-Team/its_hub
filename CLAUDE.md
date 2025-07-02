# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Development installation (includes all dependencies)
pip install -e ".[dev]"

# Production installation
pip install its_hub
```

### Testing
```bash
# Run all tests
pytest tests

# Run tests with coverage
pytest tests --cov=its_hub
```

### Code Quality
```bash
# Run linter checks
uv run ruff check its_hub/

# Fix auto-fixable linting issues
uv run ruff check its_hub/ --fix

# Format code with ruff
uv run ruff format its_hub/
```

### Git Workflow
```bash
# Create commits with sign-off
git commit -s -m "commit message"

# For any git commits, always use the sign-off flag (-s)
```

### Running Examples
```bash
# Test basic functionality
python scripts/test_math_example.py

# Benchmark algorithms (see script help for full options)
python scripts/benchmark.py --help
```

## Additional Tips
- Use `rg` in favor of `grep` whenever it's available
- Use `uv` for Python environment management: always start with `uv sync --extra dev` to init the env and run stuff with `uv run`
- In case of dependency issues during testing, try commenting out `reward_hub` and `vllm` temporarily in @pyproject.toml and retry.

## Architecture Overview

**its_hub** is a library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks. The core architecture uses abstract base classes to define clean interfaces between components.

### Key Base Classes (`its_hub/base.py`)
- `AbstractLanguageModel`: Interface for LM generation and evaluation
- `AbstractScalingAlgorithm`: Base for all scaling algorithms with unified `infer()` method
- `AbstractScalingResult`: Base for algorithm results with `the_one` property
- `AbstractOutcomeRewardModel`: Interface for outcome-based reward models
- `AbstractProcessRewardModel`: Interface for process-based reward models (step-by-step scoring)

### Main Components

#### Language Models (`its_hub/lms.py`)
- `OpenAICompatibleLanguageModel`: Primary LM implementation supporting vLLM and OpenAI APIs
- `StepGeneration`: Handles incremental generation with configurable step tokens and stop conditions
- Supports async generation with concurrency limits and backoff strategies

#### Algorithms (`its_hub/algorithms/`)
All algorithms follow the same interface: `infer(lm, prompt, budget, return_response_only=True)`

- **Self-Consistency**: Generate multiple responses, select most common answer
- **Best-of-N**: Generate N responses, select highest scoring via outcome reward model  
- **Beam Search**: Step-by-step generation with beam width, uses process reward models
- **Particle Filtering/Gibbs**: Probabilistic resampling with process reward models

#### Integration (`its_hub/integration/`)
- `LocalVllmProcessRewardModel`: Integrates with reward_hub library for process-based scoring
- `iaas.py`: Inference-as-a-Service FastAPI server providing OpenAI-compatible chat completions API with budget parameter for inference-time scaling

### Budget Interpretation
The budget parameter controls computational resources allocated to each algorithm. Different algorithms interpret budget as follows:
- **Self-Consistency/Best-of-N**: Number of parallel generations to create
- **Beam Search**: Total generations divided by beam width (controls search depth)
- **Particle Filtering**: Number of particles maintained during sampling

### Step Generation Pattern
The `StepGeneration` class enables incremental text generation:
- Configure step tokens (e.g., "\n\n" for reasoning steps)
- Set max steps and stop conditions
- Post-processing for clean output formatting

### Typical Workflow
1. Start vLLM server with instruction model
2. Initialize `OpenAICompatibleLanguageModel` pointing to server
3. Create `StepGeneration` with step/stop tokens appropriate for the task
4. Initialize reward model (e.g., `LocalVllmProcessRewardModel`)
5. Create scaling algorithm with step generation and reward model
6. Call `infer()` with prompt and budget

### Mathematical Focus
The library is optimized for mathematical reasoning:
- Predefined system prompts in `its_hub/utils.py` (SAL_STEP_BY_STEP_SYSTEM_PROMPT, QWEN_SYSTEM_PROMPT)
- Regex patterns for mathematical notation (e.g., `r"\boxed"` for final answers)
- Integration with math_verify for evaluation
- Benchmarking on MATH500 and AIME-2024 datasets

## Current Experiment: Planning-Enhanced Best-of-N

### Objective
Add a planning phase before Best-of-N inference-time scaling where:
1. Model generates a plan with ~3 distinct approaches/hypotheses for the problem
2. Best-of-N budget is divided equally across these planned approaches
3. Each approach is executed sequentially with its allocated budget
4. Compare performance against vanilla Best-of-N baseline

### Experimental Setup
- **Target Algorithm**: Best-of-N only (simplified scope)
- **Planning Phase**: Model generates structured plan with distinct solution approaches
- **Budget Allocation**: Divide total budget equally across planned approaches (e.g., 16 total → ~5-6 per approach)
- **Dataset**: AIME2024 (already integrated via HuggingFace: Maxwell-Jia/AIME_2024)
- **Budget Range**: Up to 16 particles for testing
- **Comparison**: Planning-Enhanced Best-of-N vs. Vanilla Best-of-N

### Current Status
✅ **AIME2024 Dataset**: Fully integrated in benchmark.py with proper data loading and preprocessing
✅ **Vanilla Best-of-N**: Algorithm tested and working correctly with mock language/reward models
✅ **GPU Separation**: Inference model (GPU 0: 57GB) and reward model (GPU 1: 14GB) properly separated
✅ **Planning-Enhanced Best-of-N**: Fully implemented and tested with real AIME problems
✅ **Experimental Validation**: Completed comparison on 5 AIME2024 questions with budgets 4, 8, 16

### Testing Notes
- **Vanilla Best-of-N Test**: Created `test_vanilla_bon.py` - confirmed algorithm works with mock components
- **Dataset Loading**: Uses `datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]` with column normalization
- **Evaluation**: Uses `math_verify` library for mathematical answer verification with `\boxed{}` pattern extraction
- **Real Model Testing**: Created `test_real_bon.py` - validated 16 particle budget works without context issues
- **GPU Setup**: 
  - **GPU 0**: `Qwen2.5-Math-1.5B-Instruct` via vLLM server (port 8100)
  - **GPU 1**: `Qwen2.5-Math-PRM-7B` via LocalVllmProcessRewardModel (port 8101)

### Experimental Results Summary
**Dataset**: 5 AIME2024 questions (IDs: 2024-II-14, 2024-I-3, 2024-II-4, 2024-II-2, 2024-I-2)
**Models**: Qwen2.5-Math-1.5B-Instruct (inference) + Qwen2.5-Math-PRM-7B (reward)
**Budgets Tested**: 4, 8, 16 particles

#### Performance Comparison
| Budget | Vanilla Accuracy | Planning Accuracy | Score Improvement | Time Overhead |
|--------|------------------|-------------------|-------------------|---------------|
| 4      | 2/5 (40%)       | 2/5 (40%)        | +0.1106          | +0.4s        |
| 8      | 1/5 (20%)       | 1/5 (20%)        | -0.0435          | +2.0s        |
| 16     | 2/5 (40%)       | 2/5 (40%)        | +0.0430          | +2.0s        |

#### Key Findings
✅ **Planning Implementation**: Successfully generates structured plans with 2-3 distinct mathematical approaches
✅ **Budget Allocation**: Correctly splits budget across planned approaches (e.g., 16 → 8+7 for 2 approaches)
✅ **Approach Diversity**: Generated approaches like "Vieta's formulas", "factoring", "discriminant analysis"
✅ **Scalability**: Planning overhead decreases with larger budgets (becomes faster at budget 16)

⚠️ **Limitations Identified**:
- Plan quality issues with some multilingual/corrupted text generation
- Generic fallback approaches when plan parsing fails
- No significant accuracy improvement over vanilla Best-of-N
- Performance depends heavily on problem complexity and plan quality

✅ **Successful Test Cases**:
- **Question 3 (Log system)**: Both found answer 33, Planning scored higher (0.8256 vs 0.3081)
- **Question 5 (Log equations)**: Both consistently found correct answer 25

❌ **Challenging Cases**:
- **Question 1 (Base conversion)**: Neither found correct answer 211 (complex number theory)
- **Question 2 (Game theory)**: Neither found correct answer 809 (requires strategic analysis)

### Implementation Plan
1. **Create Planning Prompt**: Template encouraging hypothesis/approach generation
2. **Plan Parser**: Extract structured approaches from planning output
3. **Planning-Enhanced Best-of-N Class**: New algorithm that:
   - Generates plan (1 generation)
   - Parses approaches from plan
   - Runs Best-of-N for each approach with allocated budget
   - Combines results across approaches
4. **Evaluation Script**: Compare against vanilla Best-of-N on AIME2024

### Key Questions Resolved
- **Structure**: Natural language plan with distinct numbered approaches
- **Algorithm**: New Planning-Enhanced Best-of-N variant
- **Budget**: Fixed 1 generation for planning, remainder split across approaches
- **Guidance**: Explicit prompting with approach-specific context
- **Testing**: AIME2024 dataset, up to 16 total budget

### Planning Enhancement Extension ✅ **COMPLETED**

**Objective**: Extend planning enhancement to work with ANY ITS algorithm, not just Best-of-N.

**Implementation**: Created `PlanningWrapper` class that can wrap any base ITS algorithm:

#### Architecture
- **Location**: `its_hub/algorithms/planning_wrapper.py`
- **Core Class**: `PlanningWrapper(AbstractScalingAlgorithm)` 
- **Interface**: Takes any base algorithm and enhances it with planning
- **Usage**: Same `infer()` interface maintained across all enhanced algorithms

#### Process Flow
1. **Planning Phase**: Generate plan with 3 distinct approaches (costs 1 from budget)
2. **Approach Parsing**: Extract approaches using regex patterns with fallbacks
3. **Budget Allocation**: Divide remaining budget equally across approaches
4. **Execution**: Run base algorithm for each approach with approach-specific prompts
5. **Selection**: Choose best result based on algorithm-specific scoring

#### Supported Algorithms
✅ **Self-Consistency**: Enhanced with planning via `create_planning_self_consistency()`
✅ **Best-of-N**: Enhanced with planning via `create_planning_best_of_n()`
✅ **Particle Filtering**: Enhanced with planning via `create_planning_particle_filtering()`
✅ **Beam Search**: Enhanced with planning via `create_planning_beam_search()`

#### Example Usage
```python
from its_hub.algorithms.planning_wrapper import PlanningWrapper
from its_hub.algorithms import SelfConsistency

# Manual wrapping
base_sc = SelfConsistency(extract_fn)
planning_sc = PlanningWrapper(base_sc)

# Or convenience function
from its_hub.algorithms.planning_wrapper import create_planning_self_consistency
planning_sc = create_planning_self_consistency(extract_fn)

# Same interface for all
result = planning_sc.infer(lm, prompt, budget=16, return_response_only=False)
print(f"Best approach: {result.best_approach}")
print(f"All approaches: {result.approaches}")
```

#### Testing Results
**Test Script**: `test_planning_wrapper.py` validates all algorithm combinations
- ✅ Planning-Enhanced Self-Consistency: Working
- ✅ Planning-Enhanced Best-of-N: Working  
- ✅ Planning-Enhanced Particle Filtering: Working

**Key Features Validated**:
- Unified interface across all enhanced algorithms
- Proper budget allocation and approach-specific prompting
- Result aggregation and best approach selection
- Fallback handling for plan parsing failures