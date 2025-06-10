# Benchmarking

its-hub includes comprehensive benchmarking tools to evaluate inference-time scaling algorithms on standard mathematical reasoning datasets.

## Quick Start

```bash
python scripts/benchmark.py --help
```

Example benchmark command:
```bash
python scripts/benchmark.py \
    --benchmark aime-2024 \
    --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
    --alg particle-filtering \
    --rm_device cuda:1 \
    --endpoint http://0.0.0.0:8000/v1 \
    --shuffle_seed 1110 \
    --does_eval \
    --budgets 1,2,4,8,16,32,64 \
    --rm_agg_method model
```

## Supported Datasets

### MATH500
A subset of 500 problems from the MATH dataset, covering various mathematical topics.

```bash
python scripts/benchmark.py --benchmark math-500 --model_name Qwen/Qwen2.5-Math-1.5B-Instruct
```

### AIME-2024
American Invitational Mathematics Examination problems from 2024.

```bash
python scripts/benchmark.py --benchmark aime-2024 --model_name Qwen/Qwen2.5-Math-1.5B-Instruct
```

## Algorithm Comparison

### Benchmarking Multiple Algorithms

```bash
# Compare all algorithms on MATH500
for alg in self-consistency best-of-n beam-search particle-filtering; do
    python scripts/benchmark.py \
        --benchmark math-500 \
        --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
        --alg $alg \
        --budgets 1,2,4,8,16 \
        --does_eval
done
```

### Budget Scaling Analysis

```bash
# Analyze performance vs computational budget
python scripts/benchmark.py \
    --benchmark math-500 \
    --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
    --alg particle-filtering \
    --budgets 1,2,4,8,16,32,64,128 \
    --does_eval
```

## Configuration Options

### Basic Parameters

- `--benchmark`: Dataset to use (`math-500`, `aime-2024`)
- `--model_name`: Model identifier (e.g., `Qwen/Qwen2.5-Math-1.5B-Instruct`)
- `--alg`: Algorithm to benchmark (`self-consistency`, `best-of-n`, `beam-search`, `particle-filtering`)
- `--budgets`: Comma-separated list of budget values
- `--endpoint`: API endpoint for model inference
- `--does_eval`: Enable automatic evaluation of results

### Advanced Parameters

- `--shuffle_seed`: Seed for shuffling problems (reproducibility)
- `--rm_device`: GPU device for reward model (e.g., `cuda:0`)
- `--rm_agg_method`: Reward aggregation method (`prod`, `mean`, `model`)
- `--beam_width`: Beam width for beam search (default: 4)
- `--max_steps`: Maximum steps for step-by-step algorithms
- `--step_token`: Token for step boundaries (default: `\\n\\n`)
- `--stop_pattern`: Regex pattern for stopping generation

## Output Format

### Results Structure

The benchmark script generates detailed results including:

```json
{
    "algorithm": "particle-filtering",
    "dataset": "math-500",
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "budget": 8,
    "accuracy": 0.756,
    "total_problems": 500,
    "correct_answers": 378,
    "average_response_time": 12.34,
    "detailed_results": [
        {
            "problem_id": "001",
            "problem": "Solve x^2 + 5x + 6 = 0",
            "correct_answer": "x = -2, -3",
            "model_response": "...",
            "is_correct": true,
            "response_time": 8.21
        }
    ]
}
```

### Evaluation Metrics

- **Accuracy**: Percentage of correctly solved problems
- **Response Time**: Average time per problem (seconds)
- **Budget Efficiency**: Accuracy improvement per unit budget
- **Error Analysis**: Breakdown of error types and frequencies

## Performance Analysis

### Plotting Results

```python
import matplotlib.pyplot as plt
import json

# Load benchmark results
with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

# Plot accuracy vs budget
budgets = [r['budget'] for r in results]
accuracies = [r['accuracy'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(budgets, accuracies, 'o-')
plt.xlabel('Budget')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Computational Budget')
plt.grid(True)
plt.show()
```

### Statistical Analysis

```python
import numpy as np
from scipy import stats

# Compare two algorithms
results_a = [r for r in results if r['algorithm'] == 'self-consistency']
results_b = [r for r in results if r['algorithm'] == 'particle-filtering']

accuracies_a = [r['accuracy'] for r in results_a]
accuracies_b = [r['accuracy'] for r in results_b]

# Perform t-test
t_stat, p_value = stats.ttest_ind(accuracies_a, accuracies_b)
print(f"T-test p-value: {p_value}")
```

## Custom Benchmarks

### Adding New Datasets

```python
# Create custom dataset
custom_problems = [
    {
        "id": "custom_001",
        "problem": "Your math problem here",
        "answer": "Expected answer",
        "category": "algebra"
    }
]

# Save as JSON
import json
with open('custom_benchmark.json', 'w') as f:
    json.dump(custom_problems, f)
```

### Custom Evaluation Metrics

```python
def custom_evaluator(predicted_answer, correct_answer):
    """Custom evaluation function"""
    # Implement your evaluation logic
    return predicted_answer.strip().lower() == correct_answer.strip().lower()

# Use in benchmark script
python scripts/benchmark.py \
    --benchmark custom_benchmark.json \
    --custom_evaluator custom_evaluator
```

## Best Practices

### Reproducibility

1. **Set Random Seeds**: Use `--shuffle_seed` for consistent problem ordering
2. **Fixed Hyperparameters**: Document all configuration options
3. **Environment Tracking**: Record GPU type, driver versions, and dependencies

### Performance Optimization

1. **GPU Memory Management**: Monitor memory usage during benchmarks
2. **Batch Processing**: Use appropriate batch sizes for your hardware
3. **Caching**: Enable model caching for faster repeated evaluations

### Result Validation

1. **Cross-Validation**: Run multiple seeds and average results
2. **Significance Testing**: Use statistical tests to validate improvements
3. **Human Evaluation**: Manually verify a sample of results

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
```bash
# Reduce batch size or budget
python scripts/benchmark.py --budgets 1,2,4 --rm_device cuda:0
```

**Slow Evaluation:**
```bash
# Disable evaluation for faster benchmarking
python scripts/benchmark.py --no_eval
```

**Model Loading Issues:**
```bash
# Verify model availability
curl http://localhost:8000/v1/models
```

### Performance Monitoring

```bash
# Monitor GPU usage during benchmarking
watch -n 1 nvidia-smi

# Monitor system resources
htop
```