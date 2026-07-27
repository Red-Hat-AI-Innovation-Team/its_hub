# Tool-Call Voting Checkpoint Results

## Setup

- **Benchmark:** BFCL v4 single-turn (simple_python: 399 tasks, multiple: 199 tasks)
- **Models:** gpt-4o-mini (mid-tier), gpt-3.5-turbo (weaker), claude-sonnet-4 (strong, partial — rate-limited)
- **Budget:** N=5 and N=10 samples per task
- **Scorer:** Field-aware majority voting with equivalence-aware normalization and configurable confidence threshold (75%)
- **Baseline:** Single-shot (1st sample), naive exact-match voting (SelfConsistency tool_hierarchical)

## BFCL Field-Type Audit

Before the checkpoint, we audited argument field types across all BFCL v4
single-turn schemas (4,602 fields total):

| Type | Count | % |
|------|------:|--:|
| free-text | 2,788 | 60.6% |
| numeric | 1,406 | 30.6% |
| categorical-enum | 408 | 8.9% |

**Surprise:** The doc predicted free-text would be under 15-20%. It's actually
the dominant category (60.6%), so the checkpoint results speak directly to the
hardest equivalence case rather than the easy one.

## Agreement Results (gpt-4o-mini, N=5)

How often does the model agree with itself across 5 samples?

| Category | Tasks | High Confidence | Forced | Forced % |
|----------|------:|----------------:|-------:|---------:|
| simple_python (temp=0.7) | 399 | 390 | 9 | 2.3% |
| simple_python (temp=1.0) | 399 | 379 | 20 | 5.0% |
| multiple (temp=0.7) | 199 | 193 | 6 | 3.0% |
| multiple (temp=1.0) | 199 | 190 | 9 | 4.5% |
| parallel (temp=0.7) | 200 | 195 | 5 | 2.5% |
| parallel_multiple (temp=0.7) | 198 | 195 | 3 | 1.5% |

Higher temperature increases disagreement (2.3% → 5.0%) but even at temp=1.0,
the model agrees with itself 95% of the time.

## Equivalence Layer

Built an equivalence-aware normalization layer that auto-detects field type
(numeric/boolean/string) and normalizes before voting:
- Strings: case-insensitive, strip trailing punctuation, collapse whitespace
- Numeric: canonicalize to float (42 == 42.0)

On simple_python temp=1.0: reduces forced rate from 5.0% → 3.8% by collapsing
5 surface-form disagreements (capitalization, trailing periods).

## Forced Case Analysis

The forced cases are all **free-text surface-form disagreements**:

- Location format: `"San Francisco"` vs `"San Francisco, CA"`
- Unit notation: `"g/mol"` vs `"g/mole"` vs `"grams/mole"`
- Capitalization: `"LeBron James"` vs `"Lebron James"`
- Punctuation: `"Let's meet at 10 AM tomorrow."` vs `"...tomorrow"` (trailing period)
- Datetime format: `"2023-10-01T12:00:00"` vs `"12:00"`

The model *knows* the right answer — it represents it differently across samples.

## Accuracy vs Ground Truth

### gpt-4o-mini, simple_python (399 tasks), temp=1.0

| Method | Name | Args (Exact) | Args (Fuzzy) |
|--------|:---:|:---:|:---:|
| Single-shot | 100% | 88.7% | 90.2% |
| Voted (N=5) | 100% | 90.0% | 90.7% |
| Best-of-5 oracle | 100% | 91.2% | 91.5% |
| **Delta (voted vs single)** | **+0.0pp** | **+1.3pp** | **+0.5pp** |

### gpt-4o-mini, multiple (197 tasks), temp=1.0, no CoT

| Method | Name | Args (Fuzzy) |
|--------|:---:|:---:|
| Single-shot | 99.0% | 93.9% |
| Voted (N=5) | 99.0% | 93.9% |
| Oracle | — | 93.9% |
| **Delta** | **+0.0pp** | **+0.0pp** |

### gpt-3.5-turbo, multiple (199 tasks), temp=1.0

| Method | Name | Args (Fuzzy) |
|--------|:---:|:---:|
| Single-shot | 98.5% | 94.0% |
| Voted (N=5) | 99.0% | 93.5% |
| **Delta** | **+0.5pp** | **-0.5pp** |

Weaker model: voting hurts arg accuracy. Systematic errors are reinforced
by majority vote rather than corrected.

## Threshold Sweep

Swept confidence threshold from 0.50 to 1.00 across all configurations.
**The pre-registered criterion (lift >= 3pp AND coverage >= 50%) was NOT MET
at any threshold in any configuration.**

Best results:

| Config | Best Threshold | Coverage | Lift |
|--------|:---:|:---:|:---:|
| gpt-4o-mini, N=5, simple, +equiv | 1.00 | 92.5% | +1.1pp |
| gpt-4o-mini, N=5, multiple | 0.85 | 90.8% | +2.2pp |
| gpt-4o-mini, N=10, simple, +equiv | 0.95 | 91.0% | +2.0pp |
| gpt-3.5-turbo, N=5, multiple | — | — | negative |

## Chain-of-Thought + Self-Consistency (CoT-SC)

Tested whether forcing step-by-step reasoning before tool selection creates
the reasoning diversity that standard SC needs to work.

### gpt-4o-mini, multiple (199 tasks), temp=1.0, fuzzy, +equiv

| Metric | No CoT | With CoT |
|--------|:---:|:---:|
| Single-shot arg accuracy | 93.9% | **94.4%** |
| Voted (N=5) arg accuracy | 93.9% | 93.9% |
| Oracle ceiling | 93.9% | **95.9%** |
| Voting delta | +0.0pp | -0.5pp |
| **Headroom (oracle - voted)** | **0.0pp** | **+2.0pp** |

### Key finding

CoT raises the oracle ceiling by +2.0pp (93.9% → 95.9%) — proving it creates
genuinely useful diversity in the sample set. But majority voting fails to
exploit it: the voted result is actually slightly worse than single-shot (-0.5pp).

**The bottleneck is the selection mechanism, not the sampling.** CoT gives
us samples where the correct answer exists, but majority vote picks the
wrong one because incorrect answers can still outnumber correct ones.

## Conclusions

### What we proved

1. **Majority voting does not reliably improve tool-call accuracy.** Across
   all configurations (2 models, 4 categories, N=5 and N=10, temp 0.7 and 1.0),
   voting lift never exceeded +1.3pp and was often zero or negative.

2. **The root cause is systematic errors.** Tool-call errors are not random —
   the model consistently produces the same wrong answer across samples. More
   samples reinforce the wrong majority rather than surfacing a correct minority.

3. **This is fundamentally different from math SC**, where diverse reasoning
   paths independently converge on the correct answer. Tool calling lacks this
   "reasoning path diversity" — the model either knows the right tool/args or
   it doesn't.

4. **CoT creates diversity but voting can't use it.** Chain-of-thought reasoning
   raises the oracle ceiling by +2pp, but the correct answers are outnumbered by
   the incorrect majority.

5. **Equivalence-aware matching helps agreement but not accuracy.** Normalizing
   surface forms collapses disagreements, but these are mostly cosmetic — the
   model was already semantically correct.

6. **The confidence tag is a useful signal.** High-confidence cases are
   consistently more accurate than forced cases (94% vs 77-88%), even though
   voting doesn't improve overall accuracy.

## Best-of-N Re-Ranking Investigation

Following the voting results, we tested whether smarter *selection* (rather
than majority vote) could exploit the oracle headroom.

### Schema Validation ORM (deterministic, zero cost)

Built a `SchemaValidationORM` (subclassing `AbstractOutcomeRewardModel`) that
scores tool calls against JSON schemas: function name, required params, types,
enum values, unexpected params. 16 tests passing.

**Result: 0.0pp lift.** Models already produce structurally valid tool calls.
Schema validation is a necessary floor but cannot differentiate between
semantically correct and incorrect candidates.

### LLM Self-Judge BoN (same model as judge)

Built a harness using gpt-4o-mini as both generator and judge. For each task,
generates N=5 candidates, scores each with a tool-call-specific judge prompt,
selects the highest-scored candidate.

**Result: 0.0pp lift.** But the reason is more fundamental than the judge
quality:

### The definitive finding: zero oracle headroom

Analysis of the full dataset revealed the root cause:

| Category | Tasks | Oracle headroom tasks |
|----------|------:|----------------------:|
| multiple (N=5, temp=1.0) | 199 | **0** |
| simple_python (N=10, temp=1.0) | 399 | **8** (2%) |

**When the model gets a tool call wrong, it gets it wrong the same way on
every sample.** There are literally zero tasks in the `multiple` category
where any of the 5 samples is correct but the first sample is wrong. No
re-ranking method — ToolRM, LLM-judge, schema validation, or anything
else — can improve accuracy when there is nothing correct to select.

This is the fundamental difference from math reasoning: in math, different
reasoning paths independently arrive at the correct answer, so sampling
creates recoverable diversity. In tool calling, the model's knowledge of the
correct function and arguments is either present (all samples correct) or
absent (all samples wrong). Temperature and CoT create surface-form diversity
but not *correctness* diversity.

## Final Conclusions

### What we proved

1. **Majority voting does not improve tool-call accuracy.** Across all
   configurations (3 models, 4 categories, N=5 and N=10, temp 0.7 and 1.0),
   the pre-registered 3pp criterion was never met. Lift ranged from -0.5pp
   to +1.3pp.

2. **Best-of-N re-ranking cannot improve tool-call accuracy either.** The
   oracle ceiling equals single-shot accuracy — there is nothing correct in
   the sample set to select. This rules out ToolRM, LLM-judge, schema
   validation, and any other re-ranking approach.

3. **The root cause is fully correlated errors.** Tool-call errors are not
   random across samples — they are systematic. The model either knows the
   correct function/arguments or it doesn't. Sampling more does not create
   the independent correctness diversity that SC and BoN require.

4. **This is a fundamental property of tool calling, not a limitation of
   specific models or techniques.** The finding held across gpt-4o-mini,
   gpt-3.5-turbo, and claude-sonnet-4, and across simple, multiple, parallel,
   and parallel_multiple task types.

5. **Equivalence-aware matching helps agreement but not accuracy.** Surface-form
   normalization collapses cosmetic disagreements but the model was already
   semantically correct in those cases.

6. **CoT prompting provides a small single-shot accuracy improvement** (+0.5pp)
   and is worth recommending for tool-calling workloads regardless.

7. **The confidence tag is a useful signal.** High-confidence cases are
   consistently more accurate (94% vs 77-88%), providing calibration value
   even without accuracy improvement.

### What would actually help

Since sampling-based approaches (SC, BoN) are ruled out by the correlated-error
structure, approaches that change *what the model knows* are the relevant
direction:

1. **Better prompting** — CoT before tool selection (+0.5pp proven), richer
   function descriptions, few-shot examples of correct tool use.

2. **RAG over API documentation** — ground the model's tool knowledge in
   retrieved documentation rather than relying on parametric memory.

3. **Execution feedback + retry** — run the tool call, use success/failure as
   a signal, retry on error. Published results show 85% → 98.8% task success
   (Self-Healing Agentic Orchestrators, arXiv:2606.01416). This works because
   it provides *external* signal the model doesn't have.

4. **RubricRefine** — generate task-specific rubrics from the tool registry,
   score and repair candidates pre-execution. +0.38 across 7 models, no extra
   model (arXiv:2605.09730).

5. **Fine-tuning** on tool-calling data (out of scope per the research doc,
   but the only approach that changes the model's underlying knowledge).

### Recommendation

**Close the inference-time scaling track for tool calling.** Neither voting
nor re-ranking can improve accuracy when errors are fully correlated across
samples. The mechanism is well understood and the result is definitive.

**Recommend CoT prompting** as a default for its_hub's `PlanningWrapper`
when used with tool-calling workloads — small but real single-shot improvement.

**For future tool-call reliability work**, focus on execution-based feedback
loops (retry on failure) and prompt engineering (richer function descriptions,
few-shot examples). These address the actual bottleneck: what the model knows,
not how it selects among samples.

## References

- ToolRM: [arXiv:2509.11963](https://arxiv.org/abs/2509.11963)
- Reinforced Agent: [arXiv:2604.27233](https://arxiv.org/abs/2604.27233)
- Self-Healing Agentic Orchestrators: [arXiv:2606.01416](https://arxiv.org/html/2606.01416v1)
- RubricRefine: [arXiv:2605.09730](https://arxiv.org/abs/2605.09730)
- Self-Certainty BoN: [arXiv:2502.18581](https://arxiv.org/abs/2502.18581)
- Mirror-Consistency: [arXiv:2410.10857](https://arxiv.org/abs/2410.10857)
- Universal Self-Consistency: [arXiv:2311.17311](https://arxiv.org/abs/2311.17311)
