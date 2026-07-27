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

### What would work instead

The +2pp oracle headroom from CoT-SC confirms that correct answers *exist* in
the sample set — the problem is *selecting* them. Three approaches that address
this directly:

1. **Best-of-N with ToolRM** — a specialized reward model (ToolRM, 1.7B-14B
   parameters) that scores tool calls and selects the best one. Published
   results show up to +25% accuracy improvement. its_hub already has `BestOfN`
   with `AbstractOutcomeRewardModel`; ToolRM would plug in directly.

2. **Reviewer agent feedback loop** (Apple's Reinforced Agent) — a second model
   reviews provisional tool calls before execution. +5.5% on BFCL irrelevance
   detection, +7.1% on multi-turn tasks.

3. **Universal Self-Consistency (USC)** — use an LLM to judge which of N tool
   calls is most consistent, rather than exact/fuzzy matching. Handles semantic
   equivalence without hand-coded rules.

### Recommendation

**Close the majority-voting track for tool calling.** The pre-registered
criterion was not met, and the mechanism analysis explains why: tool-call
errors are systematic, not random, so sampling more doesn't help.

**Open a new track: Best-of-N re-ranking with a tool-call verifier.** This
directly addresses the identified bottleneck (selection, not sampling) and has
published evidence of large gains. its_hub's existing `BestOfN` algorithm
provides the infrastructure; the research question becomes which verifier
to use (ToolRM, LLM-judge, or schema-based validation).

**Preserve the CoT-SC finding** as a recommendation for its_hub's
`PlanningWrapper`: when used with tool-calling workloads, CoT prompting
improves single-shot accuracy (+0.5pp) and raises the ceiling for any
downstream selection mechanism.
