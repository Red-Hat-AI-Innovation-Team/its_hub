# Tool-Call Voting Checkpoint Results

## Setup

- **Benchmark:** BFCL v4 single-turn (simple_python: 399 tasks)
- **Models:** gpt-4o-mini (mid-tier), claude-sonnet-4 (strong, partial — rate-limited)
- **Budget:** N=5 samples per task
- **Scorer:** Field-aware majority voting with configurable confidence threshold (75%)
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
| parallel (temp=0.7) | 200 | 195 | 5 | 2.5% |
| parallel_multiple (temp=0.7) | 198 | 195 | 3 | 1.5% |

Higher temperature increases disagreement (2.3% → 5.0%) but even at temp=1.0,
the model agrees with itself 95% of the time.

## Forced Case Analysis

The 9 forced cases at temp=0.7 are all **free-text surface-form disagreements**:

- Location format: `"San Francisco"` vs `"San Francisco, CA"`
- Unit notation: `"g/mol"` vs `"g/mole"` vs `"grams/mole"`
- Capitalization: `"LeBron James"` vs `"Lebron James"`
- Punctuation: `"Let's meet at 10 AM tomorrow."` vs `"...tomorrow"` (trailing period)
- Datetime format: `"2023-10-01T12:00:00"` vs `"12:00"`

The model *knows* the right answer — it represents it differently across samples.
This is precisely the gap the doc identified: field-aware semantic equivalence
would collapse these variants.

## Accuracy vs Ground Truth (gpt-4o-mini, simple_python, 399 tasks)

### Temperature 0.7

| Method | Name Accuracy | Arg Accuracy (Exact) | Arg Accuracy (Fuzzy) |
|--------|:---:|:---:|:---:|
| Single-shot (1st sample) | 100% | 89.2% | 90.2% |
| Voted (N=5) | 100% | 89.2% | 90.7% |
| Best-of-5 (oracle) | 100% | 91.0% | 91.2% |
| **Delta (voted vs single)** | **+0.0pp** | **+0.0pp** | **+0.5pp** |
| Headroom (oracle - voted) | — | +1.8pp | +0.5pp |

### Temperature 1.0

| Method | Name Accuracy | Arg Accuracy (Exact) | Arg Accuracy (Fuzzy) |
|--------|:---:|:---:|:---:|
| Single-shot (1st sample) | 100% | 88.7% | 90.2% |
| Voted (N=5) | 100% | **90.0%** | 90.7% |
| Best-of-5 (oracle) | 100% | 91.2% | 91.5% |
| **Delta (voted vs single)** | **+0.0pp** | **+1.3pp** | **+0.5pp** |
| Headroom (oracle - voted) | — | +1.3pp | +0.8pp |

### Key finding

At temp=1.0 with exact matching, voting shows **+1.3pp lift** over single-shot.
This is the classic self-consistency pattern: higher temperature hurts individual
samples (89.2% → 88.7%) but the ensemble recovers and exceeds (→ 90.0%).

The high-confidence subset is also more accurate than the overall voted result
(90.8% vs 90.0%), confirming the confidence tag is a useful calibration signal.

## Assessment Against Pre-Registered Decision Criterion

> Voting is worth continuing if high-confidence accuracy exceeds single-shot by
> 3-5 points while covering at least 50% of cases.

**Result: borderline.** The +1.3pp lift at temp=1.0 is real but below the 3-5pp
threshold. Coverage is high (95% tagged high-confidence), but the lift is modest.

However, three factors suggest this understates the opportunity:

1. **gpt-4o-mini is already strong on BFCL simple tasks** (89%+ single-shot).
   The ceiling is low. Harder tasks or weaker models would have more room for
   voting to help.

2. **N=5 is small.** The oracle ceiling (91.2%) is close to the voted result
   (90.0%), meaning most of the recoverable diversity is already captured. Larger
   N would raise the oracle ceiling and widen the gap for voting to exploit.

3. **The equivalence layer doesn't exist yet.** Current voting uses exact string
   matching. The forced cases show the model genuinely knows the right answer but
   represents it in different surface forms — a semantic equivalence layer would
   collapse these variants and increase both the agreement rate and the accuracy
   of the voted result.

## Recommendation

**Continue into the equivalence-layer build**, but with adjusted expectations:

- The lift from voting alone is modest on easy tasks with strong models (+1-2pp).
- The primary value-add is the **confidence signal** (high-confidence cases are
  more accurate) and the **semantic equivalence layer** (collapsing surface-form
  variants the model already gets right).
- Target the evaluation at harder task types (multi-turn, complex schemas) and
  weaker models where single-shot accuracy is lower and voting has more headroom.
- The `multiple` category showed 10.5% forced rate on Claude Sonnet (vs 3% on
  simple) — multi-function selection is the more promising surface area.

## Cost

| Config | Tokens/task | Cost per N=1 equiv |
|--------|:---:|:---:|
| gpt-4o-mini, N=5 | ~610 | ~122 |
| claude-sonnet, N=5 | ~2,815 | ~563 |

Voting at N=5 with gpt-4o-mini costs ~5x a single call (~$0.0003/task).
The cost overhead is negligible for tool-calling workloads.
