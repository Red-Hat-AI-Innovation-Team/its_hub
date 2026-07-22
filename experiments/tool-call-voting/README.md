# Tool-Call Self-Consistency Voting — Research Spike

**Goal:** Test whether self-consistency voting over N sampled tool calls beats
single-shot generation for structured function calls, before investing in a full
field-aware aggregation layer.

## Hypothesis

When an LLM generates tool calls, sampling multiple completions and voting across
them should improve reliability — especially for categorical/enum arguments where
the model may be uncertain. A field-aware scorer that votes per-argument (rather
than exact-match on the entire call) should recover correct fields even when
individual samples disagree on other fields.

## Components

| File | Purpose |
|------|---------|
| `bfcl_field_audit.py` | Audit BFCL v4 schemas — bucket argument fields by type (numeric / categorical-enum / free-text) |
| `roll_up_scorer.py` | Field-aware scorer: majority-vote tool name, then score each argument field independently |
| `sampling_harness.py` | Wire up `OpenAICompatibleLanguageModel` to sample N completions per BFCL task, compare field-aware scorer vs naive `SelfConsistency(tool_vote="tool_hierarchical")` baseline |

## Running

```bash
# Audit BFCL field types
uv run python experiments/tool-call-voting/bfcl_field_audit.py

# Run the checkpoint experiment (requires configured LM endpoint)
uv run python experiments/tool-call-voting/sampling_harness.py

# Run scorer tests
uv run pytest experiments/tool-call-voting/tests/ -v
```

## Status

Scaffolded — not yet run. Pending model endpoint configuration and BFCL data download.
