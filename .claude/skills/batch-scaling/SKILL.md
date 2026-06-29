---
name: batch-scaling
description: "Use when the user wants to run inference-time scaling on multiple prompts from a file (JSONL, CSV, or TXT). Applies to batch processing, evaluation runs, or dataset-level scaling."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_batch_scale.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)"]
---

# Batch Scaling

Run inference-time scaling on multiple prompts from a file.

## Supported Input Formats

- **JSONL** — one JSON object per line with a `prompt` or `messages` field
- **CSV** — must have a `prompt` column
- **TXT** — one prompt per line

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

If `config=missing`, tell the user to run the `setup-guide` skill first.

## Step 2: Run Batch Scaling

Call the batch scaling script with the input file and any overrides:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_batch_scale.sh" [--algorithm ALG] [--budget N] [--model KEY] [--output FILE] <input-file>
```

The script loads config once and processes all prompts in a single process. Default output path is `results/<input_name>_scaled.jsonl`.

## Step 3: Report Summary

The script outputs a JSON summary with `total`, `succeeded`, `failed`, `failures`, and `output_file`.

Report: "N/M prompts completed successfully. K failed. Results written to <output_file>"

If there were failures, list the line numbers and error messages.
