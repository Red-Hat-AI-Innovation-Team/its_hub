---
description: "Run inference-time scaling on a batch of prompts from a file"
argument-hint: "<file> [--output <file>]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)"]
---

# its-hub Batch Scale

Run inference-time scaling on multiple prompts from a file.

## Supported Input Formats

- **JSONL** — one JSON object per line with a `prompt` or `messages` field
- **CSV** — must have a `prompt` column
- **TXT** — one prompt per line

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

If `config=missing`, tell the user to run `/its-setup` first.

## Step 2: Parse Arguments

Extract from `$ARGUMENTS`:
- `file` — the input file path (required)
- `--output` — output file path (default: `<input_name>_scaled.jsonl`)

Validate the input file exists and detect its format from the extension.

## Step 3: Read and Validate Input

Read the file and extract prompts based on format:
- JSONL: parse each line, extract `prompt` or `messages` field
- CSV: read with Python csv module, extract `prompt` column
- TXT: each line is a prompt

Report: "Found N prompts in <filename>"

## Step 4: Process Prompts

For each prompt, call the scaling script:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh" "<prompt>"
```

Write each result to the output file as a JSONL line:
```json
{"prompt": "...", "selected_response": "...", "algorithm": "...", "budget": N, "metadata": {...}}
```

If a prompt fails, write an error entry and continue:
```json
{"prompt": "...", "error": "error message", "algorithm": "...", "budget": N}
```

## Step 5: Report Summary

Report: "N/M prompts completed successfully. K failed. Results written to <output_file>"

If there were failures, list the line numbers and error messages.
