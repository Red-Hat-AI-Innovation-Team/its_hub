"""Explode a cot_compare CSV (one row per prompt x item) into one row per reasoning STEP.

A "step" = the granularity PF/EPF resample on: a blank-line (``\\n\\n``) separated chunk
(so ``n_steps`` matches the ``n_chunks`` column of the source CSV). For prompts that write
steps on single newlines (e.g. method 9's ``Step 1:\\nStep 2:`` format), the blank-line
split yields one blob, so we fall back to splitting on explicit step markers
(``Step N``, ``## Step``, ``N.``) — so its individual steps are still visible.

Each ``step_text`` is whitespace-collapsed into a single clean paragraph for easy reading.

    python -m benchmarking.mmau_pro.explode_steps \
        --in benchmarking/mmau_pro/results/cot_compare_40.csv \
        --out benchmarking/mmau_pro/results/cot_compare_40_steps.csv
"""

import csv
import re

import click

# A new step begins at a line starting with a step marker (used only as a fallback
# when blank-line splitting produced a single chunk).
_STEP_MARKER = re.compile(r"(?im)^\s*(?:#{1,3}\s*step\b|step\s*\d+\b|\d+[.)]\s)")


def _as_paragraph(s: str) -> str:
    """Collapse all internal whitespace/newlines so the step reads as one paragraph."""
    return re.sub(r"\s+", " ", s).strip()


def split_steps(text: str) -> list[str]:
    """Split a full response into reasoning steps (see module docstring)."""
    text = (text or "").strip()
    if not text:
        return []
    # primary: blank-line separated chunks (PF \n\n granularity == source n_chunks)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    if len(chunks) >= 2:
        return chunks
    # fallback: single-newline step formats -> split on explicit step markers
    markers = list(_STEP_MARKER.finditer(text))
    if len(markers) >= 2:
        out = []
        if markers[0].start() > 0:  # preamble before the first marker (e.g. "Reasoning:")
            pre = text[: markers[0].start()].strip()
            if pre:
                out.append(pre)
        for i, m in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            seg = text[m.start() : end].strip()
            if seg:
                out.append(seg)
        return out
    return chunks  # genuinely a single step


@click.command()
@click.option("--in", "in_path", default="benchmarking/mmau_pro/results/cot_compare_40.csv")
@click.option("--out", "out_path", default="benchmarking/mmau_pro/results/cot_compare_40_steps.csv")
def main(in_path, out_path):
    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f))

    fields = [
        "unique_id", "category", "method", "method_name",
        "step_index", "n_steps", "step_text",
        "predicted_letter", "gold_letter", "correct", "question",
    ]
    out_rows = []
    for r in rows:
        steps = split_steps(r["response"])
        n = len(steps)
        for i, step in enumerate(steps, start=1):
            out_rows.append({
                "unique_id": r["unique_id"],
                "category": r["category"],
                "method": r["method"],
                "method_name": r["method_name"],
                "step_index": i,
                "n_steps": n,
                "step_text": _as_paragraph(step),
                "predicted_letter": r["predicted_letter"],
                "gold_letter": r["gold_letter"],
                "correct": r["correct"],
                "question": r["question"],
            })

    # contiguous per response, methods grouped: (method, item, step order)
    out_rows.sort(key=lambda d: (int(d["method"]), d["unique_id"], d["step_index"]))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"read {len(rows)} responses -> wrote {len(out_rows)} step rows -> {out_path}")


if __name__ == "__main__":
    main()
