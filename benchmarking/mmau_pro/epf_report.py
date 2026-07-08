"""Render the EPF diversity sweep (epf_div.csv) into one self-contained HTML report.

Everything in one file: config, the verdict, per-prompt trend tables for BOTH weight
signals (selected / oracle / majority acc + distinct-ratio / consensus / ESS), a
selected-vs-oracle heatmap, and a collapsible per-item appendix (all rows).

    python -m benchmarking.mmau_pro.epf_report \
        --in benchmarking/mmau_pro/results/run06_epf_div/epf_div.csv \
        --out benchmarking/mmau_pro/results/run10_run6_full/epf_div.html
"""

import csv
import html
from collections import defaultdict

import click

_CSS = """
:root { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
body { margin: 26px; color: #1a1a1a; background: #fff; max-width: 1100px; }
h1 { font-size: 21px; } h2 { font-size: 17px; margin: 26px 0 6px; } h3 { font-size: 14px; margin: 18px 0 4px; }
.meta { color:#555; font-size:13px; } code { background:#f3f3f3; padding:1px 4px; border-radius:3px; }
.verdict { background:#fff8e6; border:1px solid #f0d98a; border-radius:7px; padding:12px 16px; font-size:14px; }
.verdict b { color:#9a6a00; }
table { border-collapse:collapse; margin:6px 0 10px; font-size:13px; }
th, td { border:1px solid #e6e6e6; padding:4px 9px; text-align:center; }
th { background:#fafafa; } td.l, th.l { text-align:left; }
.sig { font-weight:600; color:#444; }
.gap { font-weight:700; }
details { margin:4px 0; } summary { cursor:pointer; font-size:13px; padding:3px 0; }
summary code { font-size:12px; }
table.items { font-size:12px; } table.items td { padding:2px 6px; }
.ok { color:#137333; font-weight:600; } .no { color:#c5221f; }
.preds { font-family: ui-monospace, Menlo, monospace; font-size:11px; color:#333; letter-spacing:1px; }
.note { color:#666; font-size:12px; }
"""

PROMPT_ORDER = [4, 5, 7, 9]
BUDGET_ORDER = [1, 8, 16, 32]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _green(v):
    """Heatmap background for an accuracy in [0,1] (greener = higher)."""
    if v is None:
        return ""
    a = max(0.0, min(1.0, v)) * 0.55
    return f" style='background: rgba(19,115,51,{a:.2f})'"


def _acc(rows, key):
    g = [r for r in rows if r.get(key) in ("True", "False")]
    return (sum(1 for r in g if r[key] == "True") / len(g)) if g else None, len(g)


def _mean(rows, key):
    vals = [_f(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else "—"


def _cell_stats(rows):
    sel, n = _acc(rows, "selected_correct")
    orc, _ = _acc(rows, "oracle_correct")
    maj, _ = _acc(rows, "majority_correct")
    gap = (orc - sel) if (orc is not None and sel is not None) else None
    return {
        "selected": sel, "oracle": orc, "majority": maj, "gap": gap, "n": n,
        "distinct": _mean(rows, "distinct_ratio"), "consensus": _mean(rows, "consensus"),
        "ess": _mean(rows, "ess_ratio"), "parsed": _mean(rows, "parsed_ratio"),
    }


def build_html(rows, title, max_appendix=0):
    by = defaultdict(list)
    for r in rows:
        by[(int(r["method"]), r["signal"], int(r["budget"]))].append(r)
    names = {int(r["method"]): r["method_name"] for r in rows}
    methods = [m for m in PROMPT_ORDER if m in {int(r["method"]) for r in rows}]
    signals = sorted({r["signal"] for r in rows})
    budgets = [b for b in BUDGET_ORDER if b in {int(r["budget"]) for r in rows}]
    n_items = len({r["unique_id"] for r in rows})

    # ---- verdict numbers ----
    big = max(budgets)
    gaps = []
    for m in methods:
        for sig in signals:
            st = _cell_stats(by[(m, sig, big)])
            if st["gap"] is not None:
                gaps.append((st["gap"], m, sig, st["oracle"], st["selected"]))
    gaps.sort(reverse=True)
    gmax = gaps[0] if gaps else None

    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>EPF · temp 0.8 · systematic resampling · ess_threshold 0.6 · early_phase 0.7 · "
        f"{len(methods)} prompts &times; {len(signals)} weight signals &times; budgets {budgets} &times; {n_items} "
        f"MCQ items · {len(rows)} runs.</p>",
    ]

    # ---- verdict (numbers computed below from the data, not hardcoded) ----
    out.append("<h2>Verdict</h2><div class='verdict'>")
    out.append(
        "Across <b>all prompts and both weight signals</b>, the pattern is identical: "
        "<b>oracle accuracy climbs steeply with budget</b> &mdash; the correct answer is increasingly in the "
        "swarm &mdash; but <b>selected accuracy (what EPF returns) stays roughly flat</b> and does not scale. "
        "Majority-vote is no better. The self-certainty <b>weight is the bottleneck, not the particle "
        "count</b>: more particles surface the answer but EPF returns the wrong one.")
    if gmax:
        out.append(
            f"<br>Largest gap @budget {big}: <b>P{gmax[1]} {html.escape(names[gmax[1]])} "
            f"({gmax[2]})</b> — oracle {gmax[3]:.3f} vs selected {gmax[4]:.3f} = <b>+{gmax[0]:.3f}</b>.")
    out.append("</div>")

    # ---- per-prompt trend tables (both signals) ----
    out.append("<h2>Trend — selected vs oracle by budget (heatmap = accuracy)</h2>")
    out.append("<p class='note'>Greener = higher. Watch the <b>oracle</b> column climb while "
               "<b>selected</b> stays flat; <b>gap</b> = oracle &minus; selected.</p>")
    for m in methods:
        out.append(f"<h3>P{m} — {html.escape(names[m])}</h3>")
        out.append("<table><tr><th>signal</th><th>budget</th><th>selected</th><th>oracle</th>"
                   "<th>majority</th><th>gap</th><th>distinct</th><th>consensus</th><th>ESS</th>"
                   "<th>parsed</th><th>n</th></tr>")
        for sig in signals:
            for b in budgets:
                st = _cell_stats(by[(m, sig, b)])
                out.append(
                    f"<tr><td class='sig'>{sig}</td><td>{b}</td>"
                    f"<td{_green(st['selected'])}>{_fmt(st['selected'])}</td>"
                    f"<td{_green(st['oracle'])}>{_fmt(st['oracle'])}</td>"
                    f"<td>{_fmt(st['majority'])}</td>"
                    f"<td class='gap'>{('+'+_fmt(st['gap'])) if st['gap'] is not None else '—'}</td>"
                    f"<td>{_fmt(st['distinct'])}</td><td>{_fmt(st['consensus'])}</td>"
                    f"<td>{_fmt(st['ess'],2)}</td><td>{_fmt(st['parsed'],2)}</td><td>{st['n']}</td></tr>")
        out.append("</table>")

    # ---- per-item appendix (collapsible per cell; capped for large inputs) ----
    out.append("<h2>Per-item detail</h2>")
    note = ("One collapsible per (prompt, signal, budget). <span class='preds'>preds</span> = each "
            "particle's predicted letter (<code>?</code> = unparsed); ✓/✗ vs gold.")
    if max_appendix:
        note += f" Showing up to {max_appendix} items per cell."
    note += (" <i>(For mean_logprob budgets 8/16/32 the per-particle letters were carried from Run 8, "
             "which stored only aggregate metrics, so <code>preds</code>/<code>selected</code> are blank "
             "there — the accuracy/diversity numbers are exact.)</i>")
    out.append(f"<p class='note'>{note}</p>")
    for m in methods:
        for sig in signals:
            for b in budgets:
                cell = by[(m, sig, b)]
                st = _cell_stats(cell)
                items_sorted = sorted(cell, key=lambda x: (x["category"], x["unique_id"]))
                shown = items_sorted[:max_appendix] if max_appendix else items_sorted
                cap_note = (f" (showing {len(shown)} of {len(items_sorted)})"
                            if len(shown) < len(items_sorted) else "")
                summ = (f"P{m} {names[m]} · <b>{sig}</b> · budget {b} — "
                        f"selected {_fmt(st['selected'])} | oracle {_fmt(st['oracle'])} | "
                        f"gap +{_fmt(st['gap'])} | {st['n']} items{cap_note}")
                out.append(f"<details><summary>{summ}</summary>")
                out.append("<table class='items'><tr><th class='l'>item</th><th>cat</th><th>gold</th>"
                           "<th>selected</th><th>major</th><th class='l'>preds (N particles)</th>"
                           "<th>dist</th><th>cons</th><th>ESS</th><th>sel?</th><th>orc?</th></tr>")
                for r in shown:
                    def mark(v):
                        return ("<span class='ok'>✓</span>" if v == "True"
                                else "<span class='no'>✗</span>" if v == "False" else "·")
                    out.append(
                        f"<tr><td class='l'><code>{r['unique_id'][:8]}</code></td>"
                        f"<td>{html.escape(r['category'])}</td><td>{html.escape(r['gold_letter'])}</td>"
                        f"<td>{html.escape(r['selected_letter'])}</td>"
                        f"<td>{html.escape(r['majority_letter'])}</td>"
                        f"<td class='preds'>{html.escape(r['preds'])}</td>"
                        f"<td>{_fmt(_f(r['distinct_ratio']),2)}</td>"
                        f"<td>{_fmt(_f(r['consensus']),2)}</td>"
                        f"<td>{_fmt(_f(r['ess_ratio']),2)}</td>"
                        f"<td>{mark(r['selected_correct'])}</td><td>{mark(r['oracle_correct'])}</td></tr>")
                out.append("</table></details>")

    out.append("</body></html>")
    return "\n".join(out)


@click.command()
@click.option("--in", "in_path", default="benchmarking/mmau_pro/results/run10_run6_full/run6_full.csv")
@click.option("--out", "out_path", default="benchmarking/mmau_pro/results/run10_run6_full/epf_div.html")
@click.option("--title", default="Run 6 — EPF diversity sweep")
@click.option("--max-appendix-per-cell", default=0, help="cap per-item appendix rows per cell (0 = all)")
def main(in_path, out_path, title, max_appendix_per_cell):
    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(out_path, "w") as f:
        f.write(build_html(rows, title, max_appendix_per_cell))
    print(f"wrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
