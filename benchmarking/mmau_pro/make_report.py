"""Render a cot_compare CSV into side-by-side HTML reports (prompts as columns).

Per question: a horizontally-scrollable table whose columns are the prompts and whose cells
are that prompt's numbered reasoning steps, color-coded green/red by correctness with the
predicted vs gold letter. A summary section gives per-prompt accuracy (overall + excluding
trivial single-choice items) and a prompt x category matrix (best per category highlighted).

Two modes:
  --out FILE                       single self-contained HTML (good for small sets)
  --out-dir DIR --paginate category   index.html (summary + links) + one file per category
                                       (good for the full 957 set, which is too big for one file)

    python -m benchmarking.mmau_pro.make_report \
        --in benchmarking/mmau_pro/results/run05_cot957/cot957.csv \
        --out-dir benchmarking/mmau_pro/results/run05_cot957/cot957_html --paginate category
"""

import csv
import html
import os
import re
from collections import defaultdict

import click

from benchmarking.mmau_pro.explode_steps import _as_paragraph, split_steps

_CSS = """
:root { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
body { margin: 24px; color: #1a1a1a; background: #fff; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin: 28px 0 6px; } h3 { font-size:14px; margin:18px 0 4px; }
.meta { color:#555; font-size:13px; margin:2px 0; }
.opts { color:#333; font-size:13px; margin:2px 0 8px; }
.legend { font-size:12px; color:#666; margin:6px 0 12px; }
.scroll { overflow-x:auto; border:1px solid #ddd; border-radius:6px; }
table.qa { border-collapse:collapse; width:max-content; }
table.qa th, table.qa td { min-width:300px; max-width:340px; vertical-align:top; padding:8px 10px;
         border:1px solid #ececec; font-size:13px; line-height:1.4; }
table.qa th { position:sticky; top:0; background:#fafafa; text-align:left; z-index:1; }
table.qa th .name { font-weight:600; } table.qa th .badge { float:right; font-weight:700; }
table.qa th.correct { background:#e7f6ec; } table.qa th.wrong { background:#fdeaea; }
table.qa td.correct { background:#f4fbf6; } table.qa td.wrong { background:#fef6f6; }
.ok { color:#137333; } .no { color:#c5221f; }
ol { margin:0; padding-left:18px; } li { margin:3px 0; }
table.matrix { border-collapse:collapse; margin:6px 0 4px; }
table.matrix th, table.matrix td { font-size:12px; padding:4px 9px; text-align:center; border:1px solid #e6e6e6; }
table.matrix td.l, table.matrix th.l { text-align:left; }
table.matrix td.best { background:#d6efde; font-weight:700; }
ul.cats { font-size:13px; line-height:1.7; }
"""


def _steps_for_display(method: str, text: str) -> list[str]:
    """Per-prompt step list for the report cells.

    Method 9 (evidence-grounded, boxed) writes ``Reasoning:`` then single-newline
    ``Step 1: ... Step 2: ...``. For a tidy display we drop the ``Reasoning:`` header
    and split on each ``Step N`` marker so every bullet reads ``Step N: ...``. Other
    prompts use the standard \\n\\n step splitter.
    """
    if str(method) == "9":
        t = re.sub(r"(?i)^\s*reasoning\s*:\s*", "", _as_paragraph(text))  # drop header
        parts = [p.strip() for p in re.split(r"(?=\bStep\s*\d+\b)", t) if p.strip()]
        return parts or [t]
    return [_as_paragraph(s) for s in split_steps(text)]


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "cat"


def _acc(rows, pred=None):
    """(accuracy or None, n_graded) over gradeable rows optionally filtered by `pred`."""
    g = [r for r in rows if r["correct"] in ("True", "False") and (pred is None or pred(r))]
    if not g:
        return None, 0
    return sum(1 for r in g if r["correct"] == "True") / len(g), len(g)


def _fmt(a):
    return f"{a:.3f}" if a is not None else "—"


def _page(title: str, body: str) -> str:
    return "\n".join([
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        body,
        "</body></html>",
    ])


def _legend(methods, names) -> str:
    return "<div class='legend'>" + " &nbsp;·&nbsp; ".join(
        f"<b>P{m}</b> {html.escape(names[m])}" for m in methods
    ) + "</div>"


def _by_method(rows):
    by_m = defaultdict(list)
    for r in rows:
        by_m[int(r["method"])].append(r)
    return by_m


def _summary_html(rows) -> str:
    """Per-prompt accuracy (all + excl single-choice) and a prompt x category matrix."""
    by_m = _by_method(rows)
    methods = sorted(by_m)
    names = {m: by_m[m][0]["method_name"] for m in methods}

    not_trivial = lambda r: str(r.get("n_choices", "2")) != "1"  # noqa: E731
    stats = {m: (_acc(by_m[m]), _acc(by_m[m], not_trivial)) for m in methods}
    best_m = max(methods, key=lambda m: (stats[m][1][0] if stats[m][1][0] is not None else -1))

    th = "<tr><th class='l'>prompt</th>" + "".join(f"<th>P{m}</th>" for m in methods) + "</tr>"
    r_all = "<tr><td class='l'>acc (all)</td>" + "".join(f"<td>{_fmt(stats[m][0][0])}</td>" for m in methods) + "</tr>"
    r_ex = "<tr><td class='l'>acc (excl. 1-choice)</td>" + "".join(
        f"<td class='{'best' if m == best_m else ''}'>{_fmt(stats[m][1][0])}</td>" for m in methods) + "</tr>"
    r_g = "<tr><td class='l'>graded</td>" + "".join(f"<td>{stats[m][0][1]}</td>" for m in methods) + "</tr>"
    ptable = f"<table class='matrix'>{th}{r_all}{r_ex}{r_g}</table>"

    cat_items = {}
    for r in rows:
        cat_items.setdefault(r["category"], set()).add(r["unique_id"])
    cats = sorted(cat_items, key=lambda c: -len(cat_items[c]))
    head = "<tr><th class='l'>category</th><th>n</th>" + "".join(f"<th>P{m}</th>" for m in methods) + "</tr>"
    mbody = ""
    for c in cats:
        accs = [_acc([r for r in by_m[m] if r["category"] == c])[0] for m in methods]
        top = max((a for a in accs if a is not None), default=None)
        tds = "".join(
            f"<td class='{'best' if (a is not None and a == top) else ''}'>{_fmt(a)}</td>"
            for a in accs
        )
        mbody += f"<tr><td class='l'>{html.escape(c)}</td><td>{len(cat_items[c])}</td>{tds}</tr>"
    mtable = f"<table class='matrix'>{head}{mbody}</table>"

    return (
        _legend(methods, names)
        + "<h2>Summary</h2>"
        + f"<p class='meta'><b>Best prompt (accuracy excl. single-choice): "
          f"P{best_m} {html.escape(names[best_m])} = {_fmt(stats[best_m][1][0])}</b></p>"
        + ptable
        + "<h3>Accuracy by category (best per row highlighted)</h3>"
        + mtable
    )


def _questions_html(rows) -> str:
    """The side-by-side per-question tables (prompts as columns)."""
    by_m = _by_method(rows)
    methods = sorted(by_m)
    names = {m: by_m[m][0]["method_name"] for m in methods}

    items, order = {}, []
    for r in rows:
        if r["unique_id"] not in items:
            items[r["unique_id"]] = []
            order.append(r["unique_id"])
        items[r["unique_id"]].append(r)

    out = [
        _legend(methods, names),
        "<p class='meta'>Columns = the CoT prompts, side by side; each cell lists that prompt's "
        "reasoning steps. Header badge = predicted letter; green = correct, red = wrong.</p>",
    ]
    for i, uid in enumerate(order, 1):
        mrows = sorted(items[uid], key=lambda r: int(r["method"]))
        r0 = mrows[0]
        out.append(f"<h2>Q{i}. {html.escape(r0['question'])}</h2>")
        out.append(
            f"<div class='meta'>item <code>{uid[:8]}</code> &nbsp;|&nbsp; "
            f"category <b>{html.escape(r0['category'])}</b> &nbsp;|&nbsp; "
            f"gold <b>{html.escape(r0['gold_letter'])}</b> "
            f"({html.escape(str(r0.get('gold_answer', '')))})</div>"
        )
        if r0.get("choices"):
            out.append(f"<div class='opts'>{html.escape(r0['choices'])}</div>")
        ths, tds = [], []
        for r in mrows:
            ok = r["correct"] == "True"
            cls = "correct" if ok else "wrong"
            if r.get("error"):
                badge = "<span class='badge no'>err</span>"
            else:
                sym = "✓" if ok else "✗"
                badge = f"<span class='badge {'ok' if ok else 'no'}'>{sym} {html.escape(r['predicted_letter'] or '—')}</span>"
            ths.append(
                f"<th class='{cls}'><span class='name'>P{r['method']} "
                f"{html.escape(r['method_name'])}</span>{badge}</th>"
            )
            if r.get("error"):
                body = f"<li><i>error: {html.escape(r['error'])}</i></li>"
            else:
                steps = [html.escape(s) for s in _steps_for_display(r["method"], r["response"])]
                body = "".join(f"<li>{s}</li>" for s in steps) or "<li><i>(empty)</i></li>"
            tds.append(f"<td class='{cls}'><ol>{body}</ol></td>")
        out.append(
            "<div class='scroll'><table class='qa'><thead><tr>"
            + "".join(ths) + "</tr></thead><tbody><tr>" + "".join(tds) + "</tr></tbody></table></div>"
        )
    return "\n".join(out)


def _write_paginated(rows, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    by_m = _by_method(rows)
    methods = sorted(by_m)
    cat_items = {}
    for r in rows:
        cat_items.setdefault(r["category"], set()).add(r["unique_id"])
    cats = sorted(cat_items, key=lambda c: -len(cat_items[c]))

    links = ["<h2>Per-category responses</h2><ul class='cats'>"]
    for c in cats:
        fname = f"cot957_{_safe(c)}.html"
        accs = {m: _acc([r for r in by_m[m] if r["category"] == c])[0] for m in methods}
        bm = max(methods, key=lambda m: (accs[m] if accs[m] is not None else -1))
        links.append(
            f"<li><a href='{fname}'>{html.escape(c)}</a> — {len(cat_items[c])} items "
            f"(best: P{bm} {_fmt(accs[bm])})</li>"
        )
        cat_rows = [r for r in rows if r["category"] == c]
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(_page(f"{title} — {c}", _questions_html(cat_rows)))
    links.append("</ul>")

    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(_page(title, _summary_html(rows) + "\n".join(links)))
    print(f"wrote index.html + {len(cats)} category files -> {out_dir}")


@click.command()
@click.option("--in", "in_path", default="benchmarking/mmau_pro/results/cot_compare_5.csv")
@click.option("--out", "out_path", default="benchmarking/mmau_pro/results/cot_compare_5_compare.html")
@click.option("--out-dir", "out_dir", default=None, help="paginated output directory (with --paginate)")
@click.option("--paginate", type=click.Choice(["none", "category"]), default="none")
@click.option("--title", default="CoT prompt comparison")
def main(in_path, out_path, out_dir, paginate, title):
    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f))
    n_items = len({r["unique_id"] for r in rows})
    n_methods = len({r["method"] for r in rows})
    full_title = f"{title} ({n_items} items x {n_methods} prompts)"

    if out_dir and paginate == "category":
        _write_paginated(rows, out_dir, full_title)
    else:
        with open(out_path, "w") as f:
            f.write(_page(full_title, _summary_html(rows) + _questions_html(rows)))
        print(f"wrote {out_path}  ({n_items} items, {n_methods} prompts)")


if __name__ == "__main__":
    main()
