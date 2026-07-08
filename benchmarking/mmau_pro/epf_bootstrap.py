"""Bootstrap the full-957 EPF accuracy numbers (Run 10) to quantify robustness.

Purely offline: we already have per-item correctness (selected/oracle/majority_correct) for every
cell (prompt x signal x budget) in run6_full.csv, so this just resamples those precomputed scores
(no model, no GPU — vectorized numpy, <1 s). For each cell & metric we report two things:

  * 100-sample std  : draw 100 questions WITHOUT replacement from the 957, recompute accuracy,
                      n times -> std = "how noisy is a 100-question eval" (~+-0.05).
  * full-957 SE     : resample 957 WITH replacement (standard bootstrap), n times -> std = SE of the
                      actually-reported number (~+-0.016). Converges to sqrt(p(1-p)/957).

Output: results/run10_run6_full/epf_div_bootstrap.html  (per-prompt tables: point +-std100 (+-SE957)).

    python -m benchmarking.mmau_pro.epf_bootstrap --n 10000 \
        --in benchmarking/mmau_pro/results/run10_run6_full/run6_full.csv \
        --out benchmarking/mmau_pro/results/run10_run6_full/epf_div_bootstrap.html
"""

import csv
import html
import json
import math
from collections import defaultdict

import click
import numpy as np
import plotly.graph_objects as go

from benchmarking.mmau_pro.epf_report import _CSS

METRICS = ["selected", "oracle", "majority"]
PROMPT_ORDER = [4, 5, 7, 9]
BUDGET_ORDER = [1, 8, 16, 32]
NAMES = {4: "plan-and-solve", 5: "least-to-most", 7: "format-forcing", 9: "evidence-grounded"}
PROMPT_COLORS = {4: "#1b9e77", 5: "#d95f02", 7: "#7570b3", 9: "#e7298a"}
SIGNAL_DASH = {"mean_logprob": "solid", "entropy": "dash"}
SIGNAL_SYMBOL = {"mean_logprob": "circle", "entropy": "square"}
METRIC_LABELS = {"selected": "Selected (what EPF returns)",
                 "oracle": "Oracle (answer in any particle)",
                 "majority": "Majority vote"}


def _to01(v):
    return 1.0 if v == "True" else 0.0 if v == "False" else np.nan  # ungradeable -> NaN


def load_cells(path):
    """cell (method,signal,budget) -> {metric: float array aligned to a fixed item order}."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    items = sorted({r["unique_id"] for r in rows})
    idx = {u: i for i, u in enumerate(items)}
    cells = defaultdict(lambda: {m: np.full(len(items), np.nan) for m in METRICS})
    for r in rows:
        key = (int(r["method"]), r["signal"], int(r["budget"]))
        i = idx[r["unique_id"]]
        for m in METRICS:
            cells[key][m][i] = _to01(r[f"{m}_correct"])
    graded = {r["unique_id"] for r in rows if r.get("selected_correct") in ("True", "False")}
    return cells, len(items), len(graded)


def bootstrap(arr, idx_matrix):
    """Mean accuracy over each resample (NaN = ungradeable, excluded from the denominator)."""
    gathered = arr[idx_matrix]                  # (n_boot, sample_size)
    return np.nanmean(gathered, axis=1)         # (n_boot,)


def _fmt(x, nd=3):
    return f"{x:.{nd}f}"


def build_plot_block(stats, n_items):
    """Interactive acc-vs-budget plot: 8 lines (4 prompts x 2 signals), toggleable per
    prompt and per signal, with a metric radio. Error bars = the bootstrap SE_full of
    each cell (the same +-SE reported in the tables below). Plotly.js inlined."""
    signals = ["mean_logprob", "entropy"]
    budgets = [b for b in BUDGET_ORDER if any(k[2] == b for k in stats)]
    xcats = [str(b) for b in budgets]

    fig = go.Figure()
    trace_info = []
    for metric in METRICS:
        for m in PROMPT_ORDER:
            for sig in signals:
                ys, ses = [], []
                for b in budgets:
                    s = stats.get((m, sig, b), {}).get(metric)
                    ys.append(round(s["point"], 4) if s else None)
                    ses.append(round(s["se957"], 4) if s else 0.0)
                fig.add_trace(go.Scatter(
                    x=xcats, y=ys, mode="lines+markers",
                    name=f"P{m} {NAMES[m]} · {sig}",
                    line=dict(color=PROMPT_COLORS[m], dash=SIGNAL_DASH[sig], width=2.4),
                    marker=dict(symbol=SIGNAL_SYMBOL[sig], size=8, color=PROMPT_COLORS[m]),
                    error_y=dict(type="data", array=ses, visible=True, thickness=1.2, width=3),
                    customdata=ses,
                    hovertemplate=(f"<b>P{m} {NAMES[m]}</b> · {sig}<br>"
                                   "budget %{x} · acc %{y:.3f} ±%{customdata:.3f} (SE)"
                                   "<extra></extra>"),
                    visible=(metric == "selected"),
                ))
                trace_info.append({"prompt": m, "signal": sig, "metric": metric})

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Budget (number of particles)", type="category",
                   categoryorder="array", categoryarray=xcats),
        yaxis=dict(title=METRIC_LABELS["selected"] + " accuracy",
                   range=[0.50, 0.70], tickformat=".2f"),
        showlegend=False, hovermode="closest",
        margin=dict(l=70, r=30, t=20, b=50), height=520,
    )
    plot_html = fig.to_html(include_plotlyjs=True, full_html=False, div_id="epfplot",
                            config={"displaylogo": False, "responsive": True})

    metric_meta = {m: {"ytitle": METRIC_LABELS[m] + " accuracy",
                       "range": [0.50, 1.0] if m == "oracle" else [0.50, 0.70]}
                   for m in METRICS}
    prompt_boxes = "".join(
        f'<label class="chk"><input type="checkbox" class="prompt" value="{m}" checked>'
        f'<span class="swatch" style="background:{PROMPT_COLORS[m]}"></span>'
        f'P{m} {NAMES[m]}</label>'
        for m in PROMPT_ORDER)
    signal_boxes = "".join(
        f'<label class="chk"><input type="checkbox" class="signal" value="{sig}" checked>'
        f'<span class="pline {("solid" if sig == "mean_logprob" else "dashed")}"></span>'
        f'{sig}</label>'
        for sig in signals)
    metric_radios = "".join(
        f'<label class="chk"><input type="radio" name="metric" class="metric" value="{m}"'
        f'{" checked" if m == "selected" else ""}>{METRIC_LABELS[m]}</label>'
        for m in METRICS)

    return f"""
<h2>Accuracy vs budget (toggle prompts / signals)</h2>
<style>
  .panel {{ display:flex; flex-wrap:wrap; gap:26px; align-items:flex-start;
            background:#fafafa; border:1px solid #eee; border-radius:8px;
            padding:12px 16px; margin:8px 0 4px; }}
  .group {{ display:flex; flex-direction:column; gap:5px; }}
  .group .ttl {{ font-size:11px; font-weight:700; letter-spacing:.04em;
                 text-transform:uppercase; color:#888; margin-bottom:2px; }}
  .prow {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .chk {{ font-size:13px; display:inline-flex; align-items:center; gap:6px; cursor:pointer; }}
  .swatch {{ width:13px; height:13px; border-radius:3px; display:inline-block; }}
  .pline {{ width:22px; height:0; display:inline-block; border-top-width:3px; }}
  .pline.solid {{ border-top-style:solid; border-color:#555; }}
  .pline.dashed {{ border-top-style:dashed; border-color:#555; }}
  .btns {{ margin-left:auto; display:flex; gap:8px; align-self:center; }}
  .btns button {{ font-size:12px; padding:4px 10px; border:1px solid #ccc; border-radius:6px;
                  background:#fff; cursor:pointer; }}
  .btns button:hover {{ background:#f0f0f0; }}
</style>
<div class="panel">
  <div class="group"><span class="ttl">Metric</span><div class="prow">{metric_radios}</div></div>
  <div class="group"><span class="ttl">Prompt</span><div class="prow">{prompt_boxes}</div></div>
  <div class="group"><span class="ttl">Signal</span><div class="prow">{signal_boxes}</div></div>
  <div class="btns"><button id="all" type="button">All on</button>
                    <button id="none" type="button">All off</button></div>
</div>
{plot_html}
<p class="meta">colour = prompt · solid / ○ = mean_logprob · dashed / □ = entropy ·
error bars = bootstrap ±SE<sub>{n_items}</sub> (same as the tables below). A line shows only if
its prompt <b>and</b> signal are checked and its metric is selected.</p>
<script>
  const TRACE_INFO = {json.dumps(trace_info)};
  const METRIC_META = {json.dumps(metric_meta)};
  const gd = document.getElementById('epfplot');
  function selMetric() {{ return document.querySelector('input.metric:checked').value; }}
  function checkedSet(cls) {{
    return new Set(Array.from(document.querySelectorAll('input.' + cls + ':checked'))
                        .map(e => e.value));
  }}
  function apply() {{
    const metric = selMetric(), prompts = checkedSet('prompt'), signals = checkedSet('signal');
    const vis = TRACE_INFO.map(t =>
      (t.metric === metric && prompts.has(String(t.prompt)) && signals.has(t.signal)));
    Plotly.restyle(gd, {{visible: vis}});
    const mm = METRIC_META[metric];
    Plotly.relayout(gd, {{'yaxis.title.text': mm.ytitle, 'yaxis.range': mm.range}});
  }}
  document.querySelectorAll('input.prompt, input.signal, input.metric')
          .forEach(e => e.addEventListener('change', apply));
  document.getElementById('all').addEventListener('click', () => {{
    document.querySelectorAll('input.prompt, input.signal').forEach(e => e.checked = true);
    apply();
  }});
  document.getElementById('none').addEventListener('click', () => {{
    document.querySelectorAll('input.prompt, input.signal').forEach(e => e.checked = false);
    apply();
  }});
  apply();
</script>
"""


def build_html(stats, n_boot, n_items, n_graded, sub_m, checks):
    methods = sorted({k[0] for k in stats})
    methods = [m for m in PROMPT_ORDER if m in methods]
    signals = sorted({k[1] for k in stats})
    budgets = [b for b in BUDGET_ORDER if b in {k[2] for k in stats}]

    # headline: mean std across cells
    mean_s100 = np.mean([stats[k]["selected"]["std100"] for k in stats])
    mean_se957 = np.mean([stats[k]["selected"]["se957"] for k in stats])

    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>EPF accuracy bootstrap</title><style>{_CSS}"
        " .s1{color:#b06000;} .se{color:#137333;} td.cellb{min-width:150px;}"
        " .leg{font-size:12.5px;color:#444;background:#f7f7f7;border:1px solid #e3e3e3;"
        "border-radius:6px;padding:10px 14px;margin:8px 0 14px;}</style></head><body>",
        f"<h1>EPF grid accuracy — bootstrap robustness (full {n_items:,} MCQ)</h1>",
        f"<p class='meta'>n = {n_boot:,} resamples · {n_items:,} items ({n_graded:,} gradeable) · "
        "EPF, temp 0.8, mean_logprob &amp; entropy.</p>",
        f"<div class='leg'>Each cell shows <b>point</b> = full-{n_items:,} accuracy, "
        f"<span class='s1'>±std<sub>{sub_m}</sub></span> = std of accuracy across "
        f"{n_boot:,} random <b>{sub_m}-question</b> samples (drawn without replacement — "
        f"\"how noisy is a {sub_m}-question eval\"), and "
        f"<span class='se'>(±SE<sub>{n_items}</sub>)</span> = std across "
        f"{n_boot:,} <b>{n_items:,}</b>-resamples with replacement = the <b>error bar on the reported "
        f"number</b>. Read as <b>point</b> <span class='s1'>±std<sub>{sub_m}</sub></span> "
        f"<span class='se'>(±SE<sub>{n_items}</sub>)</span>.</div>",
        f"<p class='meta'><b>Headline (selected acc):</b> a {sub_m}-question eval wobbles "
        f"<span class='s1'>±{mean_s100:.3f}</span> on average, but the reported full-{n_items:,} numbers are "
        f"precise to <span class='se'>±{mean_se957:.3f}</span> (~{mean_s100/mean_se957:.1f}&times; tighter).</p>",
    ]
    # closed-form cross-check note
    out.append("<p class='meta'>Validation vs closed form (binomial ± FPC): " +
               "; ".join(checks) + "</p>")

    out.append(build_plot_block(stats, n_items))

    for m in methods:
        out.append(f"<h2>P{m} — {html.escape(NAMES[m])}</h2>")
        out.append("<table class='matrix'><tr><th class='l'>signal</th><th>budget</th>"
                   + "".join(f"<th>{mt}</th>" for mt in METRICS) + "</tr>")
        for sig in signals:
            for b in budgets:
                cells = ""
                for mt in METRICS:
                    s = stats[(m, sig, b)][mt]
                    cells += (f"<td class='cellb'><b>{_fmt(s['point'])}</b> "
                              f"<span class='s1'>±{_fmt(s['std100'])}</span> "
                              f"<span class='se'>(±{_fmt(s['se957'])})</span></td>")
                out.append(f"<tr><td class='l'>{sig}</td><td>{b}</td>{cells}</tr>")
        out.append("</table>")
    out.append("</body></html>")
    return "\n".join(out)


@click.command()
@click.option("--in", "in_path", default="benchmarking/mmau_pro/results/run10_run6_full/run6_full.csv")
@click.option("--out", "out_path", default="benchmarking/mmau_pro/results/run10_run6_full/epf_div_bootstrap.html")
@click.option("--n", "n_boot", default=10000, help="number of bootstrap resamples")
@click.option("--subsample", "sub_m", default=100, help="subsample size (questions) for std100")
@click.option("--seed", default=0)
def main(in_path, out_path, n_boot, sub_m, seed):
    cells, n_items, n_graded = load_cells(in_path)
    rng = np.random.default_rng(seed)
    # shared index matrices (paired across cells): without-replacement size sub_m, with-replacement size N
    idx_sub = rng.random((n_boot, n_items)).argsort(axis=1)[:, :sub_m]   # unique per row
    idx_full = rng.integers(0, n_items, size=(n_boot, n_items))          # with replacement

    stats = {}
    checks = []
    for key in cells:
        stats[key] = {}
        for mt in METRICS:
            arr = cells[key][mt]
            point = float(np.nanmean(arr))
            b100 = bootstrap(arr, idx_sub)
            b957 = bootstrap(arr, idx_full)
            stats[key][mt] = {
                "point": point,
                "std100": float(np.std(b100)), "ci100": (float(np.percentile(b100, 2.5)), float(np.percentile(b100, 97.5))),
                "se957": float(np.std(b957)), "ci957": (float(np.percentile(b957, 2.5)), float(np.percentile(b957, 97.5))),
            }

    # closed-form cross-check on a few selected-acc cells
    fpc = math.sqrt((n_items - sub_m) / (n_items - 1))
    for key in [(4, "mean_logprob", 8), (9, "entropy", 32), (5, "mean_logprob", 16)]:
        s = stats[key]["selected"]
        p = s["point"]
        cf100 = math.sqrt(p * (1 - p) / sub_m) * fpc
        cf957 = math.sqrt(p * (1 - p) / n_items)
        checks.append(f"P{key[0]} {key[1]} b{key[2]}: std{sub_m} {s['std100']:.4f}≈{cf100:.4f}, "
                      f"SE{n_items} {s['se957']:.4f}≈{cf957:.4f}")
        print(f"[check] {checks[-1]}")

    with open(out_path, "w") as f:
        f.write(build_html(stats, n_boot, n_items, n_graded, sub_m, checks))
    print(f"wrote {out_path}  ({len(cells)} cells, n={n_boot})")


if __name__ == "__main__":
    main()
