"""Neat accuracy-vs-budget visualization for the full-957 EPF diversity sweep.

Source: results/run10_run6_full/run6_full.csv  (the full 957-MCQ run — "Run 10" in epf_bootstrap.py).
One line per (prompt, signal) cell -> 4 prompts x 2 self-certainty signals = 8 lines.
Colour encodes the prompt; line style / marker encodes the signal.
Budgets {1, 8, 16, 32} particles.

Error bars = analytic 957 standard error sqrt(p(1-p)/n) — this is exactly the SE_957
that epf_bootstrap.py reports the resampling std converges to (~+-0.016).

Emits two self-contained figures:
  results/plots/epf_acc_vs_budget.html  -- interactive (toggle prompt / signal / metric / error bars)
  results/plots/epf_acc_vs_budget.png   -- static publication figure (default metric, error bars)

    /home/exx/miniconda3/envs/epf/bin/python -m benchmarking.mmau_pro.plot_epf957
"""

import json
import math

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

PROMPT_ORDER = [4, 5, 7, 9]
BUDGET_ORDER = [1, 8, 16, 32]
SIGNALS = ["mean_logprob", "entropy"]
METRICS = [
    ("selected_correct", "Selected accuracy (what EPF returns)"),
    ("oracle_correct", "Oracle accuracy (answer in any particle)"),
    ("majority_correct", "Majority-vote accuracy"),
]
# colour-blind-friendly per prompt; style/marker per signal
PROMPT_COLORS = {4: "#1b9e77", 5: "#d95f02", 7: "#7570b3", 9: "#e7298a"}
SIGNAL_DASH = {"mean_logprob": "solid", "entropy": "dash"}
SIGNAL_SYMBOL = {"mean_logprob": "circle", "entropy": "square"}
SIGNAL_MPL = {  # matplotlib equivalents
    "mean_logprob": {"linestyle": "-", "marker": "o"},
    "entropy": {"linestyle": "--", "marker": "s"},
}


def _std100(p, n_total, m=100):
    """Std of accuracy across a random *m*-question eval drawn without replacement.

    sqrt(p(1-p)/m) * finite-population correction sqrt((N-m)/(N-1)) — i.e. "how noisy is a
    100-question eval" (~+-0.047). This is the std_100 closed form validated in epf_bootstrap.py.
    """
    if p is None or (isinstance(p, float) and math.isnan(p)) or n_total <= 1:
        return 0.0
    fpc = math.sqrt(max(0.0, n_total - m) / (n_total - 1))
    return math.sqrt(p * (1.0 - p) / m) * fpc


def load(in_path):
    """Return (short-name map, cells, n_total) with cells[(metric, prompt, signal)] = (ys, std100s, ns).

    n_total = distinct MCQs in the benchmark (957); per-cell ns is the gradeable count (952).
    std100 = std of accuracy across a random 100-question eval (the noise of a 100-item eval).
    """
    df = pd.read_csv(in_path)
    n_total = int(df["unique_id"].nunique())
    names = dict(df[["method", "method_name"]].drop_duplicates().values)
    short = {m: names[m].split(" (")[0] for m in names}
    cells = {}
    for metric_key, _ in METRICS:
        g = (df.groupby(["method", "signal", "budget"])[metric_key]
               .agg(acc="mean", n="count").reset_index())
        for m in PROMPT_ORDER:
            for sig in SIGNALS:
                sub = (g[(g.method == m) & (g.signal == sig)]
                       .set_index("budget").reindex(BUDGET_ORDER))
                ys = [None if pd.isna(v) else round(float(v), 4) for v in sub["acc"]]
                ns = [0 if pd.isna(v) else int(v) for v in sub["n"]]
                std100s = [round(_std100(y, n_total), 4) for y in ys]
                cells[(metric_key, m, sig)] = (ys, std100s, ns)
    return short, cells, n_total


def _metric_range(cells, metric_key):
    """Tight y-range with padding, computed from the data for this metric."""
    vals = [y for (mk, _, _), (ys, *_) in cells.items() if mk == metric_key
            for y in ys if y is not None]
    lo, hi = min(vals), max(vals)
    pad = 0.04
    return [max(0.0, round(lo - pad, 2)), min(1.0, round(hi + pad, 2))]


# --------------------------------------------------------------------------- HTML

def build_html(short, cells, n_total, n_grade):
    fig = go.Figure()
    info = []  # parallel to fig.data: {prompt, signal, metric}
    xcats = [str(b) for b in BUDGET_ORDER]

    for metric_key, _ in METRICS:
        for m in PROMPT_ORDER:
            for sig in SIGNALS:
                ys, std100s, ns = cells[(metric_key, m, sig)]
                fig.add_trace(go.Scatter(
                    x=xcats, y=ys, mode="lines+markers",
                    name=f"P{m} {short[m]} · {sig}",
                    legendgroup=f"P{m}",
                    line=dict(color=PROMPT_COLORS[m], dash=SIGNAL_DASH[sig], width=2.4),
                    marker=dict(symbol=SIGNAL_SYMBOL[sig], size=8, color=PROMPT_COLORS[m]),
                    customdata=[[n, s] for n, s in zip(ns, std100s)],
                    hovertemplate=(f"<b>P{m} {short[m]}</b> · {sig}<br>"
                                   "budget %{x} · acc %{y:.3f} ±%{customdata[1]:.3f} (std₁₀₀)<br>"
                                   "n=%{customdata[0]} items<extra></extra>"),
                    visible=(metric_key == "selected_correct"),
                ))
                info.append({"prompt": m, "signal": sig, "metric": metric_key})

    sel_range = _metric_range(cells, "selected_correct")
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"MMAU-Pro · EPF diversity sweep (full {n_total} MCQ) — accuracy vs budget",
                   x=0.5, xanchor="center", font=dict(size=17)),
        xaxis=dict(title="Budget (number of particles)", type="category",
                   categoryorder="array", categoryarray=xcats),
        yaxis=dict(title=METRICS[0][1], range=sel_range, tickformat=".2f"),
        showlegend=False, hovermode="closest",
        margin=dict(l=70, r=30, t=60, b=50), height=580,
    )

    plot_html = fig.to_html(include_plotlyjs=True, full_html=False, div_id="epfplot",
                            config={"displaylogo": False, "responsive": True})

    metric_meta = {k: {"ytitle": lab, "range": _metric_range(cells, k)} for k, lab in METRICS}

    metric_radios = "".join(
        f'<label class="chk"><input type="radio" name="metric" class="metric" value="{k}"'
        f'{" checked" if k=="selected_correct" else ""}>{lab.split(" accuracy")[0]}</label>'
        for k, lab in METRICS)
    prompt_boxes = "".join(
        f'<label class="chk"><input type="checkbox" class="prompt" value="{m}" checked>'
        f'<span class="swatch" style="background:{PROMPT_COLORS[m]}"></span>P{m} {short[m]}</label>'
        for m in PROMPT_ORDER)
    signal_boxes = "".join(
        f'<label class="chk"><input type="checkbox" class="signal" value="{sig}" checked>'
        f'<span class="line {("solid" if sig=="mean_logprob" else "dashed")}"></span>{sig}</label>'
        for sig in SIGNALS)

    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>EPF accuracy vs budget — full 957</title>
<style>
  :root {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
  body {{ margin: 22px; color:#1a1a1a; max-width: 1000px; }}
  h1 {{ font-size: 19px; margin: 0 0 2px; }}
  .sub {{ color:#666; font-size:13px; margin:0 0 14px; }}
  .panel {{ display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start;
            background:#fafafa; border:1px solid #eee; border-radius:8px;
            padding:12px 16px; margin-bottom:10px; }}
  .group {{ display:flex; flex-direction:column; gap:5px; }}
  .group .ttl {{ font-size:11px; font-weight:700; letter-spacing:.04em;
                 text-transform:uppercase; color:#888; margin-bottom:2px; }}
  .row {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .chk {{ font-size:13px; display:inline-flex; align-items:center; gap:6px; cursor:pointer; }}
  .swatch {{ width:13px; height:13px; border-radius:3px; display:inline-block; }}
  .line {{ width:22px; height:0; display:inline-block; border-top-width:3px; }}
  .line.solid {{ border-top-style:solid; border-color:#555; }}
  .line.dashed {{ border-top-style:dashed; border-color:#555; }}
  .btns {{ margin-left:auto; display:flex; gap:8px; align-self:center; }}
  button {{ font-size:12px; padding:4px 10px; border:1px solid #ccc; border-radius:6px;
            background:#fff; cursor:pointer; }}
  button:hover {{ background:#f0f0f0; }}
  .note {{ color:#888; font-size:12px; margin-top:6px; }}
</style></head>
<body>
  <h1>MMAU-Pro · EPF diversity sweep — full {n_total} MCQ</h1>
  <p class="sub">Accuracy vs budget · 4 prompts × 2 self-certainty signals = <b>8 lines</b> ·
     {n_total}-MCQ benchmark, n={n_grade} gradeable/cell · EPF, temp 0.8, systematic resampling.
     Hover a point for value, n and ±std₁₀₀ = √(p(1−p)/100)·FPC ≈ the noise of a random 100-question eval.</p>

  <div class="panel">
    <div class="group"><span class="ttl">Metric</span><div class="row">{metric_radios}</div></div>
    <div class="group"><span class="ttl">Prompt</span><div class="row">{prompt_boxes}</div></div>
    <div class="group"><span class="ttl">Signal</span><div class="row">{signal_boxes}</div></div>
    <div class="btns">
      <button id="all">All on</button>
      <button id="none">All off</button>
    </div>
  </div>

  {plot_html}
  <p class="note">colour = prompt · solid line / ○ = mean_logprob · dashed line / □ = entropy.
     A trace shows only if its prompt <b>and</b> signal are checked for the selected metric.</p>

<script>
  const INFO = {json.dumps(info)};
  const METRIC_META = {json.dumps(metric_meta)};
  const gd = document.getElementById('epfplot');
  const setOf = c => new Set(Array.from(document.querySelectorAll('input.'+c+':checked')).map(e=>e.value));

  function apply() {{
    const metric  = document.querySelector('input.metric:checked').value;
    const prompts = setOf('prompt');
    const signals = setOf('signal');
    const vis = INFO.map(t =>
      t.metric === metric && prompts.has(String(t.prompt)) && signals.has(t.signal));
    Plotly.restyle(gd, {{visible: vis}});
    const mm = METRIC_META[metric];
    Plotly.relayout(gd, {{'yaxis.title.text': mm.ytitle, 'yaxis.range': mm.range}});
  }}
  document.querySelectorAll('input').forEach(e => e.addEventListener('change', apply));
  document.getElementById('all').addEventListener('click', () => {{
    document.querySelectorAll('input.prompt, input.signal').forEach(e => e.checked = true); apply();
  }});
  document.getElementById('none').addEventListener('click', () => {{
    document.querySelectorAll('input.prompt, input.signal').forEach(e => e.checked = false); apply();
  }});
  apply();
</script>
</body></html>"""
    return page


# --------------------------------------------------------------------------- PNG

def build_png(short, cells, out_path, metric_key, n_total, n_grade):
    ylab = dict(METRICS)[metric_key]
    x = list(range(len(BUDGET_ORDER)))  # even categorical spacing
    fig, ax = plt.subplots(figsize=(8.6, 6), dpi=170)

    for m in PROMPT_ORDER:
        for sig in SIGNALS:
            ys, std100s, _ = cells[(metric_key, m, sig)]
            label = f"P{m} {short[m]} · {sig}"
            ax.errorbar(x, ys, yerr=std100s, color=PROMPT_COLORS[m], label=label,
                        linewidth=1.9, markersize=6, capsize=2.5, elinewidth=0.8,
                        alpha=0.95, **SIGNAL_MPL[sig])

    rng = _metric_range(cells, metric_key)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGET_ORDER])
    ax.set_xlabel("Budget (number of particles)")
    ax.set_ylabel(ylab)
    ax.set_ylim(*rng)
    ax.set_title(f"MMAU-Pro · EPF diversity sweep (full {n_total} MCQ)\n"
                 f"accuracy vs budget — 4 prompts × 2 self-certainty signals "
                 f"(n={n_grade} gradeable/cell, ±std₁₀₀)", fontsize=12)
    ax.grid(True, axis="both", linestyle=":", alpha=0.4)
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              framealpha=0.95, columnspacing=1.2, handlelength=2.4,
              title="colour = prompt   ·   solid / ○ = mean_logprob   ·   dashed / □ = entropy")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option("--in", "in_path", default="benchmarking/mmau_pro/results/run10_run6_full/run6_full.csv")
@click.option("--html-out", default="benchmarking/mmau_pro/results/plots/epf_acc_vs_budget.html")
@click.option("--png-out", default="benchmarking/mmau_pro/results/plots/epf_acc_vs_budget.png")
@click.option("--png-metric", default="selected_correct",
              type=click.Choice([k for k, _ in METRICS]),
              help="Which metric the static PNG shows (default: selected).")
def main(in_path, html_out, png_out, png_metric):
    short, cells, n_total = load(in_path)
    n_grade = max(n for (_, _, _), (_, _, ns) in cells.items() for n in ns)

    with open(html_out, "w") as f:
        f.write(build_html(short, cells, n_total, n_grade))
    print(f"wrote {html_out}")

    build_png(short, cells, png_out, png_metric, n_total, n_grade)
    print(f"wrote {png_out}  (metric={png_metric})")


if __name__ == "__main__":
    main()
