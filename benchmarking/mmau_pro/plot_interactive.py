"""Build a self-contained INTERACTIVE html of EPF accuracy vs budget (Run 6).

One trace per (prompt, signal, metric). The page has checkbox filters for prompt
and signal, plus a metric selector (selected / oracle / majority). Plotly.js is
embedded inline, so the file works offline (just open it in a browser).

    python -m benchmarking.mmau_pro.plot_interactive \
        --in benchmarking/mmau_pro/results/run06_epf_div/epf_div.csv \
        --out benchmarking/mmau_pro/results/plots/acc_vs_budget.html
"""

import json
import math

import click
import pandas as pd
import plotly.graph_objects as go

PROMPT_ORDER = [4, 5, 7, 9]
BUDGET_ORDER = [1, 8, 16, 32, 64, 128]
SIGNALS = ["mean_logprob", "entropy"]
METRICS = [
    ("selected_correct", "Selected (what EPF returns)"),
    ("oracle_correct", "Oracle (answer in any particle)"),
    ("majority_correct", "Majority vote"),
]
PROMPT_COLORS = {4: "#1b9e77", 5: "#d95f02", 7: "#7570b3", 9: "#e7298a"}
SIGNAL_DASH = {"mean_logprob": "solid", "entropy": "dash"}
SIGNAL_SYMBOL = {"mean_logprob": "circle", "entropy": "square"}


def _wilson_halfwidth(p, n, z=1.96):
    if n == 0 or p is None or (isinstance(p, float) and math.isnan(p)):
        return 0.0
    denom = 1 + z * z / n
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return margin


@click.command()
@click.option("--in", "in_path", default="benchmarking/mmau_pro/results/run06_epf_div/epf_div.csv")
@click.option("--out", "out_path", default="benchmarking/mmau_pro/results/plots/acc_vs_budget.html")
def main(in_path, out_path):
    df = pd.read_csv(in_path)
    names = dict(df[["method", "method_name"]].drop_duplicates().values)
    short = {m: names[m].split(" (")[0] for m in names}

    fig = go.Figure()
    trace_info = []  # parallel to fig.data: {prompt, signal, metric}
    xcats = [str(b) for b in BUDGET_ORDER]

    for metric_key, _ in METRICS:
        g = (df.groupby(["method", "signal", "budget"])[metric_key]
               .agg(acc="mean", n="count").reset_index())
        for m in PROMPT_ORDER:
            for sig in SIGNALS:
                sub = (g[(g.method == m) & (g.signal == sig)]
                       .set_index("budget").reindex(BUDGET_ORDER))
                ys = [None if pd.isna(v) else round(float(v), 4) for v in sub["acc"]]
                ns = [0 if pd.isna(v) else int(v) for v in sub["n"]]
                hw = [round(_wilson_halfwidth(y, n), 4) for y, n in zip(ys, ns)]
                visible = (metric_key == "selected_correct")
                fig.add_trace(go.Scatter(
                    x=xcats, y=ys, mode="lines+markers",
                    name=f"P{m} {short[m]} · {sig}",
                    legendgroup=f"P{m}",
                    line=dict(color=PROMPT_COLORS[m], dash=SIGNAL_DASH[sig], width=2.4),
                    marker=dict(symbol=SIGNAL_SYMBOL[sig], size=8,
                                color=PROMPT_COLORS[m]),
                    customdata=[[n, h] for n, h in zip(ns, hw)],
                    hovertemplate=(f"<b>P{m} {short[m]}</b> · {sig}<br>"
                                   "budget %{x} · acc %{y:.3f}"
                                   " ±%{customdata[1]:.3f}<br>"
                                   "n=%{customdata[0]}<extra></extra>"),
                    visible=visible,
                ))
                trace_info.append({"prompt": m, "signal": sig, "metric": metric_key})

    fig.update_layout(
        template="plotly_white",
        title=dict(text="MMAU-Pro · EPF diversity sweep (Run 6) — accuracy vs budget",
                   x=0.5, xanchor="center", font=dict(size=17)),
        xaxis=dict(title="Budget (number of particles)", type="category",
                   categoryorder="array", categoryarray=xcats),
        yaxis=dict(title="Selected accuracy (what EPF returns)", range=[0.30, 0.70],
                   tickformat=".2f"),
        showlegend=False, hovermode="closest",
        margin=dict(l=70, r=30, t=60, b=50), height=560,
    )

    plot_html = fig.to_html(include_plotlyjs=True, full_html=False, div_id="epfplot",
                            config={"displaylogo": False, "responsive": True})

    metric_meta = {k: {"label": lab,
                       "ytitle": {"selected_correct": "Selected accuracy (what EPF returns)",
                                  "oracle_correct": "Oracle accuracy (answer in any particle)",
                                  "majority_correct": "Majority-vote accuracy"}[k],
                       "range": [0.30, 1.0] if k == "oracle_correct" else [0.30, 0.70]}
                   for k, lab in METRICS}

    prompt_boxes = "".join(
        f'<label class="chk"><input type="checkbox" class="prompt" value="{m}" checked>'
        f'<span class="swatch" style="background:{PROMPT_COLORS[m]}"></span>'
        f'P{m} {short[m]}</label>'
        for m in PROMPT_ORDER)
    signal_boxes = "".join(
        f'<label class="chk"><input type="checkbox" class="signal" value="{sig}" checked>'
        f'<span class="line {("solid" if sig=="mean_logprob" else "dashed")}"></span>'
        f'{sig}</label>'
        for sig in SIGNALS)
    metric_radios = "".join(
        f'<label class="chk"><input type="radio" name="metric" class="metric" value="{k}"'
        f'{" checked" if k=="selected_correct" else ""}>{lab}</label>'
        for k, lab in METRICS)

    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>EPF accuracy vs budget — Run 6</title>
<style>
  :root {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
  body {{ margin: 22px; color:#1a1a1a; max-width: 1000px; }}
  h1 {{ font-size: 19px; margin: 0 0 2px; }}
  .sub {{ color:#666; font-size:13px; margin:0 0 14px; }}
  .panel {{ display:flex; flex-wrap:wrap; gap:26px; align-items:flex-start;
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
  <h1>MMAU-Pro · EPF diversity sweep (Run 6)</h1>
  <p class="sub">Accuracy vs budget · 4 prompts × 2 self-certainty signals · n=100 items/cell ·
     temp 0.8, systematic resampling. Hover a point for the value, n, and 95% CI.</p>

  <div class="panel">
    <div class="group">
      <span class="ttl">Metric</span>
      <div class="row">{metric_radios}</div>
    </div>
    <div class="group">
      <span class="ttl">Prompt</span>
      <div class="row">{prompt_boxes}</div>
    </div>
    <div class="group">
      <span class="ttl">Signal</span>
      <div class="row">{signal_boxes}</div>
    </div>
    <div class="btns">
      <button id="all">All on</button>
      <button id="none">All off</button>
    </div>
  </div>

  {plot_html}
  <p class="note">colour = prompt · solid line / ○ = mean_logprob · dashed line / □ = entropy.
     A trace shows only if its prompt <b>and</b> signal are checked and its metric is selected.</p>

<script>
  const TRACE_INFO = {json.dumps(trace_info)};
  const METRIC_META = {json.dumps(metric_meta)};
  const gd = document.getElementById('epfplot');

  function selectedMetric() {{
    return document.querySelector('input.metric:checked').value;
  }}
  function checkedSet(cls) {{
    return new Set(Array.from(document.querySelectorAll('input.' + cls + ':checked'))
                        .map(e => e.value));
  }}
  function apply() {{
    const metric = selectedMetric();
    const prompts = checkedSet('prompt');
    const signals = checkedSet('signal');
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
</body></html>"""

    with open(out_path, "w") as f:
        f.write(page)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
