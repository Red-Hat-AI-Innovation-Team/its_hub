"""Combined accuracy-vs-budget viz: EPF self-certainty sweep + self-consistency temp sweep.

Two studies on one shared log2 budget axis (1..128), full 957-MCQ benchmark:
  - EPF sweep   : 4 prompts x 2 self-certainty signals = 8 lines, budgets {1,8,16,32},
                  3 metrics (selected / oracle / majority)        [results/run10_run6_full/run6_full.csv]
  - Temp sweep  : self-consistency majority vote, T in {0.3,0.5,0.7,1.0}, N {1..128} = 4 lines
                  [/home/exx/inference-time-scaling/reports/qwen2_5_omni_7b_bootstrapped/bootstrap_std.csv]

Both studies report the SAME uncertainty: std_100 = std of accuracy across a random 100-question
eval (no vertical bars; shown in the hover). The temp CSV's `bootstrap_std` IS that std_100.

The EPF lines / std_100 / colours are imported from plot_epf957 so they stay identical to the
standalone epf_acc_vs_budget.html. This script writes a NEW file and never touches the old one.

    /home/exx/miniconda3/envs/epf/bin/python -m benchmarking.mmau_pro.plot_epf_temp
"""

import json

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from benchmarking.mmau_pro.plot_epf957 import (
    METRICS, PROMPT_COLORS, PROMPT_ORDER, SIGNAL_DASH, SIGNAL_SYMBOL, SIGNALS,
    load as load_epf,
)

# ---- temperature sweep config ----
TEMPS = ["0.3", "0.5", "0.7", "1.0"]
TEMP_COLORS = {"0.3": "#2166ac", "0.5": "#67a9cf", "0.7": "#ef8a62", "1.0": "#b2182b"}
TEMP_N_ITEMS = 957  # "full incl. open (957 items)" per the overlay report

ALL_X = [1, 2, 4, 8, 16, 32, 64, 128]  # shared log2 budget ticks


def load_temp(path):
    """temp -> (Ns, ys, std100s); accuracy & bootstrap_std are percentages in the CSV."""
    df = pd.read_csv(path)
    curves = {}
    for T in TEMPS:
        sub = df[np.isclose(df["temperature"], float(T))].sort_values("N")
        Ns = [int(n) for n in sub["N"]]
        ys = [round(a / 100.0, 4) for a in sub["accuracy"]]
        sd = [round(s / 100.0, 4) for s in sub["bootstrap_std"]]
        curves[T] = (Ns, ys, sd)
    return curves


def build_html(short, epf_cells, temp_curves, n_total, n_grade):
    fig = go.Figure()
    info = []  # parallel to fig.data

    # ---- EPF traces (8 lines x 3 metrics) ----
    for metric_key, _ in METRICS:
        for m in PROMPT_ORDER:
            for sig in SIGNALS:
                ys, std100s, ns = epf_cells[(metric_key, m, sig)]
                xs = [b for b in (1, 8, 16, 32)]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers",
                    name=f"EPF · P{m} {short[m]} · {sig}",
                    legendgroup=f"P{m}",
                    line=dict(color=PROMPT_COLORS[m], dash=SIGNAL_DASH[sig], width=2.4),
                    marker=dict(symbol=SIGNAL_SYMBOL[sig], size=8, color=PROMPT_COLORS[m]),
                    customdata=[[n, s] for n, s in zip(ns, std100s)],
                    hovertemplate=(f"<b>EPF · P{m} {short[m]}</b> · {sig}<br>"
                                   "budget %{x} · acc %{y:.3f} ±%{customdata[1]:.3f} (std₁₀₀)<br>"
                                   "n=%{customdata[0]} items<extra></extra>"),
                    visible=(metric_key == "selected_correct"),
                ))
                info.append({"study": "epf", "prompt": m, "signal": sig, "metric": metric_key})

    # ---- temp traces (4 lines) ----
    for T in TEMPS:
        Ns, ys, sd = temp_curves[T]
        fig.add_trace(go.Scatter(
            x=Ns, y=ys, mode="lines+markers",
            name=f"self-consistency · T={T}",
            line=dict(color=TEMP_COLORS[T], dash="dot", width=3),
            marker=dict(symbol="diamond", size=8, color=TEMP_COLORS[T]),
            customdata=[[s, TEMP_N_ITEMS] for s in sd],
            hovertemplate=(f"<b>self-consistency · T={T}</b> (majority vote)<br>"
                           "N %{x} · acc %{y:.3f} ±%{customdata[0]:.3f} (std₁₀₀)<br>"
                           "n=%{customdata[1]} items<extra></extra>"),
            visible=True,
        ))
        info.append({"study": "temp", "temp": T})

    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"MMAU-Pro · accuracy vs budget (full {n_total} MCQ) — "
                        "EPF self-certainty sweep + self-consistency temperature sweep",
                   x=0.5, xanchor="center", font=dict(size=15.5)),
        xaxis=dict(title="Budget N — particles (EPF) / samples majority-voted (temp), log₂ scale",
                   type="log", tickmode="array", tickvals=ALL_X,
                   ticktext=[str(x) for x in ALL_X]),
        yaxis=dict(title="Accuracy (fraction of items correct)", tickformat=".2f",
                   autorange=True),
        showlegend=False, hovermode="closest",
        margin=dict(l=70, r=30, t=64, b=50), height=600,
    )

    plot_html = fig.to_html(include_plotlyjs=True, full_html=False, div_id="cplot",
                            config={"displaylogo": False, "responsive": True})

    # ---- control panel ----
    study_boxes = (
        '<label class="chk"><input type="checkbox" class="study" value="epf" checked>'
        'EPF sweep (8 lines)</label>'
        '<label class="chk"><input type="checkbox" class="study" value="temp" checked>'
        'temp sweep (4 lines)</label>')
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
    temp_boxes = "".join(
        f'<label class="chk"><input type="checkbox" class="temp" value="{T}" checked>'
        f'<span class="swatch" style="background:{TEMP_COLORS[T]}"></span>T={T}</label>'
        for T in TEMPS)

    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>EPF + temp sweep — accuracy vs budget (957)</title>
<style>
  :root {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
  body {{ margin: 22px; color:#1a1a1a; max-width: 1060px; }}
  h1 {{ font-size: 19px; margin: 0 0 2px; }}
  .sub {{ color:#666; font-size:13px; margin:0 0 14px; }}
  .panel {{ display:flex; flex-wrap:wrap; gap:22px; align-items:flex-start;
            background:#fafafa; border:1px solid #eee; border-radius:8px;
            padding:12px 16px; margin-bottom:10px; }}
  .group {{ display:flex; flex-direction:column; gap:5px; }}
  .group .ttl {{ font-size:11px; font-weight:700; letter-spacing:.04em;
                 text-transform:uppercase; color:#888; margin-bottom:2px; }}
  .row {{ display:flex; gap:14px; flex-wrap:wrap; }}
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
  <h1>MMAU-Pro · accuracy vs budget — full {n_total} MCQ</h1>
  <p class="sub">Two studies on one log₂ budget axis (1–128) ·
     <b>EPF self-certainty sweep</b> (8 lines: 4 prompts × 2 signals, n={n_grade} gradeable/cell,
     budgets 1–32) + <b>self-consistency temperature sweep</b> (4 lines: majority vote, N 1–128,
     n={TEMP_N_ITEMS}). Both report ±std₁₀₀ = the noise of a random 100-question eval (hover a point).</p>

  <div class="panel">
    <div class="group"><span class="ttl">Study</span><div class="row">{study_boxes}</div></div>
    <div class="group"><span class="ttl">EPF metric</span><div class="row">{metric_radios}</div></div>
    <div class="group"><span class="ttl">Prompt (EPF)</span><div class="row">{prompt_boxes}</div></div>
    <div class="group"><span class="ttl">Signal (EPF)</span><div class="row">{signal_boxes}</div></div>
    <div class="group"><span class="ttl">Temperature</span><div class="row">{temp_boxes}</div></div>
    <div class="btns">
      <button id="all">All on</button>
      <button id="none">All off</button>
    </div>
  </div>

  {plot_html}
  <p class="note">EPF: colour = prompt · solid/○ = mean_logprob · dashed/□ = entropy (budgets 1–32).
     Temp sweep: dotted/◇, colour = temperature (N 1–128). The temp lines are independent of the EPF
     metric selector — switch the EPF metric to <i>majority</i> for the closest apples-to-apples
     comparison. Y-axis auto-fits the visible lines.</p>

<script>
  const INFO = {json.dumps(info)};
  const gd = document.getElementById('cplot');
  const setOf = c => new Set(Array.from(document.querySelectorAll('input.'+c+':checked')).map(e=>e.value));
  function apply() {{
    const metric  = document.querySelector('input.metric:checked').value;
    const studies = setOf('study');
    const prompts = setOf('prompt');
    const signals = setOf('signal');
    const temps   = setOf('temp');
    const vis = INFO.map(t => {{
      if (t.study === 'epf')
        return studies.has('epf') && t.metric === metric
            && prompts.has(String(t.prompt)) && signals.has(t.signal);
      if (t.study === 'temp')
        return studies.has('temp') && temps.has(t.temp);
      return false;
    }});
    Plotly.restyle(gd, {{visible: vis}});
    Plotly.relayout(gd, {{'yaxis.autorange': true}});
  }}
  document.querySelectorAll('input').forEach(e => e.addEventListener('change', apply));
  document.getElementById('all').addEventListener('click', () => {{
    document.querySelectorAll('input[type=checkbox]').forEach(e => e.checked = true); apply();
  }});
  document.getElementById('none').addEventListener('click', () => {{
    document.querySelectorAll('input.study, input.prompt, input.signal, input.temp')
            .forEach(e => e.checked = false); apply();
  }});
  apply();
</script>
</body></html>"""
    return page


@click.command()
@click.option("--epf-in", default="benchmarking/mmau_pro/results/run10_run6_full/run6_full.csv")
@click.option("--temp-in",
              default="/home/exx/inference-time-scaling/reports/qwen2_5_omni_7b_bootstrapped/bootstrap_std.csv")
@click.option("--out", "out_path",
              default="benchmarking/mmau_pro/results/plots/epf_temp_acc_vs_budget.html")
def main(epf_in, temp_in, out_path):
    short, epf_cells, n_total = load_epf(epf_in)
    n_grade = max(n for (_, _, _), (_, _, ns) in epf_cells.items() for n in ns)
    temp_curves = load_temp(temp_in)

    with open(out_path, "w") as f:
        f.write(build_html(short, epf_cells, temp_curves, n_total, n_grade))
    print(f"wrote {out_path}  (8 EPF + {len(temp_curves)} temp lines, x up to {ALL_X[-1]})")


if __name__ == "__main__":
    main()
