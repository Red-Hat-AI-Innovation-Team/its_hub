"""Combined INTERACTIVE html: EPF diversity sweep (Run 6, n=100) + self-consistency
temperature sweep (full 957) on one accuracy-vs-budget axis.

Two studies share a log2 budget axis (N / particles, 1..128):
  - EPF sweep (n=100):  4 prompts x 2 signals x {selected, oracle, majority} metric,
                        budgets {1,8,16,32}     [from epf_div.csv]
  - Temp sweep (n=957): self-consistency majority vote, T in {0.3,0.5,0.7,1.0},
                        N {1,2,4,8,16,32,64,128} [from the scaling_results.json files]
  + greedy (T=0,N=1) reference line.

Filters on the page: study, EPF metric, prompt, signal, temperature, greedy toggle.
Plotly.js is embedded inline -> the file works offline.

    python -m benchmarking.mmau_pro.plot_combined_interactive
"""

import json
import math

import click
import pandas as pd
import plotly.graph_objects as go

# ---- EPF (Run 6) config ----
PROMPT_ORDER = [4, 5, 7, 9]
EPF_BUDGETS = [1, 8, 16, 32]
SIGNALS = ["mean_logprob", "entropy"]
METRICS = [
    ("selected_correct", "Selected (what EPF returns)"),
    ("oracle_correct", "Oracle (answer in any particle)"),
    ("majority_correct", "Majority vote"),
]
PROMPT_COLORS = {4: "#1b9e77", 5: "#d95f02", 7: "#7570b3", 9: "#e7298a"}
SIGNAL_DASH = {"mean_logprob": "solid", "entropy": "dash"}
SIGNAL_SYMBOL = {"mean_logprob": "circle", "entropy": "square"}

# ---- temperature sweep config ----
TEMPS = ["0.3", "0.5", "0.7", "1.0"]
TEMP_N = [1, 2, 4, 8, 16, 32, 64, 128]
TEMP_COLORS = {"0.3": "#2166ac", "0.5": "#67a9cf", "0.7": "#ef8a62", "1.0": "#b2182b"}

ALL_X = [1, 2, 4, 8, 16, 32, 64, 128]


def _wilson_halfwidth(p, n, z=1.96):
    if n == 0 or p is None or (isinstance(p, float) and math.isnan(p)):
        return 0.0
    denom = 1 + z * z / n
    return z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom


def load_epf(in_path):
    df = pd.read_csv(in_path)
    names = dict(df[["method", "method_name"]].drop_duplicates().values)
    short = {m: names[m].split(" (")[0] for m in names}
    cells = {}
    for metric_key, _ in METRICS:
        g = (df.groupby(["method", "signal", "budget"])[metric_key]
               .agg(acc="mean", n="count").reset_index())
        for m in PROMPT_ORDER:
            for sig in SIGNALS:
                sub = (g[(g.method == m) & (g.signal == sig)]
                       .set_index("budget").reindex(EPF_BUDGETS))
                ys = [None if pd.isna(v) else round(float(v), 4) for v in sub["acc"]]
                ns = [0 if pd.isna(v) else int(v) for v in sub["n"]]
                cells[(metric_key, m, sig)] = (EPF_BUDGETS, ys, ns)
    return short, cells


def load_temp(temp_dir, stem):
    curves, greedy = {}, None
    gp = f"{temp_dir}/{stem}_T0.0_n1_scaling_results.json"
    greedy = json.load(open(gp))["results"]["1"]["accuracy_mean"] / 100.0
    for T in TEMPS:
        d = json.load(open(f"{temp_dir}/{stem}_T{T}_n128_scaling_results.json"))["results"]
        Ns = sorted(int(k) for k in d)
        ys = [round(d[str(n)]["accuracy_mean"] / 100.0, 4) for n in Ns]
        sd = [round(d[str(n)]["accuracy_std"] / 100.0, 4) for n in Ns]
        ni = json.load(open(f"{temp_dir}/{stem}_T{T}_n128_scaling_results.json")).get("n_items", 957)
        curves[T] = (Ns, ys, sd, ni)
    return curves, greedy


@click.command()
@click.option("--epf-csv", default="benchmarking/mmau_pro/results/run06_epf_div/epf_div.csv")
@click.option("--temp-dir",
              default="/home/exx/inference-time-scaling/results/qwen2_5_omni_7b/vllm")
@click.option("--stem", default="Qwen2.5-Omni-7B_testmini-00000-of-00001")
@click.option("--out", "out_path",
              default="benchmarking/mmau_pro/results/plots/acc_vs_budget_combined.html")
def main(epf_csv, temp_dir, stem, out_path):
    short, epf_cells = load_epf(epf_csv)
    temp_curves, greedy = load_temp(temp_dir, stem)

    fig = go.Figure()
    info = []  # parallel to fig.data

    # ---- EPF traces (study=epf) ----
    for metric_key, _ in METRICS:
        for m in PROMPT_ORDER:
            for sig in SIGNALS:
                xs, ys, ns = epf_cells[(metric_key, m, sig)]
                hw = [round(_wilson_halfwidth(y, n), 4) for y, n in zip(ys, ns)]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers",
                    name=f"EPF · P{m} {short[m]} · {sig}",
                    line=dict(color=PROMPT_COLORS[m], dash=SIGNAL_DASH[sig], width=2.4),
                    marker=dict(symbol=SIGNAL_SYMBOL[sig], size=8, color=PROMPT_COLORS[m]),
                    customdata=[[n, h] for n, h in zip(ns, hw)],
                    hovertemplate=(f"<b>EPF · P{m} {short[m]}</b> · {sig}<br>"
                                   "budget %{x} · acc %{y:.3f} ±%{customdata[1]:.3f}<br>"
                                   "n=%{customdata[0]} items<extra></extra>"),
                    visible=(metric_key == "selected_correct"),
                ))
                info.append({"study": "epf", "prompt": m, "signal": sig, "metric": metric_key})

    # ---- temperature traces (study=temp) ----
    for T in TEMPS:
        Ns, ys, sd, ni = temp_curves[T]
        fig.add_trace(go.Scatter(
            x=Ns, y=ys, mode="lines+markers",
            name=f"self-consistency · T={T}",
            line=dict(color=TEMP_COLORS[T], dash="dot", width=3),
            marker=dict(symbol="diamond", size=8, color=TEMP_COLORS[T]),
            customdata=[[s, ni] for s in sd],
            hovertemplate=(f"<b>self-consistency · T={T}</b> (majority vote)<br>"
                           "N %{x} · acc %{y:.3f} ±%{customdata[0]:.3f}<br>"
                           "n=%{customdata[1]} items<extra></extra>"),
            visible=True,
        ))
        info.append({"study": "temp", "temp": T})

    # ---- greedy reference line ----
    fig.add_trace(go.Scatter(
        x=[ALL_X[0], ALL_X[-1]], y=[greedy, greedy], mode="lines",
        name=f"greedy (T=0,N=1) = {greedy:.3f}",
        line=dict(color="#555", dash="dashdot", width=1.6),
        hovertemplate=f"greedy baseline = {greedy:.3f} (n=957)<extra></extra>",
        visible=True,
    ))
    info.append({"study": "greedy"})

    fig.update_layout(
        template="plotly_white",
        title=dict(text="MMAU-Pro · accuracy vs budget — EPF sweep (n=100) + "
                        "self-consistency temp sweep (n=957)",
                   x=0.5, xanchor="center", font=dict(size=16)),
        xaxis=dict(title="Budget N — particles (EPF) / samples voted (temp), log₂ scale",
                   type="log", tickmode="array", tickvals=ALL_X,
                   ticktext=[str(x) for x in ALL_X]),
        yaxis=dict(title="Accuracy (fraction of items correct)", tickformat=".2f",
                   autorange=True),
        showlegend=False, hovermode="closest",
        margin=dict(l=70, r=30, t=60, b=50), height=580,
    )

    plot_html = fig.to_html(include_plotlyjs=True, full_html=False, div_id="cplot",
                            config={"displaylogo": False, "responsive": True})

    # ---- control panel ----
    study_boxes = (
        '<label class="chk"><input type="checkbox" class="study" value="epf" checked>'
        'EPF sweep (n=100)</label>'
        '<label class="chk"><input type="checkbox" class="study" value="temp" checked>'
        'temp sweep (n=957)</label>')
    metric_radios = "".join(
        f'<label class="chk"><input type="radio" name="metric" class="metric" value="{k}"'
        f'{" checked" if k=="selected_correct" else ""}>{lab}</label>'
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
<title>EPF + temp sweep — accuracy vs budget</title>
<style>
  :root {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
  body {{ margin: 22px; color:#1a1a1a; max-width: 1040px; }}
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
  <h1>MMAU-Pro · accuracy vs budget</h1>
  <p class="sub">Two studies on one log₂ budget axis ·
     <b>EPF sweep</b> (n=100, self-certainty selected/oracle/majority, budgets 1–32) +
     <b>self-consistency temp sweep</b> (n=957, majority vote, N 1–128).
     Apples-to-oranges in n &amp; method — hover any point for its method, value, n, CI/std.</p>

  <div class="panel">
    <div class="group"><span class="ttl">Study</span><div class="row">{study_boxes}</div></div>
    <div class="group"><span class="ttl">EPF metric</span><div class="row">{metric_radios}</div></div>
    <div class="group"><span class="ttl">Prompt (EPF)</span><div class="row">{prompt_boxes}</div></div>
    <div class="group"><span class="ttl">Signal (EPF)</span><div class="row">{signal_boxes}</div></div>
    <div class="group"><span class="ttl">Temperature</span><div class="row">{temp_boxes}</div></div>
    <div class="group"><span class="ttl">Reference</span><div class="row">
      <label class="chk"><input type="checkbox" id="greedy" checked>greedy baseline</label>
    </div></div>
    <div class="btns">
      <button id="all">All on</button>
      <button id="none">All off</button>
    </div>
  </div>

  {plot_html}
  <p class="note">EPF: colour = prompt · solid/○ = mean_logprob · dashed/□ = entropy.
     Temp sweep: dotted/◇, colour = temperature. EPF metric is single-select (switch to
     <i>oracle</i> to compare EPF coverage against the temp curves).</p>

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
    const greedyOn = document.getElementById('greedy').checked;
    const vis = INFO.map(t => {{
      if (t.study === 'epf')
        return studies.has('epf') && prompts.has(String(t.prompt))
            && signals.has(t.signal) && t.metric === metric;
      if (t.study === 'temp')
        return studies.has('temp') && temps.has(t.temp);
      if (t.study === 'greedy')
        return greedyOn;
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
            .forEach(e => e.checked = false);
    document.getElementById('greedy').checked = false; apply();
  }});
  apply();
</script>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(page)
    print(f"wrote {out_path}  (greedy={greedy:.4f})")


if __name__ == "__main__":
    main()
