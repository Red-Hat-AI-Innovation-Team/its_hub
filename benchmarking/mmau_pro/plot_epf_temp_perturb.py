"""Same combined EPF + temp-sweep interactive plot, with the perturbation-robustness
figure embedded as a separate static panel below it.

The perturbation study (SAFE vs UNSAFE, N perturbed copies majority-voted) lives on a
DIFFERENT scaling axis from the particle/sample budget above — it is not apples-to-apples,
so it is shown as its own small panel (no filters), not merged into the interactive plot.

Reuses build_html() from plot_epf_temp so the top plot stays identical to
epf_temp_acc_vs_budget.html; this writes a NEW file and touches nothing existing.

    /home/exx/miniconda3/envs/epf/bin/python -m benchmarking.mmau_pro.plot_epf_temp_perturb
"""

import base64

import click

from benchmarking.mmau_pro.plot_epf957 import load as load_epf
from benchmarking.mmau_pro.plot_epf_temp import build_html, load_temp

PERTURB_DEFAULT = ("/home/exx/inference-time-scaling/reports/qwen2_5_omni_7b/"
                   "plots/perturbation/safe_vs_unsafe_with_open.png")


def perturb_section(png_path):
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"""
  <hr style="border:none;border-top:1px solid #eee;margin:30px 0 16px;">
  <h2 style="font-size:17px;margin:0 0 2px;">Perturbation robustness — SAFE vs UNSAFE
    <span style="color:#888;font-weight:400;font-size:13px;">(separate study · different axis, not apples-to-apples with the budget plot above)</span></h2>
  <p class="sub">Qwen2.5-Omni-7B · MMAU-Pro (with open). Each question is expanded into
     <b>N perturbed copies</b> and majority-voted; SAFE = 7 perturbation axes, UNSAFE = 10
     axes. The x-axis is N <i>perturbed copies</i> — a different scaling axis from the
     particles / samples above — so it is shown as its own panel.</p>
  <img alt="SAFE vs UNSAFE perturbation scaling (N perturbed copies majority-voted)"
       src="data:image/png;base64,{b64}"
       style="max-width:780px;width:100%;height:auto;border:1px solid #eee;border-radius:6px;">
"""


@click.command()
@click.option("--epf-in", default="benchmarking/mmau_pro/results/run10_run6_full/run6_full.csv")
@click.option("--temp-in",
              default="/home/exx/inference-time-scaling/reports/qwen2_5_omni_7b_bootstrapped/bootstrap_std.csv")
@click.option("--perturb-png", default=PERTURB_DEFAULT)
@click.option("--out", "out_path",
              default="benchmarking/mmau_pro/results/plots/epf_temp_acc_pertubs_vs_budget.html")
def main(epf_in, temp_in, perturb_png, out_path):
    short, epf_cells, n_total = load_epf(epf_in)
    n_grade = max(n for (_, _, _), (_, _, ns) in epf_cells.items() for n in ns)
    temp_curves = load_temp(temp_in)

    page = build_html(short, epf_cells, temp_curves, n_total, n_grade)
    page = page.replace("</body></html>", perturb_section(perturb_png) + "\n</body></html>")

    with open(out_path, "w") as f:
        f.write(page)
    print(f"wrote {out_path}  (combined plot + embedded perturbation panel)")


if __name__ == "__main__":
    main()
