"""Plot EPF selected accuracy vs budget for the Run 6 diversity sweep (epf_div.csv).

One line per (prompt, signal) cell -> 4 prompts x 2 signals = 8 lines.
Colour encodes the prompt, line style/marker encodes the self-certainty signal.

    python -m benchmarking.mmau_pro.plot_acc_vs_budget \
        --in benchmarking/mmau_pro/results/run06_epf_div/epf_div.csv \
        --out benchmarking/mmau_pro/results/plots/acc_vs_budget.png
"""

import math

import click
import matplotlib.pyplot as plt
import pandas as pd

PROMPT_ORDER = [4, 5, 7, 9]
BUDGET_ORDER = [1, 8, 16, 32, 64, 128]
# colour per prompt (colour-blind-friendly), style/marker per signal
PROMPT_COLORS = {4: "#1b9e77", 5: "#d95f02", 7: "#7570b3", 9: "#e7298a"}
SIGNAL_STYLE = {
    "mean_logprob": {"linestyle": "-", "marker": "o"},
    "entropy": {"linestyle": "--", "marker": "s"},
}
SIGNAL_LABEL = {"mean_logprob": "mean_logprob", "entropy": "entropy"}


def _wilson_halfwidth(p, n, z=1.96):
    """95% Wilson interval half-width, for light error bars."""
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    # report a symmetric-ish bar around p using the half-interval width
    lo, hi = centre - margin, centre + margin
    return (hi - lo) / 2


@click.command()
@click.option("--in", "in_path", default="benchmarking/mmau_pro/results/run06_epf_div/epf_div.csv")
@click.option("--out", "out_path", default="benchmarking/mmau_pro/results/plots/acc_vs_budget.png")
@click.option("--metric", default="selected_correct",
              type=click.Choice(["selected_correct", "oracle_correct", "majority_correct"]),
              help="Which accuracy to plot (default: selected = what EPF returns).")
@click.option("--errorbars/--no-errorbars", default=False, help="Show 95% Wilson CI bars.")
@click.option("--bands/--no-bands", default=False, help="Show faint 95% Wilson CI shaded bands.")
def main(in_path, out_path, metric, errorbars, bands):
    df = pd.read_csv(in_path)
    names = dict(df[["method", "method_name"]].drop_duplicates().values)
    signals = [s for s in ("mean_logprob", "entropy") if s in set(df["signal"])]

    # accuracy + n per (prompt, signal, budget)
    g = (df.groupby(["method", "signal", "budget"])[metric]
           .agg(acc="mean", n="count").reset_index())

    x = list(range(len(BUDGET_ORDER)))  # even categorical spacing
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=160)

    for m in PROMPT_ORDER:
        for sig in signals:
            sub = (g[(g.method == m) & (g.signal == sig)]
                   .set_index("budget").reindex(BUDGET_ORDER))
            ys = sub["acc"].tolist()
            ns = sub["n"].tolist()
            style = SIGNAL_STYLE[sig]
            label = f"P{m} {names[m].split(' (')[0]} · {SIGNAL_LABEL[sig]}"
            if errorbars:
                errs = [_wilson_halfwidth(p, int(n)) if pd.notna(p) else 0.0
                        for p, n in zip(ys, ns)]
                ax.errorbar(x, ys, yerr=errs, color=PROMPT_COLORS[m], label=label,
                            linewidth=1.8, markersize=6, capsize=2.5,
                            elinewidth=0.8, alpha=0.9, **style)
            else:
                ax.plot(x, ys, color=PROMPT_COLORS[m], label=label,
                        linewidth=1.8, markersize=6, **style)
            if bands:
                errs = [_wilson_halfwidth(p, int(n)) if pd.notna(p) else 0.0
                        for p, n in zip(ys, ns)]
                lo = [y - e if pd.notna(y) else None for y, e in zip(ys, errs)]
                hi = [y + e if pd.notna(y) else None for y, e in zip(ys, errs)]
                ax.fill_between(x, lo, hi, color=PROMPT_COLORS[m], alpha=0.07, linewidth=0)

    metric_title = {"selected_correct": "Selected accuracy (what EPF returns)",
                    "oracle_correct": "Oracle accuracy (answer in any particle)",
                    "majority_correct": "Majority-vote accuracy"}[metric]
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGET_ORDER])
    ax.set_xlabel("Budget (number of particles)")
    ax.set_ylabel(metric_title)
    ax.set_title("MMAU-Pro · EPF diversity sweep (Run 6)\n"
                 "accuracy vs budget — 4 prompts × 2 self-certainty signals (n=100/cell)",
                 fontsize=12)
    ax.grid(True, axis="both", linestyle=":", alpha=0.4)
    ax.set_ylim(0.30, 1.0 if metric == "oracle_correct" else 0.70)

    # two-group legend below the axes so it never overlaps the lines
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              framealpha=0.95, columnspacing=1.2, handlelength=2.4,
              title="colour = prompt   ·   solid line / ○ = mean_logprob   ·   dashed line / □ = entropy")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
