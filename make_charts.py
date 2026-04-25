"""
Generates two publication-quality bar charts comparing 4 policies on the
held-out net_zero_profit task. Saves PNGs to current directory.

Run this in a fresh Colab cell or standalone Python — only needs matplotlib.
Numbers come from the 3-run averaged eval (eval_3run_averaged_net_zero_profit.json).

Outputs:
  - chart_grader_comparison.png   (headline metric, with error bars)
  - chart_profit_carbon_ratio.png (the 25.6× story, log scale)
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Data — net_zero_profit, held-out seeds 500-509 ─────────────────────────
policies = ["Random", "Base Qwen-1.5B", "GRPO Qwen (ours)", "Heuristic"]
colors   = ["#bdbdbd", "#7a7a7a", "#2e7d32", "#1565c0"]   # gray, dark gray, green (us), blue

# Grader scores
grader_means = [0.001, 0.259, 0.273, 0.292]
grader_stds  = [0.000, 0.082, 0.019, 0.037]   # σ from 3-run averaging where available

# Profit / carbon ratio (aggregate)
ratio_values = [2.85, 2.65, 67.96, 1e6]   # heuristic is ∞ — use a large finite for log scale
ratio_labels = ["2.85", "2.65", "67.96", "∞ (carbon=0)"]


# ─── Chart 1: Grader scores with error bars ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(policies))
bars = ax.bar(x, grader_means, yerr=grader_stds, capsize=8,
              color=colors, edgecolor="black", linewidth=0.8,
              error_kw={"elinewidth": 1.5, "ecolor": "#222"})

# Highlight our bar
bars[2].set_edgecolor("#1b5e20")
bars[2].set_linewidth(2.5)

# Value labels on top of each bar
for i, (mean, std) in enumerate(zip(grader_means, grader_stds)):
    if std > 0:
        label = f"{mean:.3f} ± {std:.3f}"
    else:
        label = f"{mean:.3f}"
    ax.text(i, mean + std + 0.012, label, ha="center", fontsize=10,
            fontweight="bold" if i == 2 else "normal")

ax.set_xticks(x)
ax.set_xticklabels(policies, fontsize=11)
ax.set_ylabel("Grader score (net_zero_profit)", fontsize=12)
ax.set_title("Held-out grader score — 3-run averaged, 30 episodes total\n"
             "(seeds 500–509, never seen during training)",
             fontsize=13, pad=15)
ax.set_ylim(0, max(grader_means) * 1.35)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Annotate the headline
ax.annotate("Our headline:\n0.273 ± 0.019\n(σ across 3 runs)",
            xy=(2, 0.273), xytext=(2.7, 0.18),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#2e7d32"))

plt.tight_layout()
plt.savefig("chart_grader_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Saved chart_grader_comparison.png")
plt.show()
plt.close()


# ─── Chart 2: Profit/carbon ratio (log scale) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
# Use log scale so we can show 2.65 → 67.96 without flattening the small ones
plot_values = [v if v < 1e5 else 1e3 for v in ratio_values]   # cap heuristic for plotting
bars = ax.bar(x, plot_values, color=colors, edgecolor="black", linewidth=0.8)
bars[2].set_edgecolor("#1b5e20")
bars[2].set_linewidth(2.5)

for i, (val, lbl) in enumerate(zip(plot_values, ratio_labels)):
    ax.text(i, val * 1.15, lbl, ha="center", fontsize=11,
            fontweight="bold" if i == 2 else "normal")

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(policies, fontsize=11)
ax.set_ylabel("Profit / carbon ratio (log scale)", fontsize=12)
ax.set_title("Profit per unit of carbon emitted — held-out net_zero_profit\n"
             "GRPO improves 25.6× over base Qwen",
             fontsize=13, pad=15)
ax.set_ylim(1, 5000)
ax.grid(axis="y", which="both", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Annotate the multiplier story
ax.annotate("",
            xy=(2, 67.96), xytext=(1, 2.65),
            arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=2,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(1.5, 12, "25.6×",
        fontsize=18, fontweight="bold", color="#2e7d32", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#2e7d32", lw=1.5))

plt.tight_layout()
plt.savefig("chart_profit_carbon_ratio.png", dpi=150, bbox_inches="tight")
print("✓ Saved chart_profit_carbon_ratio.png")
plt.show()
plt.close()

print("\nDone — 2 charts ready for the README/blog.")
