import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

scenarios = {
    "Baseline": 976_657_703.50,
    "Mild stress": 1_200_039_584.84,
    "Adverse": 1_462_982_396.67,
    "Severe": 1_791_015_732.81,
}

stages = {
    "Stage 1": 34.51,
    "Stage 2": 43.01,
    "Stage 3": 22.48,
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=220)

# Left: ECL by scenario
ax0 = axes[0]
labels = list(scenarios.keys())
values = list(scenarios.values())
colors = ["#2F5597", "#4F81BD", "#C0504D", "#943634"]
bars = ax0.bar(labels, values, color=colors)

ax0.set_title("ECL IFRS9 por Escenario", fontsize=16, fontweight="bold")
ax0.set_ylabel("ECL (USD)", fontsize=12)
ax0.set_xlabel("Escenario", fontsize=12)
ax0.grid(axis="y", linestyle="--", alpha=0.3)
ax0.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x/1e9:.2f}B"))

for bar, v in zip(bars, values):
    ax0.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"${v/1e9:.2f}B",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Right: Baseline stage composition
ax1 = axes[1]
stage_labels = list(stages.keys())
stage_vals = list(stages.values())
stage_colors = ["#9BBB59", "#F79646", "#8064A2"]
wedges, texts, autotexts = ax1.pie(
    stage_vals,
    labels=stage_labels,
    colors=stage_colors,
    autopct="%1.2f%%",
    startangle=90,
    wedgeprops={"width": 0.45, "edgecolor": "white"},
    textprops={"fontsize": 11},
)

for t in autotexts:
    t.set_fontweight("bold")

ax1.set_title("Composición Baseline por Stage", fontsize=16, fontweight="bold")
ax1.text(0, 0, "IFRS9\nBaseline", ha="center", va="center", fontsize=11, fontweight="bold")

fig.suptitle(
    "Sensibilidad Prudencial IFRS9 en la Tesis de Especialización",
    fontsize=18,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig("thesis_poster/figures/figure3.png", bbox_inches="tight")
