"""Publication-quality figure helpers for the Quarto book.

Provides consistent styling for matplotlib and plotly figures across all
chapters. Uses a project-standard color palette and font sizes suitable
for both HTML (interactive) and PDF (static, vector) output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project palette
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "info": "#0891B2",
    "muted": "#6B7280",
    "grades": {
        "A": "#059669",
        "B": "#0891B2",
        "C": "#2563EB",
        "D": "#7C3AED",
        "E": "#D97706",
        "F": "#DC2626",
        "G": "#BE185D",
    },
}

GRADE_COLORS = list(PALETTE["grades"].values())
GRADE_ORDER = list(PALETTE["grades"].keys())

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------

_RC_DEFAULTS = {
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.format": "svg",
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Source Sans 3", "Inter", "DejaVu Sans", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "text.color": "#111827",
    "axes.labelcolor": "#111827",
    "axes.titlecolor": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
}


def apply_style() -> None:
    """Apply project-standard matplotlib style in-place."""
    plt.rcParams.update(_RC_DEFAULTS)


def get_grade_color(grade: str) -> str:
    """Return the color for a credit grade letter."""
    return PALETTE["grades"].get(grade, PALETTE["muted"])


def save_figure(
    fig: plt.Figure,
    output_path: str | Path,
    *,
    transparent: bool = False,
    close: bool = False,
) -> Path:
    """Save a matplotlib figure using project defaults.

    Defaults to SVG-first output so Quarto can render publication-quality vector
    graphics in HTML and PDF with a single asset.
    """
    apply_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=transparent)
    if close:
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Convenience figure constructors
# ---------------------------------------------------------------------------


def bar_chart(
    labels: list[str],
    values: list[float],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str | list[str] | None = None,
    horizontal: bool = False,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Create a styled bar chart and return the Figure."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    c = color or PALETTE["primary"]
    if horizontal:
        ax.barh(labels, values, color=c)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.bar(labels, values, color=c)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def metric_table_md(data: dict[str, str | float], title: str = "") -> str:
    """Return a markdown table from a flat dict for inline rendering."""
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    lines.append("| Métrica | Valor |")
    lines.append("|---------|-------|")
    for k, v in data.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
