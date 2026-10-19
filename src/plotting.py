"""Shared chart style so notebooks, report figures and dashboard read as one system.

Colors follow the entity (a model or campaign keeps its hue in every chart).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a02524"]

CAMPAIGN_COLORS = {"Womens E-Mail": BLUE, "Mens E-Mail": ORANGE, "No E-Mail": MUTED,
                   "womens": BLUE, "mens": ORANGE}
POLICY_COLORS = {
    "Random": MUTED,
    "Treat all": INK_2,
    "Purchase propensity": ORANGE,
    "S-learner": YELLOW,
    "T-learner": BLUE,
    "X-learner": AQUA,
    "DR-learner": MAGENTA,
    "Calibrated propensity": VIOLET,
    "Causal forest": GREEN,
}


def color_for(name: str) -> str:
    """Model colour by family, so tuned variants ("Causal forest (leaf 200)") keep their hue."""
    for key, c in POLICY_COLORS.items():
        if name.startswith(key):
            return c
    return MUTED


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]),
    })


def savefig(fig, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def md(text: str):
    """Markdown for display() with literal dollar signs escaped (Jupyter renders $...$ as LaTeX)."""
    import re

    from IPython.display import Markdown

    return Markdown(re.sub(r"(?<!\\)\$", r"\\$", text))


def money0(v: float) -> str:
    """Format dollars for tables: $1,234 / −$1,234."""
    return ("−" if v < 0 else "") + f"${abs(v):,.0f}"


def dollar(ax, axis: str = "y", decimals: int = 2) -> None:
    fmt = mpl.ticker.FuncFormatter(
        lambda v, _: ("−" if v < 0 else "") + f"${abs(v):,.{decimals}f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def percent(ax, axis: str = "x", decimals: int = 0) -> None:
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
