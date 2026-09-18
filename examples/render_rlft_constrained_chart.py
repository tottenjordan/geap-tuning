"""Re-render the RLFT constrained-generation report chart.

NO TUNING COST, NO GCP. This is a pure offline figure renderer for the chart
embedded in ``docs/doe/rlft-constrained-generation/README.md``. The measured
values below are transcribed from that report's before -> after table (n = 30
held out); rerunning this script reproduces the committed PNG byte-for-byte on
the same matplotlib version. It needs the optional viz group:

    uv run --group viz python examples/render_rlft_constrained_chart.py
    uv run --group viz python examples/render_rlft_constrained_chart.py --out /tmp

Why a script and not the PaperBanana MCP tool (which made the first version):
image models re-render the whole figure each pass, so layout constraints do not
converge — three attempts each fixed the legend but broke something else. A data
chart needs exact numbers and deterministic placement, so it is drawn here
instead. See docs/notes/doe-and-visualization.md.

Flow: draw the grouped bars with matplotlib, then downscale to 1800px wide and
quantize to a 256-color palette (the repo's image convention, matching the
sibling chart in docs/doe/dpo-concise-email/).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path("docs/doe/rlft-constrained-generation")
FILENAME = "metrics.png"
TARGET_WIDTH = 1800

# Transcribed from the report table; keep in sync with the README if it is re-scored.
COMPONENTS = (
    "Accuracy",
    "Keywords",
    "Forbidden",
    "Word\nCount",
    "Sentence\nCount",
    "Full Satisfaction\nRate",
)
BASE = (0.823, 0.986, 1.000, 0.000, 1.000, 0.000)
TUNED = (0.828, 1.000, 1.000, 0.000, 1.000, 0.000)

# The one component with real headroom, and the one that never moved.
HEADROOM_INDEX = 3
ANNOTATION = "the headroom axis:\n0/30 before and after\nno advantage signal,\nno gradient"

TITLE = "RLFT constrained generation — a predicted null: RL can't bootstrap a near-never behavior"
SUBTITLE = "Experiment: gemini-3.5-flash, us-central1, n=30 held out"

NAVY = "#1b3a5c"
AMBER = "#dd8a0b"
SLATE = "#2e3d4d"
GREY = "#6b7785"
DARK_RED = "#9c1f22"

_VIZ_HINT = "run: uv sync --group viz"


def _import(module: str) -> Any:  # noqa: ANN401 - returns the imported module
    """Return ``module``, or raise an actionable error if the viz group is missing."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        msg = f"{module} is required to render this chart; {_VIZ_HINT}"
        raise RuntimeError(msg) from exc


def _arg(flag: str, default: str) -> str:
    """Return the value following ``flag`` in argv, or ``default``."""
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def _build_figure() -> Any:  # noqa: ANN401 - matplotlib Figure
    """Draw the grouped base-vs-tuned bars with nothing overlapping anything."""
    plt = _import("matplotlib.pyplot")
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    width = 0.36
    positions = range(len(COMPONENTS))
    base_x = [pos - width / 2 for pos in positions]
    tuned_x = [pos + width / 2 for pos in positions]
    bar_style = {"width": width, "edgecolor": SLATE, "linewidth": 0.8}
    ax.bar(base_x, BASE, color=NAVY, label="Base Model", **bar_style)
    ax.bar(tuned_x, TUNED, color=AMBER, hatch="///", label="Tuned Model", **bar_style)

    for x, value in [*zip(base_x, BASE, strict=True), *zip(tuned_x, TUNED, strict=True)]:
        ax.text(
            x,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=SLATE,
        )

    # The annotation sits in the empty column above the zero-height headroom pair,
    # so it never crosses the neighbouring bars.
    ax.annotate(
        ANNOTATION,
        xy=(HEADROOM_INDEX, 0.075),
        xytext=(HEADROOM_INDEX, 0.60),
        ha="center",
        va="top",
        fontsize=9.5,
        fontstyle="italic",
        color=DARK_RED,
        arrowprops={"arrowstyle": "-|>", "color": DARK_RED, "linewidth": 1.4, "shrinkB": 2},
    )

    _style_axes(ax)
    fig.tight_layout()
    return fig


def _style_axes(ax: Any) -> None:  # noqa: ANN401 - matplotlib Axes
    """Apply the report chart styling: wrapped labels, title block, legend below."""
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(list(range(len(COMPONENTS))))
    ax.set_xticklabels(COMPONENTS, fontsize=10.5, color=SLATE)
    ax.set_ylabel("Satisfaction Rate / Mean Reward", fontsize=11, fontweight="bold", color=SLATE)
    ax.tick_params(axis="y", labelcolor=SLATE, labelsize=10)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#d9dde2", linewidth=0.7, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9c0c8")

    ax.set_title(TITLE, fontsize=13.5, fontweight="bold", color=SLATE, pad=26)
    ax.text(
        0.5,
        1.018,
        SUBTITLE,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=GREY,
    )
    # Legend in its own strip below the axis — inside the plot it collided with
    # the 1.000 value labels.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=1.8,
        columnspacing=2.4,
    )


def _optimize(path: Path) -> None:
    """Downscale to ``TARGET_WIDTH`` and quantize to 256 colors, in place."""
    image_module = _import("PIL.Image")
    image = image_module.open(path).convert("RGB")
    if image.width != TARGET_WIDTH:
        height = round(image.height * TARGET_WIDTH / image.width)
        image = image.resize((TARGET_WIDTH, height), image_module.LANCZOS)
    image.quantize(colors=256, method=image_module.Quantize.MEDIANCUT).save(path, optimize=True)


def main() -> None:
    """Render the chart and write the optimized PNG next to its report."""
    out_dir = Path(_arg("--out", str(DEFAULT_OUT)))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FILENAME

    figure = _build_figure()
    figure.savefig(path, facecolor="white", bbox_inches="tight")
    _optimize(path)

    print(f"Saved {path} ({path.stat().st_size // 1024}K)")


if __name__ == "__main__":
    main()
