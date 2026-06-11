#!/usr/bin/env python3
"""Generate the paper's figures as PDF for direct \\includegraphics use."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon,
)
import numpy as np

import os as _os
OUT = _os.environ.get("FIG_OUT_DIR") or _os.path.dirname(_os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#333",
    "axes.linewidth": 0.8,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#222",
    "ytick.color": "#222",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
})

PRIMARY = "#1f4e79"
ACCENT  = "#c0504d"
GREEN   = "#2e7d32"
ORANGE  = "#d77f33"
PURPLE  = "#6a4a8c"
NEUTRAL = "#8a8a8a"
LIGHT   = "#dbe5f1"


# ============================================================
# Figure 1: workflow — five phases on a clean horizontal pipeline,
# with the LLM agent annotated as the source of probing code (Phase 1)
# and candidate strategies (Phases 2-3).
# ============================================================
def _draw_icon_calibrate(ax, cx, cy, color):
    """Three small ascending bars — calibration / measurement icon."""
    base_y = cy - 0.30
    widths = 0.22
    for i, h in enumerate((0.30, 0.45, 0.60)):
        ax.add_patch(Rectangle((cx - 0.35 + i * 0.30, base_y),
                               widths, h, fc=color, ec='none', zorder=4))
    ax.plot([cx - 0.50, cx + 0.55], [base_y, base_y],
            color='#555', lw=0.7, zorder=4)


def _draw_icon_code_box(ax, cx, cy, color, scale=1.0, offset=(0, 0)):
    """A small code-file icon (folded corner + horizontal text lines)."""
    w, h = 0.65 * scale, 0.80 * scale
    x = cx - w / 2 + offset[0]
    y = cy - h / 2 + offset[1]
    # Page outline with folded corner
    ax.add_patch(Polygon(
        [(x, y), (x, y + h), (x + w * 0.7, y + h),
         (x + w, y + h * 0.75), (x + w, y), (x, y)],
        fc='white', ec=color, lw=0.9, zorder=4))
    # Fold triangle
    ax.add_patch(Polygon(
        [(x + w * 0.7, y + h), (x + w * 0.7, y + h * 0.75),
         (x + w, y + h * 0.75)],
        fc='#f0f0f0', ec=color, lw=0.6, zorder=5))
    # Code lines
    for i, frac in enumerate((0.55, 0.40, 0.25)):
        ax.plot([x + 0.10 * scale, x + w - 0.10 * scale],
                [y + h * frac, y + h * frac],
                color=color, lw=0.6, alpha=0.7, zorder=5)


def _draw_icon_pool(ax, cx, cy, color):
    """Stacked code modules — a pool of candidate strategies."""
    for i, off in enumerate(((-0.10, -0.10), (0.05, 0.0), (0.20, 0.10))):
        _draw_icon_code_box(ax, cx, cy, color, scale=0.85, offset=off)


def _draw_icon_hardware(ax, cx, cy, color):
    """Server / rack icon — three stacked horizontal bays with LEDs."""
    w, h = 0.90, 0.90
    x = cx - w / 2
    y = cy - h / 2
    ax.add_patch(Rectangle((x, y), w, h, fc='white', ec=color, lw=0.9,
                           zorder=4))
    for i in range(3):
        by = y + h - (i + 1) * (h / 3) + 0.04
        ax.add_patch(Rectangle((x + 0.06, by), w - 0.12, h / 3 - 0.08,
                               fc='#f4f4f4', ec=color, lw=0.5, zorder=5))
        ax.add_patch(Circle((x + w - 0.13, by + h / 6 - 0.04), 0.04,
                            fc=color, ec='none', zorder=6))


def _draw_icon_deployed(ax, cx, cy, color):
    """File icon with a green check — the deployed strategy."""
    _draw_icon_code_box(ax, cx, cy, color, scale=1.1)
    # Green check overlay
    cxc, cyc = cx + 0.27, cy - 0.27
    ax.add_patch(Circle((cxc, cyc), 0.18, fc=color, ec='none', zorder=6))
    ax.plot([cxc - 0.08, cxc - 0.02, cxc + 0.09],
            [cyc + 0.00, cyc - 0.06, cyc + 0.05],
            color='white', lw=1.4, solid_capstyle='round', zorder=7)


def _draw_icon_llm_bot(ax, cx, cy, color):
    """Small stylized robot head — LLM agent indicator."""
    # Antenna
    ax.plot([cx, cx], [cy + 0.30, cy + 0.50], color=color, lw=1.0, zorder=5)
    ax.add_patch(Circle((cx, cy + 0.55), 0.05, fc=color, ec='none', zorder=6))
    # Head
    ax.add_patch(FancyBboxPatch((cx - 0.30, cy - 0.25), 0.60, 0.55,
                                boxstyle="round,pad=0.04",
                                fc='white', ec=color, lw=1.0, zorder=5))
    # Eyes
    ax.add_patch(Circle((cx - 0.12, cy + 0.07), 0.05, fc=color,
                        ec='none', zorder=6))
    ax.add_patch(Circle((cx + 0.12, cy + 0.07), 0.05, fc=color,
                        ec='none', zorder=6))
    # Mouth
    ax.plot([cx - 0.12, cx + 0.12], [cy - 0.10, cy - 0.10],
            color=color, lw=1.0, zorder=6)


def _draw_icon_simulator(ax, cx, cy, color):
    """Sparkline graph — the predictive simulator."""
    import numpy as np
    xs = np.linspace(-0.30, 0.30, 7)
    ys_off = np.array([-0.05, 0.08, -0.02, 0.15, 0.05, 0.18, -0.08])
    ax.plot(cx + xs, cy + ys_off, color=color, lw=1.2, zorder=5)
    ax.scatter(cx + xs, cy + ys_off, s=10, c=color, edgecolors='none', zorder=6)
    # Horizontal baseline
    ax.plot([cx - 0.35, cx + 0.35], [cy - 0.18, cy - 0.18],
            color='#888', lw=0.6, zorder=4)


def _draw_icon_runner(ax, cx, cy, color):
    """Play triangle inside a circle — execution runner."""
    ax.add_patch(Circle((cx, cy), 0.28, fc='white', ec=color, lw=1.0,
                        zorder=5))
    tri = Polygon([(cx - 0.09, cy - 0.13),
                   (cx + 0.15, cy),
                   (cx - 0.09, cy + 0.13)],
                  fc=color, ec='none', zorder=6)
    ax.add_patch(tri)


def _draw_icon_dev_baselines(ax, cx, cy, color):
    """Three small books on a shelf — developer-supplied baselines."""
    book_w = 0.13
    base_y = cy - 0.22
    heights = (0.40, 0.50, 0.36)
    cols = ('#bbb', color, '#bbb')
    for i, (h, c) in enumerate(zip(heights, cols)):
        x_book = cx - book_w * 1.6 + i * book_w * 1.05
        ax.add_patch(Rectangle((x_book, base_y), book_w, h,
                               fc=c, ec=color, lw=0.6, zorder=5))
    ax.plot([cx - 0.28, cx + 0.28], [base_y, base_y],
            color='#555', lw=0.7, zorder=4)


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.set_xlim(0, 23.5); ax.set_ylim(-1.4, 9.6); ax.set_axis_off()

    # ---- Five phase boxes (taller to give icons their own row) ----
    bw, bh = 3.85, 3.80
    y_main = 0.6
    gap = 0.85  # wider gap so forward arrows are clearly visible
    xs = [0.45 + i * (bw + gap) for i in range(5)]
    phase_centers = [x + bw / 2 for x in xs]

    phases = [
        ("Phase 1", "Calibrate\nsimulator",
         "fit cost constants\nfrom LLM probes",
         "#fff1d4", ORANGE, "#7a4a00",
         "icon_calibrate"),
        ("Phase 2", "Seed pool",
         "time the\nbaseline strategy",
         "#dbe5f1", PRIMARY, "#15375a",
         "icon_code"),
        ("Phase 3", "Propose &\nrefine",
         "LLM mutates;\nsimulator scores",
         "#efe7f5", PURPLE, "#3d2d4f",
         "icon_pool"),
        ("Phase 4", "Validate on\nhardware",
         "compile + run\nsurvivors",
         "#fbe3e3", ACCENT, "#7a2b2b",
         "icon_hw"),
        ("Phase 5", "Emit deployed\nstrategy",
         "drop-in runtime\nmodule",
         "#e7f3ec", GREEN, "#1b5e20",
         "icon_deployed"),
    ]

    for x, (phase, title, body, fc, ec, tc, icon) in zip(xs, phases):
        ax.add_patch(FancyBboxPatch((x, y_main), bw, bh,
                                    boxstyle="round,pad=0.06",
                                    lw=1.4, ec=ec, fc=fc, zorder=2))
        # Phase label (top of box)
        ax.text(x + bw/2, y_main + bh - 0.32, phase,
                ha='center', va='center', fontsize=7.0,
                color=tc, weight='bold')
        # Title (just below phase label)
        ax.text(x + bw/2, y_main + bh - 1.00, title,
                ha='center', va='center', fontsize=8.6,
                color=tc, weight='bold')
        # Body description (middle of box)
        ax.text(x + bw/2, y_main + bh - 2.00, body,
                ha='center', va='center', fontsize=6.4,
                color='#333')
        # Icon: own row at the bottom, clear of all text above
        icon_cy = y_main + 0.55
        icon_cx = x + bw / 2
        if icon == "icon_calibrate":
            _draw_icon_calibrate(ax, icon_cx, icon_cy, ec)
        elif icon == "icon_code":
            _draw_icon_code_box(ax, icon_cx, icon_cy, ec)
        elif icon == "icon_pool":
            _draw_icon_pool(ax, icon_cx, icon_cy, ec)
        elif icon == "icon_hw":
            _draw_icon_hardware(ax, icon_cx, icon_cy, ec)
        elif icon == "icon_deployed":
            _draw_icon_deployed(ax, icon_cx, icon_cy, ec)

    # Forward arrows
    for i in range(4):
        x1 = xs[i] + bw
        x2 = xs[i + 1]
        ax.add_patch(FancyArrowPatch((x1, y_main + bh/2),
                                     (x2, y_main + bh/2),
                                     arrowstyle="-|>,head_length=5,head_width=3.5",
                                     lw=1.6, color='#555', zorder=1))

    # 2-line "once per environment" annotation under Phase 1
    ax.text(xs[0] + bw / 2, y_main - 0.10,
            "(run once per\n(cluster, model, library) environment)",
            ha='center', va='top', fontsize=7.2,
            color='#7a4a00', style='italic', linespacing=1.2)

    # ---- Support row above the phases: four small boxes, one per role ----
    # Stacked layout per box: icon at top, label at bottom. Sits higher to
    # give the dashed connectors more room to breathe.
    sup_y = y_main + bh + 1.50
    sup_bh = 1.45
    sup_bw = 2.90
    supports = [
        ("LLM agent",     PURPLE,  "#3d2d4f", "icon_llm_bot",
         [0, 2], phase_centers[0]),
        ("Dev baselines", "#666",  "#222",    "icon_dev",
         [1],    phase_centers[1]),
        ("Simulator",     PRIMARY, "#15375a", "icon_sim",
         [0, 2, 3], phase_centers[2]),
        ("Runner",        ACCENT,  "#7a2b2b", "icon_runner",
         [1, 3], phase_centers[3]),
    ]

    sup_anchors = []  # bottom-center of each support box (for dashed arrows)
    for name, ec, tc, icon, conn, cx in supports:
        sx = cx - sup_bw / 2
        ax.add_patch(FancyBboxPatch((sx, sup_y), sup_bw, sup_bh,
                                    boxstyle="round,pad=0.05",
                                    lw=1.1, ec=ec, fc='white', zorder=3))
        # Icon centered in the upper portion of the box
        icon_cx = sx + sup_bw / 2
        icon_cy = sup_y + sup_bh * 0.66
        if icon == "icon_llm_bot":
            _draw_icon_llm_bot(ax, icon_cx, icon_cy, ec)
        elif icon == "icon_dev":
            _draw_icon_dev_baselines(ax, icon_cx, icon_cy, ec)
        elif icon == "icon_sim":
            _draw_icon_simulator(ax, icon_cx, icon_cy, ec)
        elif icon == "icon_runner":
            _draw_icon_runner(ax, icon_cx, icon_cy, ec)
        # Label centered in the lower portion of the box
        ax.text(sx + sup_bw / 2, sup_y + sup_bh * 0.18, name,
                ha='center', va='center', fontsize=7.8,
                color=tc, weight='bold')
        # Bottom-center anchor for line departures
        sup_anchors.append((cx, sup_y, conn, ec))

    # Dashed connectors from each support to its phases. We anchor the line
    # at the bottom-center of the support, and the top of the phase box.
    for cx, by, conn, ec in sup_anchors:
        for p_idx in conn:
            tx = phase_centers[p_idx]
            ty = y_main + bh + 0.02
            ax.add_patch(FancyArrowPatch((cx, by),
                                         (tx, ty),
                                         arrowstyle="-|>,head_length=3.0,head_width=2.0",
                                         lw=0.9, color=ec, ls=(0, (3, 2)),
                                         alpha=0.85, zorder=1))

    plt.savefig(f"{OUT}/workflow.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()


# ============================================================
# Figure 2: HW vs training "where does the agent win?"
# ============================================================
def fig_disagreement():
    """Cross-scope inversion chart: per-problem agent/baseline ratio at
    three measurement scopes. Agent column is the median across the
    three agent search styles (strategy-enumerate, cc-react,
    multi-island). Bench scopes use the cold-NEFF developer-faithful
    v6/v7 benches (WARMUP=0, warm_med of post-cold iterations). The
    7-node training scope reports the primitive's per-step contribution
    (per-call latency * calls-per-step), so the M:1 bundling
    amortisation is directly visible.

    Uniform A2A is included via the 1-node \texttt{train_ua2a_sweep_7node.py}
    probe (chunk=16384) — the agent's dispatch-ratio over baseline is
    1.81$\\times$ and is shape-stable from 1n to 7n on the
    AG+slice vs AG+T+RS pair."""
    fig, ax = plt.subplots(figsize=(10.0, 3.4))

    problems = ["AllToAllV", "Uniform\nA2A", "Ring KV", "dxe",
                "PP cross-\nstage", "TP MLP", "FSDP",
                "Layer\nblock\nAR"]
    # baseline/agent ratios (>1 = agent faster, <1 = agent slower).
    # 1-node and 7-node bench: warm_med per call from v6/v7.
    # 7-node training: per-step contribution ratio (per-call * calls/step).
    # Uniform A2A: r125 probe gives 1.81x agent-faster per dispatch (shape-stable 1n-7n);
    # 1-node and 7-node bench cells from tab:perproblem (0.98 = 46.95/47.87, 1.06 = 55.4/52.3).
    # Ring KV and dxe bars updated to match tab:perproblem (strat-enum reference):
    #   Ring KV  1n = 3.57/1.07 = 3.34, 7n = 2.95/2.02 = 1.46, train = 13.49/10.32 = 1.31
    #   dxe      1n = 1.64/0.37 = 4.43, 7n = 1.87/0.57 = 3.28, train = 16.74/14.63 = 1.14
    hw_1node = [0.98, 0.98, 3.34, 4.43, 0.75, 0.20, 0.61, 0.45]
    hw_7node = [0.87, 1.06, 1.46, 3.28, 0.57, 0.51, 0.99, 0.26]
    training = [1.30, 1.81, 1.31, 1.14, 5.22, 4.16, 4.20, 4.16]

    x = np.arange(len(problems))
    w = 0.26

    bars1 = ax.bar(x - w, hw_1node, w, label="1-node HW microbench",
                   color=NEUTRAL)
    bars2 = ax.bar(x, hw_7node, w, label="7-node HW microbench",
                   color=ORANGE)
    bars3 = ax.bar(x + w, training, w, label="7-node real training (per call)",
                   color=PRIMARY)

    ax.axhline(1.0, color="#444", lw=0.8, ls="--")
    ax.text(len(problems) + 0.05, 1.0, "agent = baseline",
            fontsize=8, color="#444",
            ha="left", va="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    for b in list(bars1) + list(bars2) + list(bars3):
        v = b.get_height()
        if not np.isfinite(v):
            continue
        ax.text(b.get_x() + b.get_width()/2, v * 1.06, f"{v:.2f}$\\times$",
                ha="center", va="bottom", color="#222",
                fontsize=5.5)

    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    ax.set_yticklabels(["0.05$\\times$", "0.1$\\times$", "0.2$\\times$",
                        "0.5$\\times$", "1$\\times$", "2$\\times$",
                        "5$\\times$"])
    ax.set_ylim(0.04, 10.0)
    ax.set_xticks(x); ax.set_xticklabels(problems, fontsize=9)
    ax.set_xlim(-0.6, len(problems) + 1.1)
    ax.set_ylabel("speedup (baseline$/$agent)")
    ax.set_title("Where the agent wins: 1-node bench $\\to$ 7-node bench $\\to$ 7-node training (per-step contribution)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3, fontsize=8)

    plt.savefig(f"{OUT}/disagreement.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()


# ============================================================
# Figure 3: 7-node OLMoE end-to-end speedup
# ============================================================
def fig_speedup():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.9),
                              gridspec_kw=dict(wspace=0.45))

    panels = [
        ("Wall clock",          "min", 23.9,    15.9),
        ("Avg step",            "ms",  4778.0,  3658.9),
        ("Steady step ($\\geq$200)", "ms", 4399.1, 3130.4),   # ONE-LINE title
    ]
    for ax, (title, unit, b, a) in zip(axes, panels):
        ax.bar([0, 1], [b, a], width=0.55, color=[NEUTRAL, PRIMARY])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["baseline", "agent"])
        ax.set_ylabel(unit)
        ax.set_title(title, pad=4)
        ax.set_ylim(0, b * 1.30)
        for i, v in enumerate([b, a]):
            ax.text(i, v + b * 0.02,
                    f"{v:,.1f}".rstrip("0").rstrip("."),
                    ha="center", va="bottom", fontsize=9)
        ax.text(0.5, b * 1.18, f"{b/a:.3f}$\\times$",
                ha="center", va="center",
                fontsize=12, color=ACCENT, weight="bold")

    fig.suptitle("7-node OLMoE-style training (300 steps, 224 ranks): "
                 "baseline vs full-agent stack",
                 fontsize=10, y=1.10)
    plt.subplots_adjust(top=0.80)
    plt.savefig(f"{OUT}/speedup.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close()


# ============================================================
# Figure 4: contiguity — DENSE permute vs SUB-REGION narrow
# ============================================================
def fig_contiguity():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6),
                              gridspec_kw=dict(wspace=0.35))

    def draw_grid(ax, ncols, nrows, fill_cols, label, cost_str, cost_color):
        for r in range(nrows):
            for c in range(ncols):
                fc = PRIMARY if c in fill_cols else "#dde6f0"
                ec = "#15375a" if c in fill_cols else "#aab"
                ax.add_patch(Rectangle((c, nrows - 1 - r), 1, 1,
                                       fc=fc, ec=ec, lw=0.5))
        ax.set_xlim(-0.3, ncols + 0.3)
        ax.set_ylim(-1.5, nrows + 0.5)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.text(ncols / 2, nrows + 0.15, label, ha="center", va="bottom",
                fontsize=9, weight="bold", color="#15375a")
        ax.text(ncols / 2, -0.7, cost_str, ha="center", va="top",
                fontsize=8.5, color=cost_color, weight="bold")

    ax = axes[0]
    draw_grid(ax, 6, 3, list(range(6)),
              "DENSE permute(...).reshape(-1)",
              "bytes / sequential bw",
              GREEN)

    ax = axes[1]
    draw_grid(ax, 6, 3, [2],
              "SUB-REGION narrow(dim=1, k, 1).reshape(-1)",
              "bytes / strided bw ($\\sim$10$\\times$ slower)",
              ACCENT)

    plt.savefig(f"{OUT}/contiguity.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()


if __name__ == "__main__":
    fig_workflow()
    fig_disagreement()
    fig_speedup()
    fig_contiguity()
    print("ok")
