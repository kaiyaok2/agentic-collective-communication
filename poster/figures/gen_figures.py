"""Generate the four poster figures as SVG (vector, prints crisply at any
poster size). Run from this directory:
    python gen_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

# Common style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333",
    "axes.linewidth": 1.0,
    "axes.labelcolor": "#222",
    "xtick.color": "#222",
    "ytick.color": "#222",
})

PRIMARY = "#1f4e79"   # deep blue
ACCENT  = "#c0504d"   # red
NEUTRAL = "#7f7f7f"
LIGHT   = "#dbe5f1"
GREEN   = "#2e7d32"


# ============================================================
# Figure 1: 5-phase workflow OVERVIEW (vivid, paper-style)
# Central LLM agent surrounded by tools, simulator, hardware, runtime,
# with annotated data-flow arrows between phases.
# ============================================================
def make_workflow():
    # Compact wide-landscape aspect (~16:6) so the band doesn't dominate
    # the poster vertically. Body text is intentionally terse — the
    # poster's text sections explain the details.
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()

    # ---- color palette ----
    BLUE   = PRIMARY        # cool / search side
    DARK   = PRIMARY_DK = "#15375a"
    ORANGE = "#d77f33"      # profiling
    PURPLE = "#6a4a8c"      # simulator
    RED    = ACCENT         # hardware
    GREENC = GREEN          # codegen / output
    LLM    = "#3b6ea8"

    def card(x, y, w, h, color, title, body, body_size=9.2, title_color="white"):
        # body
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            linewidth=1.6, edgecolor=color, facecolor="#fff"))
        # title strip
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, y + h - 0.85), w - 0.08, 0.78,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0, facecolor=color))
        ax.text(x + w/2, y + h - 0.46, title,
                ha="center", va="center", fontsize=11.0,
                fontweight="bold", color=title_color)
        ax.text(x + w/2, y + (h - 0.85)/2 + 0.05, body,
                ha="center", va="center", fontsize=body_size,
                color="#1a1a1a")

    def arrow(p1, p2, color="#333", lw=1.8, mut=14, rad=0.0,
              label=None, label_offset=(0, 0), label_color="#333",
              label_size=8.5, style="italic"):
        ar = FancyArrowPatch(
            p1, p2, arrowstyle="-|>",
            mutation_scale=mut, linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(ar)
        if label:
            mx = (p1[0] + p2[0]) / 2 + label_offset[0]
            my = (p1[1] + p2[1]) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center",
                    fontsize=label_size, color=label_color,
                    style=style)

    # ---- Layout for a compact 16 × 6.2 canvas ----
    # Rows (y ranges):
    #   TOP   y = 4.50..5.95   profiling | hardware | composition-space
    #   GAP                    arrows (rows are 0.55 tall, no overlap)
    #   MID   y = 2.40..3.95   search/evolution | LLM agent | simulator
    #   GAP
    #   LOW   y = 0.20..1.85   Phase 4 (wide) | Phase 5
    # Body text is terse — the poster sections explain the details.

    # ---- TOP row ---------------------------------------------------------
    # Card layout: narrower cards (4.20 / 4.80 / 4.20) with WIDER inter-
    # card gaps (0.95 each). Wide gaps give the arrows + labels room
    # without overlapping card borders.
    top_y, top_h = 4.50, 1.45

    card(0.30, top_y, 4.20, top_h, ORANGE,
         "Phase 1 · Hardware Profiling Tools",
         "13 measurement tools probe real Trainium:\n"
         "latency · per-op µs · graph_launch ·\n"
         "NEFF cost · seq + strided memcpy bw",
         body_size=9.0)

    card(5.60, top_y, 4.80, top_h, RED,
         "Real AWS Trainium  (trn1.32xlarge)",
         "16 NeuronDevices · 4×4 torus\n"
         "NeuronLink ~192 GB/s · 8× EFA\n"
         "closed: AG · RS · AR · CP · A2A",
         body_size=9.0)

    card(11.50, top_y, 4.20, top_h, "#7f7f7f",
         "Composition space  (the search space)",
         "how many xm.* dispatches · in what\n"
         "order · against what tensor layout ·\n"
         "with what local XLA work in between",
         body_size=9.0)

    # ---- MID row ---------------------------------------------------------
    # Same x-positions as TOP row.
    mid_y, mid_h = 2.40, 1.55

    card(0.30, mid_y, 4.20, mid_h, BLUE,
         "Phase 2/3 · Search & Evolution",
         "Phase 2 — score builtin templates\n"
         "Phase 3 — multi-island LLM evolution\n"
         "sandbox → correct → sim → feedback",
         body_size=9.0)

    cx, cy = 8.0, mid_y + mid_h / 2
    cr_w, cr_h = 4.80, mid_h
    ax.add_patch(FancyBboxPatch(
        (cx - cr_w/2, cy - cr_h/2), cr_w, cr_h,
        boxstyle="round,pad=0.06,rounding_size=0.30",
        linewidth=2.4, edgecolor=LLM, facecolor="#eaf2fb"))
    ax.add_patch(FancyBboxPatch(
        (cx - cr_w/2 + 0.08, cy + cr_h/2 - 0.55), cr_w - 0.16, 0.50,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=0, facecolor=LLM))
    ax.text(cx, cy + cr_h/2 - 0.30,
            "LLM Agent  (Sonnet 4.6)",
            ha="center", va="center", fontsize=12.5,
            fontweight="bold", color="white")
    ax.text(cx, cy - 0.15,
            "Phase 1 — calls profiling tools\n"
            "Phase 3 — proposes candidate code; iterates\n"
            "on simulator-score feedback "
            "(no HW reward signal)",
            ha="center", va="center", fontsize=9.4, color="#1a1a1a")

    card(11.50, mid_y, 4.20, mid_h, PURPLE,
         "Reward-hacking-free Simulator",
         "predicts per-step TRAINING latency:\n"
         "• graph_launch per mark_step pair\n"
         "• contiguity-aware reshape / contig copy\n"
         "• volume-scaled index_select · tensor",
         body_size=8.8)

    # ---- LOW row ---------------------------------------------------------
    # Phase 4 width chosen so its right edge aligns with the LLM card
    # (10.40), giving a wide 1.10 gap before Phase 5 — same as every
    # other inter-card gap. The "winner" arrow + label fit cleanly.
    low_y, low_h = 0.30, 1.55

    card(0.30, low_y, 10.10, low_h, RED,
         "Phase 4 · Hardware Validation  (gates only — not the ranker)",
         "On-Trainium 20-iter microbench  AND  10-step bf16 training\n"
         "validation. Both are correctness / feasibility GATES; the\n"
         "simulator's per-step prediction is the ranking signal.",
         body_size=9.5)

    card(11.50, low_y, 4.20, low_h, GREENC,
         "Phase 5 · Code Generation",
         "winner emitted to\n"
         "runtime/trainium_<problem>.py\n"
         "drop-in for training scripts",
         body_size=9.5)

    # ---- ARROWS  (placed entirely in the GAPS — never on box borders) ----
    y_gap1 = (mid_y + mid_h + top_y) / 2.0   # gap between TOP and MID rows
    y_gap2 = (low_y + low_h + mid_y) / 2.0   # gap between MID and LOW rows

    def labeled_arrow(x_from, y_from, x_to, y_to, color, label,
                      label_xy, lw=1.5, fontsize=8.6, weight="normal"):
        """Arrow + label placed at an explicit (x, y); label_xy is in
        figure coords. Caller is responsible for placing label_xy
        cleanly between cards."""
        ax.annotate("",
                    xy=(x_to, y_to), xytext=(x_from, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
        ax.text(label_xy[0], label_xy[1], label,
                ha="center", va="center", fontsize=fontsize,
                color=color, style="italic", fontweight=weight)

    # ---- TOP-row arrows: Phase 1 ↔ Hardware (probe / measure) ----
    # Gap between cards is now 1.10 wide (was 0.55); labels fit easily.
    g_left  = 4.50 + 0.05          # right edge of Phase-1 card body
    g_right = 5.60 - 0.05          # left edge of Trainium card body
    y_top_arrow = top_y + top_h - 0.45
    y_bot_arrow = top_y + 0.45
    ax.annotate("", xy=(g_right + 0.03, y_top_arrow),
                    xytext=(g_left - 0.03, y_top_arrow),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=ORANGE, lw=1.4))
    ax.text((g_left + g_right) / 2, y_top_arrow + 0.13, "probe",
            ha="center", va="bottom", fontsize=8.2, color=ORANGE,
            style="italic")
    ax.annotate("", xy=(g_left - 0.03, y_bot_arrow),
                    xytext=(g_right + 0.03, y_bot_arrow),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=RED, lw=1.4))
    ax.text((g_left + g_right) / 2, y_bot_arrow - 0.13, "measure",
            ha="center", va="top", fontsize=8.2, color=RED,
            style="italic")

    # ---- TOP → MID: Phase-1 Tools → Phase 2/3 Search ----
    labeled_arrow(2.30, top_y - 0.04,
                  2.30, mid_y + mid_h + 0.04,
                  ORANGE, "cost-model parameters",
                  label_xy=(3.40, y_gap1))

    # ---- MID-row inter-card arrows: Phase 2/3 ↔ LLM, LLM ↔ Simulator ----
    # Gaps are now 1.10 wide so labels like "candidate" / "sim time" fit
    # comfortably without overlapping card borders.
    g_l = 4.50
    g_r = cx - cr_w/2                  # = 5.60
    ax.annotate("", xy=(g_r - 0.02, mid_y + mid_h - 0.45),
                    xytext=(g_l + 0.02, mid_y + mid_h - 0.45),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    ax.text((g_l + g_r)/2, mid_y + mid_h - 0.30, "code",
            ha="center", va="bottom", fontsize=8.2, color=BLUE,
            style="italic")
    ax.annotate("", xy=(g_l + 0.02, mid_y + 0.45),
                    xytext=(g_r - 0.02, mid_y + 0.45),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    ax.text((g_l + g_r)/2, mid_y + 0.60, "score",
            ha="center", va="top", fontsize=8.2, color=BLUE,
            style="italic")

    g_l = cx + cr_w/2                  # = 10.40
    g_r = 11.50
    ax.annotate("", xy=(g_r - 0.02, mid_y + mid_h - 0.45),
                    xytext=(g_l + 0.02, mid_y + mid_h - 0.45),
                    arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4))
    ax.text((g_l + g_r)/2, mid_y + mid_h - 0.30, "candidate",
            ha="center", va="bottom", fontsize=8.2, color=PURPLE,
            style="italic")
    ax.annotate("", xy=(g_l + 0.02, mid_y + 0.45),
                    xytext=(g_r - 0.02, mid_y + 0.45),
                    arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4))
    ax.text((g_l + g_r)/2, mid_y + 0.60, "sim time",
            ha="center", va="top", fontsize=8.2, color=PURPLE,
            style="italic")

    # ---- MID → LOW ----
    labeled_arrow(3.0, mid_y - 0.04,
                  3.0, low_y + low_h + 0.04,
                  RED, "top-K candidates",
                  label_xy=(4.30, y_gap2))
    labeled_arrow(8.0, mid_y - 0.04,
                  8.0, low_y + low_h + 0.04,
                  PURPLE, "rank by simulator",
                  label_xy=(6.50, y_gap2))

    # ---- Phase 4 → Phase 5 (horizontal in the 1.10-wide card gap) ----
    labeled_arrow(10.45, low_y + low_h/2,
                  11.45, low_y + low_h/2,
                  GREENC, "winner",
                  label_xy=(10.95, low_y + low_h/2 + 0.25),
                  lw=2.2, weight="bold", fontsize=9.5)

    plt.tight_layout()
    plt.savefig("workflow.svg", bbox_inches="tight", pad_inches=0.10)
    plt.close()
    print("  wrote workflow.svg")


# ============================================================
# Figure 2: Speedup bar chart (4 problems)
# ============================================================
def make_speedup():
    problems = ["AllToAllV", "Uniform AllToAll",
                "Fused\nReduceScatter", "Ring Attention\nKV"]
    agent =    [655.1, 655.8, 421.2, 466.9]
    baseline = [816.0, 816.5, 428.2, 474.0]
    speedups = [b/a for a, b in zip(agent, baseline)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4),
                                    gridspec_kw={"width_ratios": [1.6, 1]})

    # ---- Left panel: per-step latency comparison
    x = np.arange(len(problems))
    w = 0.36
    bars1 = ax1.bar(x - w/2, baseline, w, color=NEUTRAL,
                     label="developer baseline", edgecolor="white")
    bars2 = ax1.bar(x + w/2, agent, w, color=PRIMARY,
                     label="agent runtime", edgecolor="white")
    for b, v in zip(bars1, baseline):
        ax1.text(b.get_x() + b.get_width()/2, v + 7, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9, color="#222")
    for b, v in zip(bars2, agent):
        ax1.text(b.get_x() + b.get_width()/2, v + 7, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9,
                 color=PRIMARY, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(problems, fontsize=10)
    ax1.set_ylabel("avg per-step latency (ms)", fontsize=11)
    ax1.set_ylim(0, max(baseline) * 1.18)
    ax1.set_title("5000-step DeepSeek-MoE-Lite (1× trn1.32xlarge, bf16)",
                   fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.95)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ---- Right panel: speedup factor
    bars = ax2.barh(np.arange(len(problems)), speedups,
                     color=PRIMARY, edgecolor="white")
    for i, s in enumerate(speedups):
        ax2.text(s + 0.005, i, f"{s:.3f}×",
                 va="center", ha="left", fontsize=10,
                 color=PRIMARY, fontweight="bold")
    ax2.set_yticks(np.arange(len(problems)))
    ax2.set_yticklabels([p.replace("\n", " ") for p in problems],
                         fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("speedup vs baseline", fontsize=11)
    ax2.set_xlim(0.95, max(speedups) * 1.10)
    ax2.axvline(1.0, color="#333", linewidth=1)
    ax2.set_title("Per-problem speedup", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("speedup.svg", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print("  wrote speedup.svg")


# ============================================================
# Figure 3: Per-call HW vs 5000-step disagreement
# ============================================================
def make_disagreement():
    problems = ["AllToAllV", "Uniform A2A", "Fused RS", "Ring KV"]
    agent_call =     [2.75,  3.67,  2.33,  0.87]
    baseline_call =  [1.66,  8.63,  1.46,  4.73]
    agent_step =     [655.1, 655.8, 421.2, 466.9]
    baseline_step =  [816.0, 816.5, 428.2, 474.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    # Per-call
    x = np.arange(len(problems))
    w = 0.36
    ax1.bar(x - w/2, baseline_call, w, color=NEUTRAL,
            label="baseline", edgecolor="white")
    ax1.bar(x + w/2, agent_call, w, color=PRIMARY,
            label="agent", edgecolor="white")
    for i, (a, b) in enumerate(zip(agent_call, baseline_call)):
        flip = "✗" if a > b else "✓"
        color = ACCENT if a > b else GREEN
        ax1.text(i, max(a, b) + 0.4, flip,
                 ha="center", va="bottom", fontsize=14,
                 color=color, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(problems, fontsize=10)
    ax1.set_ylabel("per-call HW microbench (ms)", fontsize=11)
    ax1.set_title("Per-call HW (20-iter isolated, world=32)",
                   fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.95)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim(0, max(baseline_call) * 1.25)

    # Per-step
    ax2.bar(x - w/2, baseline_step, w, color=NEUTRAL,
            label="baseline", edgecolor="white")
    ax2.bar(x + w/2, agent_step, w, color=PRIMARY,
            label="agent", edgecolor="white")
    for i, (a, b) in enumerate(zip(agent_step, baseline_step)):
        flip = "✗" if a > b else "✓"
        color = ACCENT if a > b else GREEN
        ax2.text(i, max(a, b) + 30, flip,
                 ha="center", va="bottom", fontsize=14,
                 color=color, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(problems, fontsize=10)
    ax2.set_ylabel("5000-step training avg (ms / step)", fontsize=11)
    ax2.set_title("Real training (DeepSeek-MoE-Lite, 5000 steps)",
                   fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.95)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(0, max(baseline_step) * 1.18)

    fig.suptitle("Why per-call ≠ per-step:  isolated microbench prefers "
                 "baseline for 2 of 4 problems, but training prefers agent "
                 "in all 4",
                 fontsize=12, color="#444", y=1.02)

    plt.tight_layout()
    plt.savefig("disagreement.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print("  wrote disagreement.svg")


# ============================================================
# Figure 4: Cost-model dense vs sub-region copy schematic
# ============================================================
def make_cost_model():
    # Compact canvas (13 × 3.6) — small ylim so the title sits just above
    # the grid figure instead of leaving a big white gap.
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))

    YMAX = 3.8

    # ---- LEFT: dense (permute) copy ----
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, YMAX)

    ax.text(5.5, 3.50, "DENSE view: permute(...).reshape(-1)",
            ha="center", fontsize=12, fontweight="bold", color=PRIMARY)
    ax.text(5.5, 3.13, "source covers full storage with permuted strides",
            ha="center", fontsize=10, color="#444", style="italic")

    # Source storage (4x4 grid, all cells filled)
    sx, sy = 0.6, 0.6
    cs = 0.55
    rows, cols = 4, 4
    for i in range(rows):
        for j in range(cols):
            r = Rectangle((sx + j * cs, sy + (rows - 1 - i) * cs), cs, cs,
                          facecolor=LIGHT, edgecolor=PRIMARY, linewidth=1.0)
            ax.add_patch(r)
            ax.text(sx + j * cs + cs/2, sy + (rows - 1 - i) * cs + cs/2,
                    f"{i*cols+j}", ha="center", va="center", fontsize=8,
                    color=PRIMARY)
    ax.text(sx + cols * cs / 2, sy - 0.30,
            "source: shape (4,4),\nstride (4,1) — contiguous",
            ha="center", fontsize=9, color="#222")

    # Arrow
    arrow_y = sy + (rows * cs) / 2
    ax.annotate("", xy=(5.0, arrow_y),
                xytext=(sx + cols * cs + 0.1, arrow_y),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    # Label sits above the arrow, pushed left so it doesn't crowd the
    # output-cell strip that starts at x = dx = 5.4.
    ax.text(4.10, arrow_y + 0.40,
            "permute(1,0)\n.reshape(-1)",
            ha="center", fontsize=9, color="#444")

    # Output storage — vertically centered around the arrow
    dx = 5.4
    out_h = 0.55
    out_y = arrow_y - out_h / 2 + 0.5     # slight upward bias
    for k in range(16):
        r = Rectangle((dx + k * 0.32, out_y), 0.32, out_h,
                      facecolor=LIGHT, edgecolor=PRIMARY, linewidth=1.0)
        ax.add_patch(r)
    ax.text(dx + 16 * 0.32 / 2, out_y - 0.40,
            "every byte read once · predictable strides\n"
            "→ compiler vectorizes",
            ha="center", fontsize=9, color=GREEN, fontweight="bold")
    ax.text(dx + 16 * 0.32 / 2, out_y - 0.95,
            "charged at  bytes / sequential_memcpy_bw",
            ha="center", fontsize=9.5, color=PRIMARY, fontweight="bold")

    # ---- RIGHT: sub-region (narrow) copy ----
    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, YMAX)

    ax.text(5.5, 3.50, "SUB-REGION view: narrow(dim=1, k, 1).reshape(-1)",
            ha="center", fontsize=12, fontweight="bold", color=ACCENT)
    ax.text(5.5, 3.13, "source covers MORE storage than tensor needs",
            ha="center", fontsize=10, color="#444", style="italic")

    # Source storage (4x4 grid, only column-1 highlighted)
    sx, sy = 0.6, 0.6
    cs = 0.55
    rows, cols = 4, 4
    for i in range(rows):
        for j in range(cols):
            highlighted = (j == 1)
            r = Rectangle((sx + j * cs, sy + (rows - 1 - i) * cs), cs, cs,
                          facecolor=("#fdd" if highlighted else "#eee"),
                          edgecolor=(ACCENT if highlighted else NEUTRAL),
                          linewidth=(1.4 if highlighted else 0.9))
            ax.add_patch(r)
            ax.text(sx + j * cs + cs/2, sy + (rows - 1 - i) * cs + cs/2,
                    f"{i*cols+j}", ha="center", va="center", fontsize=8,
                    color=(ACCENT if highlighted else NEUTRAL))
    ax.text(sx + cols * cs / 2, sy - 0.30,
            "source: shape (4,4),\nnarrowed to (4,1) — non-leading dim",
            ha="center", fontsize=9, color="#222")

    # Arrow
    arrow_y = sy + (rows * cs) / 2
    ax.annotate("", xy=(5.0, arrow_y),
                xytext=(sx + cols * cs + 0.1, arrow_y),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    # Label sits above the arrow, pushed left so it doesn't crowd the
    # output-cell strip that starts at x = dx = 5.4.
    ax.text(4.10, arrow_y + 0.40,
            "narrow(1,1,1)\n.reshape(-1)",
            ha="center", fontsize=9, color="#444")

    # Output storage
    dx = 5.4
    out_h = 0.55
    out_y = arrow_y - out_h / 2 + 0.5
    for k in range(4):
        r = Rectangle((dx + k * 1.28, out_y), 1.28, out_h,
                      facecolor="#fdd", edgecolor=ACCENT, linewidth=1.4)
        ax.add_patch(r)
    ax.text(dx + 4 * 1.28 / 2, out_y - 0.40,
            "every read skips 3 elements of source\n"
            "→ sub-cache-line gather, low effective bw",
            ha="center", fontsize=9, color=ACCENT, fontweight="bold")
    ax.text(dx + 4 * 1.28 / 2, out_y - 0.95,
            "charged at  bytes / strided_memcpy_bw  (≈ 10× slower)",
            ha="center", fontsize=9.5, color=ACCENT, fontweight="bold")

    fig.suptitle("Detected at trace time via PyTorch tensor strides — "
                 "no leak to the agent",
                 fontsize=11, color="#444", y=1.02, style="italic")

    plt.tight_layout()
    plt.savefig("cost_model.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print("  wrote cost_model.svg")


if __name__ == "__main__":
    make_workflow()
    make_speedup()
    make_disagreement()
    make_cost_model()
    print("done")
