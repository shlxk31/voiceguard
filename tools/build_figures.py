"""
tools/build_figures.py

Generates the paper's figures as high-resolution PNGs for embedding in the
.docx. All plotted values are either measured in this project or taken from
the cited published tables (Jung et al. ICASSP 2022; Müller et al.
Interspeech 2022) -- no illustrative/invented numbers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#2f855a"
AMBER = "#b7791f"
GRID = "#d8d8d8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": "#999999",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


# ---------------------------------------------------- Fig 1: architecture --
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, label, sub="", fc="#ffffff", ec=INK, lw=1.1, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.6,rounding_size=1.2",
                     linewidth=lw, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), label,
                ha="center", va="center", fontsize=fs, weight="bold", color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center",
                    fontsize=7, color=MUTED)

    def arrow(x1, y1, x2, y2, color=INK, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=9, linewidth=1.0, color=color))

    box(1, 16, 13, 11, "Voice input", "16 kHz mono", fc="#f4f7fb")
    arrow(14.4, 21.5, 18.6, 21.5)

    box(19, 16, 17, 11, "Layer 2", "ECAPA-TDNN\nspeaker verification", fc="#eef4fb", ec=BLUE)
    arrow(36.4, 21.5, 40.6, 21.5)

    box(41, 16, 17, 11, "Layer 2.5", "AASIST-L\nanti-spoofing", fc="#fdf0ef", ec=RED)
    arrow(58.4, 21.5, 62.6, 21.5)

    box(63, 16, 16, 11, "Decision", "trust score +\nspoof gate", fc="#f6f6f6")
    arrow(79.4, 24.5, 84.6, 30.0, color=GREEN)
    arrow(79.4, 18.5, 84.6, 13.0, color=RED)

    box(85, 26, 14, 8.5, "APPROVE", fc="#eaf6ee", ec=GREEN, fs=8)
    box(85, 8, 14, 8.5, "DENY", fc="#fdeaea", ec=RED, fs=8)

    ax.text(27.5, 12.6, "“is this the\nright voice?”", ha="center", va="top",
            fontsize=7, style="italic", color=BLUE)
    ax.text(49.5, 12.6, "“is this a\nreal human?”", ha="center", va="top",
            fontsize=7, style="italic", color=RED)

    ax.text(50, 38.5, "Both layers must pass; either can deny independently",
            ha="center", fontsize=7.5, color=MUTED)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ Fig 2: the domain gap ----
def fig_domain_gap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                    gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) literature + ours: benchmark vs real-world
    labels = ["RawGAT-ST\n[Müller'22]", "RawNet2\n[Müller'22]", "12-model avg\n[Müller'22]",
              "AASIST-L\n(this work)"]
    bench = [1.229, 3.154, 9.85, 0.99]
    wild = [37.154, 37.819, 60.10, np.nan]
    x = np.arange(len(labels)); w = 0.36

    ax1.bar(x - w / 2, bench, w, label="Benchmark (ASVspoof19 LA)", color=BLUE)
    ax1.bar(x + w / 2, wild, w, label="Real-world audio", color=RED)
    ax1.scatter([x[3] + w / 2], [100], marker="v", s=42, color=RED, zorder=5)
    ax1.text(x[3] + w / 2, 92, "100%\nFPR", ha="center", va="top", fontsize=6.5,
             color=RED, weight="bold")
    for i, v in enumerate(bench):
        ax1.text(x[i] - w / 2, v + 1.5, f"{v:.2f}", ha="center", fontsize=6.2, color=BLUE)
    for i, v in enumerate(wild[:3]):
        ax1.text(x[i] + w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.2, color=RED)

    ax1.set_ylabel("EER (%)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=6.4)
    ax1.set_ylim(0, 108)
    ax1.legend(fontsize=6.3, frameon=False, loc="upper left")
    ax1.set_title("(a) Benchmark vs. real-world performance", fontsize=8)
    ax1.grid(axis="y", color=GRID, linewidth=0.6)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # (b) spectral diagnosis
    bands = ["0–300 Hz", "6–8 kHz", "Noise floor\n(×100)"]
    lab_bona = [39.9, 0.38, 0.43]
    lab_spoof = [46.8, 0.48, 1.86]
    real = [53.8, 0.04, 7.02]
    x2 = np.arange(len(bands)); w2 = 0.26
    ax2.bar(x2 - w2, lab_bona, w2, label="ASVspoof bona fide", color=BLUE)
    ax2.bar(x2, lab_spoof, w2, label="ASVspoof spoof", color=AMBER)
    ax2.bar(x2 + w2, real, w2, label="Real phone (ours)", color=RED)
    ax2.set_ylabel("Energy share (%) / scaled")
    ax2.set_xticks(x2); ax2.set_xticklabels(bands, fontsize=7)
    ax2.legend(fontsize=6.3, frameon=False)
    ax2.set_title("(b) Why: spectral & noise-floor mismatch", fontsize=8)
    ax2.grid(axis="y", color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_domain_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------- Fig 3: degradation + streaming ----
def fig_degradation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    conds = ["Clean", "16 kbps\n5% loss", "8 kbps\n15% loss"]
    lfcc = [18.30, 21.03, 26.83]
    aasist = [1.17, 5.23, 14.00]
    x = np.arange(len(conds))
    ax1.plot(x, lfcc, "o-", color=MUTED, label="LFCC-GMM baseline", linewidth=1.4, markersize=4)
    ax1.plot(x, aasist, "s-", color=RED, label="AASIST-L", linewidth=1.4, markersize=4)
    for i, (a, b) in enumerate(zip(lfcc, aasist)):
        ax1.annotate(f"{b:.2f}", (i, b), textcoords="offset points", xytext=(0, -11),
                     ha="center", fontsize=6.3, color=RED)
        ax1.annotate(f"{a:.2f}", (i, a), textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=6.3, color=MUTED)
    ax1.annotate("advantage:\n15.6× → 1.9×", xy=(1.55, 21), fontsize=6.6,
                 color=INK, ha="center")
    ax1.set_xticks(x); ax1.set_xticklabels(conds, fontsize=7)
    ax1.set_ylabel("EER (%)"); ax1.set_ylim(0, 32)
    ax1.set_title("(a) VoIP channel degradation", fontsize=8)
    ax1.legend(fontsize=6.4, frameon=False, loc="upper left")
    ax1.grid(color=GRID, linewidth=0.6); ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    wins = ["Full\nutt.", "1000\nms", "500\nms", "250\nms"]
    eer = [0.99, 23.62, 33.28, 36.43]
    x2 = np.arange(len(wins))
    ax2.bar(x2, eer, 0.55, color=[GREEN, AMBER, "#d97706", RED])
    for i, v in enumerate(eer):
        ax2.text(i, v + 1.2, f"{v:.2f}", ha="center", fontsize=6.6)
    ax2.axhline(50, color=MUTED, linestyle=":", linewidth=0.9)
    ax2.text(3.35, 51, "chance", fontsize=6.2, color=MUTED, ha="right")
    ax2.set_xticks(x2); ax2.set_xticklabels(wins, fontsize=7)
    ax2.set_ylabel("Chunk-level EER (%)"); ax2.set_ylim(0, 58)
    ax2.set_title("(b) Streaming window size", fontsize=8)
    ax2.grid(axis="y", color=GRID, linewidth=0.6); ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_degradation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------- Fig 4: attack vs both layers --
def fig_attack():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                                    gridspec_kw={"width_ratios": [1, 1]})

    kinds = ["Genuine\nuser", "AI clone\nof user", "Different\nperson"]
    sim = [0.726, 0.476, 0.124]
    colors = [GREEN, RED, BLUE]
    x = np.arange(len(kinds))
    ax1.bar(x, sim, 0.5, color=colors)
    ax1.axhline(0.50, color=INK, linestyle="--", linewidth=1.0)
    ax1.text(2.45, 0.53, "accept threshold", fontsize=6.3, ha="right", color=INK)
    for i, v in enumerate(sim):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
    ax1.set_xticks(x); ax1.set_xticklabels(kinds, fontsize=7)
    ax1.set_ylabel("Mean speaker similarity"); ax1.set_ylim(0, 0.85)
    ax1.set_title("(a) Layer 2 alone: clones sit above the gate", fontsize=8)
    ax1.grid(axis="y", color=GRID, linewidth=0.6); ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    tiers = ["<₹500", "₹500–\n2k", "₹2k–\n10k", ">₹10k"]
    sv_only = [54.0, 54.0, 11.0, 0.0]
    both = [7.5, 7.5, 1.8, 0.0]
    x2 = np.arange(len(tiers)); w = 0.36
    ax2.bar(x2 - w / 2, sv_only, w, label="SV only", color=RED)
    ax2.bar(x2 + w / 2, both, w, label="SV + anti-spoofing", color=GREEN)
    for i, (a, b) in enumerate(zip(sv_only, both)):
        if a: ax2.text(i - w / 2, a + 1.2, f"{a:.0f}", ha="center", fontsize=6.4)
        if b: ax2.text(i + w / 2, b + 1.2, f"{b:.1f}", ha="center", fontsize=6.4)
    ax2.set_xticks(x2); ax2.set_xticklabels(tiers, fontsize=7)
    ax2.set_ylabel("Attacks approved (%)"); ax2.set_ylim(0, 62)
    ax2.legend(fontsize=6.4, frameon=False)
    ax2.set_title("(b) Authorization outcome by tier", fontsize=8)
    ax2.grid(axis="y", color=GRID, linewidth=0.6); ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_attack.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


fig_architecture()
fig_domain_gap()
fig_degradation()
fig_attack()
print("figures written to", OUT.resolve())
for f in sorted(OUT.glob("*.png")):
    print(" ", f.name, f.stat().st_size // 1024, "KB")
