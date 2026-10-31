"""Generate Hero Figure: Dual-Mechanism Discovery Engine.

Combines:
  Panel (A): The Problem — Multiplicity Amnesia & Selection Bias
             (Exploration budget under-reporting ratio M_true / M_self and holdout failure spike)
  Panel (B): The Solution — Phase 2 Verified Signal Yield & Discovery F1
             (SignalGuard governance delivering high verified yield while cutting FDR)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIRS = [
    REPO_ROOT / "paper" / "figures",
    REPO_ROOT / "notebooks" / "figures",
]

for fd in FIG_DIRS:
    fd.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 14,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=300)

# ==========================================
# PANEL (A): The Problem — Multiplicity Amnesia & False Discovery Spikes
# ==========================================
models = [
    "Claude Sonnet 5.5\n(Anthropic Flagship)",
    "GPT-6.1 Sol\n(OpenAI Flagship)",
    "Gemini Flash Lite\n(Google Scout)",
    "GPT-6 Luna\n(OpenAI Scout)",
]

x = np.arange(len(models))
width = 0.36

# Metric 1: Ratio of True Tested Hypotheses to Self-Reported (M_true / M_self)
# In Phase 1 & Phase 2, unassisted agents under-count by 3.5x to 7.7x
under_reporting = [4.6, 5.1, 6.7, 5.3]

# Metric 2: Peak Unassisted False Discovery Rate on Complex Slices (%)
peak_fdr = [12.1, 38.9, 32.1, 12.5]  # Lending Club / King County unassisted peak FDR

color_under = "#6366f1"  # Indigo
color_fdr = "#ef4444"    # Red / Spurious

rects1 = ax1.bar(x - width/2, under_reporting, width, label="Search Amnesia Ratio (M_true / M_self)", color=color_under, alpha=0.9, edgecolor="black", linewidth=0.6)

# Create twin axis for FDR %
ax1_twin = ax1.twinx()
rects2 = ax1_twin.bar(x + width/2, peak_fdr, width, label="Unassisted Peak Out-of-Sample FDR (%)", color=color_fdr, alpha=0.85, hatch="//", edgecolor="black", linewidth=0.6)

ax1.set_ylabel("Hypothesis Under-Counting Ratio (x)", color=color_under, fontweight="bold")
ax1_twin.set_ylabel("Peak Empirical False Discovery Rate (%)", color=color_fdr, fontweight="bold")
ax1.set_title("(A) The Vulnerability: Multiplicity Amnesia & Noise Exploitation\nAgents Under-Report Search Spaces by 4.6x–6.7x, Spiking Real FDR up to 39%", fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 8.5)
ax1_twin.set_ylim(0, 50)
ax1.axhline(1.0, color="#10b981", linestyle="--", linewidth=1.5, alpha=0.8, label="Ideal 1.0x Calibration")

# Value annotations
for r in rects1:
    h = r.get_height()
    ax1.annotate(f"{h:.1f}x", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color=color_under)

for r in rects2:
    h = r.get_height()
    ax1_twin.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                      textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color=color_fdr)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)
ax1_twin.grid(False)

# ==========================================
# PANEL (B): The Solution — Phase 2 Verified Signal Yield & Discovery F1
# ==========================================
p2_models = [
    "GPT-6.1 Sol\n(Flagship)",
    "Gemini Flash Lite\n(Scout)",
]

x2 = np.arange(len(p2_models))
w2 = 0.22

# Verified Discoveries (Count)
verified_unassisted = [16, 13]
verified_signalguard = [23, 19]

# Discovery F1 Score
f1_unassisted = [0.542, 0.389]
f1_signalguard = [0.594, 0.499]

c_unass = "#94a3b8"  # Slate Gray
c_guard = "#1d4ed8"  # Royal Blue

b1 = ax2.bar(x2 - w2*1.5, verified_unassisted, w2, label="Unassisted Replications (Count)", color=c_unass, alpha=0.8, edgecolor="black", linewidth=0.6)
b2 = ax2.bar(x2 - w2*0.5, verified_signalguard, w2, label="SignalGuard Replications (Count)", color=c_guard, alpha=0.9, edgecolor="black", linewidth=0.6)

# Twin axis for F1
ax2_twin = ax2.twinx()
b3 = ax2_twin.bar(x2 + w2*0.5, f1_unassisted, w2, label="Unassisted Discovery F1", color="#64748b", hatch="..", alpha=0.7, edgecolor="black", linewidth=0.6)
b4 = ax2_twin.bar(x2 + w2*1.5, f1_signalguard, w2, label="SignalGuard Discovery F1", color="#059669", alpha=0.85, edgecolor="black", linewidth=0.6)

ax2.set_ylabel("Verified Holdout Discoveries ($p < 0.05$ on Vault)", color=c_guard, fontweight="bold")
ax2_twin.set_ylabel("Discovery F1 Score (Harmonic Mean)", color="#059669", fontweight="bold")
ax2.set_title("(B) The Resolution: Stateful Governance in Live Agentic SQL (Phase 2)\nSignalGuard Expands Verified Discoveries (+44%) While Elevating Discovery F1", fontweight="bold")

ax2.set_xticks(x2)
ax2.set_xticklabels(p2_models)
ax2.set_ylim(0, 30)
ax2_twin.set_ylim(0, 0.8)

for b in b1:
    h = b.get_height()
    ax2.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
for b in b2:
    h = b.get_height()
    ax2.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color=c_guard)

for b in b3:
    h = b.get_height()
    ax2_twin.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5)
for b in b4:
    h = b.get_height()
    ax2_twin.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#059669")

lines_b1, labels_b1 = ax2.get_legend_handles_labels()
lines_b2, labels_b2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines_b1 + lines_b2, labels_b1 + labels_b2, loc="upper left", framealpha=0.9, fontsize=9.5)
ax2_twin.grid(False)

plt.tight_layout()

for fd in FIG_DIRS:
    out_png = fd / "fig_hero_dual_mechanism.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")

plt.close()
