"""Generate Hero Figure: Dual-Mechanism Discovery Engine (Decluttered & Polished).

Combines:
  Panel (A): The 2D Landscape — Resolving the Amnesia vs. Paralysis Dilemma
             (2D Scatter of Empirical FDR vs. Signal Recall showing Paralysis Trap, Amnesia Trap, and SignalGuard Zone)
  Panel (B): The Empirical Resolution — Phase 2 Verified Signal Yield & Discovery F1
             (SignalGuard governance delivering high verified yield while cutting FDR)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.4), dpi=300)

# ==========================================
# PANEL (A): The 2D Landscape — Amnesia vs. Paralysis Traps
# ==========================================

# 1. Shaded Background Zones
# (a) Paralysis Trap (Bottom-Left: Low FDR, but near-zero Recall due to over-penalization)
ax1.axvspan(-1.5, 12.5, ymin=0, ymax=0.32, color='#fef3c7', alpha=0.6, zorder=1)
ax1.text(5.5, 27, "PARALYSIS TRAP (Recall ≤ 17%)\nOver-Penalized: Power Collapse", 
         color='#92400e', fontsize=8.5, fontweight='bold', ha='center', va='top', zorder=2)

# (b) Amnesia Trap (Right: High FDR due to unassisted multi-testing and noise exploitation)
ax1.axvspan(17.5, 43.5, ymin=0, ymax=1.0, color='#fee2e2', alpha=0.55, zorder=1)
ax1.text(30.5, 20, "AMNESIA TRAP (FDR ≥ 18%)\nUnder-Penalized: Noise Exploitation", 
         color='#991b1b', fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=2)

# (c) SignalGuard Optimal Sweet Spot (Upper-Left: Controlled FDR ≤ 10% + High Recall ≥ 70%)
ax1.axvspan(-1.5, 12.5, ymin=0.65, ymax=1.0, color='#dcfce7', alpha=0.65, zorder=1)
ax1.text(5.5, 96, "SIGNALGUARD OPTIMAL ZONE\n(FDR ≤ 10%, Recall ≥ 70%)\nCalibrated M_eff Governance", 
         color='#166534', fontsize=8.5, fontweight='bold', ha='center', va='top', zorder=2)

# 2. Points configuration:
# To prevent vertical stacking at FDR = 0.0%, we apply a minor visual offset (jitter) for distinct visibility:
# Sonnet Bonf: (0.2, 5.6)
# Haiku BH: (1.2, 11.1)
# Sonnet Unassisted: (0.4, 16.7)
# Sonnet SignalGuard: (1.8, 16.7)

# Scatter points:
# (FDR, Recall, Label, Arm, Color, Marker, Size)
pts = [
    # Paralysis Trap Points (Multiple models collapse here!)
    (0.2, 5.6, "Sonnet (Bonf: 5.6%)", "bonf", "#d97706", "s", 70),
    (1.4, 11.1, "Haiku (BH: 11.1%)", "bh", "#ea580c", "^", 70),
    (0.4, 16.7, "Sonnet (Unassisted: 16.7%)", "unassist", "#64748b", "o", 65),
    (2.0, 16.7, "Sonnet (SignalGuard)", "sg", "#1e40af", "D", 85),
    
    # Amnesia Trap Points
    (23.8, 77.8, "Sol (Unassisted)", "unassist", "#64748b", "o", 75),
    (22.2, 77.8, "Sol (BH)", "bh", "#ea580c", "^", 75),
    (19.0, 61.1, "Flash-Lite (BH)", "bh", "#ea580c", "^", 75),
    (38.9, 72.2, "Sol Peak Outlier\n(Lending Club: 38.9% FDR)", "outlier", "#dc2626", "X", 100),
    
    # Intermediate / Other Points
    (0.0, 50.0, "Flash-Lite (Unassisted)", "unassist", "#64748b", "o", 75),
    (4.2, 61.1, "Flash-Lite (Bonf)", "bonf", "#d97706", "s", 70),
    (9.7, 83.3, "Sol (Bonf)", "bonf", "#d97706", "s", 75),
    (10.4, 50.0, "Haiku (SignalGuard)", "sg", "#1e40af", "D", 75),
    
    # SignalGuard Sweet Spot Points
    (8.0, 77.8, "Sol (SignalGuard)", "sg", "#1e40af", "D", 95),
    (9.0, 72.2, "Flash-Lite (SignalGuard)", "sg", "#1e40af", "D", 95),
]

for fdr, rec, lbl, arm, col, m, sz in pts:
    ax1.scatter(fdr, rec, color=col, s=sz, marker=m, zorder=5, edgecolor='black', linewidth=0.7)

# 3. Clean, Non-Overlapping Callout Labels & Annotations

# (a) Bottom-Left Paralysis Callouts
ax1.annotate("Sonnet (Bonf: 5.6%)", xy=(0.2, 5.6), xytext=(5.5, 4.5),
             arrowprops=dict(arrowstyle="->", color="#92400e", lw=0.8),
             fontsize=7.5, color="#92400e", fontweight='bold', va='center')
ax1.annotate("Haiku (BH: 11.1%)", xy=(1.4, 11.1), xytext=(5.5, 11.1),
             arrowprops=dict(arrowstyle="->", color="#92400e", lw=0.8),
             fontsize=7.5, color="#92400e", fontweight='bold', va='center')
ax1.annotate("Sonnet (Unassisted: 16.7%)", xy=(0.4, 16.7), xytext=(5.5, 17.5),
             arrowprops=dict(arrowstyle="->", color="#475569", lw=0.8),
             fontsize=7.5, color="#475569", va='center')

# (b) Red '+' / 'X' Outlier Callout
ax1.annotate("Sol Peak Task Outlier\n(Lending Club: 38.9% FDR)",
             xy=(38.9, 72.2), xytext=(35.5, 62.0),
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
             fontsize=8, color="#991b1b", fontweight='bold', ha='center',
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2", edgecolor="#fca5a5", alpha=0.9))

# (c) Trajectory Rescue Arrows
# Sol rescue
ax1.annotate("", xy=(8.0, 77.8), xytext=(23.8, 77.8),
             arrowprops=dict(arrowstyle="->", color="#1e40af", lw=2.2, mutation_scale=15), zorder=6)
ax1.text(16.0, 81.5, "Sol: -16% FDR Spike\n(16 -> 23 Replicated)", color="#1e40af", fontsize=8, fontweight='bold', ha='center')

# Flash-Lite rescue
ax1.annotate("", xy=(9.0, 72.2), xytext=(0.0, 50.0),
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.2, mutation_scale=15), zorder=6)
ax1.text(7.2, 59.0, "Flash-Lite: +22% Recall\n(13 -> 19 Replicated)", color="#16a34a", fontsize=8, fontweight='bold', ha='center')

# Axis formatting
ax1.set_xlim(-1.5, 43.5)
ax1.set_ylim(-2, 102)
ax1.set_xlabel("Empirical False Discovery Rate (Real FDR % on Holdout)", fontweight="bold")
ax1.set_ylabel("Out-of-Sample Signal Recall (%)", fontweight="bold")
ax1.set_title("(A) The 2D Landscape: Resolving Amnesia vs. Paralysis Traps\nSignalGuard Navigates Models to the Low-FDR / High-Recall Sweet Spot", fontweight="bold")

# 4. Legend moved to TOP RIGHT (decluttered)
legend_elements_a = [
    Line2D([0], [0], marker='D', color='w', label='SignalGuard (Ours)', markerfacecolor='#1e40af', markeredgecolor='black', markersize=9),
    Line2D([0], [0], marker='o', color='w', label='Unassisted Explorer', markerfacecolor='#64748b', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='s', color='w', label='Bonferroni (FWER)', markerfacecolor='#d97706', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='^', color='w', label='Benjamini-Hochberg', markerfacecolor='#ea580c', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='X', color='w', label='Peak Task Outlier (Lending Club)', markerfacecolor='#dc2626', markeredgecolor='black', markersize=9),
]
ax1.legend(handles=legend_elements_a, loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=8.5)
ax1.grid(True, linestyle="--", alpha=0.35)


# ==========================================
# PANEL (B): The Resolution — Phase 2 Verified Signal Yield & F1 Lift
# ==========================================
models_b = [
    "GPT-6.1 Sol\n(Flagship)",
    "Gemini Flash Lite\n(Scout)",
]

x_b = np.arange(len(models_b))
width_b = 0.22

# Verified Replicated Signals Count
unassisted_reps = [16, 13]
signalguard_reps = [23, 19]

# Out-of-sample Discovery F1 Score
unassisted_f1 = [0.542, 0.389]
signalguard_f1 = [0.594, 0.499]

color_unassist_bar = "#94a3b8"  # Slate gray
color_sg_bar = "#2563eb"        # Royal blue
color_unassist_f1 = "#64748b"   # Dark slate
color_sg_f1 = "#16a34a"         # Emerald green

# Plot Counts (Left Axis)
b1 = ax2.bar(x_b - width_b*1.5, unassisted_reps, width_b, label="Unassisted Replications (Count)", color=color_unassist_bar, edgecolor="black", linewidth=0.6)
b2 = ax2.bar(x_b - width_b*0.5, signalguard_reps, width_b, label="SignalGuard Replications (Count)", color=color_sg_bar, edgecolor="black", linewidth=0.6)

# Create twin axis for Discovery F1
ax2_twin = ax2.twinx()
b3 = ax2_twin.bar(x_b + width_b*0.5, unassisted_f1, width_b, label="Unassisted Discovery F1", color=color_unassist_f1, hatch="...", alpha=0.85, edgecolor="black", linewidth=0.6)
b4 = ax2_twin.bar(x_b + width_b*1.5, signalguard_f1, width_b, label="SignalGuard Discovery F1", color=color_sg_f1, alpha=0.9, edgecolor="black", linewidth=0.6)

# Add exact value labels on bars
def autolabel_counts(rects, ax_target):
    for rect in rects:
        h = rect.get_height()
        ax_target.annotate(f"{int(h)}",
                           xy=(rect.get_x() + rect.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9.5, fontweight="bold")

def autolabel_floats(rects, ax_target, color):
    for rect in rects:
        h = rect.get_height()
        ax_target.annotate(f"{h:.3f}",
                           xy=(rect.get_x() + rect.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight="bold", color=color)

autolabel_counts(b1, ax2)
autolabel_counts(b2, ax2)
autolabel_floats(b3, ax2_twin, "#1e293b")
autolabel_floats(b4, ax2_twin, "#15803d")

ax2.set_xticks(x_b)
ax2.set_xticklabels(models_b, fontweight="bold")
ax2.set_ylabel("Verified Holdout Discoveries (p < 0.05 on Vault)", color="#1e3a8a", fontweight="bold")
ax2_twin.set_ylabel("Discovery F1 Score (Harmonic Mean)", color="#15803d", fontweight="bold")
ax2.set_ylim(0, 30)
ax2_twin.set_ylim(0, 0.8)

ax2.set_title("(B) The Resolution: Stateful Governance in Live Agentic SQL (Phase 2)\nSignalGuard Expands Verified Discoveries (+44%) While Elevating Discovery F1", fontweight="bold")

# Combine legends for Panel B
lines_b, labels_b = ax2.get_legend_handles_labels()
lines_b2, labels_b2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines_b + lines_b2, labels_b + labels_b2, loc="upper left", framealpha=0.9, fontsize=8.5)

plt.tight_layout()

for fd in FIG_DIRS:
    out_file = fd / "fig_hero_dual_mechanism.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved: {out_file}")

print("Hero figure regenerated successfully.")
