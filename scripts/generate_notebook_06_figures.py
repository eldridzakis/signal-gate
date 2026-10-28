"""Generates publication-quality figures for Notebook 06:
Real-World Datasets & Agentune AutoML Overfitting Benchmarks.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data"
FIG_DIR = REPO_ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
})


def generate_fig4_real_world_frontier():
    """Figure 4: Real-World Frontier LLM Replication & Under-Reporting Audit."""
    lc = pd.read_csv(DATA_DIR / "real_world_eval_lending_club_gemini-3.5-flash-lite_summary.csv")
    kc = pd.read_csv(DATA_DIR / "real_world_eval_king_county_gemini-3.5-flash-lite_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Holdout Replication Rate across Personas
    personas = ["Business Analyst\n(Unassisted)", "Statistician\n(Unassisted)", "Statistician\n(SignalGate)"]
    x = np.arange(len(personas))
    width = 0.35

    lc_rep = [
        lc[lc["Persona"] == "business_analyst"]["Holdout_Replication_Rate (%)"].values[0],
        lc[lc["Persona"] == "statistician_unassisted"]["Holdout_Replication_Rate (%)"].values[0],
        lc[lc["Persona"] == "statistician_signalgate"]["Holdout_Replication_Rate (%)"].values[0],
    ]
    kc_rep = [
        kc[kc["Persona"] == "business_analyst"]["Holdout_Replication_Rate (%)"].values[0],
        kc[kc["Persona"] == "statistician_unassisted"]["Holdout_Replication_Rate (%)"].values[0],
        kc[kc["Persona"] == "statistician_signalgate"]["Holdout_Replication_Rate (%)"].values[0],
    ]

    bars1 = axes[0].bar(x - width/2, lc_rep, width, label="Lending Club Defaults", color="#e74c3c", alpha=0.9, edgecolor="black")
    bars2 = axes[0].bar(x + width/2, kc_rep, width, label="King County Housing", color="#2980b9", alpha=0.9, edgecolor="black")

    axes[0].set_ylabel("Holdout Replication Rate (%) [p < 0.05 on Test]")
    axes[0].set_title("(A) Out-of-Sample Holdout Replication Rate", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(personas)
    axes[0].set_ylim(0, 115)
    axes[0].axhline(100, color="gray", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")

    for bar in bars1 + bars2:
        h = bar.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Panel B: Under-Reporting Ratio (Search Space Amnesia)
    lc_under = [
        lc[lc["Persona"] == "business_analyst"]["Under_Reporting_Ratio"].values[0],
        lc[lc["Persona"] == "statistician_unassisted"]["Under_Reporting_Ratio"].values[0],
        lc[lc["Persona"] == "statistician_signalgate"]["Under_Reporting_Ratio"].values[0],
    ]
    kc_under = [
        kc[kc["Persona"] == "business_analyst"]["Under_Reporting_Ratio"].values[0],
        kc[kc["Persona"] == "statistician_unassisted"]["Under_Reporting_Ratio"].values[0],
        kc[kc["Persona"] == "statistician_signalgate"]["Under_Reporting_Ratio"].values[0],
    ]

    bars3 = axes[1].bar(x - width/2, lc_under, width, label="Lending Club Defaults", color="#e67e22", alpha=0.9, edgecolor="black")
    bars4 = axes[1].bar(x + width/2, kc_under, width, label="King County Housing", color="#8e44ad", alpha=0.9, edgecolor="black")

    axes[1].set_ylabel("Under-Reporting Ratio (M_evaluated / M_reported)")
    axes[1].set_title("(B) Search Space Amnesia vs Ledger Tracking", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(personas)
    axes[1].set_ylim(0, 12)
    axes[1].axhline(1.0, color="green", linestyle="-.", label="Perfect Ledger (1.0x)", linewidth=2)
    axes[1].legend(loc="upper right")

    for bar in bars3 + bars4:
        h = bar.get_height()
        axes[1].annotate(f"{h:.1f}x", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    out_path = FIG_DIR / "fig4_real_world_frontier_replication.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_fig5_agentune_decay():
    """Figure 5: Agentune-Style Feature Overfitting & Holdout Decay."""
    unassisted = pd.read_csv(DATA_DIR / "agentune_benchmark_lending_club_unassisted_topk.csv")
    signalgate = pd.read_csv(DATA_DIR / "agentune_benchmark_lending_club_signalgate_topk.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Train RIG vs Holdout Test RIG (Scatter + 45-deg line)
    axes[0].plot([0, 0.07], [0, 0.07], 'k--', label="Perfect Replication (Train = Test)", alpha=0.6)

    axes[0].scatter(
        unassisted["train_rig"], unassisted["holdout_rig"],
        color="#e74c3c", s=80, alpha=0.85, edgecolors="black", label=f"Unassisted Agentune (FDR = 26.7%)"
    )
    axes[0].scatter(
        signalgate["train_rig"], signalgate["holdout_rig"],
        color="#27ae60", s=90, alpha=0.85, marker="s", edgecolors="black", label=f"SignalGate Gated (FDR = 0.0%)"
    )

    axes[0].set_xlabel("Train Relative Information Gain (RIG)")
    axes[0].set_ylabel("Holdout Test RIG (Pure Out-of-Sample)")
    axes[0].set_title("(A) Feature Replication: Train RIG vs Holdout RIG", fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].set_xlim(0.015, 0.065)
    axes[0].set_ylim(0.005, 0.055)

    # Panel B: Holdout Decay % Distribution
    decay_data = [unassisted["decay_pct"], signalgate["decay_pct"]]
    bp = axes[1].boxplot(decay_data, patch_artist=True, tick_labels=["Unassisted Agentune\n(Greedy AutoML)", "SignalGate-Governed\nSelection"])
    colors = ["#f8d7da", "#d4edda"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    axes[1].set_ylabel("Out-of-Sample Decay (%)")
    axes[1].set_title("(B) Distribution of Out-of-Sample Decay", fontweight="bold")
    axes[1].axhline(unassisted["decay_pct"].mean(), color="#e74c3c", linestyle=":", label=f"Unassisted Mean: {unassisted['decay_pct'].mean():.1f}%")
    axes[1].axhline(signalgate["decay_pct"].mean(), color="#27ae60", linestyle=":", label=f"SignalGate Mean: {signalgate['decay_pct'].mean():.1f}%")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    out_path = FIG_DIR / "fig5_agentune_holdout_decay.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_fig6_downstream_auc():
    """Figure 6: Downstream Model Overfitting Gap (Train AUC vs Test AUC)."""
    lc_down = pd.read_csv(DATA_DIR / "agentune_benchmark_lending_club_downstream.csv")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(lc_down))
    width = 0.35

    bars_tr = ax.bar(x - width/2, lc_down["Train_AUC"], width, label="Train ROC AUC", color="#3498db", alpha=0.9, edgecolor="black")
    bars_te = ax.bar(x + width/2, lc_down["Holdout_Test_AUC"], width, label="Holdout Test ROC AUC", color="#2ecc71", alpha=0.9, edgecolor="black")

    ax.set_ylabel("ROC AUC Score")
    ax.set_title("Downstream Classifier Generalization: The AutoML Overfitting Gap", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "Baseline Raw Features\n(119 cols)",
        "Baseline + Unassisted Agentune Top-K\n(134 cols - Greedy)",
        "Baseline + SignalGate Top-K\n(134 cols - Governed)"
    ])
    ax.set_ylim(0.65, 0.76)
    ax.legend(loc="upper left")

    # Annotate overfitting gap delta
    for i, row in lc_down.iterrows():
        gap = row["Overfitting_Gap"]
        te_auc = row["Holdout_Test_AUC"]
        ax.annotate(f"Holdout: {te_auc:.4f}\n(Gap: {gap:.4f})",
                    xy=(x[i] + width/2, te_auc),
                    xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9.5)

    plt.tight_layout()
    out_path = FIG_DIR / "fig6_agentune_downstream_overfitting_gap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    generate_fig4_real_world_frontier()
    generate_fig5_agentune_decay()
    generate_fig6_downstream_auc()
