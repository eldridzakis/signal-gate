"""Generates publication-quality figures for Notebook 06:
Real-World Datasets, Multi-Model Frontier LLM Audits & Agentune AutoML Overfitting Benchmarks.
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
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
})


def generate_fig4_real_world_frontier():
    """Figure 4: Real-World Multi-Model Frontier LLM Replication & False Discovery Audit."""
    df_all = pd.read_csv(DATA_DIR / "real_world_eval_all_models_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    # Panel A: Holdout Replication Rate across Datasets & Personas (Averaged across 4 Frontier Models)
    ds_avg = df_all.groupby(["Dataset", "Persona"], as_index=False).agg({
        "Holdout_Replication_Rate (%)": "mean",
        "Real_FDR (%)": "mean",
    })

    personas = ["business_analyst", "statistician_unassisted", "statistician_signalgate"]
    persona_labels = ["Business Analyst\n(Unassisted)", "Statistician\n(Unassisted)", "Statistician\n(SignalGate)"]
    x = np.arange(len(personas))
    width = 0.25

    datasets = [
        ("lending_club", "Lending Club Defaults", "#e74c3c", -width),
        ("king_county", "King County Housing", "#2980b9", 0.0),
        ("bank_marketing", "UCI Bank Marketing", "#8e44ad", width),
    ]

    for ds_key, ds_label, color, offset in datasets:
        vals = [
            ds_avg[(ds_avg["Dataset"] == ds_key) & (ds_avg["Persona"] == p)]["Holdout_Replication_Rate (%)"].values[0]
            for p in personas
        ]
        bars = axes[0].bar(x + offset, vals, width, label=ds_label, color=color, alpha=0.9, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    axes[0].set_ylabel("Holdout Replication Rate (%) [p < 0.05 on Full Test Vault]")
    axes[0].set_title("(A) Out-of-Sample Replication Across 3 Real-World Datasets\n(Pooled Across 4 Frontier LLMs)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(persona_labels)
    axes[0].set_ylim(0, 118)
    axes[0].axhline(100, color="green", linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    # Panel B: Real False Discovery Rate (%) by Frontier Model (Business Analyst vs SignalGate)
    models = ["claude-opus-5-5", "claude-sonnet-5-5", "gpt-6.1-sol", "gemini-3.5-flash-lite"]
    model_labels = ["Claude Opus 5.5\n(Anthropic)", "Claude Sonnet 5.5\n(Anthropic)", "GPT-6.1-Sol\n(OpenAI)", "Gemini 3.5 Flash Lite\n(Google)"]
    mod_avg = df_all.groupby(["Model", "Persona"], as_index=False)["Real_FDR (%)"].mean()

    xm = np.arange(len(models))
    w_m = 0.34

    ba_fdr = [mod_avg[(mod_avg["Model"] == m) & (mod_avg["Persona"] == "business_analyst")]["Real_FDR (%)"].values[0] for m in models]
    sg_fdr = [mod_avg[(mod_avg["Model"] == m) & (mod_avg["Persona"] == "statistician_signalgate")]["Real_FDR (%)"].values[0] for m in models]

    bars_ba = axes[1].bar(xm - w_m / 2, ba_fdr, w_m, label="Unassisted Business Analyst", color="#e67e22", alpha=0.9, edgecolor="black")
    bars_sg = axes[1].bar(xm + w_m / 2, sg_fdr, w_m, label="SignalGate-Governed Agent", color="#27ae60", alpha=0.9, edgecolor="black")

    for bar in list(bars_ba) + list(bars_sg):
        h = bar.get_height()
        axes[1].annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )

    axes[1].set_ylabel("Mean Real False Discovery Rate (%) on Holdout")
    axes[1].set_title("(B) Out-of-Sample False Discovery Rate by Frontier Model\n(Averaged Across All 3 Production Datasets)", fontweight="bold")
    axes[1].set_xticks(xm)
    axes[1].set_xticklabels(model_labels)
    axes[1].set_ylim(0, 75)
    axes[1].legend(loc="upper right")

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
        color="#e74c3c", s=80, alpha=0.85, edgecolors="black", label="Unassisted Agentune (FDR = 26.7%)"
    )
    axes[0].scatter(
        signalgate["train_rig"], signalgate["holdout_rig"],
        color="#27ae60", s=90, alpha=0.85, marker="s", edgecolors="black", label="SignalGate Gated (FDR = 0.0%)"
    )

    axes[0].set_xlabel("Train Relative Information Gain (RIG)")
    axes[0].set_ylabel("Holdout Test RIG (Pure Out-of-Sample)")
    axes[0].set_title("(A) Feature Replication: Train RIG vs Holdout RIG", fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].set_xlim(0.015, 0.065)
    axes[0].set_ylim(0.005, 0.055)

    # Panel B: Holdout Decay % Distribution
    decay_data = [unassisted["decay_pct"], signalgate["decay_pct"]]
    bp = axes[1].boxplot(
        decay_data,
        patch_artist=True,
        tick_labels=["Unassisted Agentune\n(Greedy AutoML)", "SignalGate-Governed\nSelection"]
    )
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

    ax.bar(x - width/2, lc_down["Train_AUC"], width, label="Train ROC AUC", color="#3498db", alpha=0.9, edgecolor="black")
    ax.bar(x + width/2, lc_down["Holdout_Test_AUC"], width, label="Holdout Test ROC AUC", color="#2ecc71", alpha=0.9, edgecolor="black")

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
