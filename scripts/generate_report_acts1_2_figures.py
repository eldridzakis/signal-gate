"""Generates fig1_scaling_laws.png and fig2_agentic_trajectory.png for REPORT.md."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from signalgate.agent_sandbox import generate_benchmark_datasets, InstrumentedAgentSandbox, run_exploratory_agent_simulation

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.5,
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
})


def generate_fig1_scaling_laws():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4))

    # Panel A: Expected Null RIG vs. Sample Size N
    N_vals = np.logspace(1.5, 4.5, 200)  # 30 to 30,000
    h_y = 0.529  # binary target with 12% minority rate
    k_df = 1     # binary feature (df=1)

    c_scale = 2.0 * np.log(2.0) * h_y
    expected_rig = k_df / (c_scale * N_vals)
    p95_rig = stats.chi2.ppf(0.95, df=k_df) / (c_scale * N_vals)
    p99_rig = stats.chi2.ppf(0.99, df=k_df) / (c_scale * N_vals)

    ax1.loglog(N_vals, expected_rig, "b-", lw=2.2, label=r"Expected Null RIG $\mathbb{E}[\mathrm{RIG}_{\mathrm{null}}] = \mathcal{O}(k/N)$")
    ax1.loglog(N_vals, p95_rig, "#e67e22", ls="--", lw=2, label=r"95th Percentile Noise Floor ($p=0.05$)")
    ax1.loglog(N_vals, p99_rig, "#c0392b", ls=":", lw=2.2, label=r"99th Percentile Noise Floor ($p=0.01$)")

    ax1.axvspan(30, 200, color="red", alpha=0.12, label="Agent Subgroup Slice Zone ($N=30-200$)")
    ax1.axhline(0.02, color="gray", ls="-.", alpha=0.7, label="Naive Rule-of-Thumb Cutoff (RIG = 2%)")

    ax1.set_xlabel("Sample Size $N$ (Active Cohort Rows)")
    ax1.set_ylabel("Null Relative Information Gain (RIG)")
    ax1.set_title("(A) Finite-Sample Noise Explosion:\nDrilling Into Subgroups ($N \\to 0$) Multiplies Spurious RIG", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    # Panel B: Gumbel EVT Noise Ceiling vs Search Space M
    M_vals = np.logspace(0, 4, 200)  # M = 1 to 10,000
    N_fixed = 3000

    for K, color, ls in [(2, "#2ecc71", "-"), (5, "#3498db", "--"), (10, "#e74c3c", "-.")]:
        k_deg = K - 1
        chi2_evt = 2 * np.log(np.maximum(2, M_vals)) + (k_deg - 2) * np.log(np.maximum(1.1, 2 * np.log(np.maximum(2, M_vals)))) - 2 * np.log(np.maximum(1e-5, stats.gamma.pdf(k_deg / 2, 1))) + 2 * 0.5772
        tau_evt = chi2_evt / (2 * N_fixed * np.log(2.0) * h_y)
        ax2.semilogx(M_vals, tau_evt, color=color, ls=ls, lw=2.2, label=f"EVT Max Ceiling ($K={K}$ categories, $df={k_deg}$)")

    ax2.axvline(1, color="gray", ls=":", label="Single Test ($M=1$)")
    ax2.axvline(50, color="purple", ls="--", alpha=0.7, label="Standard Tabular Scan ($M=50$)")
    ax2.axvline(1000, color="crimson", ls="--", alpha=0.7, label="AutoML / Slice Explorer ($M=1,000$)")

    ax2.set_xlabel("Cumulative Hypotheses Evaluated ($M$)")
    ax2.set_ylabel(r"Extreme Value Theory Max RIG Ceiling $\tau_{\mathrm{EVT}}$")
    ax2.set_title("(B) Multiplicity Explosion ($N=3,000$):\nCumulative Testing ($M$) Systematically Breaches Static Gates", fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    out_path = FIG_DIR / "fig1_scaling_laws.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_fig2_agentic_trajectory():
    df, _ = generate_benchmark_datasets(mode="null_honeypot", nrows=3000, minority_frac=0.12, seed=42)
    sb = InstrumentedAgentSandbox(df=df, target_col="is_fraud", guardrail_enabled=True)
    run_exploratory_agent_simulation(sb, seed=42, include_slices=False)

    history = sb.telemetry_history
    turns = np.arange(1, len(history) + 1)
    cum_best_rig = [h["best_rig_in_turn"] for h in history]
    cum_m = [h["cumulative_M"] for h in history]
    evt_ceil = [h.get("evt_expected_max_rig", 0.005) for h in history]
    bonf_gate = [h.get("bonferroni_gate_rig", 0.008) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.5))

    # Panel A: Hypothesis Testing Explosion
    ax1.plot(turns, cum_m, marker="o", color="#1f77b4", lw=2.5, markersize=7, label=r"Cumulative Hypotheses Tested ($M$)")
    ax1.fill_between(turns, 0, cum_m, color="#1f77b4", alpha=0.15)
    ax1.set_xlabel("Agent Exploratory Turn")
    ax1.set_ylabel(r"Cumulative Hypotheses Tested ($M$)")
    ax1.set_title("(A) Agentic Multiplicity Inflation:\nA Brief 15-Turn Session Evaluates $M > 500$ Silent Hypotheses", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True)
    for i in [2, 7, 11, 14]:
        t, m = turns[i], cum_m[i]
        ax1.annotate(f"Turn {t}\nM={m}", (t, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, fontweight="bold")

    # Panel B: Dynamic SignalGate Escalation vs Naive Static Alpha
    n_base = 3000
    p_base = float(df["is_fraud"].mean())
    h_y = float(stats.entropy([1 - p_base, p_base], base=2))
    c_scale = 2.0 * n_base * np.log(2.0) * h_y
    tau_naive = float(stats.chi2.ppf(0.95, df=1)) / c_scale

    ax2.axhline(tau_naive, color="#e74c3c", linestyle="--", lw=2, label=f"Naive Single-Test Cutoff ($\\alpha=0.05, M=1$): $\\tau={tau_naive:.5f}$")
    ax2.plot(turns, evt_ceil, color="#f39c12", linestyle="--", lw=2, marker="s", markersize=6, label=r"Dynamic EVT Expected Noise Max $\tau_{\mathrm{EVT}}(M)$")
    ax2.plot(turns, bonf_gate, color="#27ae60", linestyle="-", lw=2.5, marker="^", markersize=7, label=r"SignalGate Bonferroni Gate $\tau_{\mathrm{sig}}(M_{\mathrm{eff}}, \alpha=0.05)$")
    ax2.plot(turns, cum_best_rig, color="#2980b9", linestyle="-", lw=2.2, marker="o", markersize=7, label="Candidate Discovered by Agent in Turn")

    ax2.fill_between(turns, tau_naive, evt_ceil, color="#f39c12", alpha=0.18, label="Agentic Hallucination Zone (Winner's Curse Trap)")

    ax2.set_xlabel("Agent Exploratory Turn")
    ax2.set_ylabel("Relative Information Gain (RIG)")
    ax2.set_title("(B) Dynamic Threshold Escalation:\nSignalGate Closes the Hallucination Trap as $M$ Grows", fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout()
    out_path = FIG_DIR / "fig2_agentic_trajectory.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    generate_fig1_scaling_laws()
    generate_fig2_agentic_trajectory()
