from signalgate.real_world_benchmark import run_real_world_frontier_eval
import pandas as pd

for persona in ["statistician_unassisted", "statistician_signalgate"]:
    print(f"\n==================================================")
    print(f"PERSONA: {persona}")
    print(f"==================================================")
    for seed in [42, 59, 76]:
        res = run_real_world_frontier_eval(
            dataset_name="king_county",
            persona=persona,
            model_name="gemini-3.5-flash-lite",
            seed=seed,
            sample_train=5000,
            sample_test=2500,
            use_cache=True
        )
        audit = res["holdout_audit"]
        print(f"\n--- Seed {seed} ---")
        print(f"True Tests Evaluated: {res['true_M_raw']} | Self-Reported Tests: {res['self_reported_M']}")
        print(f"Under-reporting ratio: {audit['under_reporting_ratio']:.2f}x")
        print(f"Promoted drivers count: {audit['n_reported']}")
        print(f"Replication: {audit['replicated_count']}/{audit['n_reported']} ({audit['replication_rate']*100:.1f}%) | Real FDR: {audit['real_fdr']*100:.1f}%")
        print(f"Mean Train RIG: {audit['mean_train_rig']:.4f} | Mean Holdout RIG: {audit['mean_holdout_rig']:.4f}")
        df = pd.DataFrame(audit["audit_table"])
        print(df[["feature", "train_rig", "holdout_rig", "bias", "decay_pct", "holdout_p_value", "replicated_on_holdout"]].to_string(index=False))
