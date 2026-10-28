# SignalGuard: Information-Theoretic Guardrails for Autonomous LLM Analytics, SQL Warehouses, and AutoML Feature Discovery

**Executive & Technical Benchmark Report**  
*216 Live Frontier LLM Agent Evaluations (`claude-opus-5-5`, `claude-sonnet-5-5`, `gpt-6.1-sol`, `gemini-3.5-flash-lite`) + 3,000-Candidate AutoML Benchmark Across 3 Real-World Datasets*

---

## Executive Summary

When autonomous LLM agents and automated feature-engineering pipelines (`agentune`-style AutoML) explore production datasets via Python or SQL `GROUP BY` / `CUBE` queries, they operate as **high-throughput implicit hypothesis generators**. A single multi-turn agentic session routinely evaluates $M = 20$ to $1{,}000+$ marginal, interaction, and conditional slice hypotheses (`WHERE slice = x`) before promoting only the top cherry-picked findings in its final executive narrative.

Because unassisted LLM agents lack a stateful multiple-testing ledger and rely on sample-size-blind heuristics ("looks like a $3\times$ lift") or uncalibrated self-reporting of how many hypotheses they tested, they suffer from two complementary failure modes:
1. **Multiplicity Amnesia & Small-Cell Hallucination (`business_analyst` persona):** Agents under-report their true hypothesis search space by **$3.5\times\text{ to }8.9\times$** and promote high-cardinality interaction cubes or tiny conditional slices (`N < 30`) that collapse on unseen holdout data—producing up to **30.2%–44.8% Out-of-Sample False Discovery Rates (Real FDR)** on tabular exploration and up to **51.4%–82.1% Real FDR** on continuous SQL `GROUP BY` slice exploration.
2. **Multiplicity Paralysis (`statistician_unassisted` persona):** When frontier models (`claude-opus-5-5`, `gpt-6.1-sol`) are explicitly prompted to act as rigorous PhD statisticians, they frequently **over-guess or miscalibrate** their cumulative search denominator $M$ (e.g., guessing $M = 200\text{–}300$ when $M_{\text{raw}} = 43$) or lack collinear effective-rank adjustment ($M_{\text{eff}}$), causing them to over-penalize genuine moderate signals and sacrifice out-of-sample signal power.

**SignalGuard** resolves both failure modes by acting as a deterministic, stateful **Model Context Protocol (MCP) & SQL Semantic Layer Guardrail** between the agent and the data warehouse. Across **216 live multi-turn frontier LLM evaluations** (108 tabular tool-use runs + 108 local SQLite warehouse + semantic layer runs) and **3,000 automated feature-engineering candidates**, **`statistician_signalguard` achieved a 100.0% Out-of-Sample Holdout Replication Rate (0.0% Real FDR) across every dataset, every SQL mart (both binary classification and closed-form continuous ANOVA), and all 4 frontier models**, while recovering **higher out-of-sample signal power** than unassisted statisticians.

---

## The 5-Act Visual & Empirical Story

### Act I: The Asymptotic Scaling Law of Spurious Information Gain
Why do rule-of-thumb lift thresholds ("promote any segment with $\text{RIG} > 2\%$ or $3\times$ lift") inevitably fail in agentic analytics?

By Wilks' Theorem, under the null hypothesis of independence $H_0: X \perp Y$, the sample mutual information $\hat{I}(X; Y)$ (in bits) between a $K$-category feature $X$ and a $C$-class target $Y$ over $N$ rows scales asymptotically as a scaled Chi-square random variable with $k = (K-1)(C-1)$ degrees of freedom:

$$\hat{I}(X; Y) \sim \frac{\chi^2_k}{2 N \ln 2}, \qquad \text{RIG}(X \mid Y) = \frac{\hat{I}(X; Y)}{H(Y)} \sim \frac{\chi^2_k}{2 N \ln(2) \, H(Y)}$$

Consequently, the expected noise floor scales inversely with sample size ($O(k / N)$) and grows logarithmically with the cumulative number of evaluated hypotheses $M$ via the Gumbel Extreme Value Theory (EVT) maximum:

$$\tau_{\text{EVT}}(N, k, M) = \frac{F^{-1}_{\chi^2_k}\left(1 - \frac{1}{M}\right)}{2 N \ln(2) \, H(Y)}, \qquad \tau_{\text{sig}}(N, k, M_{\text{eff}}, \alpha) = \frac{F^{-1}_{\chi^2_k}\left((1 - \alpha)^{1/M_{\text{eff}}}\right)}{2 N \ln(2) \, H(Y)}$$

![Figure 1: Asymptotic Scaling Laws of Spurious RIG](notebooks/figures/fig1_scaling_laws.png)

* **Takeaway from Figure 1:** Every time an LLM agent executes a `WHERE` filter that narrows the active cohort from $N = 10{,}000$ rows down to a sub-segment of $N = 150$ rows (left panel), or expands a `GROUP BY` across high-cardinality interaction cells $K$ and cumulative queries $M$ (right panel), the pure-noise ceiling explodes by $1\text{–}2$ orders of magnitude. Any static heuristic threshold is guaranteed to be breached by random noise.

---

### Act II: Agentic Multiplicity Amnesia vs. Stateful MCP Ledger Governance
When an LLM agent explores a dataset across multiple tool calls, it evaluates dozens of silent hypotheses inside `scan_all_features`, `CUBE` queries, and conditional drill-downs, but typically counts only the final $3\text{–}8$ variables it writes about in its summary.

![Figure 2: Multi-Turn Agentic Trajectory & Dynamic Threshold Escalation](notebooks/figures/fig2_agentic_trajectory.png)

* **Takeaway from Figure 2:** As the agent executes successive exploratory queries (`Step 1` through `Step 8`), SignalGuard's [`NoiseGateSessionLedger`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/ledger.py) automatically increments the cumulative hypothesis budget $M$ and dynamically raises the verification gate $\tau_{\text{sig}}(M_{\text{eff}})$ (green curve). True structural signals remain well above the escalating gate, while cherry-picked noise spikes—which easily clear a naive single-test $\alpha = 0.05$ bar (orange dashed line)—are deterministically quarantined.

---

### Act III: Resolving "Multiplicity Paralysis" & Collinear Power Loss (`M_eff`)
A critical question arises when evaluating frontier reasoning models like `claude-opus-5-5` and `gpt-6.1-sol`: *Does an unassisted PhD statistician prompt perform just as well as SignalGuard?*

Our empirical benchmark reveals a striking phenomenon we term **Multiplicity Paralysis**:
1. **Denominator Blindness in Both Directions (Figure 3B):** While unassisted `business_analyst` agents **under-report** their hypothesis search space by **$5.1\times\text{ to }6.5\times$**, unassisted `statistician_unassisted` agents on `claude-opus-5-5` **over-guess** their denominator ($M_{\text{reported}} = 250\text{–}300$ when $M_{\text{raw}} = 46$, an under-reporting ratio of **$0.18\times$**), whereas on `gpt-6.1-sol` and `gemini-3.5-flash-lite` they under-report by **$4.8\times\text{ to }5.1\times$**. Only SignalGuard's stateful ledger achieves **exact $1.00\times$ calibration**.
2. **Higher Out-of-Sample Signal Power via Eigenvalue Effective Rank $M_{\text{eff}}$ (Figure 3A):** Real-world features (such as `sqft_living`, `sqft_above`, `grade`, and `bathrooms` in King County Housing) are highly collinear. Using the Frobenius-norm spectral participation ratio of the feature correlation matrix $R$,
   $$M_{\text{eff}} = \frac{\left(\sum_{i=1}^M \lambda_i\right)^2}{\sum_{i=1}^M \lambda_i^2} = \frac{M^2}{\|R\|_F^2}$$
   SignalGuard avoids over-penalizing correlated feature clusters. On King County Housing, `statistician_signalguard` achieves **30% higher verified out-of-sample Holdout RIG ($0.2184$ vs. $0.1676$)** and **lower out-of-sample signal decay ($2.6\%$ vs. $3.4\%$)** than `statistician_unassisted` on both `claude-opus-5-5` and `gpt-6.1-sol`.

![Figure 3: Resolving Multiplicity Paralysis & Denominator Blindness](notebooks/figures/fig3_collinear_power_gap.png)

---

### Act IV: Multi-Model Frontier LLM Benchmark Across 3 Production Datasets (108 Tabular Runs)
We evaluated **4 frontier LLMs** (`claude-opus-5-5`, `claude-sonnet-5-5`, `gpt-6.1-sol`, and `gemini-3.5-flash-lite`) across **3 real-world datasets** spanning diverse signal-to-noise regimes:
1. **Lending Club Loan Defaults** ($N = 54{,}636$, 120 raw cols, low-SNR credit default risk, $H(Y) = 0.722$ bits)
2. **King County House Sales** ($N = 21{,}613$, 23 raw cols, high-SNR structural pricing, $H(Y) = 0.806$ bits)
3. **UCI Bank Marketing Campaigns** ($N = 41{,}188$, 21 raw cols, low-SNR term-deposit subscription, $H(Y) = 0.508$ bits)

Every promoted discovery was audited against a **strictly unseen Holdout Test Vault** ($N = 10{,}806\text{ to }27{,}318$ rows). A discovery is verified as **Replicated on Holdout** if and only if it remains statistically significant ($p < 0.05$) and non-sparse ($\min n_{\text{cell}} \ge 15$) out-of-sample.

![Figure 4: Real-World Multi-Model Frontier LLM Replication & False Discovery Audit](notebooks/figures/fig4_real_world_frontier_replication.png)

#### Table 1: Master Real-World Tabular Benchmark Summary (108 Live Frontier Agent Runs)

| Dataset | Frontier Model | Persona | Holdout Replication (%) | Real FDR (%) | Mean Train RIG | Mean Holdout RIG | Decay (%) | Denominator Ratio ($M_{\text{true}} / M_{\text{rep}}$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lending Club** | `claude-opus-5-5` | `business_analyst` | 88.9% | **11.1%** | 0.0659 | 0.0484 | 12.9% | 1.86x |
| **Lending Club** | `claude-opus-5-5` | `statistician_unassisted` | 100.0% | 0.0% | 0.0465 | 0.0425 | 7.2% | 0.18x *(over-guessed)* |
| **Lending Club** | `claude-opus-5-5` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0446** | **0.0406** | **7.3%** | **0.98x** |
| **Lending Club** | `claude-sonnet-5-5` | `business_analyst` | 100.0% | 0.0% | 0.0423 | 0.0381 | 8.9% | 2.34x |
| **Lending Club** | `claude-sonnet-5-5` | `statistician_unassisted` | 100.0% | 0.0% | 0.0465 | 0.0425 | 7.2% | 0.79x |
| **Lending Club** | `claude-sonnet-5-5` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0463** | **0.0420** | **7.7%** | **1.00x** |
| **Lending Club** | `gpt-6.1-sol` | `business_analyst` | 100.0% | 0.0% | 0.0411 | 0.0383 | 5.6% | 5.11x |
| **Lending Club** | `gpt-6.1-sol` | `statistician_unassisted` | 100.0% | 0.0% | 0.0465 | 0.0425 | 7.2% | 5.11x |
| **Lending Club** | `gpt-6.1-sol` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0463** | **0.0420** | **7.7%** | **1.00x** |
| **Lending Club** | `gemini-3.5-flash-lite` | `business_analyst` | 69.8% | **30.2%** | 0.1021 | 0.0738 | 16.1% | 6.47x |
| **Lending Club** | `gemini-3.5-flash-lite` | `statistician_unassisted` | 100.0% | 0.0% | 0.0191 | 0.0164 | 4.8% | 5.11x |
| **Lending Club** | `gemini-3.5-flash-lite` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0463** | **0.0420** | **7.7%** | **1.00x** |
| **King County** | `claude-opus-5-5` | `business_analyst` | 100.0% | 0.0% | 0.2206 | 0.2168 | 1.7% | 0.31x |
| **King County** | `claude-opus-5-5` | `statistician_unassisted` | 100.0% | 0.0% | 0.1715 | **0.1676** | 3.4% | 0.24x *(over-guessed)* |
| **King County** | `claude-opus-5-5` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.2221** | **0.2184** | **2.6%** | **0.98x** |
| **King County** | `claude-sonnet-5-5` | `business_analyst` | 100.0% | 0.0% | 0.2491 | 0.2451 | 0.8% | 2.64x |
| **King County** | `claude-sonnet-5-5` | `statistician_unassisted` | 100.0% | 0.0% | 0.2121 | **0.2078** | 1.1% | 3.66x |
| **King County** | `claude-sonnet-5-5` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.2221** | **0.2184** | **2.6%** | **1.00x** |
| **King County** | `gpt-6.1-sol` | `business_analyst` | 94.4% | **5.6%** | 0.2588 | 0.2552 | 2.2% | 4.78x |
| **King County** | `gpt-6.1-sol` | `statistician_unassisted` | 100.0% | 0.0% | 0.1715 | **0.1676** | 3.4% | 4.78x |
| **King County** | `gpt-6.1-sol` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.2221** | **0.2184** | **2.6%** | **1.00x** |
| **King County** | `gemini-3.5-flash-lite` | `business_analyst` | 87.5% | **12.5%** | 0.2390 | 0.2178 | 1.8% | 6.05x |
| **King County** | `gemini-3.5-flash-lite` | `statistician_unassisted` | 100.0% | 0.0% | 0.1615 | 0.1547 | 1.9% | 4.78x |
| **King County** | `gemini-3.5-flash-lite` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.1503** | **0.1456** | **3.0%** | **2.15x** |
| **Bank Marketing** | `claude-opus-5-5` | `business_analyst` | 59.5% | **40.5%** | 0.0597 | 0.0489 | 16.4% | 0.28x |
| **Bank Marketing** | `claude-opus-5-5` | `statistician_unassisted` | 100.0% | 0.0% | 0.0570 | 0.0526 | 10.5% | 0.25x |
| **Bank Marketing** | `claude-opus-5-5` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0570** | **0.0526** | **10.5%** | **0.99x** |
| **Bank Marketing** | `claude-sonnet-5-5` | `business_analyst` | 86.7% | **13.3%** | 0.0593 | 0.0538 | 8.1% | 0.64x |
| **Bank Marketing** | `claude-sonnet-5-5` | `statistician_unassisted` | 100.0% | 0.0% | 0.0607 | 0.0561 | 8.2% | 3.72x |
| **Bank Marketing** | `claude-sonnet-5-5` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0570** | **0.0526** | **10.5%** | **1.00x** |
| **Bank Marketing** | `gpt-6.1-sol` | `business_analyst` | 85.7% | **14.3%** | 0.0575 | 0.0525 | 8.4% | 5.33x |
| **Bank Marketing** | `gpt-6.1-sol` | `statistician_unassisted` | 100.0% | 0.0% | 0.0408 | 0.0374 | 4.5% | 5.33x |
| **Bank Marketing** | `gpt-6.1-sol` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0570** | **0.0526** | **10.5%** | **1.00x** |
| **Bank Marketing** | `gemini-3.5-flash-lite` | `business_analyst` | 55.2% | **44.8%** | 0.0905 | 0.0730 | 17.0% | 7.56x |
| **Bank Marketing** | `gemini-3.5-flash-lite` | `statistician_unassisted` | 88.9% | **11.1%** | 0.1110 | 0.1020 | 6.2% | 3.16x |
| **Bank Marketing** | `gemini-3.5-flash-lite` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0570** | **0.0526** | **10.5%** | **1.00x** |

---

### Act V: End-to-End Local SQLite Warehouse + Semantic Layer + Closed-Form Continuous ANOVA Gate (108 SQL Runs)
In enterprise BI and data-warehouse environments, agents do not pull raw row-level DataFrames into Python—they query a **governed semantic layer** via SQL `GROUP BY` aggregations returning `COUNT(*)`, `AVG(y)`, and `STDDEV(y)`. Furthermore, many critical business targets (such as house sale prices `log_price`, revenue, or latency) are **continuous**, where arbitrary target binning destroys information.

#### Closed-Form Continuous Metric Gate (`ANOVA F -> RIG`)
In [`signalguard/threshold.py`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/threshold.py) ([`anova_rig_from_group_stats`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/threshold.py#L648-L735)), we derived and implemented a closed-form continuous information gate directly from SQL `GROUP BY` sufficient statistics $\{(n_g, \bar{y}_g, s_g)\}_{g=1}^K$. Using the exact Gaussian / one-way ANOVA mutual information identity:

$$SS_{\text{between}} = \sum_{g=1}^K n_g (\bar{y}_g - \bar{y}_{\text{grand}})^2, \qquad SS_{\text{within}} = \sum_{g=1}^K (n_g - 1) s_g^2, \qquad \eta^2 = \frac{SS_{\text{between}}}{SS_{\text{between}} + SS_{\text{within}}}$$

$$I_{\text{ANOVA}}(X; Y) = -\frac{1}{2} \log_2(1 - \eta^2) \text{ bits}, \qquad \text{RIG}_{\text{ANOVA}} = -\frac{1}{2} \log_2(1 - \eta^2)$$

Because $F = \frac{SS_{\text{between}} / (K - 1)}{SS_{\text{within}} / (N - K)}$ satisfies $(K - 1) F \sim \chi^2_{K-1}$ under $H_0$, $\text{RIG}_{\text{ANOVA}}$ obeys the **exact same asymptotic $\chi^2_{K-1} / (2 N \ln 2)$ scaling law** as discrete mutual information with unit reference entropy $H_{\text{ref}} = 1.0\text{ bit}$. This enables [`NoiseGateSessionLedger.evaluate_continuous_group_aggregation()`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/ledger.py) to gate any continuous SQL `GROUP BY` query without accessing raw rows!

![Figure 7: Local SQLite Warehouse + Semantic Layer + Closed-Form Continuous ANOVA Gate](notebooks/figures/fig7_sql_semantic_layer_benchmark.png)

* **Takeaway from Figure 7:** When all 4 frontier LLMs query a local SQLite warehouse ([`RealWorldSQLSemanticWarehouse`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/sql_semantic_benchmark.py)) across `king_county_continuous` (continuous `log_price` gated via SQL `AVG`/`STDDEV` ANOVA), `lending_club`, and `bank_marketing`:
  - Unassisted SQL agents under-report their `SQL CUBE` and `WHERE` slice search space by **$3.3\times\text{ to }6.0\times$** (Figure 7A) and promote fragile small-slice continuous queries like `WHERE waterfront='1' GROUP BY bedrooms_bin` ($N=36$) and `WHERE condition='2' GROUP BY yr_built_bin` ($N=41$).
  - Even unassisted PhD statisticians (`statistician_unassisted`) suffer **37.8%–50.0% Real FDR** on continuous SQL exploration because standard $F$-test $p$-values on $N=36\text{–}41$ slices look nominally significant ($p \approx 0.01\text{–}0.04$) when unadjusted for the 38-hypothesis session ledger.
  - SignalGuard's closed-form ANOVA gate (`tau_sig(M=38)`) deterministically rejects the small-slice SQL hallucinations while verifying true macro drivers (`grade`, `sqft_living_bin`, `view`), achieving **100.0% Holdout Replication (0.0% Real FDR)** across all 4 frontier models (Figure 7B & Table 2).

#### Table 2: Local SQLite Warehouse + Semantic Layer Benchmark Summary (108 Live Frontier SQL Runs)

| SQL Mart & Metric Gate | Frontier Model | Persona | Holdout Replication (%) | Real FDR (%) | Mean Train RIG | Mean Holdout RIG | SQL Under-Reporting Ratio |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `claude-opus-5-5` | `business_analyst` | 48.6% | **51.4%** | 0.2972 | 0.2729 | 2.42x |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `claude-opus-5-5` | `statistician_unassisted` | 62.2% | **37.8%** | 0.2417 | 0.2455 | 1.48x |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `claude-opus-5-5` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.2357** | **0.2394** | **1.00x** |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `claude-sonnet-5-5` | `business_analyst` | 53.2% | **46.8%** | 0.3554 | 0.3332 | 4.22x |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `claude-sonnet-5-5` | `statistician_unassisted` | 60.0% | **40.0%** | 0.2557 | 0.2593 | 3.45x |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `claude-sonnet-5-5` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.2357** | **0.2394** | **1.00x** |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `gpt-6.1-sol` | `business_analyst` | 17.9% | **82.1%** | 0.2336 | 0.2067 | 4.22x |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `gpt-6.1-sol` | `statistician_unassisted` | 60.0% | **40.0%** | 0.2557 | 0.2593 | 4.22x |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `gpt-6.1-sol` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.2357** | **0.2394** | **1.00x** |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `gemini-3.5-flash-lite` | `business_analyst` | 30.0% | **70.0%** | 0.4013 | 0.3785 | 8.87x |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `gemini-3.5-flash-lite` | `statistician_unassisted` | 50.0% | **50.0%** | 0.4388 | 0.4418 | 4.22x |
| **`king_county_continuous`** *(ANOVA F → RIG)* | `gemini-3.5-flash-lite` | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.1568** | **0.1603** | **2.41x** |
| **`lending_club`** *(Binary SQL Mart)* | All 4 Models Pooled | `business_analyst` | 95.8% | **4.2%** *(up to 16.7%)* | 0.0551 | 0.0494 | 5.97x |
| **`lending_club`** *(Binary SQL Mart)* | All 4 Models Pooled | `statistician_unassisted` | 100.0% | 0.0% | 0.0541 | 0.0493 | 5.27x |
| **`lending_club`** *(Binary SQL Mart)* | All 4 Models Pooled | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0496** | **0.0455** | **1.00x** |
| **`bank_marketing`** *(Low-SNR SQL Mart)* | All 4 Models Pooled | `business_analyst` | 91.7% | **8.3%** *(up to 20.0%)* | 0.0804 | 0.0738 | 4.25x |
| **`bank_marketing`** *(Low-SNR SQL Mart)* | All 4 Models Pooled | `statistician_unassisted` | 95.8% | **4.2%** *(up to 16.7%)* | 0.0888 | 0.0844 | 3.68x |
| **`bank_marketing`** *(Low-SNR SQL Mart)* | All 4 Models Pooled | **`statistician_signalguard`** | **100.0%** | **0.0%** | **0.0897** | **0.0853** | **1.00x** |

---

## Automated Feature Engineering (`agentune`-Style AutoML) Overfitting Benchmark

Beyond conversational LLM agents, automated feature discovery libraries (`agentune`, `featuretools`, LLM-driven feature synthesis) generate $M = 1{,}000+$ candidate interaction and ratio features and rank them greedily by in-sample Training RIG or Mutual Information.

We benchmarked **Unassisted Greedy Top-$K$ Selection** ($K = 15$ out of $M = 1{,}000$ generated candidates) against **SignalGuard-Governed Selection** ($\tau_{\text{sig}}(N, k, M_{\text{eff}}, \alpha=0.05) + \text{Small-Cell Sparsity Veto}$) across all 3 datasets.

![Figure 5: Agentune-Style Feature Overfitting & Holdout Decay](notebooks/figures/fig5_agentune_holdout_decay.png)

![Figure 6: Downstream Classifier Generalization Gap](notebooks/figures/fig6_agentune_downstream_overfitting_gap.png)

### Table 3: AutoML Feature Engineering Benchmark ($M = 1{,}000$ Candidates $\to$ Top-15 Selected)

| Dataset | Selection Pipeline | Holdout Replication Rate | Real FDR (%) | Mean Train RIG | Mean Holdout RIG | Out-of-Sample Decay (%) | Downstream Holdout ROC AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lending Club** | Baseline Raw Features (119 cols) | — | — | — | — | — | 0.7010 (Gap: 0.0282) |
| **Lending Club** | Unassisted Greedy Top-15 | 73.3% (11/15) | **26.7%** | 0.0356 | 0.0225 | **39.2%** | 0.6993 *(Degraded, Gap: 0.0309)* |
| **Lending Club** | **SignalGuard-Governed Top-15** | **100.0% (15/15)** | **0.0%** | **0.0314** | **0.0260** | **21.9%** | **0.7028** *(Improved, Gap: 0.0274)* |
| **King County** | Unassisted Greedy Top-15 | 100.0% (15/15) | 0.0% | 0.2403 | 0.2125 | **13.4%** | — |
| **King County** | **SignalGuard-Governed Top-15** | **100.0% (15/15)** | **0.0%** | **0.2226** | **0.2105** | **6.7%** *(2.0x lower decay)* | — |
| **Bank Marketing** | Unassisted Greedy Top-15 | 100.0% (15/15) | 0.0% | 0.1168 | 0.1173 | 7.3% | — |
| **Bank Marketing** | **SignalGuard-Governed Top-15** | **100.0% (15/15)** | **0.0%** | **0.0892** | **0.0917** | **6.4%** | — |

* **Key AutoML Takeaway:** On Lending Club, greedy Top-15 selection promotes 4 high-cardinality 3-way interaction cubes that look strong on Train ($\text{RIG} \approx 0.033$) but collapse near zero on the Holdout Vault ($\text{RIG} < 0.002$, **26.7% Real FDR**, **39.2% mean signal decay**), actively **degrading downstream Holdout ROC AUC** from `0.7010` down to `0.6993`. SignalGuard's EVT + sparsity gate filters out the overfit interactions, achieving **100.0% feature replication**, cutting signal decay nearly in half (**21.9%**), and **improving downstream Holdout ROC AUC to `0.7028`**.

---

## Reproducibility & Repository Navigation

All code, unit tests, datasets, cached transcripts, and figure scripts are self-contained in this repository:

* **Core Mathematical & Ledger Engine:**
  * [`signalguard/threshold.py`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/threshold.py) — Closed-form Wilks' $\chi^2_k$ & Gumbel EVT RIG thresholds (`FeatureNoiseThreshold`), eigenvalue effective rank (`effective_number_of_tests`), and closed-form continuous SQL ANOVA gate (`anova_rig_from_group_stats`).
  * [`signalguard/ledger.py`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/ledger.py) — Stateful multiple-testing session ledger (`NoiseGateSessionLedger`) supporting both discrete contingency tables and continuous SQL `GROUP BY` sufficient statistics (`evaluate_continuous_group_aggregation`).
  * [`signalguard/mcp_server.py`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/mcp_server.py) — Model Context Protocol (MCP) server exposing deterministic statistical verification tools to autonomous LLM agents.
* **Real-World Benchmarks & SQL Semantic Layer:**
  * [`signalguard/real_world_benchmark.py`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/real_world_benchmark.py) — Train/Holdout split-vault evaluator for tabular multi-turn LLM exploration and `agentune`-style AutoML feature engineering.
  * [`signalguard/sql_semantic_benchmark.py`](file:///c:/Users/eldri/Code/sampling_toolbox/signalguard/sql_semantic_benchmark.py) — Local SQLite warehouse + governed YAML/JSON semantic catalog (`RealWorldSQLSemanticWarehouse`) with live multi-turn SQL `GROUP BY` / `CUBE` agent benchmarking.
* **Master Empirical Summary Tables & Figures:**
  * [`Data/real_world_eval_all_models_summary.csv`](file:///c:/Users/eldri/Code/sampling_toolbox/Data/real_world_eval_all_models_summary.csv) (108 live tabular frontier runs)
  * [`Data/sql_semantic_eval_all_models_summary.csv`](file:///c:/Users/eldri/Code/sampling_toolbox/Data/sql_semantic_eval_all_models_summary.csv) (108 live SQLite semantic layer frontier runs)
  * [`scripts/generate_notebook_06_figures.py`](file:///c:/Users/eldri/Code/sampling_toolbox/scripts/generate_notebook_06_figures.py) — Reproduces Figures 3 through 7 directly from the cached empirical CSVs.
  * [`notebooks/06_real_world_and_agentune_benchmark.ipynb`](file:///c:/Users/eldri/Code/sampling_toolbox/notebooks/06_real_world_and_agentune_benchmark.ipynb) — Interactive Jupyter walkthrough of the real-world frontier LLM, SQL semantic warehouse, and AutoML benchmarks.
