# SignalGuard: Executive Brief for Engineering & Business Leadership

**Why AI Data Agents Report Insights That Don't Hold Up, and How a Statistical Guardrail Helps**

*A 3-minute guide for Engineering Directors, VPs of Operations, and Finance Leaders.*

> 🎨 **Interactive Visual Dashboard:** An interactive visual explainer version of this brief featuring live multi-testing simulations, visual side-by-side diagrams, and 1-click PDF export is available live at **[eldridzakis.github.io/signal-guard](https://eldridzakis.github.io/signal-guard/)** (or via [docs/index.html](docs/index.html)).

---

## 1. The Problem: "The Ghost in the Data Warehouse"

Many companies are giving Large Language Models (LLMs), via Claude Code, Cursor, LangGraph, or internal Text-to-SQL tools, direct access to their data warehouse (Snowflake, BigQuery, Databricks, PostgreSQL).

The pitch: *"Ask your data questions in plain English, and the AI will uncover hidden revenue leaks, operational bottlenecks, and customer risks."*

**The trap is easy to miss:**

```
           WHAT THE BUSINESS SEES                           WHAT ACTUALLY HAPPENED
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│  AI Agent Slack / Dashboard Output:      │    │  Hidden SQL Session Behind the Scenes:   │
│                                          │    │  • Query 1: region = 'East' (no trend)   │
│  "We discovered that customer segment    │    │  • Query 2: tenure > 3 years (no trend)  │
│   X has a 42% higher default rate        │ ◄──┼─ • ...                                   │
│   (p = 0.03, statistically significant)! │    │  • Query 38: region='West' AND age<25    │
│   Recommendation: Restructure pricing."  │    │    (N=35 rows, p=0.03, plausibly chance) │
└──────────────────────────────────────────┘    └──────────────────────────────────────────┘
```

When a careful analyst runs 40 queries, they know they tested 40 hypotheses and adjust for it. An AI agent exploring a database often doesn't:
* It may run **dozens of SQL queries** behind the scenes.
* Some of those queries drill into small cohorts (e.g. $N = 35$ customers).
* With enough small cuts, some will look "statistically significant" purely by chance.
* The agent reports the most striking result as a finding, without accounting for the queries that came up empty.

In our benchmark (described in Section 3), frontier LLMs reading such an exploration transcript frequently promoted small-slice results that **did not replicate on held-out data**.

---

## 2. Why Leadership Should Care: The Cost of Error by Role

| Executive Role | The Risk | Business Consequence |
| :--- | :--- | :--- |
| **Director of Finance / Head of FP&A** | **The Winner's Curse**<br>The most impressive-looking segment effects are, on average, overstated, and some are noise. | **Misallocated budget:** pricing, credit, or marketing decisions anchored on effects that shrink or vanish out-of-sample. |
| **VP of Operations & Supply Chain** | **Over-reacting to small samples**<br>"Urgent bottlenecks" flagged from a few dozen records. | **Wasted effort:** teams re-engineer processes to fix problems that may not exist at scale. |
| **Director of Engineering / Platform** | **Building on weak signals**<br>AI-generated findings get turned into pipelines, features, and alerts. | **Technical debt:** sprints spent on dbt models, features, and dashboards around signals that provide little real lift. |

---

## 3. What Our Benchmark Shows (and What It Doesn't)

We ran 216 evaluations across four frontier models (`claude-opus-5-5`, `claude-sonnet-5-5`, `gpt-6.1-sol`, `gemini-3.5-flash-lite`) on three public datasets (Lending Club loans, King County housing, UCI Bank Marketing). Each model read a scripted multi-step exploration transcript (single-dimension queries, cross-tabs, and small filtered slices) and wrote a findings report. We then checked whether each reported finding held up on a held-out portion of the data.

Results on the continuous King County SQL mart, where the effect is clearest:

| Configuration | What Happened | Findings That Failed Holdout Replication |
| :--- | :--- | :---: |
| **Unassisted "business analyst" prompt** | Under-counted how many queries were run; promoted small-slice results ($N \approx 36\text{-}41$). | **14% - 41%** (0% - 27% under a plain holdout test) |
| **"Act as a PhD statistician" prompt** | Declined to report the small slices in every run; reported only large, well-supported effects. | **0%** |
| **SignalGuard verdicts provided** | Model reported only findings SignalGuard verified. | **0%** |

On the binary-target SQL marts (Lending Club, Bank Marketing) the `business_analyst` failure rates were about 4-8% (0-2% under a plain holdout test).

> [!IMPORTANT]
> **How to read these numbers.** The exploration path was scripted (the models chose what to *report*, not what to *query*), and the small slices were deliberately included as traps. The holdout check also treats very small samples as non-replicating, which overlaps with SignalGuard's own small-sample rule, so the 0% figure partly reflects that shared criterion. These results show the failure mode is real and that the guardrail mechanism works as designed. They are **not** an independent estimate of false discovery rates in production. Some findings SignalGuard rejects may be real but under-powered effects. See [Benchmark Design & Limitations](REPORT.md#benchmark-design--limitations).

**Takeaway:** in this benchmark, a model prompted as a skeptical statistician mostly avoided the small-slice traps on its own, while a business-analyst-style prompt did not. A guardrail's value is therefore less "fixing a model that can't be careful" and more **enforcing carefulness deterministically and auditing it** (a fixed hypothesis count and a rising bar, regardless of how the model is prompted). Verified-only reporting is also more conservative: it surfaced fewer of the holdout-significant candidates on the King County SQL mart (34% recall vs 53% for the unassisted statistician prompt).

---

## 4. How SignalGuard Works (Without Math)

### The Casino Analogy
Imagine a roulette table:
* If one person bets on number 17 and wins, that's a 1-in-38 event, impressive.
* If 50 people each pick a number, it's very likely *someone* hits 17. That's not skill; it's the number of bets.

An AI agent querying your database is often making many bets. When it reports back, it can look like the one lucky bettor.

**SignalGuard keeps count of the bets:**
1. **Session ledger:** every hypothesis the agent tests is recorded.
2. **Rising bar:** the more hypotheses tested, and the smaller the sample, the stronger the evidence required before something is called a finding.
3. **Verdicts back to the agent:** results that don't clear the bar are flagged as likely noise (or "needs validation") before they reach a report.

---

## 5. Architecture

```
┌────────────────────────┐      ┌──────────────────────────┐      ┌────────────────────────┐
│  AI Agent / Workflow   │ ───► │   SignalGuard Guardrail  │ ───► │  SQL Warehouse / Lake  │
│ (Claude, LangGraph)    │ ◄─── │ (MCP Server / Tool APIs) │ ◄─── │ (Snowflake, BigQuery)  │
└────────────────────────┘      └──────────────────────────┘      └────────────────────────┘
                                 • Works on aggregate statistics
                                 • Closed-form math (no simulations needed)
                                 • Stateful hypothesis count per session
```

1. **Lightweight:** for SQL `GROUP BY` results, the check uses the aggregate statistics the query already returns (`COUNT`, `AVG`, `STDDEV`) and a closed-form formula, so it doesn't need to re-scan tables. (We have not yet published formal latency benchmarks.)
2. **Works on aggregates:** the continuous-metric check needs only group-level summaries, not row-level records, which can simplify data-governance review.
3. **Fits agent toolchains:** ships as a **Model Context Protocol (MCP)** server (Claude Desktop, Cursor, Claude Code) and as OpenAI/Anthropic tool schemas for custom Python agents.

---

## 6. Three Questions to Ask Your Team

If your team is deploying LLMs or automated analytics against company data:

1. **"Does our agent track how many hypotheses it tests across a session, or is each query judged in isolation?"**  
   *(If in isolation, expect more false positives the longer the agent explores.)*
2. **"When the agent drills into a small cohort (say $N < 100$), does the bar for calling something significant go up?"**  
   *(Fixed thresholds like $p < 0.05$ or "3× lift" are easily cleared by chance in small groups.)*
3. **"How do we check whether AI-recommended insights hold up on data the agent didn't see?"**  
   *(A holdout check, plus a guardrail like SignalGuard during exploration, substantially reduces the risk.)*

---

## Next Steps

* **For the full mathematical derivations, benchmark charts, and limitations:** Read [`REPORT.md`](REPORT.md).
* **To install and run the MCP server:** Read [`README.md`](README.md).
