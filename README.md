---
title: DataOnCallEnv
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
tags:
  - openenv
---

# DataOnCallEnv

Every data team has a 3am Slack message: *"the numbers look wrong."*  
**DataOnCallEnv** is an RL benchmark for the debugging workflow that follows — hypothesis, query, trace, fix — evaluated against real root causes planted in a synthetic but realistic data stack.

---

## Why this environment exists

AI agents are being deployed as data analysts. Companies like Airbnb, Stripe, and every fintech team are building agents that monitor pipelines and investigate anomalies. But there is no benchmark that measures whether these agents can actually *debug* — not just query.

DataOnCallEnv fills that gap. It simulates exactly the workflow a real on-call analyst performs, scored against deterministic ground truth.

---

## Key Features (v2.0)

| Feature | Description |
|---------|-------------|
| **Partial Observability** | Tables are hidden at reset. Agent must call `list_tables()` to discover what's available before inspecting or querying. |
| **Query Cost Budget** | Each tool has a fixed cost (e.g., `run_sql`=2.0). Total budget: 20.0 per episode. Forces efficient investigation. |
| **Realistic Logs** | Expanded dbt pipeline logs (8-15 rows per task) with noise entries + Airflow DAG run history. |
| **Anti-Cheat** | `SELECT *` blocked, results capped at 50 rows, minimum 2 tool calls before submit, memorized-answer detection. |
| **Tiered Evaluation** | Diagnosis scored by depth of understanding (exact/category/symptom), not binary. Investigation quality bonus. |

---

## Environment description

The agent receives a stakeholder Slack message describing a broken report. It has access to a SQLite database (sales, products, pipeline logs, Airflow runs) and a set of tools to investigate. It must diagnose the root cause and propose a corrected query — without being told where the bug is or even what tables exist.

**State:** The current task scenario, tool results accumulated so far, steps remaining, cost budget remaining, and discovered tables.  
**Episode:** Ends when the agent calls `submit()`, exhausts its 15-step budget, or spends its 20.0 cost budget.  
**Reward:** Multi-dimensional — diagnosis accuracy (tiered), fix correctness, efficiency (steps + cost), reasoning quality, investigation methodology, minus penalties.

---

## Action space

| Tool | Query format | Cost | Description |
|------|-------------|------|-------------|
| `list_tables` | *(empty)* | 0.5 | Discover all available tables (START HERE) |
| `inspect_schema` | Table name | 1.0 | See column names and types (table must be discovered first) |
| `check_logs` | *(empty)* | 1.0 | Read the dbt pipeline changelog |
| `check_airflow` | *(empty)* | 1.0 | Read Airflow DAG run history |
| `run_sql` | SQL SELECT string | 2.0 | Execute a query (no SELECT *, must specify columns) |
| `diff_report` | `"date1,date2"` | 1.5 | Compare report output between two dates |
| `submit` | Your answer string | 0.0 | Submit root cause + fix. Ends episode. |

Optional field on every action: `reasoning` — the agent's explanation of *why* it is making this call. The grader rewards agents that explain their thinking.

---

## Observation space

```json
{
  "task_id": 1,
  "result": { "...tool output..." },
  "steps_taken": 3,
  "done": false,
  "max_steps": 15,
  "cost_spent": 4.5,
  "budget_remaining": 15.5
}
```

---

## Tasks

### Task 1 — Easy: Silent NULL JOIN
**Scenario:** Weekly revenue report shows $0 for all international sales. USD sales are fine.  
**Root cause:** `currency_rates` table stores codes as `"usd"` but `sales` table uses `"USD"`. The JOIN silently returns NULLs — no error in the pipeline logs.  
**Optimal steps:** 5 | **Optimal cost:** 7.0

### Task 2 — Medium: Timezone Ghost
**Scenario:** Monthly active users dropped 8% on February 1st. Nothing changed in the product.  
**Root cause:** Pipeline migrated from UTC to local time on Jan 31. Events near midnight were double-counted in January and missed in February. The migration is recorded in `dbt_log`.  
**Optimal steps:** 7 | **Optimal cost:** 10.0

### Task 3 — Hard: Fanout Inflation
**Scenario:** Cloud Storage revenue is overstated by exactly 3.7x. Other product lines are fine.  
**Root cause:** `product_promotions` table has 3–4 rows per product (non-unique key). A JOIN multiplies every sale row by the number of promo rows.  
**Optimal steps:** 8 | **Optimal cost:** 13.0

---

## Reward function

```
score = diagnosis_correct     (0.00–0.25)   # tiered: exact/category/symptom
      + fix_valid              (0.00–0.25)   # proposed fix returns correct output
      + efficiency             (0.00–0.15)   # step efficiency + cost efficiency
      + reasoning_quality      (0.00–0.10)   # fraction of actions with reasoning
      + investigation_quality  (0.00–0.10)   # logical debugging methodology
      - false_positive_penalty (0.00–0.15)   # irrelevant tables, duplicates, cheating
```

Reward is dense — every step contributes signal. An agent that diagnoses correctly but writes a broken fix scores ~0.55, not 0 or 1.

---

## Setup and usage

### Run locally

```bash
git clone https://huggingface.co/spaces/your-username/dataoncallenv
cd dataoncallenv
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```

### Quick API test

```bash
# Health check
curl http://localhost:8000/health

# Start Task 1
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'

# Discover tables first (partial observability)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "list_tables", "query": "", "reasoning": "Need to discover what tables exist"}'

# Inspect a table
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "inspect_schema", "query": "sales", "reasoning": "Check sales table structure"}'
```

### Run Docker

```bash
docker build -t dataoncallenv .
docker run -p 7860:7860 dataoncallenv
```

### Run baseline inference

```bash
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

---

## Project structure

```
dataoncallenv/
├── models.py        # Pydantic types: Action, Observation, Reward, EnvState
├── database.py      # SQLite builder + tool implementations + anti-cheat
├── tasks.py         # Task definitions with ground truth + diagnosis tiers
├── graders.py       # Tiered scoring, investigation quality, penalties
├── environment.py   # Core env: partial obs, query costs, anti-cheat
├── api/
│   └── app.py       # FastAPI server
├── inference.py     # OpenAI agent baseline
├── test_env.py      # Integration tests
├── openenv.yaml     # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```