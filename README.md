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
**DataOnCallEnv** is the first RL benchmark for the debugging workflow that follows — hypothesis, query, trace, fix — evaluated against real root causes planted in a synthetic but realistic data stack.

---

## Why this environment exists

AI agents are being deployed as data analysts. Companies like Airbnb, Stripe, and every fintech team are building agents that monitor pipelines and investigate anomalies. But there is no benchmark that measures whether these agents can actually *debug* — not just query.

DataOnCallEnv fills that gap. It simulates exactly the workflow a real on-call analyst performs, scored against deterministic ground truth.

---

## Environment description

The agent receives a stakeholder Slack message describing a broken report. It has access to a SQLite database (sales, products, pipeline logs) and a set of tools to investigate. It must diagnose the root cause and propose a corrected query — without being told where the bug is.

**State:** The current task scenario, tool results accumulated so far, and steps remaining.  
**Episode:** Ends when the agent calls `submit()` or exhausts its 10-step budget.  
**Reward:** Multi-dimensional — diagnosis accuracy, fix correctness, efficiency, reasoning quality.

---

## Action space

| Tool | Query format | Description |
|------|-------------|-------------|
| `run_sql` | SQL SELECT string | Execute a query against the broken database |
| `inspect_schema` | Table name | See column names and types |
| `check_logs` | *(empty)* | Read the dbt pipeline changelog |
| `diff_report` | `"date1,date2"` | Compare report output between two dates |
| `list_tables` | *(empty)* | List all available tables |
| `submit` | Your answer string | Submit root cause + fix. Ends episode. |

Optional field on every action: `reasoning` — the agent's explanation of *why* it is making this call. The grader rewards agents that explain their thinking.

---

## Observation space

```json
{
  "task_id": 1,
  "result": { "...tool output..." },
  "steps_taken": 3,
  "done": false,
  "max_steps": 10
}
```

---

## Tasks

### Task 1 — Easy: Silent NULL JOIN
**Scenario:** Weekly revenue report shows $0 for all international sales. USD sales are fine.  
**Root cause:** `currency_rates` table stores codes as `"usd"` but `sales` table uses `"USD"`. The JOIN silently returns NULLs — no error in the pipeline logs.  
**Expected GPT-4o score:** ~0.72  
**Optimal steps:** 4

### Task 2 — Medium: Timezone Ghost
**Scenario:** Monthly active users dropped 8% on February 1st. Nothing changed in the product.  
**Root cause:** Pipeline migrated from UTC to local time on Jan 31. Events near midnight were double-counted in January and missed in February. The migration is recorded in `dbt_log`.  
**Expected GPT-4o score:** ~0.51  
**Optimal steps:** 6

### Task 3 — Hard: Fanout Inflation
**Scenario:** Cloud Storage revenue is overstated by exactly 3.7x. Other product lines are fine.  
**Root cause:** `product_promotions` table has 3–4 rows per product (non-unique key). A JOIN multiplies every sale row by the number of promo rows. Only affects products launched after a schema change.  
**Expected GPT-4o score:** ~0.28  
**Optimal steps:** 8

---

## Reward function

```
score = diagnosis_correct   (0.00–0.30)  # named the right root cause
      + fix_valid            (0.00–0.25)  # proposed fix returns correct output
      + efficiency           (0.00–0.20)  # fewer steps = higher score
      + reasoning_quality    (0.00–0.15)  # fraction of actions with reasoning field
      - false_positive_penalty (0.00–0.20) # destructive or irrelevant queries
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

# Send a tool call
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "list_tables", "query": "", "reasoning": "First I will see what tables exist"}'
```

### Run Docker

```bash
docker build -t dataoncallenv .
docker run -p 7860:7860 dataoncallenv
```

### Run baseline inference

```bash
export OPENAI_API_KEY="your_key"
export MODEL_NAME="gpt-4o-mini"
export API_BASE_URL="https://api.openai.com/v1"
python inference.py
```

---

## Baseline scores

| Task | Difficulty | GPT-4o-mini | GPT-4o |
|------|-----------|-------------|--------|
| 1    | Easy      | 0.65        | 0.72   |
| 2    | Medium    | 0.42        | 0.51   |
| 3    | Hard      | 0.18        | 0.28   |
| **avg** | —      | **0.42**    | **0.50** |

---

## Project structure

```
dataoncallenv/
├── models.py        # Pydantic types: Action, Observation, Reward
├── database.py      # SQLite builder + tool implementations
├── tasks.py         # Task definitions with ground truth
├── graders.py       # Deterministic scoring functions
├── environment.py   # Core env: reset() step() state()
├── api/
│   └── app.py       # FastAPI server
├── inference.py     # OpenAI agent baseline
├── openenv.yaml     # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```