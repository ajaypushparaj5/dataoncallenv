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

An RL benchmark that simulates the workflow of an on-call data analyst debugging broken reports. The agent investigates realistic data pipeline bugs across three difficulty tiers — diagnosing root causes, writing corrected SQL, and earning a multi-dimensional reward score.

## Why This Exists

AI agents are being deployed as data analysts. But there is no benchmark that measures whether these agents can actually debug — not just query. DataOnCallEnv fills that gap by simulating exactly the workflow a real on-call analyst performs, scored against deterministic ground truth.

## Features

| Feature | Description |
|---------|-------------|
| Partial Observability | Tables are hidden at reset. Agent must call `list_tables()` to discover schema before inspecting or querying. |
| Query Cost Budget | Each tool has a fixed cost (`run_sql`=2.0, `inspect_schema`=1.0, etc). Total budget: 20.0 per episode. |
| Realistic Logs | Expanded dbt pipeline logs (8–15 rows per task) with noise entries + Airflow DAG run history table. |
| Anti-Cheat | `SELECT *` blocked, results capped at 50 rows, minimum 2 tool calls before submit, memorized-answer detection. |
| Tiered Evaluation | Diagnosis scored by depth of understanding (exact → category → symptom), not binary pass/fail. |
| Deterministic | All tasks and grading are fully deterministic. Same actions always produce the same score. |

## Tasks

| ID | Difficulty | Title | Root Cause | Optimal Steps | Optimal Cost |
|----|-----------|-------|------------|---------------|-------------|
| 1 | Easy | Revenue shows $0 for international sales | Currency code casing mismatch (`USD` vs `usd`) causes silent NULL JOIN | 5 | 7.0 |
| 2 | Medium | MAU dropped 8% on Feb 1st | UTC→local timezone migration double-counts events at month boundary | 7 | 10.0 |
| 3 | Hard | Cloud Storage revenue overstated by 3.7x | Non-unique key in `product_promotions` causes fanout on JOIN | 8 | 13.0 |

## Action Space

Every action has `tool`, `query`, and an optional `reasoning` field (rewarded by the grader).

| Tool | Query Format | Cost | Description |
|------|-------------|------|-------------|
| `list_tables` | `""` | 0.5 | Discover available tables (must be called first) |
| `inspect_schema` | `"table_name"` | 1.0 | Column names and types for a discovered table |
| `check_logs` | `""` | 1.0 | dbt pipeline changelog (ordered by most recent) |
| `check_airflow` | `""` | 1.0 | Airflow DAG run history |
| `run_sql` | `"SELECT col FROM ..."` | 2.0 | Execute a SELECT query (no `SELECT *`) |
| `diff_report` | `"date1,date2"` | 1.5 | Compare revenue totals between two dates |
| `submit` | `"ROOT CAUSE: ... CORRECTED SQL: ..."` | 0.0 | Submit diagnosis and fix. Ends episode. |

## Observation Space

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

## Reward Function

| Component | Range | Description |
|-----------|-------|-------------|
| `diagnosis_correct` | 0.00–0.25 | Tiered: exact root cause (0.25), category match (0.15), symptom only (0.08) |
| `fix_valid` | 0.00–0.25 | Agent's proposed SQL returns correct output vs ground truth |
| `efficiency` | 0.00–0.15 | Combined step + cost efficiency relative to optimal |
| `reasoning_quality` | 0.00–0.10 | Fraction of actions that include a reasoning field |
| `investigation_quality` | 0.00–0.10 | Logical methodology: discovery → schema → logs → hypothesis → verify |
| `false_positive_penalty` | 0.00–0.15 | Deducted for irrelevant table access, duplicate queries, or cheating |

**Total: 0.0–1.0** (dense reward — every action contributes signal)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check, returns env metadata and version |
| `GET` | `/` | Root info with available endpoints |
| `GET` | `/tasks` | List all tasks with metadata |
| `GET` | `/state` | Full current episode state |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/reset` | Start a fresh episode. Body: `{"task_id": 1}` |
| `POST` | `/step` | Send one action. Body: `{"tool": "...", "query": "...", "reasoning": "..."}` |

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install and Run Locally

```bash
git clone https://github.com/ajaypushparaj5/dataoncallenv.git
cd dataoncallenv
pip install -r requirements.txt

# Start the API server
uvicorn api.app:app --reload --port 8000
```

### Quick API Test

```bash
# Health check
curl http://localhost:8000/health

# Start Task 1
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'

# Discover tables
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "list_tables", "query": "", "reasoning": "Discover available tables"}'

# Inspect schema
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "inspect_schema", "query": "sales", "reasoning": "Check sales table structure"}'
```

### Run with Docker

```bash
docker build -t dataoncallenv .
docker run -p 7860:7860 dataoncallenv
```

### Run Baseline Inference

Requires an API key for an OpenAI-compatible inference provider.

```bash
# Create .env file
cat > .env << EOF
HF_TOKEN=your_hf_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
EOF

python inference.py
```

### Run Tests

```bash
python test_env.py
# Expected: 36 passed, 0 failed
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | Hugging Face API token (used as `OPENAI_API_KEY`) |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | OpenAI-compatible inference endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for the inference provider |

## Hugging Face Spaces Deployment

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Upload the project files (or connect via Git)
4. Add `HF_TOKEN` as a secret in Space Settings
5. The Space will auto-build using the `Dockerfile` and expose the API on port 7860

## Project Structure

```
dataoncallenv/
├── models.py          # Pydantic types: Action, Observation, Reward, EnvState
├── tasks.py           # Task definitions with ground truth and diagnosis tiers
├── database.py        # SQLite database builder, tool implementations, anti-cheat
├── environment.py     # Core RL env: partial obs, query costs, step/reset/state
├── graders.py         # Tiered scoring, investigation quality, penalties
├── inference.py       # Baseline agent using OpenAI-compatible API
├── test_env.py        # 36 integration tests
├── api/
│   └── app.py         # FastAPI server with /health, /reset, /step, /state
├── openenv.yaml       # OpenEnv spec metadata
├── Dockerfile         # HF Spaces container (port 7860)
├── requirements.txt   # Python dependencies
├── baseline_scores.json  # Reproducible baseline results
└── .env               # API credentials (not committed)
```

## OpenEnv Spec

This environment implements the full [OpenEnv](https://github.com/open-env/openenv) specification:
- Typed Pydantic models for `Action`, `Observation`, `Reward`, and `EnvState`
- `reset()` / `step()` / `state()` API
- `openenv.yaml` with full metadata
- Minimum 3 tasks with agent graders (easy → medium → hard)
- Scores in range 0.0–1.0 with partial progress signals
- Deterministic evaluation
- Baseline inference script with reproducible scores

## Anti-Cheat Constraints

| Constraint | Effect |
|------------|--------|
| `SELECT *` blocked | Agent must specify columns explicitly |
| Row cap (50) | Large result dumps are truncated |
| Min 2 tools before submit | Prevents skipping investigation |
| Memorized-answer detection | Correct diagnosis without investigation incurs penalty |
| Duplicate query penalty | Repeated identical queries are penalized |
| Irrelevant table penalty | Accessing tables unrelated to the task is penalized |
