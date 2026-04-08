"""Integration tests for DataOnCallEnv.
"""

import sys
import json

# Ensure project root is on path
sys.path.insert(0, ".")

from environment import DataOnCallEnv
from models import Action, Observation, Reward

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {detail}")
        failed += 1

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST SUITE: DataOnCallEnv v2.0")
print("="*60)

# ── 1. Partial Observability ─────────────────────────────────────────────────
print("\n── 1. Partial Observability ──")

env = DataOnCallEnv()
obs = env.reset(task_id=1)

# No tables should be visible in initial observation
test("Reset hides tables",
     "available_tables" not in str(obs.result) or "tables" not in obs.result,
     f"Tables found in reset: {obs.result}")

test("Discovered tables empty at start",
     len(env.discovered_tables) == 0)

# Try to inspect schema before discovering tables
action = Action(tool="inspect_schema", query="sales", reasoning="test")
obs, reward, done, info = env.step(action)
test("inspect_schema before discovery → error",
     "error" in str(obs.result).lower() or "not discovered" in str(obs.result).lower(),
     f"Got: {obs.result}")

# Try to run SQL before discovering tables
action = Action(tool="run_sql", query="SELECT sale_id FROM sales LIMIT 5", reasoning="test")
obs, reward, done, info = env.step(action)
test("run_sql before discovery → error",
     "error" in str(obs.result).lower(),
     f"Got: {obs.result}")

# Now discover tables
action = Action(tool="list_tables", query="", reasoning="discover tables")
obs, reward, done, info = env.step(action)
test("list_tables discovers tables",
     len(env.discovered_tables) > 0,
     f"Discovered: {env.discovered_tables}")

# Now inspect_schema should work
action = Action(tool="inspect_schema", query="sales", reasoning="check schema after discovery")
obs, reward, done, info = env.step(action)
test("inspect_schema after discovery → works",
     "columns" in str(obs.result),
     f"Got: {obs.result}")

# ── 2. Query Cost System ─────────────────────────────────────────────────────
print("\n── 2. Query Cost System ──")

env2 = DataOnCallEnv()
obs = env2.reset(task_id=1)

test("Cost starts at 0", obs.cost_spent == 0.0)
test("Budget starts at 20.0", obs.budget_remaining == 20.0)

# list_tables costs 0.5
action = Action(tool="list_tables", query="", reasoning="test cost")
obs, _, _, info = env2.step(action)
test("list_tables costs 0.5", obs.cost_spent == 0.5, f"Got: {obs.cost_spent}")
test("Budget remaining correct", obs.budget_remaining == 19.5, f"Got: {obs.budget_remaining}")

# run_sql costs 2.0
action = Action(tool="run_sql", query="SELECT sale_id, amount FROM sales LIMIT 3", reasoning="test cost")
obs, _, _, info = env2.step(action)
test("run_sql costs 2.0", obs.cost_spent == 2.5, f"Got: {obs.cost_spent}")

# Cost appears in info
test("Cost in info dict", "cost_spent" in info and "budget_remaining" in info)

# ── 3. Real Logs (dbt + Airflow) ─────────────────────────────────────────────
print("\n── 3. Real Logs (dbt + Airflow) ──")

env3 = DataOnCallEnv()
env3.reset(task_id=1)

# Check dbt logs are expanded
action = Action(tool="check_logs", query="", reasoning="test logs")
obs, _, _, _ = env3.step(action)
log_result = obs.result
test("dbt logs have multiple entries",
     isinstance(log_result, list) and len(log_result) >= 8,
     f"Got {len(log_result) if isinstance(log_result, list) else 'non-list'} entries")

# Check logs have realistic fields
if isinstance(log_result, list) and len(log_result) > 0:
    test("dbt logs have duration_s field",
         "duration_s" in log_result[0],
         f"Keys: {log_result[0].keys()}")

# Check Airflow logs
action = Action(tool="check_airflow", query="", reasoning="test airflow")
obs, _, _, _ = env3.step(action)
af_result = obs.result
test("Airflow runs available",
     isinstance(af_result, list) and len(af_result) >= 3,
     f"Got {len(af_result) if isinstance(af_result, list) else 'non-list'} entries")

if isinstance(af_result, list) and len(af_result) > 0:
    test("Airflow has dag_id field",
         "dag_id" in af_result[0],
         f"Keys: {af_result[0].keys()}")

# Test all 3 tasks have logs
for tid in [1, 2, 3]:
    env_t = DataOnCallEnv()
    env_t.reset(task_id=tid)
    obs_l, _, _, _ = env_t.step(Action(tool="check_logs", query="", reasoning="test"))
    obs_a, _, _, _ = env_t.step(Action(tool="check_airflow", query="", reasoning="test"))
    test(f"Task {tid} has dbt logs",
         isinstance(obs_l.result, list) and len(obs_l.result) >= 8)
    test(f"Task {tid} has airflow runs",
         isinstance(obs_a.result, list) and len(obs_a.result) >= 3)

# ── 4. Anti-Cheating ─────────────────────────────────────────────────────────
print("\n── 4. Anti-Cheating ──")

env4 = DataOnCallEnv()
env4.reset(task_id=1)

# Try to submit before minimum steps
action = Action(tool="submit", query="case mismatch", reasoning="test early submit")
obs, reward, done, info = env4.step(action)
test("Early submit blocked",
     "error" in str(obs.result).lower() and not done,
     f"Done={done}, result={obs.result}")

# SELECT * should be blocked
action = Action(tool="list_tables", query="", reasoning="discover")
env4.step(action)
action = Action(tool="run_sql", query="SELECT * FROM sales", reasoning="test select star")
obs, _, _, _ = env4.step(action)
test("SELECT * blocked",
     "error" in str(obs.result).lower() and "select *" in str(obs.result).lower(),
     f"Got: {obs.result}")

# Specific columns should work
action = Action(tool="run_sql", query="SELECT sale_id, amount FROM sales LIMIT 3", reasoning="test")
obs, _, _, _ = env4.step(action)
test("Specific columns work",
     isinstance(obs.result, list),
     f"Got: {type(obs.result)}")

# ── 5. Better Evaluation ─────────────────────────────────────────────────────
print("\n── 5. Better Evaluation (Graders) ──")

# Test with a good investigation path
env5 = DataOnCallEnv()
env5.reset(task_id=1)

good_actions = [
    Action(tool="list_tables", query="", reasoning="First, discover what tables exist"),
    Action(tool="inspect_schema", query="sales", reasoning="Check sales table structure"),
    Action(tool="inspect_schema", query="currency_rates", reasoning="Check currency rates structure"),
    Action(tool="check_logs", query="", reasoning="Check pipeline logs for errors"),
    Action(tool="run_sql", query="SELECT currency FROM sales LIMIT 5", reasoning="Check currency format in sales"),
    Action(tool="run_sql", query="SELECT currency_code FROM currency_rates", reasoning="Checking if there's a case mismatch between tables"),
    Action(tool="run_sql",
           query="SELECT ROUND(SUM(s.amount * cr.rate_to_usd), 2) as total_revenue_usd FROM sales s JOIN currency_rates cr ON LOWER(s.currency) = cr.currency_code",
           reasoning="Verify fix by joining with LOWER()"),
    Action(tool="submit",
           query="ROOT CAUSE: The currency codes have a case mismatch. The sales table stores currency as uppercase (USD, EUR, GBP) but currency_rates stores them as lowercase (usd, eur, gbp). The JOIN fails silently because of this case-sensitive mismatch, returning NULLs for non-USD currencies. CORRECTED SQL: SELECT ROUND(SUM(s.amount * cr.rate_to_usd), 2) as total_revenue_usd FROM sales s JOIN currency_rates cr ON LOWER(s.currency) = cr.currency_code",
           reasoning="Found the root cause: case mismatch in currency codes"),
]

for action in good_actions:
    obs, reward, done, info = env5.step(action)

test("Good agent gets reward", reward is not None)
if reward:
    test("Diagnosis scored > 0", reward.breakdown.diagnosis_correct > 0,
         f"Got: {reward.breakdown.diagnosis_correct}")
    test("Fix scored > 0", reward.breakdown.fix_valid > 0,
         f"Got: {reward.breakdown.fix_valid}")
    test("Investigation quality > 0", reward.breakdown.investigation_quality > 0,
         f"Got: {reward.breakdown.investigation_quality}")
    test("Reasoning quality > 0", reward.breakdown.reasoning_quality > 0,
         f"Got: {reward.breakdown.reasoning_quality}")
    test("Total score > 0.5", reward.score > 0.5,
         f"Got: {reward.score}")
    print(f"  📊 Full score: {reward.score:.4f}")
    print(f"     Diagnosis: {reward.breakdown.diagnosis_correct}")
    print(f"     Fix:       {reward.breakdown.fix_valid}")
    print(f"     Efficiency:{reward.breakdown.efficiency}")
    print(f"     Reasoning: {reward.breakdown.reasoning_quality}")
    print(f"     Investig:  {reward.breakdown.investigation_quality}")
    print(f"     Penalty:  -{reward.false_positive_penalty}")

# ── 6. Determinism ────────────────────────────────────────────────────────────
print("\n── 6. Determinism ──")

# Run same actions twice, should get identical scores
scores = []
for run in range(2):
    env_det = DataOnCallEnv()
    env_det.reset(task_id=1)
    for action in good_actions:
        obs, reward, done, info = env_det.step(action)
    if reward:
        scores.append(reward.score)

test("Two runs produce identical scores",
     len(scores) == 2 and scores[0] == scores[1],
     f"Run 1: {scores[0] if scores else 'N/A'}, Run 2: {scores[1] if len(scores) > 1 else 'N/A'}")

# ── 7. State endpoint ────────────────────────────────────────────────────────
print("\n── 7. State ──")

env7 = DataOnCallEnv()
env7.reset(task_id=2)
state = env7.state()
test("State includes discovered_tables", hasattr(state, "discovered_tables"))
test("State includes cost_spent", hasattr(state, "cost_spent"))
test("State includes budget_remaining", hasattr(state, "budget_remaining"))

# After discovering tables, state should reflect it
env7.step(Action(tool="list_tables", query="", reasoning="test"))
state = env7.state()
test("State shows discovered tables after list_tables",
     len(state.discovered_tables) > 0,
     f"Got: {state.discovered_tables}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print('='*60)

if failed > 0:
    sys.exit(1)
else:
    print("🎉 All tests passed!")
    sys.exit(0)
