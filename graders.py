# graders.py
# Deterministic scoring functions. Given what the agent did, return a Reward.
# These must be deterministic — same inputs always produce same score.
# The hackathon validators will run these and check scores are in 0.0–1.0.

from models import Reward, RewardBreakdown
from tasks import get_task
from database import run_sql

def _check_keywords(text: str, keywords: list) -> bool:
    """Returns True if any keyword appears in the agent's text (case-insensitive)."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

def _efficiency_score(steps_taken: int, optimal_steps: int) -> float:
    """
    Score drops as agent uses more steps than optimal.
    At optimal steps: 0.20. Double the steps: 0.10. Triple: 0.0.
    """
    if steps_taken <= optimal_steps:
        return 0.20
    ratio = optimal_steps / steps_taken
    return round(max(0.0, 0.20 * ratio), 4)

def _reasoning_score(actions: list) -> float:
    """
    Rewards agents that explain their thinking in the 'reasoning' field.
    0.15 if most actions include reasoning. 0.0 if none do.
    """
    if not actions:
        return 0.0
    with_reasoning = sum(1 for a in actions if a.get("reasoning"))
    ratio = with_reasoning / len(actions)
    return round(0.15 * ratio, 4)

def _false_positive_check(actions: list, safe_tables: list) -> float:
    """
    Penalizes agent if it ran queries touching tables unrelated to the bug.
    -0.20 if agent modified or queried tables it had no reason to touch.
    We check by looking at SQL queries in actions for unexpected table names.
    """
    penalty = 0.0
    all_queries = " ".join(
        a.get("query", "") for a in actions
        if a.get("tool") == "run_sql"
    ).lower()
    # If agent queried a "fix" that touched completely unrelated tables, penalize
    # (light check — we just look for DROP/DELETE which run_sql blocks anyway)
    if "drop" in all_queries or "delete" in all_queries:
        penalty = 0.20
    return penalty

# ── Task 1 grader ─────────────────────────────────────────────────────────────

def grade_task1(conn, actions: list, final_answer: str) -> Reward:
    """
    Scores the NULL JOIN task.
    diagnosis_correct: did agent identify the currency code case mismatch?
    fix_valid:         does agent's proposed fix query return ~$2851?
    efficiency:        how many steps vs optimal 4?
    reasoning_quality: did agent explain each tool call?
    """
    task = get_task(1)

    # 1. Diagnosis — did they name the root cause?
    diagnosis_correct = 0.30 if _check_keywords(
        final_answer, task["root_cause_keywords"]
    ) else 0.0

    # 2. Fix validity — find the last run_sql in actions and check its output
    fix_valid = 0.0
    # Look for any SQL that uses LOWER() or UPPER() to fix the join
    for action in reversed(actions):
        if action.get("tool") == "run_sql":
            q = action.get("query", "").upper()
            if "LOWER" in q or "UPPER" in q or "COLLATE" in q:
                # Run their query against the real db and check result
                result = run_sql(conn, action["query"])
                if isinstance(result, list) and len(result) > 0:
                    # Check if result is close to the correct total
                    try:
                        val = list(result[0].values())[0]
                        if val and abs(float(val) - task["expected_total_revenue"]) < 10:
                            fix_valid = 0.25
                    except (ValueError, TypeError):
                        pass
                break

    efficiency = _efficiency_score(len(actions), task["optimal_steps"])
    reasoning_quality = _reasoning_score(actions)
    false_positive_penalty = _false_positive_check(actions, ["sales", "currency_rates"])

    total = diagnosis_correct + fix_valid + efficiency + reasoning_quality - false_positive_penalty

    return Reward(
        score=total,
        breakdown=RewardBreakdown(
            diagnosis_correct=diagnosis_correct,
            fix_valid=fix_valid,
            efficiency=efficiency,
            reasoning_quality=reasoning_quality,
        ),
        false_positive_penalty=false_positive_penalty,
    )

# ── Task 2 grader ─────────────────────────────────────────────────────────────

def grade_task2(conn, actions: list, final_answer: str) -> Reward:
    """
    Scores the TIMEZONE GHOST task.
    diagnosis_correct: did agent find the timezone migration in dbt_log?
    fix_valid:         did agent mention deduplication or UTC-only filtering?
    """
    task = get_task(2)

    diagnosis_correct = 0.30 if _check_keywords(
        final_answer, task["root_cause_keywords"]
    ) else 0.0

    # Partial credit: did they at least find the dbt_log smoking gun?
    if diagnosis_correct == 0.0:
        # Check if agent queried dbt_log at all
        queried_logs = any(
            "dbt_log" in a.get("query", "").lower()
            for a in actions if a.get("tool") in ("run_sql", "check_logs")
        )
        if queried_logs and _check_keywords(final_answer, ["log", "migration", "pipeline"]):
            diagnosis_correct = 0.10  # partial credit — found the log but missed the insight

    # Fix validity — did they suggest filtering or deduplication?
    fix_keywords = ["distinct", "utc", "deduplicate", "dedup", "where tz_source", "filter"]
    fix_valid = 0.25 if _check_keywords(final_answer, fix_keywords) else 0.0

    efficiency = _efficiency_score(len(actions), task["optimal_steps"])
    reasoning_quality = _reasoning_score(actions)
    false_positive_penalty = _false_positive_check(actions, ["user_events", "dbt_log"])

    total = diagnosis_correct + fix_valid + efficiency + reasoning_quality - false_positive_penalty

    return Reward(
        score=total,
        breakdown=RewardBreakdown(
            diagnosis_correct=diagnosis_correct,
            fix_valid=fix_valid,
            efficiency=efficiency,
            reasoning_quality=reasoning_quality,
        ),
        false_positive_penalty=false_positive_penalty,
    )

# ── Task 3 grader ─────────────────────────────────────────────────────────────

def grade_task3(conn, actions: list, final_answer: str) -> Reward:
    """
    Scores the FANOUT bug task.
    This is hard — agent must connect 3.7x inflation to non-unique JOIN key.
    Partial credit for noticing the precise ratio or querying product_promotions.
    """
    task = get_task(3)

    diagnosis_correct = 0.30 if _check_keywords(
        final_answer, task["root_cause_keywords"]
    ) else 0.0

    # Partial credit if agent noticed the fanout ratio or queried promotions table
    if diagnosis_correct == 0.0:
        noticed_ratio = _check_keywords(final_answer, ["3.7", "multiplied", "inflated", "ratio"])
        queried_promotions = any(
            "product_promotions" in a.get("query", "").lower()
            for a in actions
        )
        if noticed_ratio and queried_promotions:
            diagnosis_correct = 0.15  # close but didn't name it fully

    # Fix validity — run their corrected query if present
    fix_valid = 0.0
    for action in reversed(actions):
        if action.get("tool") == "run_sql":
            q = action.get("query", "")
            # Look for a query that avoids joining product_promotions
            # or uses a subquery/distinct to handle fanout
            if ("product_promotions" not in q.lower() or
                    "distinct" in q.lower() or
                    "group by" in q.lower()):
                result = run_sql(conn, q)
                if isinstance(result, list) and len(result) > 0:
                    try:
                        val = list(result[0].values())[0]
                        if val and abs(float(val) - task["expected_true_revenue"]) < 50:
                            fix_valid = 0.25
                            break
                    except (ValueError, TypeError):
                        pass

    efficiency = _efficiency_score(len(actions), task["optimal_steps"])
    reasoning_quality = _reasoning_score(actions)
    false_positive_penalty = _false_positive_check(actions, ["sales", "product_promotions", "products"])

    total = diagnosis_correct + fix_valid + efficiency + reasoning_quality - false_positive_penalty

    return Reward(
        score=total,
        breakdown=RewardBreakdown(
            diagnosis_correct=diagnosis_correct,
            fix_valid=fix_valid,
            efficiency=efficiency,
            reasoning_quality=reasoning_quality,
        ),
        false_positive_penalty=false_positive_penalty,
    )

# ── Router ─────────────────────────────────────────────────────────────────────

def grade(task_id: int, conn, actions: list, final_answer: str) -> Reward:
    """Call the right grader for the given task."""
    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_id not in graders:
        raise ValueError(f"No grader for task_id={task_id}")
    return graders[task_id](conn, actions, final_answer)