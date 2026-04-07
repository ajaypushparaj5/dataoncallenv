# graders.py
# Production-grade graders. Key design decisions::
#
# 1. fix_valid: runs agent's SQL against the real db and compares result SET
#    to ground truth query result. No magic numbers. Works for any correct query.
#
# 2. diagnosis_correct: checks submit() answer text for root cause keywords.
#    Also scans all reasoning fields — model gets credit for showing understanding
#    even if its submit() wording is slightly different.
#
# 3. efficiency: based on steps taken vs optimal. Graceful degradation.
#
# 4. Grader is called BOTH when submit() is called AND when steps run out.
#    In both cases it scans all SQL the agent ran to find the best fix attempt.

import re
from models import Reward, RewardBreakdown
from tasks import get_task
from database import run_sql

# ── Helpers ───────────────────────────────────────────────────────────────────

def _keywords_found(text: str, keywords: list) -> bool:
    """Case-insensitive keyword search across a text string."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

def _extract_all_text(actions: list) -> str:
    """Pull all text from actions — query, reasoning, and tool names — into one string."""
    parts = []
    for a in actions:
        if a.get("reasoning"):
            parts.append(a["reasoning"])
        if a.get("query"):
            parts.append(a["query"])
    return " ".join(parts)

def _efficiency_score(steps_taken: int, optimal_steps: int) -> float:
    """
    0.20 at or under optimal. Decays linearly. 0.0 at 3x optimal.
    """
    if steps_taken <= optimal_steps:
        return 0.20
    # Linear decay between optimal and 3x optimal
    max_steps = optimal_steps * 3
    if steps_taken >= max_steps:
        return 0.0
    ratio = 1.0 - (steps_taken - optimal_steps) / (max_steps - optimal_steps)
    return round(0.20 * ratio, 4)

def _reasoning_score(actions: list) -> float:
    """0.15 if all actions have reasoning. Prorated."""
    if not actions:
        return 0.0
    with_reasoning = sum(1 for a in actions if a.get("reasoning", "").strip())
    return round(0.15 * (with_reasoning / len(actions)), 4)

def _run_ground_truth(conn, ground_truth_query: str):
    """Run the canonical correct query and return its result."""
    result = run_sql(conn, ground_truth_query.strip())
    if isinstance(result, dict) and "error" in result:
        return None
    return result

def _results_match(agent_result, ground_truth_result, tolerance=0.05) -> bool:
    """
    Compare agent's query result to ground truth result.
    Handles: same row count, same numeric values within tolerance,
    ignores column name differences (agent may name column differently).
    """
    if not agent_result or not ground_truth_result:
        return False
    if isinstance(agent_result, dict):  # error dict
        return False
    if not isinstance(agent_result, list) or not isinstance(ground_truth_result, list):
        return False
    if len(agent_result) != len(ground_truth_result):
        return False

    for agent_row, truth_row in zip(agent_result, ground_truth_result):
        agent_vals = list(agent_row.values())
        truth_vals = list(truth_row.values())
        if len(agent_vals) != len(truth_vals):
            # different column count — try comparing just numerics
            agent_nums = [v for v in agent_vals if isinstance(v, (int, float)) and v is not None]
            truth_nums = [v for v in truth_vals if isinstance(v, (int, float)) and v is not None]
            if not agent_nums or not truth_nums:
                return False
            # Check if the key numeric value is close enough
            if abs(agent_nums[0] - truth_nums[0]) / max(abs(truth_nums[0]), 1) > tolerance:
                return False
        else:
            for av, tv in zip(agent_vals, truth_vals):
                if isinstance(tv, (int, float)) and isinstance(av, (int, float)):
                    if tv == 0:
                        if av != 0:
                            return False
                    elif abs(av - tv) / abs(tv) > tolerance:
                        return False
    return True

def _find_best_fix_sql(actions: list) -> list[str]:
    """
    Extract all SELECT queries the agent ran that look like fix attempts.
    Returns them newest-first so we try the most recent one first.
    """
    fix_queries = []
    for action in reversed(actions):
        if action.get("tool") == "run_sql":
            q = action.get("query", "")
            q_upper = q.upper()
            # Skip trivial inspection queries
            if any(skip in q_upper for skip in ["LIMIT 5", "LIMIT 10", "SELECT *", "PRAGMA"]):
                continue
            # Prefer queries that look like aggregations (SUM, COUNT) — these are fix attempts
            if any(agg in q_upper for agg in ["SUM", "COUNT", "TOTAL", "LOWER", "UPPER"]):
                fix_queries.append(q)
    # Also include submit() query if it contains SQL
    for action in actions:
        if action.get("tool") == "submit":
            q = action.get("query", "")
            if "SELECT" in q.upper():
                fix_queries.insert(0, q)  # highest priority
    return fix_queries

def _also_check_submit_for_sql(actions: list) -> str:
    """Extract SQL from the submit() call if the agent embedded it there."""
    for action in reversed(actions):
        if action.get("tool") == "submit":
            q = action.get("query", "")
            # Find SQL block in the submit text
            match = re.search(r"SELECT.+?(?:;|$)", q, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()
    return ""

# ── Task graders ──────────────────────────────────────────────────────────────

def grade_task1(conn, actions: list, final_answer: str) -> Reward:
    task = get_task(1)

    # --- diagnosis_correct ---
    # Check submit() answer + ALL reasoning text
    all_text = final_answer + " " + _extract_all_text(actions)
    diagnosis_correct = 0.30 if _keywords_found(all_text, task["root_cause_keywords"]) else 0.0

    # Partial credit: if agent ran LOWER/UPPER but didn't name it in final answer
    if diagnosis_correct == 0.0:
        sql_text = " ".join(a.get("query","") for a in actions if a.get("tool")=="run_sql")
        if _keywords_found(sql_text, ["lower(", "upper(", "collate"]):
            diagnosis_correct = 0.15  # found it implicitly

    # --- fix_valid ---
    # Run ground truth, then try agent's fix queries
    fix_valid = 0.0
    ground_truth_result = _run_ground_truth(conn, task["ground_truth_query"])

    fix_queries = _find_best_fix_sql(actions)
    # Also check if submit contained embedded SQL
    submit_sql = _also_check_submit_for_sql(actions)
    if submit_sql:
        fix_queries.insert(0, submit_sql)

    for q in fix_queries:
        agent_result = run_sql(conn, q)
        if _results_match(agent_result, ground_truth_result):
            fix_valid = 0.25
            break

    efficiency       = _efficiency_score(len(actions), task["optimal_steps"])
    reasoning_quality = _reasoning_score(actions)

    total = diagnosis_correct + fix_valid + efficiency + reasoning_quality
    return Reward(
        score=total,
        breakdown=RewardBreakdown(
            diagnosis_correct=diagnosis_correct,
            fix_valid=fix_valid,
            efficiency=efficiency,
            reasoning_quality=reasoning_quality,
        ),
    )

def grade_task2(conn, actions: list, final_answer: str) -> Reward:
    task = get_task(2)

    all_text = final_answer + " " + _extract_all_text(actions)
    diagnosis_correct = 0.30 if _keywords_found(all_text, task["root_cause_keywords"]) else 0.0

    # Partial credit: did they check dbt_log and find the migration?
    if diagnosis_correct == 0.0:
        checked_logs = any(
            a.get("tool") in ("check_logs",) or "dbt_log" in a.get("query","").lower()
            for a in actions
        )
        if checked_logs and _keywords_found(all_text, ["log", "migration", "pipeline", "changed"]):
            diagnosis_correct = 0.12

    fix_valid = 0.0
    ground_truth_result = _run_ground_truth(conn, task["ground_truth_query"])
    fix_queries = _find_best_fix_sql(actions)
    submit_sql = _also_check_submit_for_sql(actions)
    if submit_sql:
        fix_queries.insert(0, submit_sql)

    for q in fix_queries:
        agent_result = run_sql(conn, q)
        if _results_match(agent_result, ground_truth_result):
            fix_valid = 0.25
            break

    # Partial fix credit: did they suggest deduplication at all?
    if fix_valid == 0.0:
        if _keywords_found(all_text, ["distinct", "deduplicate", "dedup", "tz_source = 'utc'", "where tz"]):
            fix_valid = 0.12

    efficiency        = _efficiency_score(len(actions), task["optimal_steps"])
    reasoning_quality = _reasoning_score(actions)

    total = diagnosis_correct + fix_valid + efficiency + reasoning_quality
    return Reward(
        score=total,
        breakdown=RewardBreakdown(
            diagnosis_correct=diagnosis_correct,
            fix_valid=fix_valid,
            efficiency=efficiency,
            reasoning_quality=reasoning_quality,
        ),
    )

def grade_task3(conn, actions: list, final_answer: str) -> Reward:
    task = get_task(3)

    all_text = final_answer + " " + _extract_all_text(actions)
    diagnosis_correct = 0.30 if _keywords_found(all_text, task["root_cause_keywords"]) else 0.0

    # Partial credit: noticed inflation + queried product_promotions
    if diagnosis_correct == 0.0:
        queried_promos = any("product_promotions" in a.get("query","").lower() for a in actions)
        noticed_inflation = _keywords_found(all_text, ["3.", "4.", "multiply", "inflat", "overstated"])
        if queried_promos and noticed_inflation:
            diagnosis_correct = 0.15

    fix_valid = 0.0
    ground_truth_result = _run_ground_truth(conn, task["ground_truth_query"])
    fix_queries = _find_best_fix_sql(actions)
    submit_sql = _also_check_submit_for_sql(actions)
    if submit_sql:
        fix_queries.insert(0, submit_sql)

    for q in fix_queries:
        agent_result = run_sql(conn, q)
        if _results_match(agent_result, ground_truth_result):
            fix_valid = 0.25
            break

    efficiency        = _efficiency_score(len(actions), task["optimal_steps"])
    reasoning_quality = _reasoning_score(actions)

    total = diagnosis_correct + fix_valid + efficiency + reasoning_quality
    return Reward(
        score=total,
        breakdown=RewardBreakdown(
            diagnosis_correct=diagnosis_correct,
            fix_valid=fix_valid,
            efficiency=efficiency,
            reasoning_quality=reasoning_quality,
        ),
    )

# ── Router ────────────────────────────────────────────────────────────────────

def grade(task_id: int, conn, actions: list, final_answer: str) -> Reward:
    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_id not in graders:
        raise ValueError(f"No grader for task_id={task_id}")
    return graders[task_id](conn, actions, final_answer)