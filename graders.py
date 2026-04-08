"""Grading logic for DataOnCallEnv.
Includes diagnosis, fix validation, efficiency, and quality scoring.
"""

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

def _extract_all_sql(actions: list) -> list[str]:
    """Extract all SQL queries the agent ran."""
    return [a.get("query", "") for a in actions if a.get("tool") == "run_sql"]

# ── Tiered Diagnosis Scoring ──────────────────────────────────────────────────

def _tiered_diagnosis_score(all_text: str, task: dict) -> float:
    """
    Tiered diagnosis scoring:
      0.25 — exact root cause identified (specific keywords)
      0.15 — correct root cause category (general keywords)
      0.08 — noticed the symptom but didn't identify cause
      0.00 — wrong or no diagnosis
    """
    tiers = task.get("diagnosis_tiers", {})

    # Check exact match first (highest tier)
    exact_keywords = tiers.get("exact", [])
    if exact_keywords and _keywords_found(all_text, exact_keywords):
        return 0.25

    # Check category match
    category_keywords = tiers.get("category", [])
    if category_keywords and _keywords_found(all_text, category_keywords):
        return 0.15

    # Check symptom recognition
    symptom_keywords = tiers.get("symptom", [])
    if symptom_keywords and _keywords_found(all_text, symptom_keywords):
        return 0.08

    # Fallback: check old-style root_cause_keywords for partial credit
    if _keywords_found(all_text, task.get("root_cause_keywords", [])):
        return 0.15

    return 0.0

# ── Efficiency Scoring (combined step + cost) ─────────────────────────────────

def _efficiency_score(steps_taken: int, optimal_steps: int,
                      cost_spent: float, optimal_cost: float) -> float:
    """
    Combined efficiency: step efficiency (max 0.075) + cost efficiency (max 0.075) = max 0.15
    Linear decay: full score at optimal, 0 at 3× optimal.
    """
    def _single_efficiency(actual, optimal, max_score):
        if actual <= optimal:
            return max_score
        max_val = optimal * 3
        if actual >= max_val:
            return 0.0
        ratio = 1.0 - (actual - optimal) / (max_val - optimal)
        return round(max_score * ratio, 4)

    step_eff = _single_efficiency(steps_taken, optimal_steps, 0.075)
    cost_eff = _single_efficiency(cost_spent, optimal_cost, 0.075)
    return round(step_eff + cost_eff, 4)

# ── Reasoning Quality ────────────────────────────────────────────────────────

def _reasoning_score(actions: list) -> float:
    """0.10 if all actions have reasoning. Prorated."""
    if not actions:
        return 0.0
    with_reasoning = sum(1 for a in actions if a.get("reasoning", "").strip())
    return round(0.10 * (with_reasoning / len(actions)), 4)

# ── Investigation Quality ────────────────────────────────────────────────────

def _investigation_quality_score(actions: list) -> float:
    """
    Rewards logical debugging methodology (max 0.10):
    - Did agent start with discovery (list_tables)? +0.02
    - Did agent inspect schema? +0.02
    - Did agent check logs or airflow? +0.02
    - Did agent form hypothesis (reasoning mentions hypothesis/cause)? +0.02
    - Did agent verify with SQL before submitting? +0.02
    """
    score = 0.0
    tools_used = [a.get("tool") for a in actions]
    all_reasoning = " ".join(a.get("reasoning", "") for a in actions if a.get("reasoning"))

    # Started with discovery
    if tools_used and tools_used[0] in ("list_tables",):
        score += 0.02

    # Inspected schema at some point
    if "inspect_schema" in tools_used:
        score += 0.02

    # Checked logs or airflow
    if "check_logs" in tools_used or "check_airflow" in tools_used:
        score += 0.02

    # Formed hypothesis (reasoning mentions investigative language)
    hypothesis_words = ["hypothesis", "suspect", "might be", "could be", "because",
                        "root cause", "investigate", "checking", "verify", "confirms"]
    if _keywords_found(all_reasoning, hypothesis_words):
        score += 0.02

    # Ran verification SQL before submitting (not just inspection queries)
    sql_queries = _extract_all_sql(actions)
    has_verification = any(
        any(kw in q.upper() for kw in ["SUM", "COUNT", "LOWER", "UPPER", "DISTINCT", "GROUP BY"])
        for q in sql_queries
    )
    if has_verification:
        score += 0.02

    return round(score, 4)

# ── False Positive Penalty ────────────────────────────────────────────────────

def _false_positive_penalty(actions: list, task: dict) -> float:
    """
    Penalize:
    - Queries on irrelevant tables: -0.04 per irrelevant table accessed
    - Duplicate identical queries: -0.03 per duplicate
    Max penalty: -0.15
    """
    penalty = 0.0
    relevant_tables = {t.lower() for t in task.get("relevant_tables", [])}

    # Track accessed tables from SQL queries
    accessed_tables = set()
    seen_queries = set()
    duplicate_count = 0

    for action in actions:
        if action.get("tool") == "run_sql":
            query = action.get("query", "")

            # Detect duplicate queries
            q_normalized = " ".join(query.upper().split())
            if q_normalized in seen_queries:
                duplicate_count += 1
            seen_queries.add(q_normalized)

            # Extract tables from query
            tokens = re.findall(r'(?:FROM|JOIN)\s+(\w+)', query, re.IGNORECASE)
            for token in tokens:
                accessed_tables.add(token.lower())

        elif action.get("tool") == "inspect_schema":
            accessed_tables.add(action.get("query", "").lower())

    # Penalize irrelevant table access
    irrelevant = accessed_tables - relevant_tables
    penalty += len(irrelevant) * 0.04

    # Penalize duplicate queries
    penalty += duplicate_count * 0.03

    return round(min(penalty, 0.15), 4)

# ── Anti-Cheat Detection ─────────────────────────────────────────────────────

def _detect_cheating(actions: list, task: dict, final_answer: str) -> float:
    """
    Detect suspicious behavior:
    - Correct diagnosis keywords with zero SQL queries run → penalty
    - Submit is the only meaningful action → penalty
    Returns additional penalty (0.0 to 0.10)
    """
    sql_queries = _extract_all_sql(actions)
    tools_used = [a.get("tool") for a in actions]

    # Check if agent submitted correct answer without any investigation
    all_text = final_answer + " " + _extract_all_text(actions)
    has_correct_keywords = _keywords_found(all_text, task.get("root_cause_keywords", []))

    # No SQL queries but correct diagnosis — suspicious
    if has_correct_keywords and len(sql_queries) == 0:
        # Did they at least check logs?
        checked_something = "check_logs" in tools_used or "check_airflow" in tools_used
        if not checked_something:
            return 0.10  # maximum suspicion penalty

    return 0.0

# ── Ground Truth Helpers ──────────────────────────────────────────────────────

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
    ignores column name differences.
    """
    if not agent_result or not ground_truth_result:
        return False
    if isinstance(agent_result, dict):  # error dict or truncated result
        if "rows" in agent_result:
            agent_result = agent_result["rows"]
        else:
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
            if any(skip in q_upper for skip in ["LIMIT 5", "LIMIT 10", "PRAGMA"]):
                continue
            # Prefer queries that look like aggregations — these are fix attempts
            if any(agg in q_upper for agg in ["SUM", "COUNT", "TOTAL", "LOWER", "UPPER"]):
                fix_queries.append(q)
    # Also include submit() query if it contains SQL
    for action in actions:
        if action.get("tool") == "submit":
            q = action.get("query", "")
            if "SELECT" in q.upper():
                fix_queries.insert(0, q)
    return fix_queries

def _also_check_submit_for_sql(actions: list) -> str:
    """Extract SQL from the submit() call if the agent embedded it there."""
    for action in reversed(actions):
        if action.get("tool") == "submit":
            q = action.get("query", "")
            match = re.search(r"SELECT.+?(?:;|$)", q, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()
    return ""

# ── Unified task grader ───────────────────────────────────────────────────────

def _grade_task(task_id: int, conn, actions: list, final_answer: str) -> Reward:
    """
    Universal grader for all tasks. Uses tiered scoring, investigation quality,
    false positive penalty, and anti-cheat detection.
    """
    task = get_task(task_id)
    all_text = final_answer + " " + _extract_all_text(actions)

    # ── 1. Diagnosis (max 0.25) — tiered ──
    diagnosis_correct = _tiered_diagnosis_score(all_text, task)

    # Task-specific partial credit fallbacks
    if diagnosis_correct == 0.0:
        if task_id == 1:
            sql_text = " ".join(a.get("query", "") for a in actions if a.get("tool") == "run_sql")
            if _keywords_found(sql_text, ["lower(", "upper(", "collate"]):
                diagnosis_correct = 0.12  # found it implicitly in SQL
        elif task_id == 2:
            checked_logs = any(
                a.get("tool") in ("check_logs",) or "dbt_log" in a.get("query", "").lower()
                for a in actions
            )
            if checked_logs and _keywords_found(all_text, ["log", "migration", "pipeline", "changed"]):
                diagnosis_correct = 0.10
        elif task_id == 3:
            queried_promos = any("product_promotions" in a.get("query", "").lower() for a in actions)
            noticed_inflation = _keywords_found(all_text, ["3.", "4.", "multiply", "inflat", "overstated"])
            if queried_promos and noticed_inflation:
                diagnosis_correct = 0.12

    # ── 2. Fix Valid (max 0.25) ──
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

    # Task 2 partial fix credit
    if fix_valid == 0.0 and task_id == 2:
        if _keywords_found(all_text, ["distinct", "deduplicate", "dedup", "tz_source = 'utc'", "where tz"]):
            fix_valid = 0.10

    # ── 3. Efficiency (max 0.15) — combined step + cost ──
    # Calculate cost from actions
    from environment import DataOnCallEnv
    cost_spent = sum(DataOnCallEnv.TOOL_COSTS.get(a.get("tool", ""), 0.0) for a in actions)
    efficiency = _efficiency_score(
        len(actions), task["optimal_steps"],
        cost_spent, task["optimal_cost"],
    )

    # ── 4. Reasoning Quality (max 0.10) ──
    reasoning_quality = _reasoning_score(actions)

    # ── 5. Investigation Quality (max 0.10) ──
    investigation_quality = _investigation_quality_score(actions)

    # ── 6. Penalties ──
    fp_penalty = _false_positive_penalty(actions, task)
    cheat_penalty = _detect_cheating(actions, task, final_answer)
    total_penalty = round(min(fp_penalty + cheat_penalty, 0.15), 4)

    # ── Final score ──
    raw_total = diagnosis_correct + fix_valid + efficiency + reasoning_quality + investigation_quality
    total = raw_total - total_penalty

    return Reward(
        score=total,
        breakdown=RewardBreakdown(
            diagnosis_correct=diagnosis_correct,
            fix_valid=fix_valid,
            efficiency=efficiency,
            reasoning_quality=reasoning_quality,
            investigation_quality=investigation_quality,
        ),
        false_positive_penalty=total_penalty,
    )

# ── Router ────────────────────────────────────────────────────────────────────

def grade(task_id: int, conn, actions: list, final_answer: str) -> Reward:
    if task_id not in (1, 2, 3):
        raise ValueError(f"No grader for task_id={task_id}")
    return _grade_task(task_id, conn, actions, final_answer)