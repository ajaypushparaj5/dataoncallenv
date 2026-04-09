"""Grading logic for DataOnCallEnv.
Includes diagnosis, fix validation, efficiency, and quality scoring.

DEFENSIVE: grade() and _grade_task() never raise — all errors return score=0.0.
"""

import re
import logging
import traceback
from models import Reward, RewardBreakdown
from tasks import get_task
from database import run_sql

logger = logging.getLogger(__name__)

# ── Safe zero reward ──────────────────────────────────────────────────────────

def _zero_reward(msg: str = "grading error") -> Reward:
    """Return a safe 0.0 Reward. Never raises."""
    return Reward(
        score=0.0,
        breakdown=RewardBreakdown(
            diagnosis_correct=0.0,
            fix_valid=0.0,
            efficiency=0.0,
            reasoning_quality=0.0,
            investigation_quality=0.0,
        ),
        false_positive_penalty=0.0,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def _keywords_found(text: str, keywords: list) -> bool:
    """Case-insensitive keyword search across a text string."""
    if not text:
        return False
    try:
        text_lower = str(text).lower()
        return any(str(kw).lower() in text_lower for kw in (keywords or []))
    except Exception:
        return False

def _extract_all_text(actions: list) -> str:
    """Pull all text from actions — query, reasoning, and tool names — into one string."""
    try:
        parts = []
        for a in (actions or []):
            if not isinstance(a, dict):
                continue
            if a.get("reasoning"):
                parts.append(str(a["reasoning"]))
            if a.get("query"):
                parts.append(str(a["query"]))
        return " ".join(parts)
    except Exception:
        return ""

def _extract_all_sql(actions: list) -> list:
    """Extract all SQL queries the agent ran."""
    try:
        return [
            str(a.get("query", ""))
            for a in (actions or [])
            if isinstance(a, dict) and a.get("tool") == "run_sql"
        ]
    except Exception:
        return []

# ── Tiered Diagnosis Scoring ──────────────────────────────────────────────────

def _tiered_diagnosis_score(all_text: str, task: dict) -> float:
    """
    Tiered diagnosis scoring:
      0.25 — exact root cause identified (specific keywords)
      0.15 — correct root cause category (general keywords)
      0.08 — noticed the symptom but didn't identify cause
      0.00 — wrong or no diagnosis
    """
    try:
        tiers = task.get("diagnosis_tiers", {})

        if _keywords_found(all_text, tiers.get("exact", [])):
            return 0.25
        if _keywords_found(all_text, tiers.get("category", [])):
            return 0.15
        if _keywords_found(all_text, tiers.get("symptom", [])):
            return 0.08

        # Fallback: check old-style root_cause_keywords for partial credit
        if _keywords_found(all_text, task.get("root_cause_keywords", [])):
            return 0.15
    except Exception:
        pass
    return 0.0

# ── Efficiency Scoring (combined step + cost) ─────────────────────────────────

def _efficiency_score(steps_taken: int, optimal_steps: int,
                      cost_spent: float, optimal_cost: float) -> float:
    """
    Combined efficiency: step efficiency (max 0.075) + cost efficiency (max 0.075) = max 0.15
    Linear decay: full score at optimal, 0 at 3× optimal.
    """
    try:
        def _single_efficiency(actual, optimal, max_score):
            try:
                actual  = float(actual)
                optimal = float(optimal)
            except (TypeError, ValueError):
                return 0.0
            if optimal <= 0:
                return 0.0
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
    except Exception:
        return 0.0

# ── Reasoning Quality ────────────────────────────────────────────────────────

def _reasoning_score(actions: list) -> float:
    """0.10 if all actions have reasoning. Prorated."""
    try:
        if not actions:
            return 0.0
        with_reasoning = sum(
            1 for a in actions
            if isinstance(a, dict) and a.get("reasoning", "").strip()
        )
        return round(0.10 * (with_reasoning / len(actions)), 4)
    except Exception:
        return 0.0

# ── Investigation Quality ────────────────────────────────────────────────────

def _investigation_quality_score(actions: list) -> float:
    """
    Rewards logical debugging methodology (max 0.10).
    """
    try:
        score = 0.0
        tools_used = [a.get("tool") for a in (actions or []) if isinstance(a, dict)]
        all_reasoning = " ".join(
            str(a.get("reasoning", "")) for a in (actions or [])
            if isinstance(a, dict) and a.get("reasoning")
        )

        if tools_used and tools_used[0] in ("list_tables",):
            score += 0.02
        if "inspect_schema" in tools_used:
            score += 0.02
        if "check_logs" in tools_used or "check_airflow" in tools_used:
            score += 0.02

        hypothesis_words = [
            "hypothesis", "suspect", "might be", "could be", "because",
            "root cause", "investigate", "checking", "verify", "confirms",
        ]
        if _keywords_found(all_reasoning, hypothesis_words):
            score += 0.02

        sql_queries = _extract_all_sql(actions)
        has_verification = any(
            any(kw in q.upper() for kw in ["SUM", "COUNT", "LOWER", "UPPER", "DISTINCT", "GROUP BY"])
            for q in sql_queries if q
        )
        if has_verification:
            score += 0.02

        return round(score, 4)
    except Exception:
        return 0.0

# ── False Positive Penalty ────────────────────────────────────────────────────

def _false_positive_penalty(actions: list, task: dict) -> float:
    """
    Penalize:
    - Queries on irrelevant tables: -0.04 per irrelevant table accessed
    - Duplicate identical queries: -0.03 per duplicate
    Max penalty: -0.15
    """
    try:
        penalty = 0.0
        relevant_tables = {t.lower() for t in task.get("relevant_tables", [])}

        accessed_tables = set()
        seen_queries    = set()
        duplicate_count = 0

        for action in (actions or []):
            if not isinstance(action, dict):
                continue
            if action.get("tool") == "run_sql":
                query = str(action.get("query", ""))

                q_normalized = " ".join(query.upper().split())
                if q_normalized in seen_queries:
                    duplicate_count += 1
                seen_queries.add(q_normalized)

                tokens = re.findall(r'(?:FROM|JOIN)\s+(\w+)', query, re.IGNORECASE)
                for token in tokens:
                    accessed_tables.add(token.lower())

            elif action.get("tool") == "inspect_schema":
                tbl = action.get("query", "")
                if tbl:
                    accessed_tables.add(str(tbl).lower())

        irrelevant  = accessed_tables - relevant_tables
        penalty    += len(irrelevant) * 0.04
        penalty    += duplicate_count * 0.03

        return round(min(penalty, 0.15), 4)
    except Exception:
        return 0.0

# ── Anti-Cheat Detection ─────────────────────────────────────────────────────

def _detect_cheating(actions: list, task: dict, final_answer: str) -> float:
    """
    Detect suspicious behavior. Returns additional penalty (0.0 to 0.10).
    """
    try:
        sql_queries = _extract_all_sql(actions)
        tools_used  = [a.get("tool") for a in (actions or []) if isinstance(a, dict)]

        all_text = str(final_answer or "") + " " + _extract_all_text(actions)
        has_correct_keywords = _keywords_found(all_text, task.get("root_cause_keywords", []))

        if has_correct_keywords and len(sql_queries) == 0:
            checked_something = "check_logs" in tools_used or "check_airflow" in tools_used
            if not checked_something:
                return 0.10
    except Exception:
        pass
    return 0.0

# ── Ground Truth Helpers ──────────────────────────────────────────────────────

def _run_ground_truth(conn, ground_truth_query: str):
    """Run the canonical correct query and return its result."""
    try:
        if not ground_truth_query or not ground_truth_query.strip():
            return None
        result = run_sql(conn, ground_truth_query.strip())
        if isinstance(result, dict) and "error" in result:
            return None
        return result
    except Exception:
        return None

def _results_match(agent_result, ground_truth_result, tolerance=0.05) -> bool:
    """
    Compare agent's query result to ground truth result.
    Handles: same row count, same numeric values within tolerance,
    ignores column name differences.
    """
    try:
        if not agent_result or not ground_truth_result:
            return False
        if isinstance(agent_result, dict):
            if "rows" in agent_result:
                agent_result = agent_result["rows"]
            else:
                return False
        if not isinstance(agent_result, list) or not isinstance(ground_truth_result, list):
            return False
        if len(agent_result) != len(ground_truth_result):
            return False

        for agent_row, truth_row in zip(agent_result, ground_truth_result):
            if not isinstance(agent_row, dict) or not isinstance(truth_row, dict):
                continue
            agent_vals = list(agent_row.values())
            truth_vals = list(truth_row.values())
            if len(agent_vals) != len(truth_vals):
                agent_nums = [v for v in agent_vals if isinstance(v, (int, float)) and v is not None]
                truth_nums = [v for v in truth_vals if isinstance(v, (int, float)) and v is not None]
                if not agent_nums or not truth_nums:
                    return False
                denom = max(abs(truth_nums[0]), 1)
                if abs(agent_nums[0] - truth_nums[0]) / denom > tolerance:
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
    except Exception:
        return False

def _find_best_fix_sql(actions: list) -> list:
    """
    Extract all SELECT queries the agent ran that look like fix attempts.
    Returns them newest-first.
    """
    try:
        fix_queries = []
        for action in reversed(actions or []):
            if not isinstance(action, dict):
                continue
            if action.get("tool") == "run_sql":
                q = str(action.get("query", ""))
                q_upper = q.upper()
                if any(skip in q_upper for skip in ["LIMIT 5", "LIMIT 10", "PRAGMA"]):
                    continue
                if any(agg in q_upper for agg in ["SUM", "COUNT", "TOTAL", "LOWER", "UPPER"]):
                    fix_queries.append(q)
        for action in (actions or []):
            if not isinstance(action, dict):
                continue
            if action.get("tool") == "submit":
                q = str(action.get("query", ""))
                if "SELECT" in q.upper():
                    fix_queries.insert(0, q)
        return fix_queries
    except Exception:
        return []

def _also_check_submit_for_sql(actions: list) -> str:
    """Extract SQL from the submit() call if the agent embedded it there."""
    try:
        for action in reversed(actions or []):
            if not isinstance(action, dict):
                continue
            if action.get("tool") == "submit":
                q = str(action.get("query", ""))
                match = re.search(r"SELECT.+?(?:;|$)", q, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(0).strip()
    except Exception:
        pass
    return ""

# ── Cost calculation (no import from environment to avoid circular imports) ───

# Tool cost map — mirrored here to avoid importing DataOnCallEnv
_TOOL_COSTS = {
    "list_tables":    0.5,
    "inspect_schema": 1.0,
    "check_logs":     1.0,
    "check_airflow":  1.0,
    "run_sql":        2.0,
    "diff_report":    1.5,
    "submit":         0.0,
}

def _calculate_cost(actions: list) -> float:
    """Sum tool costs from the action history."""
    try:
        return sum(_TOOL_COSTS.get(str(a.get("tool", "")), 0.0) for a in (actions or []) if isinstance(a, dict))
    except Exception:
        return 0.0

# ── Unified task grader ───────────────────────────────────────────────────────

def _grade_task(task_id: int, conn, actions: list, final_answer: str) -> Reward:
    """
    Universal grader for all tasks. Uses tiered scoring, investigation quality,
    false positive penalty, and anti-cheat detection.

    DEFENSIVE: wraps all logic in try/except — never raises.
    """
    try:
        # Normalise inputs
        actions      = actions if isinstance(actions, list) else []
        final_answer = str(final_answer) if final_answer is not None else ""

        task     = get_task(task_id)
        all_text = final_answer + " " + _extract_all_text(actions)

        # ── 1. Diagnosis (max 0.25) — tiered ──
        diagnosis_correct = _tiered_diagnosis_score(all_text, task)

        # Task-specific partial credit fallbacks
        if diagnosis_correct == 0.0:
            try:
                if task_id == 1:
                    sql_text = " ".join(
                        str(a.get("query", "")) for a in actions if isinstance(a, dict) and a.get("tool") == "run_sql"
                    )
                    if _keywords_found(sql_text, ["lower(", "upper(", "collate"]):
                        diagnosis_correct = 0.12
                elif task_id == 2:
                    checked_logs = any(
                        a.get("tool") in ("check_logs",) or "dbt_log" in str(a.get("query", "")).lower()
                        for a in actions if isinstance(a, dict)
                    )
                    if checked_logs and _keywords_found(all_text, ["log", "migration", "pipeline", "changed"]):
                        diagnosis_correct = 0.10
                elif task_id == 3:
                    queried_promos = any(
                        "product_promotions" in str(a.get("query", "")).lower()
                        for a in actions if isinstance(a, dict)
                    )
                    noticed_inflation = _keywords_found(all_text, ["3.", "4.", "multiply", "inflat", "overstated"])
                    if queried_promos and noticed_inflation:
                        diagnosis_correct = 0.12
            except Exception:
                pass

        # ── 2. Fix Valid (max 0.25) ──
        fix_valid = 0.0
        try:
            ground_truth_result = _run_ground_truth(conn, task.get("ground_truth_query", ""))

            fix_queries  = _find_best_fix_sql(actions)
            submit_sql   = _also_check_submit_for_sql(actions)
            if submit_sql:
                fix_queries.insert(0, submit_sql)

            for q in fix_queries:
                try:
                    agent_result = run_sql(conn, q)
                    if _results_match(agent_result, ground_truth_result):
                        fix_valid = 0.25
                        break
                except Exception:
                    continue

            # Task 2 partial fix credit
            if fix_valid == 0.0 and task_id == 2:
                if _keywords_found(all_text, ["distinct", "deduplicate", "dedup", "tz_source = 'utc'", "where tz"]):
                    fix_valid = 0.10
        except Exception as exc:
            logger.error("fix_valid scoring raised: %s", exc)
            fix_valid = 0.0

        # ── 3. Efficiency (max 0.15) — combined step + cost ──
        try:
            cost_spent = _calculate_cost(actions)
            efficiency = _efficiency_score(
                len(actions), task.get("optimal_steps", 10),
                cost_spent, task.get("optimal_cost", 15.0),
            )
        except Exception:
            efficiency = 0.0

        # ── 4. Reasoning Quality (max 0.10) ──
        reasoning_quality = _reasoning_score(actions)

        # ── 5. Investigation Quality (max 0.10) ──
        investigation_quality = _investigation_quality_score(actions)

        # ── 6. Penalties ──
        try:
            fp_penalty    = _false_positive_penalty(actions, task)
            cheat_penalty = _detect_cheating(actions, task, final_answer)
            total_penalty = round(min(fp_penalty + cheat_penalty, 0.15), 4)
        except Exception:
            total_penalty = 0.0

        # ── Final score ──
        raw_total = (
            diagnosis_correct
            + fix_valid
            + efficiency
            + reasoning_quality
            + investigation_quality
        )
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

    except Exception as exc:
        logger.error(
            "_grade_task(task_id=%s) raised: %s\n%s",
            task_id, exc, traceback.format_exc(),
        )
        return _zero_reward(f"grading failed: {exc}")


# ── Router ────────────────────────────────────────────────────────────────────

def grade(task_id: int, conn, actions: list, final_answer: str) -> Reward:
    """
    Public grading entry point.
    DEFENSIVE: never raises, never returns None.
    """
    try:
        task_id_int = int(task_id)
    except (TypeError, ValueError):
        logger.error("grade() called with invalid task_id=%s", task_id)
        return _zero_reward(f"invalid task_id: {task_id}")

    if task_id_int not in (1, 2, 3):
        logger.error("grade() called with out-of-range task_id=%s", task_id_int)
        return _zero_reward(f"no grader for task_id={task_id_int}")

    try:
        result = _grade_task(task_id_int, conn, actions, final_answer)
        if result is None:
            return _zero_reward("_grade_task returned None")
        return result
    except Exception as exc:
        logger.error("grade() raised: %s\n%s", exc, traceback.format_exc())
        return _zero_reward(str(exc))