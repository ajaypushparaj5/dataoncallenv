"""Main RL Environment implementation for DataOnCallEnv.
"""

from models import Action, Observation, Reward, EnvState
from tasks import get_task
from graders import grade
from database import (
    build_task1_db, build_task2_db, build_task3_db,
    run_sql, inspect_schema, check_logs, check_airflow,
    diff_report, list_tables
)
import re

class DataOnCallEnv:

    MAX_STEPS = 15


    # Configuration
    TOOL_COSTS = {
        "list_tables":     0.5,
        "inspect_schema":  1.0,
        "check_logs":      1.0,
        "check_airflow":   1.0,
        "run_sql":         2.0,
        "diff_report":     1.5,
        "submit":          0.0,
    }
    COST_BUDGET = 20.0

    VALID_TOOLS = set(TOOL_COSTS.keys())

    # Minimum tool calls before submit() is allowed (anti-cheat)
    MIN_STEPS_BEFORE_SUBMIT = 2

    def __init__(self):
        self.task_id          = None
        self.conn             = None
        self.steps_taken      = 0
        self.done             = False
        self.actions          = []
        self.last_observation = None
        self.final_answer     = ""
        self.discovered_tables = set()  # partial observability tracking
        self.cost_spent       = 0.0

    # ── reset() ───────────────────────────────────────────────────────────────

    def reset(self, task_id: int = 1) -> Observation:
        """Fresh episode. Rebuilds database. Returns scenario as first observation."""
        if task_id not in (1, 2, 3):
            raise ValueError(f"task_id must be 1, 2, or 3. Got {task_id}")

        self.task_id           = task_id
        self.steps_taken       = 0
        self.done              = False
        self.actions           = []
        self.final_answer      = ""
        self.discovered_tables = set()
        self.cost_spent        = 0.0

        builders = {1: build_task1_db, 2: build_task2_db, 3: build_task3_db}
        self.conn = builders[task_id]()

        task = get_task(task_id)

        # PARTIAL OBSERVABILITY: no table listing in initial observation
        # Agent must call list_tables() to discover what's available
        obs = Observation(
            task_id=task_id,
            result={
                "scenario": task["description"],
                "available_tools": {
                    "run_sql":        "run_sql(query)         — SELECT query (specify columns, no SELECT *)",
                    "inspect_schema": "inspect_schema(table)  — column names + types (discover tables first)",
                    "check_logs":     "check_logs()           — dbt pipeline changelog",
                    "check_airflow":  "check_airflow()        — Airflow DAG run history",
                    "diff_report":    "diff_report(d1,d2)     — compare two dates, query='date1,date2'",
                    "list_tables":    "list_tables()          — discover available tables (START HERE)",
                    "submit":         "submit(answer)         — your final answer. CALL THIS WHEN DONE.",
                },
                "rules": {
                    "step_budget": self.MAX_STEPS,
                    "cost_budget": self.COST_BUDGET,
                    "select_star": "SELECT * is blocked. Specify columns explicitly.",
                    "discovery": "You must discover tables with list_tables() before inspecting or querying them.",
                    "submit_rule": f"You must use at least {self.MIN_STEPS_BEFORE_SUBMIT} tools before submitting.",
                },
            },
            steps_taken=0,
            done=False,
            max_steps=self.MAX_STEPS,
            cost_spent=0.0,
            budget_remaining=self.COST_BUDGET,
        )
        self.last_observation = obs
        return obs

    # ── step() ────────────────────────────────────────────────────────────────

    def step(self, action: Action):
        """
        Execute one action. Returns (observation, reward, done, info).
        reward is None until done=True.
        Enforces: partial observability, query costs, anti-cheat rules.
        """
        if self.done:
            raise RuntimeError("Episode over. Call reset() first.")

        # Validate tool name up front
        if action.tool not in self.VALID_TOOLS:
            obs = Observation(
                task_id=self.task_id,
                result={"error": f"Unknown tool '{action.tool}'. Valid: {sorted(self.VALID_TOOLS)}"},
                steps_taken=self.steps_taken,
                done=False,
                max_steps=self.MAX_STEPS,
                cost_spent=self.cost_spent,
                budget_remaining=self.COST_BUDGET - self.cost_spent,
            )
            self.last_observation = obs
            return obs, None, False, self._make_info(False)

        # Anti-cheat: block early submit
        if action.tool == "submit" and self.steps_taken < self.MIN_STEPS_BEFORE_SUBMIT:
            obs = Observation(
                task_id=self.task_id,
                result={
                    "error": f"Cannot submit yet. You must investigate first. "
                             f"Use at least {self.MIN_STEPS_BEFORE_SUBMIT} tools before submitting. "
                             f"Steps taken so far: {self.steps_taken}."
                },
                steps_taken=self.steps_taken,
                done=False,
                max_steps=self.MAX_STEPS,
                cost_spent=self.cost_spent,
                budget_remaining=self.COST_BUDGET - self.cost_spent,
            )
            self.last_observation = obs
            return obs, None, False, self._make_info(False)


        # Cost tracking
        tool_cost = self.TOOL_COSTS.get(action.tool, 0.0)
        self.cost_spent += tool_cost

        self.actions.append(action.model_dump())
        self.steps_taken += 1

        result = self._run_tool(action)

        # Check if budget (cost or steps) exhausted — auto-submit
        budget_exhausted = self.cost_spent >= self.COST_BUDGET
        steps_exhausted = self.steps_taken >= self.MAX_STEPS

        if (budget_exhausted or steps_exhausted) and not self.done:
            if not self.final_answer:
                all_reasoning = " | ".join(
                    a.get("reasoning", "") for a in self.actions if a.get("reasoning")
                )
                all_queries = " | ".join(
                    a.get("query", "") for a in self.actions if a.get("tool") == "run_sql"
                )
                exhaust_type = "cost budget" if budget_exhausted else "step budget"
                self.final_answer = (
                    f"AUTO-SUBMITTED ({exhaust_type} exhausted). "
                    f"Reasoning: {all_reasoning}. Queries run: {all_queries}"
                )

        episode_done = action.tool == "submit" or budget_exhausted or steps_exhausted
        self.done = episode_done

        obs = Observation(
            task_id=self.task_id,
            result=result,
            steps_taken=self.steps_taken,
            done=episode_done,
            max_steps=self.MAX_STEPS,
            cost_spent=self.cost_spent,
            budget_remaining=max(0.0, self.COST_BUDGET - self.cost_spent),
        )
        self.last_observation = obs

        reward = None
        if episode_done:
            reward = grade(
                task_id=self.task_id,
                conn=self.conn,
                actions=self.actions,
                final_answer=self.final_answer,
            )

        return obs, reward, episode_done, self._make_info(episode_done)

    # ── state() ───────────────────────────────────────────────────────────────

    def state(self) -> EnvState:
        return EnvState(
            task_id=self.task_id or 0,
            steps_taken=self.steps_taken,
            done=self.done,
            agent_actions=self.actions,
            current_observation=self.last_observation,
            discovered_tables=sorted(self.discovered_tables),
            cost_spent=self.cost_spent,
            budget_remaining=max(0.0, self.COST_BUDGET - self.cost_spent),
        )

    # ── Info helper ───────────────────────────────────────────────────────────

    def _make_info(self, done: bool) -> dict:
        return {
            "steps_remaining":  self.MAX_STEPS - self.steps_taken,
            "task_id":          self.task_id,
            "done":             done,
            "cost_spent":       self.cost_spent,
            "budget_remaining": max(0.0, self.COST_BUDGET - self.cost_spent),
            "discovered_tables": sorted(self.discovered_tables),
        }

    # ── Partial observability checks ──────────────────────────────────────────

    def _check_table_access(self, query: str) -> str | None:
        """
        Check if the SQL query references any undiscovered tables.
        Returns error message if violation found, None if OK.
        """
        if not self.discovered_tables:
            return (
                "No tables discovered yet. Call list_tables() first to discover "
                "available tables before running SQL queries."
            )

        # Extract table names from the query (simple heuristic)
        # Look for FROM/JOIN followed by table name
        q_upper = query.upper()
        tokens = re.findall(r'(?:FROM|JOIN)\s+(\w+)', q_upper, re.IGNORECASE)

        for token in tokens:
            # Check against discovered tables (case-insensitive)
            if token.lower() not in {t.lower() for t in self.discovered_tables}:
                return (
                    f"Table '{token.lower()}' has not been discovered yet. "
                    f"Discovered tables: {sorted(self.discovered_tables)}. "
                    f"Use list_tables() to discover tables first."
                )
        return None

    # ── Tool router ───────────────────────────────────────────────────────────

    def _run_tool(self, action: Action):
        tool  = action.tool
        query = action.query.strip()

        if tool == "list_tables":
            tables = list_tables(self.conn)
            self.discovered_tables = set(tables)
            return {
                "tables": tables,
                "message": f"Found {len(tables)} tables. Use inspect_schema(table_name) to see columns."
            }

        elif tool == "inspect_schema":
            # Partial observability: must discover tables first
            if query.lower() not in {t.lower() for t in self.discovered_tables}:
                return {
                    "error": f"Table '{query}' not discovered yet. Call list_tables() first. "
                             f"Discovered so far: {sorted(self.discovered_tables)}"
                }
            return inspect_schema(self.conn, query)

        elif tool == "run_sql":
            # Partial observability: check table access
            access_error = self._check_table_access(query)
            if access_error:
                return {"error": access_error}
            return run_sql(self.conn, query)

        elif tool == "check_logs":
            return check_logs(self.conn)

        elif tool == "check_airflow":
            return check_airflow(self.conn)

        elif tool == "diff_report":
            parts = [p.strip() for p in query.split(",")]
            if len(parts) != 2:
                return {"error": "diff_report query must be 'date1,date2'"}
            return diff_report(self.conn, parts[0], parts[1])

        elif tool == "submit":
            self.final_answer = query
            return {
                "message": "Answer submitted. Grading now.",
                "steps_used": self.steps_taken,
                "cost_spent": self.cost_spent,
                "your_answer_preview": query[:200],
            }