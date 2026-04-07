# environment.py
# Key production changes vs v1:
#  - MAX_STEPS raised to 15 (model needs room to both investigate AND submit)
#  - When steps run out, env auto-submits with everything the agent said so far
#  - step() validates the tool name and returns a clear error (not a crash)
#  - state() always works even before reset() is called

from models import Action, Observation, Reward, EnvState
from tasks import get_task
from graders import grade
from database import (
    build_task1_db, build_task2_db, build_task3_db,
    run_sql, inspect_schema, check_logs, diff_report, list_tables
)

class DataOnCallEnv:

    MAX_STEPS = 15  # raised from 10 — model needs space to investigate + submit

    VALID_TOOLS = {"run_sql", "inspect_schema", "check_logs",
                   "diff_report", "list_tables", "submit"}

    def __init__(self):
        self.task_id          = None
        self.conn             = None
        self.steps_taken      = 0
        self.done             = False
        self.actions          = []
        self.last_observation = None
        self.final_answer     = ""

    # ── reset() ───────────────────────────────────────────────────────────────

    def reset(self, task_id: int = 1) -> Observation:
        """Fresh episode. Rebuilds database. Returns scenario as first observation."""
        if task_id not in (1, 2, 3):
            raise ValueError(f"task_id must be 1, 2, or 3. Got {task_id}")

        self.task_id      = task_id
        self.steps_taken  = 0
        self.done         = False
        self.actions      = []
        self.final_answer = ""

        builders = {1: build_task1_db, 2: build_task2_db, 3: build_task3_db}
        self.conn = builders[task_id]()

        task = get_task(task_id)

        obs = Observation(
            task_id=task_id,
            result={
                "scenario": task["description"],
                "available_tools": {
                    "run_sql":        "run_sql(query)         — SELECT query only",
                    "inspect_schema": "inspect_schema(table)  — column names + types",
                    "check_logs":     "check_logs()           — pipeline changelog",
                    "diff_report":    "diff_report(d1,d2)     — compare two dates, query='date1,date2'",
                    "list_tables":    "list_tables()          — all table names",
                    "submit":         "submit(answer)         — your final answer. CALL THIS WHEN DONE.",
                },
                "available_tables": list_tables(self.conn),
                "steps_budget":     self.MAX_STEPS,
            },
            steps_taken=0,
            done=False,
            max_steps=self.MAX_STEPS,
        )
        self.last_observation = obs
        return obs

    # ── step() ────────────────────────────────────────────────────────────────

    def step(self, action: Action):
        """
        Execute one action. Returns (observation, reward, done, info).
        reward is None until done=True.
        When steps run out, auto-submits using accumulated reasoning.
        """
        if self.done:
            raise RuntimeError("Episode over. Call reset() first.")

        # Validate tool name up front — return clean error, don't crash
        if action.tool not in self.VALID_TOOLS:
            obs = Observation(
                task_id=self.task_id,
                result={"error": f"Unknown tool '{action.tool}'. Valid: {sorted(self.VALID_TOOLS)}"},
                steps_taken=self.steps_taken,
                done=False,
                max_steps=self.MAX_STEPS,
            )
            self.last_observation = obs
            return obs, None, False, {"steps_remaining": self.MAX_STEPS - self.steps_taken}

        self.actions.append(action.model_dump())
        self.steps_taken += 1

        result = self._run_tool(action)

        # Check if budget exhausted — auto-submit with everything agent said
        if self.steps_taken >= self.MAX_STEPS and not self.done:
            if not self.final_answer:
                # Collect all reasoning the agent produced as the "answer"
                all_reasoning = " | ".join(
                    a.get("reasoning", "") for a in self.actions if a.get("reasoning")
                )
                all_queries = " | ".join(
                    a.get("query", "") for a in self.actions if a.get("tool") == "run_sql"
                )
                self.final_answer = f"AUTO-SUBMITTED (budget exhausted). Reasoning: {all_reasoning}. Queries run: {all_queries}"

        episode_done = action.tool == "submit" or self.steps_taken >= self.MAX_STEPS
        self.done = episode_done

        obs = Observation(
            task_id=self.task_id,
            result=result,
            steps_taken=self.steps_taken,
            done=episode_done,
            max_steps=self.MAX_STEPS,
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

        info = {
            "steps_remaining": self.MAX_STEPS - self.steps_taken,
            "task_id":         self.task_id,
            "done":            episode_done,
        }

        return obs, reward, episode_done, info

    # ── state() ───────────────────────────────────────────────────────────────

    def state(self) -> EnvState:
        return EnvState(
            task_id=self.task_id or 0,
            steps_taken=self.steps_taken,
            done=self.done,
            agent_actions=self.actions,
            current_observation=self.last_observation,
        )

    # ── Tool router ───────────────────────────────────────────────────────────

    def _run_tool(self, action: Action):
        tool  = action.tool
        query = action.query.strip()

        if tool == "run_sql":
            return run_sql(self.conn, query)

        elif tool == "inspect_schema":
            return inspect_schema(self.conn, query)

        elif tool == "check_logs":
            return check_logs(self.conn)

        elif tool == "list_tables":
            return list_tables(self.conn)

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
                "your_answer_preview": query[:200],
            }