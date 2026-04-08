"""Baseline inference script for DataOnCallEnv.
"""

import os
import json
import time
import httpx
from openai import OpenAI, APIStatusError
from dataoncallenv.environment import DataOnCallEnv
from dataoncallenv.models import Action
from dotenv import load_dotenv
load_dotenv()


# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()

API_KEY = (
    os.environ.get("HF_TOKEN") or
    os.environ.get("OPENAI_API_KEY")
)

if not API_KEY:
    raise EnvironmentError("Set HF_TOKEN or OPENAI_API_KEY before running.")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# Tool definitions

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a SELECT SQL query against the database. "
                "You MUST specify column names — SELECT * is blocked. "
                "You must discover tables first with list_tables(). "
                "Cost: 2.0 per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "The SELECT SQL query (no SELECT *)"},
                    "reasoning": {"type": "string", "description": "Why you are running this query"},
                },
                "required": ["query", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": (
                "See column names and data types for a specific table. "
                "Table must be discovered first via list_tables(). "
                "Cost: 1.0 per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "The table name to inspect"},
                    "reasoning": {"type": "string", "description": "Why you are inspecting this table"},
                },
                "required": ["query", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_logs",
            "description": (
                "Read the dbt pipeline changelog. Often contains the smoking gun. "
                "Cost: 1.0 per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "Pass empty string"},
                    "reasoning": {"type": "string", "description": "Why you are checking logs"},
                },
                "required": ["query", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_airflow",
            "description": (
                "Read Airflow DAG run history. Shows pipeline orchestration, "
                "manual triggers, and deployment events. "
                "Cost: 1.0 per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "Pass empty string"},
                    "reasoning": {"type": "string", "description": "Why you are checking Airflow runs"},
                },
                "required": ["query", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_report",
            "description": (
                "Compare total revenue between two dates. "
                "Query format: 'YYYY-MM-DD,YYYY-MM-DD'. "
                "Cost: 1.5 per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "Two dates: 'date1,date2'"},
                    "reasoning": {"type": "string", "description": "Why you are comparing these dates"},
                },
                "required": ["query", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "Discover all tables in the database. "
                "You MUST call this first — tables are hidden until discovered. "
                "Cost: 0.5 per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "Pass empty string"},
                    "reasoning": {"type": "string", "description": "Why you are listing tables"},
                },
                "required": ["query", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit your final answer and end the episode. "
                "You must have used at least 2 tools before submitting. "
                "Your answer MUST include: (1) the root cause, (2) a corrected SQL query. "
                "Cost: 0.0 (free)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Your complete answer. Include: "
                            "ROOT CAUSE: [what caused the bug] "
                            "CORRECTED SQL: [SELECT query that returns the right answer]"
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief summary of your investigation path",
                    },
                },
                "required": ["query", "reasoning"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a Data Analyst debugging reports.
RULES:
- Budget: 15 steps, 20.0 cost. Tables HIDDEN until list_tables() used.
- SELECT * is BLOCKED. Specify columns. min 2 steps before submit.
- Tools: list_tables(0.5), inspect_schema(1.0), check_logs(1.0), check_airflow(1.0), run_sql(2.0), diff_report(1.5).

STRATEGY:
1. list_tables() and inspect_schema() first.
2. check_logs()/check_airflow() for root causes.
3. run_sql() to confirm hypothesis (NO SELECT *).
4. submit() immediately when diagnosed.

SUBMIT FORMAT:
ROOT CAUSE: [explanation]
CORRECTED SQL: [full SELECT query fixing the bug]"""

# Agent loop

def run_agent(env: DataOnCallEnv, task_id: int) -> dict:
    print(f"\n{'='*60}")
    print(f"Task {task_id} — {['','easy','medium','hard'][task_id]}")
    print('='*60)

    obs = env.reset(task_id=task_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": json.dumps(obs.result, indent=2)},
    ]

    final_reward = None
    steps        = 0
    MAX_STEPS    = env.MAX_STEPS

    while not env.done and steps < MAX_STEPS:

        # Warn the model when it's running low on steps or budget
        steps_remaining = MAX_STEPS - steps
        budget_remaining = env.COST_BUDGET - env.cost_spent

        if steps_remaining == 5 or budget_remaining <= 5.0:
            messages.append({
                "role": "user",
                "content": (
                    f"WARNING: {steps_remaining} steps remaining, "
                    f"budget remaining: {budget_remaining:.1f}. "
                    "If you know the root cause, call submit() NOW with your answer and corrected SQL."
                )
            })
        elif steps_remaining == 2:
            messages.append({
                "role": "user",
                "content": (
                    "FINAL WARNING: 2 steps left. You MUST call submit() on your next step "
                    "or the episode will end without a proper score. Submit your best answer now."
                )
            })

        # Retry loop for model loading (Serverless API 503 errors)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                )
                break
            except Exception as e:
                # Handle 503 (model loading) or 429 (rate limit)
                err_msg = str(e).lower()
                if "503" in err_msg or "loading" in err_msg:
                    wait_time = 15 * (attempt + 1)
                    print(f"\n  [Model is loading... waiting {wait_time}s (attempt {attempt+1}/{max_retries})]")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        raise e
                elif "429" in err_msg:
                    print("\n  [Rate limit hit. Waiting 30s...]")
                    time.sleep(30)
                    if attempt == max_retries - 1:
                        raise e
                else:
                    raise e

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                query     = fn_args.get("query", "")
                reasoning = fn_args.get("reasoning", "")

                print(f"\nStep {steps+1}/{MAX_STEPS}: {fn_name} (cost: {env.TOOL_COSTS.get(fn_name, 0)})")
                print(f"  Query:     {query}")
                if reasoning:
                    print(f"  Reasoning: {reasoning}")

                action = Action(tool=fn_name, query=query, reasoning=reasoning)
                obs, reward, done, info = env.step(action)
                steps += 1

                if reward:
                    final_reward = reward

                result_preview = str(obs.result)
                if len(result_preview) > 500:
                    result_preview = result_preview[:500] + "... [truncated for console]"
                print(f"  Result:    {result_preview}")
                print(f"  Remaining: {info['steps_remaining']} steps, ${info['budget_remaining']:.1f} budget")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      json.dumps(obs.result),
                })

                if done:
                    break
        else:
            # Model gave text instead of tool call — nudge it
            content = msg.content or ""
            print(f"\n  [Model text, no tool call]: {content[:100]}")
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role":    "user",
                "content": "Use a tool to continue, or call submit() with your answer.",
            })

    # If episode ended without a reward (safety net)
    if final_reward is None:
        from graders import grade
        final_reward = grade(env.task_id, env.conn, env.actions, env.final_answer)

    print(f"\n--- Task {task_id} Score: {final_reward.score:.4f} ---")
    print(f"  diagnosis_correct:     {final_reward.breakdown.diagnosis_correct}")
    print(f"  fix_valid:             {final_reward.breakdown.fix_valid}")
    print(f"  efficiency:            {final_reward.breakdown.efficiency}")
    print(f"  reasoning_quality:     {final_reward.breakdown.reasoning_quality}")
    print(f"  investigation_quality: {final_reward.breakdown.investigation_quality}")
    print(f"  false_positive_penalty: -{final_reward.false_positive_penalty}")

    return {
        "task_id":     task_id,
        "difficulty":  ["","easy","medium","hard"][task_id],
        "score":       final_reward.score,
        "steps_taken": steps,
        "breakdown":   final_reward.breakdown.model_dump(),
        "penalty":     final_reward.false_positive_penalty,
    }

# Entry point

if __name__ == "__main__":
    env     = DataOnCallEnv()
    results = []

    for task_id in [1, 2, 3]:
        result = run_agent(env, task_id)
        results.append(result)

    print(f"\n{'='*60}")
    print("FINAL BASELINE SCORES")
    print('='*60)
    total = 0
    for r in results:
        print(f"Task {r['task_id']} ({r['difficulty']:6s}): {r['score']:.4f}  [{r['steps_taken']} steps]")
        total += r["score"]
    avg = total / len(results)
    print(f"Average:              {avg:.4f}")
    print('='*60)

    with open("baseline_scores.json", "w") as f:
        json.dump({"results": results, "average": avg}, f, indent=2)
    print("\nSaved to baseline_scores.json")