# inference.py
# Runs an OpenAI model as an agent against all 3 tasks.
# Reads credentials from environment variables — never hardcode keys.
#
# Usage:
#   export API_BASE_URL="https://api.openai.com/v1"
#   export MODEL_NAME="gpt-4o"
#   export HF_TOKEN="your_hf_token"
#   export OPENAI_API_KEY="your_openai_key"
#   python inference.py

import os
import json
from openai import OpenAI
from environment import DataOnCallEnv
from models import Action

# ── Setup ─────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")  # cheaper default
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY environment variable before running.")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

# ── Tool schema — tells the model what tools it can call ──────────────────────
# This is the OpenAI function-calling format.
# The model reads this and decides which tool to call next.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a SELECT SQL query against the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "The SQL SELECT query"},
                    "reasoning": {"type": "string", "description": "Why you are running this query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": "See column names and types for a table",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "The table name"},
                    "reasoning": {"type": "string", "description": "Why you want to inspect this table"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_logs",
            "description": "Read the dbt pipeline changelog",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "Pass empty string"},
                    "reasoning": {"type": "string", "description": "Why you are checking logs"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_report",
            "description": "Compare report output between two dates. query format: 'date1,date2'",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "Two dates: 'YYYY-MM-DD,YYYY-MM-DD'"},
                    "reasoning": {"type": "string", "description": "Why you are diffing these dates"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all tables in the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Pass empty string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your final answer and end the episode. Use this when you know the root cause.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Your full answer: root cause, explanation, and corrected SQL query",
                    },
                    "reasoning": {"type": "string", "description": "Summary of your investigation"},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an expert data analyst debugging a broken data pipeline.
You have tools to query a SQLite database, inspect schemas, read pipeline logs, and compare reports.
Think step by step. Always explain your reasoning before each tool call.
When you know the root cause, call submit() with your full explanation and a corrected SQL query.
Be precise — name the exact table, column, and type of bug."""

# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(env: DataOnCallEnv, task_id: int, max_steps: int = 10) -> dict:
    """
    Run the OpenAI model as an agent against one task.
    Returns a dict with the task result and final score.
    """
    print(f"\n{'='*60}")
    print(f"Running Task {task_id}")
    print('='*60)

    # Start fresh episode
    obs = env.reset(task_id=task_id)

    # First message to the model — the scenario
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": json.dumps(obs.result, indent=2)},
    ]

    final_reward = None
    steps = 0

    while not env.done and steps < max_steps:
        # Ask the model what to do next
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,  # deterministic — same input = same output
        )

        msg = response.choices[0].message

        # If model chose to call a tool
        if msg.tool_calls:
            # Add model's response to message history
            messages.append(msg)

            # Execute each tool call
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"\nStep {steps+1}: {fn_name}({fn_args.get('query', '')[:60]})")
                if fn_args.get("reasoning"):
                    print(f"  Reasoning: {fn_args['reasoning'][:100]}")

                # Send action to environment
                action = Action(
                    tool=fn_name,
                    query=fn_args.get("query", ""),
                    reasoning=fn_args.get("reasoning"),
                )
                obs, reward, done, info = env.step(action)
                steps += 1

                if reward:
                    final_reward = reward

                print(f"  Result: {str(obs.result)[:120]}")
                print(f"  Steps remaining: {info['steps_remaining']}")

                # Add tool result to message history so model sees it
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(obs.result),
                })

                if done:
                    break
        else:
            # Model gave a text response instead of a tool call
            # This shouldn't happen often — nudge it to use tools
            print(f"\nModel response (no tool call): {msg.content[:100]}")
            messages.append({"role": "assistant", "content": msg.content})
            messages.append({
                "role": "user",
                "content": "Please use one of the available tools to continue your investigation, or call submit() with your answer."
            })

    # Print final score
    if final_reward:
        print(f"\nFinal Score: {final_reward.score}")
        print(f"  Diagnosis correct: {final_reward.breakdown.diagnosis_correct}")
        print(f"  Fix valid:         {final_reward.breakdown.fix_valid}")
        print(f"  Efficiency:        {final_reward.breakdown.efficiency}")
        print(f"  Reasoning quality: {final_reward.breakdown.reasoning_quality}")
        if final_reward.false_positive_penalty:
            print(f"  Penalty:          -{final_reward.false_positive_penalty}")
    else:
        print("\nEpisode ended without scoring (no submit called)")
        # Force grade with what we have
        from graders import grade
        final_reward = grade(env.task_id, env.conn, env.actions, env.final_answer)
        print(f"Final Score: {final_reward.score}")

    return {
        "task_id": task_id,
        "score": final_reward.score if final_reward else 0.0,
        "steps_taken": steps,
        "breakdown": final_reward.breakdown.model_dump() if final_reward else {},
    }

# ── Main — run all 3 tasks ────────────────────────────────────────────────────

if __name__ == "__main__":
    env = DataOnCallEnv()
    results = []

    for task_id in [1, 2, 3]:
        result = run_agent(env, task_id)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("BASELINE SCORES")
    print('='*60)
    total = 0
    for r in results:
        print(f"Task {r['task_id']} ({['','easy','medium','hard'][r['task_id']]}): "
              f"{r['score']:.4f}  ({r['steps_taken']} steps)")
        total += r["score"]
    print(f"Average: {total/len(results):.4f}")
    print('='*60)

    # Save results for reproducibility
    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to baseline_scores.json")