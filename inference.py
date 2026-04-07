# inference.py
# Production inference script.
# Key fixes vs v1:
#  - System prompt explicitly tells model to submit() before running out of steps
#  - Warning injected at step 10 of 15 — "you have 5 steps left, submit soon"
#  - If model still doesn't submit, we force a submit() with its best reasoning
#  - Scores are saved to baseline_scores.json for reproducibility

import os
import json
from openai import OpenAI
from environment import DataOnCallEnv
from models import Action
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# ── Credentials from environment variables ────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

API_KEY = (
    os.environ.get("HF_TOKEN") or
    os.environ.get("OPENAI_API_KEY")
)

if not API_KEY:
    raise EnvironmentError("Set HF_TOKEN or OPENAI_API_KEY before running.")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── Tool schema ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a SELECT SQL query against the database. Use this to explore data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "The SELECT SQL query to run"},
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
            "description": "See column names and data types for a specific table.",
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
            "description": "Read the dbt pipeline changelog. Always check this — it often contains the smoking gun.",
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
            "name": "diff_report",
            "description": "Compare total revenue between two dates. Query format: 'YYYY-MM-DD,YYYY-MM-DD'",
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
            "description": "List all tables in the database. Good first step.",
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
                "Call this as soon as you know the root cause. "
                "Your answer MUST include: (1) the root cause, (2) a corrected SQL query that fixes the problem."
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

SYSTEM_PROMPT = """You are an expert data analyst debugging a broken data pipeline.

You have a budget of 15 tool calls. Use them efficiently.

STRATEGY:
1. Start with list_tables() to understand the schema
2. inspect_schema() the relevant tables
3. check_logs() — the pipeline log often contains the root cause directly
4. Use run_sql() to verify your hypothesis
5. Call submit() AS SOON AS you know the root cause — do not waste steps

IMPORTANT:
- Always include a 'reasoning' field explaining WHY you are making each call
- Your submit() answer MUST include the root cause AND a corrected SQL query
- Format your submit answer as:
    ROOT CAUSE: [clear explanation]
    CORRECTED SQL: SELECT ... (the query that returns the right answer)
- Submit before you run out of steps. If you have 3 steps left and you know the answer, submit NOW."""

# ── Agent loop ────────────────────────────────────────────────────────────────

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

        # Warn the model when it's running low on steps
        steps_remaining = MAX_STEPS - steps
        if steps_remaining == 5:
            messages.append({
                "role": "user",
                "content": (
                    f"WARNING: You have only {steps_remaining} steps remaining. "
                    "If you know the root cause, call submit() NOW with your answer and corrected SQL. "
                    "Do not spend more steps investigating unless you are completely unsure."
                )
            })
        elif steps_remaining == 2:
            messages.append({
                "role": "user",
                "content": (
                    "FINAL WARNING: 2 steps left. You MUST call submit() on your next step "
                    "or the episode will end without a score. Submit your best answer now."
                )
            })

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                query     = fn_args.get("query", "")
                reasoning = fn_args.get("reasoning", "")

                print(f"\nStep {steps+1}/{MAX_STEPS}: {fn_name}")
                print(f"  Query:     {query[:80]}")
                if reasoning:
                    print(f"  Reasoning: {reasoning[:100]}")

                action = Action(tool=fn_name, query=query, reasoning=reasoning)
                obs, reward, done, info = env.step(action)
                steps += 1

                if reward:
                    final_reward = reward

                result_preview = str(obs.result)[:150]
                print(f"  Result:    {result_preview}")
                print(f"  Remaining: {info['steps_remaining']} steps")

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

    # If episode ended without a reward (shouldn't happen, but safety net)
    if final_reward is None:
        from graders import grade
        final_reward = grade(env.task_id, env.conn, env.actions, env.final_answer)

    print(f"\n--- Task {task_id} Score: {final_reward.score:.4f} ---")
    print(f"  diagnosis_correct:  {final_reward.breakdown.diagnosis_correct}")
    print(f"  fix_valid:          {final_reward.breakdown.fix_valid}")
    print(f"  efficiency:         {final_reward.breakdown.efficiency}")
    print(f"  reasoning_quality:  {final_reward.breakdown.reasoning_quality}")

    return {
        "task_id":     task_id,
        "difficulty":  ["","easy","medium","hard"][task_id],
        "score":       final_reward.score,
        "steps_taken": steps,
        "breakdown":   final_reward.breakdown.model_dump(),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

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