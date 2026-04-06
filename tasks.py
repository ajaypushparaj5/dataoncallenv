# tasks.py
# Defines the 3 tasks — the scenario description the agent receives,
# the ground truth root cause, and what a valid fix looks like.
# The grader compares agent output against these ground truths.

TASKS = {
    1: {
        "id": 1,
        "difficulty": "easy",
        "title": "Weekly Revenue Report Shows $0 for International Sales",
        "description": """
You are a data analyst on-call. A Slack message just arrived:

    "Hey, the weekly revenue report is showing $0 for all non-USD sales.
     USD sales look fine. The pipeline ran successfully this morning.
     Can you investigate?"

Available tables: sales, products, currency_rates, dbt_log

Your goal: identify the root cause and write a SQL query that produces
the correct total revenue in USD across all currencies.
        """.strip(),

        # What the agent must say to get diagnosis_correct = 0.30
        # We check if any of these phrases appear in the agent's final answer
        "root_cause_keywords": [
            "case mismatch",
            "lowercase",
            "uppercase",
            "currency_code",
            "null join",
            "join fails",
            "usd vs USD",
            "case sensitive",
        ],

        # A query that actually fixes the problem — we run this and compare output
        "correct_fix_query": """
            SELECT SUM(s.amount * cr.rate_to_usd) as total_revenue_usd
            FROM sales s
            JOIN currency_rates cr
              ON LOWER(s.currency) = cr.currency_code
        """,

        # The correct answer value — grader runs fix query and checks this
        "expected_total_revenue": 2851.0,  # 500 + 450*1.08 + 300*1.27 + 600 + 200*1.08 + 800

        # How many steps an expert analyst would need
        "optimal_steps": 4,
    },

    2: {
        "id": 2,
        "difficulty": "medium",
        "title": "Monthly Active Users Dropped 8% on Feb 1st",
        "description": """
You are a data analyst on-call. A Slack message just arrived:

    "Our MAU metric dropped 8% going into February. Leadership is asking
     questions. January looked great. Nothing changed in the product.
     The pipeline ran fine. What happened?"

Available tables: user_events, dbt_log

Hint: look at event timestamps around the month boundary.
Your goal: identify the root cause and explain what happened to the numbers.
        """.strip(),

        "root_cause_keywords": [
            "timezone",
            "utc",
            "local time",
            "migration",
            "double count",
            "duplicate",
            "boundary",
            "jan 31",
            "january 31",
            "timestamp",
        ],

        "correct_fix_query": """
            SELECT COUNT(DISTINCT user_id) as mau, substr(event_ts, 1, 7) as month
            FROM user_events
            WHERE tz_source = 'utc'
               OR event_ts >= '2024-02-01'
            GROUP BY month
        """,

        # Grader checks agent found the dbt_log entry about timezone migration
        "smoking_gun": "Migrated timestamp handling from UTC to local timezone",

        "optimal_steps": 6,
    },

    3: {
        "id": 3,
        "difficulty": "hard",
        "title": "Cloud Storage Revenue Overstated by Exactly 3.7x",
        "description": """
You are a data analyst on-call. A Slack message just arrived:

    "Something is very wrong. Cloud Storage revenue for January is showing
     $5,920 but our finance team says actual bookings were about $1,500.
     The ratio is suspiciously precise — almost exactly 3.7x overstated.
     Other product lines look fine. This started last month."

Available tables: sales, products, product_promotions, dbt_log

Your goal: identify exactly why the number is inflated and write a
corrected query that returns the true revenue figure.
        """.strip(),

        "root_cause_keywords": [
            "fanout",
            "fan-out",
            "non-unique",
            "duplicate rows",
            "product_promotions",
            "multiple rows",
            "join multiplies",
            "not unique",
            "one to many",
            "cardinality",
        ],

        "correct_fix_query": """
            SELECT SUM(s.amount) as true_revenue
            FROM sales s
            WHERE s.product_id IN ('P003', 'P004')
        """,

        "expected_true_revenue": 1800.0,  # 700 + 800 + 300

        # The precise multiplier — agent should mention this is a fanout signal
        "fanout_ratio": 3.7,

        "optimal_steps": 8,
    }
}

def get_task(task_id: int) -> dict:
    """Return task definition. Raises if task_id is invalid."""
    if task_id not in TASKS:
        raise ValueError(f"Task ID must be 1, 2, or 3. Got: {task_id}")
    return TASKS[task_id]