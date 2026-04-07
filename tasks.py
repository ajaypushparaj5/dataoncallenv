# tasks.py
# Ground truth stored as canonical SQL queries.
# Grader runs BOTH the agent's query AND the ground truth query,
# then compares the actual result sets. No magic numbers. No string matching for SQL.

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

Your goal:
1. Find the root cause of why non-USD revenue shows as $0
2. Write a corrected SQL query that returns total revenue in USD across all currencies
3. Call submit() with your full explanation AND your corrected SQL query
        """.strip(),

        "root_cause_keywords": [
            "case", "lowercase", "uppercase", "lower(", "upper(",
            "currency_code", "null", "mismatch", "usd vs",
            "case sensitive", "case-sensitive", "join fail", "not matching",
        ],

        # This is the canonical correct answer.
        # Grader runs this and stores the result, then compares agent's SQL result to it.
        "ground_truth_query": """
            SELECT ROUND(SUM(s.amount * cr.rate_to_usd), 2) as total_revenue_usd
            FROM sales s
            JOIN currency_rates cr ON LOWER(s.currency) = cr.currency_code
        """,

        "optimal_steps": 5,
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

Your goal:
1. Find why MAU differs between January and February
2. Find the root cause in the pipeline
3. Call submit() with: your explanation AND a corrected query that counts MAU correctly
        """.strip(),

        "root_cause_keywords": [
            "timezone", "utc", "local time", "migration",
            "double count", "duplicate", "boundary", "jan 31",
            "timestamp", "tz_source", "twice", "counted twice",
        ],

        "ground_truth_query": """
            SELECT COUNT(DISTINCT user_id) as mau, substr(event_ts, 1, 7) as month
            FROM user_events
            WHERE tz_source = 'utc'
            GROUP BY month
            ORDER BY month
        """,

        "smoking_gun_text": "Migrated timestamp handling from UTC to local timezone",

        "optimal_steps": 7,
    },

    3: {
        "id": 3,
        "difficulty": "hard",
        "title": "Cloud Storage Revenue Overstated by Exactly 3–4x",
        "description": """
You are a data analyst on-call. A Slack message just arrived:

    "Something is very wrong. Revenue for our newer product lines
     is showing 3–4x higher than actual bookings.
     Other product lines look fine. This started after a schema
     change about 6 weeks ago."

Available tables: sales, products, product_promotions, dbt_log

Your goal:
1. Find exactly why revenue is inflated for newer products
2. Identify which table and join causes the multiplication
3. Call submit() with: root cause + corrected SQL query
        """.strip(),

        "root_cause_keywords": [
            "fanout", "fan-out", "fan out", "non-unique", "not unique",
            "duplicate rows", "product_promotions", "multiple rows",
            "join multiply", "multiply", "inflat", "promo",
            "one to many", "cardinality", "several rows", "many rows",
        ],

        "ground_truth_query": """
            SELECT ROUND(SUM(s.amount), 2) as true_revenue
            FROM sales s
            WHERE s.product_id IN (
                SELECT product_id FROM products WHERE launched_at >= '2023-11-01'
            )
        """,

        "optimal_steps": 8,
    }
}

def get_task(task_id: int) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Task ID must be 1, 2, or 3. Got: {task_id}")
    return TASKS[task_id]