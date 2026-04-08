"""Task definitions and ground truth for DataOnCallEnv.
"""

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


        # Grading tiers
        "diagnosis_tiers": {
            "exact": ["case sensitive", "case-sensitive", "lower(", "upper(", "lowercase", "uppercase"],
            "category": ["mismatch", "join fail", "not matching", "null", "currency"],
            "symptom": ["$0", "zero", "missing", "no rows"],
        },


        # Tables relevant to investigation (used for penalties)
        "relevant_tables": ["sales", "currency_rates", "products", "dbt_log", "airflow_runs"],


        # Canonical solution
        "ground_truth_query": """
            SELECT ROUND(SUM(s.amount * cr.rate_to_usd), 2) as total_revenue_usd
            FROM sales s
            JOIN currency_rates cr ON LOWER(s.currency) = cr.currency_code
        """,

        "optimal_steps": 5,
        "optimal_cost": 7.0,  # list_tables(0.5) + 2×inspect_schema(2.0) + check_logs(1.0) + run_sql(2.0) + submit(0) + verify(1.5)
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

        "diagnosis_tiers": {
            "exact": ["timezone", "utc", "local time", "migration", "tz_source"],
            "category": ["double count", "duplicate", "boundary", "counted twice"],
            "symptom": ["drop", "fewer", "missing users", "jan 31"],
        },

        "relevant_tables": ["user_events", "dbt_log", "airflow_runs"],

        "ground_truth_query": """
            SELECT COUNT(DISTINCT user_id) as mau, substr(event_ts, 1, 7) as month
            FROM user_events
            WHERE tz_source = 'utc'
            GROUP BY month
            ORDER BY month
        """,

        "smoking_gun_text": "Migrated timestamp handling from UTC to local timezone",

        "optimal_steps": 7,
        "optimal_cost": 10.0,
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

Available tables: You must discover them first.

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

        "diagnosis_tiers": {
            "exact": ["fanout", "fan-out", "fan out", "non-unique", "not unique",
                      "cardinality", "one to many", "product_promotions"],
            "category": ["duplicate rows", "multiple rows", "multiply", "inflat", "promo"],
            "symptom": ["3.", "4.", "overstated", "inflated", "higher"],
        },

        "relevant_tables": ["sales", "products", "product_promotions", "dbt_log", "airflow_runs"],

        "ground_truth_query": """
            SELECT ROUND(SUM(s.amount), 2) as true_revenue
            FROM sales s
            WHERE s.product_id IN (
                SELECT product_id FROM products WHERE launched_at >= '2023-11-01'
            )
        """,

        "optimal_steps": 8,
        "optimal_cost": 13.0,
    }
}

def get_task(task_id: int) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Task ID must be 1, 2, or 3. Got: {task_id}")
    return TASKS[task_id]