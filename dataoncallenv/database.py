"""Database initialization and tool implementations for DataOnCallEnv.
"""

import sqlite3
import re

# Anti-cheat
MAX_RESULT_ROWS = 50

# Table helpers

def create_base_tables(conn):
    """Creates the core tables every task shares."""
    cur = conn.cursor()

    # Products table — what we sell
    cur.execute("""
        CREATE TABLE products (
            product_id   TEXT PRIMARY KEY,
            product_name TEXT,
            category     TEXT,
            launched_at  TEXT   -- ISO date string
        )
    """)

    # Sales fact table — one row per transaction
    cur.execute("""
        CREATE TABLE sales (
            sale_id      INTEGER PRIMARY KEY,
            product_id   TEXT,
            amount       REAL,
            currency     TEXT,  -- the currency code used in THIS table
            sale_date    TEXT
        )
    """)

    # DBT run log — simulates a real pipeline changelog (expanded with noise)
    cur.execute("""
        CREATE TABLE dbt_log (
            run_id      INTEGER PRIMARY KEY,
            run_at      TEXT,
            model_name  TEXT,
            status      TEXT,
            message     TEXT,
            duration_s  REAL DEFAULT 0.0,
            rows_affected INTEGER DEFAULT 0
        )
    """)

    # Airflow DAG run history — realistic pipeline orchestration logs
    cur.execute("""
        CREATE TABLE airflow_runs (
            run_id       TEXT PRIMARY KEY,
            dag_id       TEXT,
            execution_date TEXT,
            state        TEXT,
            start_date   TEXT,
            end_date     TEXT,
            duration_s   REAL,
            conf         TEXT
        )
    """)

    conn.commit()

# Task 1

def build_task1_db():
    """
    The weekly revenue report shows $0 for all international sales.
    Root cause: sales table stores currency as 'USD', 'EUR', 'GBP'
                currency_rates table stores them as 'usd', 'eur', 'gbp'
    The JOIN silently produces NULLs — no error, just missing data.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    create_base_tables(conn)
    cur = conn.cursor()

    # Currency conversion table — NOTE: lowercase codes. This is the bug.
    cur.execute("""
        CREATE TABLE currency_rates (
            currency_code TEXT PRIMARY KEY,  -- 'usd', 'eur', 'gbp' (lowercase — BUG)
            rate_to_usd   REAL
        )
    """)

    # Seed products
    cur.executemany("INSERT INTO products VALUES (?,?,?,?)", [
        ("P001", "Analytics Pro", "SaaS", "2023-01-15"),
        ("P002", "Data Studio",   "SaaS", "2023-03-10"),
        ("P003", "Cloud Storage", "Infra","2023-06-01"),
    ])

    # Seed sales — currency stored as UPPERCASE in this table
    cur.executemany("INSERT INTO sales VALUES (?,?,?,?,?)", [
        (1, "P001", 500.0,  "USD", "2024-01-10"),
        (2, "P001", 450.0,  "EUR", "2024-01-11"),
        (3, "P002", 300.0,  "GBP", "2024-01-12"),
        (4, "P002", 600.0,  "USD", "2024-01-13"),
        (5, "P003", 200.0,  "EUR", "2024-01-14"),
        (6, "P003", 800.0,  "USD", "2024-01-15"),
    ])


    # Seed data with casing mismatch bug
    cur.executemany("INSERT INTO currency_rates VALUES (?,?)", [
        ("usd", 1.0),
        ("eur", 1.08),
        ("gbp", 1.27),
    ])


    # dbt history
    cur.executemany("INSERT INTO dbt_log VALUES (?,?,?,?,?,?,?)", [
        (1,  "2024-01-16 03:00:00", "stg_sales",          "success", "Compiled successfully. Loaded 6 rows.",                    2.1, 6),
        (2,  "2024-01-16 03:00:03", "stg_products",       "success", "Compiled successfully. Loaded 3 rows.",                    1.4, 3),
        (3,  "2024-01-16 03:00:05", "stg_currency_rates", "success", "Compiled successfully. Loaded 3 rates.",                   0.9, 3),
        (4,  "2024-01-16 03:00:07", "int_revenue_joined", "success", "JOIN completed. Note: 2 currency codes did not match.",    3.2, 4),
        (5,  "2024-01-16 03:00:11", "fct_revenue_report", "success", "Aggregation complete. Revenue report refreshed.",          4.2, 1),
        (6,  "2024-01-16 03:00:16", "test_not_null_revenue", "pass", "All non-null checks passed on fct_revenue_report.",        1.1, 0),
        (7,  "2024-01-15 03:00:00", "stg_sales",          "success", "Compiled successfully. Loaded 6 rows.",                    2.3, 6),
        (8,  "2024-01-15 03:00:04", "fct_revenue_report", "success", "Aggregation complete. Revenue report refreshed.",          4.0, 1),
        (9,  "2024-01-14 03:00:00", "stg_sales",          "success", "Compiled successfully. Loaded 5 rows.",                    1.9, 5),
        (10, "2024-01-14 03:00:05", "fct_revenue_report", "success", "Aggregation complete. Revenue report refreshed.",          3.8, 1),
        (11, "2024-01-13 12:00:00", "deprecation_check",  "warn",    "WARNING: column 'legacy_rate' in currency_rates is unused. Consider removing.", 0.5, 0),
    ])


    # Airflow history
    cur.executemany("INSERT INTO airflow_runs VALUES (?,?,?,?,?,?,?,?)", [
        ("run_101", "etl_daily_pipeline", "2024-01-16", "success", "2024-01-16 02:55:00", "2024-01-16 03:00:20", 320.0, '{"env": "prod"}'),
        ("run_100", "etl_daily_pipeline", "2024-01-15", "success", "2024-01-15 02:55:00", "2024-01-15 03:00:10", 310.0, '{"env": "prod"}'),
        ("run_099", "etl_daily_pipeline", "2024-01-14", "success", "2024-01-14 02:55:00", "2024-01-14 03:00:08", 308.0, '{"env": "prod"}'),
        ("run_098", "dbt_test_suite",     "2024-01-16", "success", "2024-01-16 03:01:00", "2024-01-16 03:02:30", 90.0,  '{"suite": "revenue"}'),
        ("run_097", "data_quality_check", "2024-01-16", "success", "2024-01-16 03:03:00", "2024-01-16 03:03:45", 45.0,  '{"check": "completeness"}'),
    ])

    conn.commit()
    return conn

# Task 2

def build_task2_db():
    """
    Monthly Active Users dropped 8% on Feb 1st.
    Root cause: pipeline migrated from UTC to local time on Jan 31.
    Events near midnight were double-counted in January and missed in February.
    The dbt_log records the migration — agent must find it.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_base_tables(conn)
    cur = conn.cursor()

    # User events table
    cur.execute("""
        CREATE TABLE user_events (
            event_id    INTEGER PRIMARY KEY,
            user_id     INTEGER,
            event_type  TEXT,
            event_ts    TEXT,   -- timestamp — some UTC, some local after migration
            tz_source   TEXT    -- 'utc' or 'local' — reveals the bug
        )
    """)

    # January events (UTC) — legitimate
    jan_events = [(i, 1000+i, "page_view", f"2024-01-{15+(i%16):02d} {(i*3)%24:02d}:00:00", "utc")
                  for i in range(1, 41)]

    # Boundary events — these exist in BOTH months due to timezone shift (the bug)
    boundary_events = [
        (41, 2001, "page_view", "2024-01-31 23:30:00", "utc"),   # counted in Jan
        (42, 2001, "page_view", "2024-01-31 23:30:00", "local"), # same event, counted again
        (43, 2002, "page_view", "2024-01-31 23:45:00", "utc"),
        (44, 2002, "page_view", "2024-01-31 23:45:00", "local"),
    ]

    # February events (local time after migration) — smaller count looks like drop
    feb_events = [(50+i, 3000+i, "page_view", f"2024-02-{1+(i%14):02d} {(i*2)%24:02d}:00:00", "local")
                  for i in range(1, 31)]

    cur.executemany("INSERT INTO user_events VALUES (?,?,?,?,?)",
                    jan_events + boundary_events + feb_events)

    # dbt history
    cur.executemany("INSERT INTO dbt_log VALUES (?,?,?,?,?,?,?)", [
        (1,  "2024-02-01 03:00:00", "stg_user_events",   "success", "Compiled successfully. Loaded 30 events.",                3.1, 30),
        (2,  "2024-02-01 03:00:04", "int_mau_calc",      "success", "MAU aggregation complete for 2024-02.",                   2.8, 1),
        (3,  "2024-02-01 03:00:08", "fct_mau_report",    "success", "Report refreshed. MAU = 30 (down from 42 in Jan).",       4.1, 1),
        (4,  "2024-02-01 03:00:13", "test_mau_not_null", "pass",    "All MAU values are non-null.",                            0.9, 0),
        (5,  "2024-01-31 22:00:00", "user_events_etl",   "success", "Migrated timestamp handling from UTC to local timezone. All future events will use local tz.", 5.2, 0),
        (6,  "2024-01-31 22:05:00", "stg_user_events",   "success", "Re-processed 4 boundary events with new timezone config.", 2.1, 4),
        (7,  "2024-01-31 03:00:00", "stg_user_events",   "success", "Compiled successfully. Loaded 40 January events.",        2.9, 40),
        (8,  "2024-01-31 03:00:04", "fct_mau_report",    "success", "Report refreshed. MAU = 42.",                             3.5, 1),
        (9,  "2024-01-30 03:00:00", "stg_user_events",   "success", "Compiled successfully. Loaded 38 events.",                2.7, 38),
        (10, "2024-01-30 03:00:04", "fct_mau_report",    "success", "Report refreshed. MAU = 40.",                             3.2, 1),
        (11, "2024-01-29 10:00:00", "deprecation_check",  "warn",   "WARNING: event_type 'page_view' will be split into 'page_load' and 'page_interaction' in v3.", 0.4, 0),
        (12, "2024-01-28 03:00:00", "data_freshness_check", "pass", "All source tables refreshed within SLA (< 6 hours).",     1.2, 0),
    ])

    # Airflow history
    cur.executemany("INSERT INTO airflow_runs VALUES (?,?,?,?,?,?,?,?)", [
        ("run_201", "etl_daily_pipeline",    "2024-02-01", "success", "2024-02-01 02:55:00", "2024-02-01 03:00:15", 315.0, '{"env": "prod"}'),
        ("run_200", "etl_daily_pipeline",    "2024-01-31", "success", "2024-01-31 02:55:00", "2024-01-31 03:00:10", 310.0, '{"env": "prod"}'),
        ("run_199", "tz_migration_manual",   "2024-01-31", "success", "2024-01-31 21:50:00", "2024-01-31 22:05:30", 930.0, '{"triggered_by": "eng_team", "reason": "timezone standardization", "target": "user_events_etl"}'),
        ("run_198", "etl_daily_pipeline",    "2024-01-30", "success", "2024-01-30 02:55:00", "2024-01-30 03:00:08", 308.0, '{"env": "prod"}'),
        ("run_197", "dbt_test_suite",        "2024-02-01", "success", "2024-02-01 03:01:00", "2024-02-01 03:02:15", 75.0,  '{"suite": "engagement"}'),
        ("run_196", "data_quality_check",    "2024-02-01", "success", "2024-02-01 03:03:00", "2024-02-01 03:03:30", 30.0,  '{"check": "mau_bounds"}'),
    ])

    conn.commit()
    return conn

# Task 3

def build_task3_db():
    """
    Revenue for 'Cloud Storage' product line is overstated by exactly 3.7x.
    Root cause: product_promotions table has 3–4 rows per product (non-unique key).
    A JOIN on product_id multiplies every sale row by the number of promo rows.
    This only affects products launched after a schema change 6 weeks ago.
    The ratio 3.7 is suspicious — it's not a round number, which hints at fanout.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_base_tables(conn)
    cur = conn.cursor()

    # Product promotions — product_id is NOT unique here. This is the bug.
    # Old products have 1 row each (fine). New products have 3-4 rows (broken).
    cur.execute("""
        CREATE TABLE product_promotions (
            promo_id    INTEGER PRIMARY KEY,
            product_id  TEXT,   -- NOT unique — multiple promos per product
            promo_name  TEXT,
            discount    REAL,
            created_at  TEXT
        )
    """)

    cur.executemany("INSERT INTO products VALUES (?,?,?,?)", [
        ("P001", "Analytics Pro", "SaaS",  "2023-01-15"),
        ("P002", "Data Studio",   "SaaS",  "2023-03-10"),
        ("P003", "Cloud Storage", "Infra", "2023-11-20"),  # new — post schema change
        ("P004", "Edge Compute",  "Infra", "2023-11-25"),  # new — post schema change
    ])

    cur.executemany("INSERT INTO sales VALUES (?,?,?,?,?)", [
        (1,  "P001", 500.0, "USD", "2024-01-10"),
        (2,  "P001", 600.0, "USD", "2024-01-11"),
        (3,  "P002", 400.0, "USD", "2024-01-12"),
        (4,  "P003", 700.0, "USD", "2024-01-13"),  # these will be multiplied
        (5,  "P003", 800.0, "USD", "2024-01-14"),  # these will be multiplied
        (6,  "P004", 300.0, "USD", "2024-01-15"),  # these will be multiplied
    ])

    # Old products: 1 promo row each — no fanout
    # New products: 3-4 promo rows each — causes fanout on JOIN
    cur.executemany("INSERT INTO product_promotions VALUES (?,?,?,?,?)", [
        (1, "P001", "Early Bird",      0.10, "2023-02-01"),   # P001: 1 row — fine
        (2, "P002", "Spring Sale",     0.15, "2023-04-01"),   # P002: 1 row — fine
        (3, "P003", "Launch Promo",    0.20, "2023-11-20"),   # P003: row 1 of 4 — BUG
        (4, "P003", "Black Friday",    0.25, "2023-11-24"),   # P003: row 2 of 4
        (5, "P003", "Cyber Monday",    0.20, "2023-11-27"),   # P003: row 3 of 4
        (6, "P003", "Year End",        0.15, "2023-12-31"),   # P003: row 4 of 4
        (7, "P004", "Launch Promo",    0.20, "2023-11-25"),   # P004: row 1 of 3
        (8, "P004", "Partner Deal",    0.10, "2023-12-01"),   # P004: row 2 of 3
        (9, "P004", "Q1 Special",      0.05, "2024-01-01"),   # P004: row 3 of 3
    ])

    # dbt history
    cur.executemany("INSERT INTO dbt_log VALUES (?,?,?,?,?,?,?)", [
        (1,  "2024-01-16 03:00:00", "stg_sales",              "success", "Compiled successfully. Loaded 6 sales rows.",           2.1, 6),
        (2,  "2024-01-16 03:00:03", "stg_products",           "success", "Compiled successfully. Loaded 4 products.",             1.3, 4),
        (3,  "2024-01-16 03:00:05", "stg_product_promotions", "success", "Compiled successfully. Loaded 9 promo rows.",           1.8, 9),
        (4,  "2024-01-16 03:00:08", "int_revenue_with_promos","success", "JOIN sales × product_promotions complete. 15 result rows.", 4.5, 15),
        (5,  "2024-01-16 03:00:13", "fct_revenue_report",     "success", "Revenue report refreshed. Total: $6,600.",              3.9, 1),
        (6,  "2024-01-16 03:00:18", "test_revenue_positive",  "pass",    "All revenue values > 0.",                               0.8, 0),
        (7,  "2023-11-19 10:00:00", "product_promotions",     "success", "Schema updated: added multi-promo support per product. product_id is no longer unique in product_promotions.", 6.3, 0),
        (8,  "2023-11-19 10:05:00", "stg_product_promotions", "success", "Backfilled 2 promo rows for existing products.",        2.4, 2),
        (9,  "2024-01-15 03:00:00", "stg_sales",              "success", "Compiled successfully. Loaded 5 sales rows.",           2.0, 5),
        (10, "2024-01-15 03:00:05", "fct_revenue_report",     "success", "Revenue report refreshed.",                             3.7, 1),
        (11, "2024-01-14 12:00:00", "deprecation_check",      "warn",    "WARNING: product_promotions.discount column has mixed precision. Consider ROUND().", 0.6, 0),
        (12, "2024-01-13 03:00:00", "data_freshness_check",   "pass",    "All source tables refreshed within SLA.",               1.0, 0),
    ])

    # Airflow history
    cur.executemany("INSERT INTO airflow_runs VALUES (?,?,?,?,?,?,?,?)", [
        ("run_301", "etl_daily_pipeline",   "2024-01-16", "success", "2024-01-16 02:55:00", "2024-01-16 03:00:25", 325.0, '{"env": "prod"}'),
        ("run_300", "etl_daily_pipeline",   "2024-01-15", "success", "2024-01-15 02:55:00", "2024-01-15 03:00:12", 312.0, '{"env": "prod"}'),
        ("run_299", "etl_daily_pipeline",   "2024-01-14", "success", "2024-01-14 02:55:00", "2024-01-14 03:00:08", 308.0, '{"env": "prod"}'),
        ("run_250", "schema_migration",     "2023-11-19", "success", "2023-11-19 09:50:00", "2023-11-19 10:06:00", 960.0, '{"triggered_by": "product_team", "reason": "multi-promo support", "tables_affected": "product_promotions"}'),
        ("run_302", "dbt_test_suite",       "2024-01-16", "success", "2024-01-16 03:01:00", "2024-01-16 03:02:00", 60.0,  '{"suite": "revenue"}'),
        ("run_303", "data_quality_check",   "2024-01-16", "success", "2024-01-16 03:03:00", "2024-01-16 03:03:20", 20.0,  '{"check": "row_counts"}'),
    ])

    conn.commit()
    return conn

# Tool implementations
# These are the actual functions called when the agent uses a tool.

def run_sql(conn, query):
    """
    Run a SELECT query. Returns list of dicts.
    Anti-cheat:
      - Blocks destructive statements
      - Blocks SELECT * (must specify columns)
      - Caps results at MAX_RESULT_ROWS
    """
    q_stripped = query.strip()
    q_upper = q_stripped.upper()

    # Safety: only allow read operations
    for forbidden in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]:
        if q_upper.startswith(forbidden):
            return {"error": f"Tool only allows SELECT queries. Got: {forbidden}"}

    # Anti-cheat: block SELECT * — force agent to think about columns
    # Allow SELECT * only in subqueries, not as the main query
    if re.match(r'^\s*SELECT\s+\*\s+FROM', q_stripped, re.IGNORECASE):
        return {
            "error": "SELECT * is not allowed. Please specify the columns you need. "
                     "Example: SELECT column1, column2 FROM table_name"
        }

    try:
        cur = conn.cursor()
        cur.execute(q_stripped)
        rows = cur.fetchall()
        result = [dict(row) for row in rows]

        # Anti-cheat: cap results
        if len(result) > MAX_RESULT_ROWS:
            return {
                "rows": result[:MAX_RESULT_ROWS],
                "truncated": True,
                "total_rows": len(result),
                "message": f"Results capped at {MAX_RESULT_ROWS} rows. Use WHERE/LIMIT to narrow your query."
            }

        return result
    except Exception as e:
        return {"error": str(e)}

def inspect_schema(conn, table_name):
    """Return column names and types for a table."""
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        cols = cur.fetchall()
        if not cols:
            return {"error": f"Table '{table_name}' not found or not yet discovered. Use list_tables() first."}
        return {"table": table_name,
                "columns": [{"name": c["name"], "type": c["type"]} for c in cols]}
    except Exception as e:
        return {"error": str(e)}

def check_logs(conn):
    """Return the full dbt pipeline log, ordered by most recent first."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM dbt_log ORDER BY run_at DESC")
    return [dict(row) for row in cur.fetchall()]

def check_airflow(conn):
    """Return Airflow DAG run history, ordered by most recent first."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM airflow_runs ORDER BY start_date DESC")
    return [dict(row) for row in cur.fetchall()]

def diff_report(conn, date1, date2):
    """
    Compare total revenue between two dates.
    Returns both totals so agent can see the discrepancy.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT SUM(amount) as total FROM sales WHERE sale_date = ?", (date1,))
        r1 = cur.fetchone()
        cur.execute("SELECT SUM(amount) as total FROM sales WHERE sale_date = ?", (date2,))
        r2 = cur.fetchone()
        return {
            date1: r1["total"] if r1["total"] else 0,
            date2: r2["total"] if r2["total"] else 0,
        }
    except Exception as e:
        return {"error": str(e)}

def list_tables(conn):
    """List all tables in the database — useful first step for any agent."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row["name"] for row in cur.fetchall()]