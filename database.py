# database.py
# Builds a synthetic in-memory SQLite database that looks like a real data stack.
# Each task gets its own database connection with a specific bug planted in it.
#
# The 3 bugs:
#   Task 1 (easy)   — NULL JOIN: currency table uses "usd" but fact table uses "USD"
#   Task 2 (medium) — TIMEZONE GHOST: events double-counted at month boundary
#   Task 3 (hard)   — FANOUT: joining on a non-unique key inflates revenue by 3.7x

import sqlite3

# ── Shared table creation helpers ─────────────────────────────────────────────

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

    # DBT run log — simulates a real pipeline changelog
    cur.execute("""
        CREATE TABLE dbt_log (
            run_id      INTEGER PRIMARY KEY,
            run_at      TEXT,
            model_name  TEXT,
            status      TEXT,
            message     TEXT
        )
    """)

    conn.commit()

# ── Task 1: NULL JOIN bug ─────────────────────────────────────────────────────

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

    # Currency rates — LOWERCASE codes (mismatched with sales table — the actual bug)
    cur.executemany("INSERT INTO currency_rates VALUES (?,?)", [
        ("usd", 1.0),
        ("eur", 1.08),
        ("gbp", 1.27),
    ])

    # DBT log shows the pipeline ran fine — no errors logged (makes bug harder to spot)
    cur.executemany("INSERT INTO dbt_log VALUES (?,?,?,?,?)", [
        (1, "2024-01-16 03:00:00", "revenue_report", "success", "Completed in 4.2s"),
        (2, "2024-01-15 03:00:00", "revenue_report", "success", "Completed in 3.8s"),
    ])

    conn.commit()
    return conn

# ── Task 2: TIMEZONE GHOST bug ────────────────────────────────────────────────

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

    # DBT log — this is the smoking gun. Agent must find this entry.
    cur.executemany("INSERT INTO dbt_log VALUES (?,?,?,?,?)", [
        (1, "2024-01-31 22:00:00", "user_events_etl", "success",
            "Migrated timestamp handling from UTC to local timezone"),
        (2, "2024-02-01 03:00:00", "mau_report",      "success", "Completed in 5.1s"),
        (3, "2024-01-31 03:00:00", "mau_report",      "success", "Completed in 4.9s"),
    ])

    conn.commit()
    return conn

# ── Task 3: FANOUT bug ────────────────────────────────────────────────────────

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

    cur.executemany("INSERT INTO dbt_log VALUES (?,?,?,?,?)", [
        (1, "2023-11-19 10:00:00", "product_promotions", "success",
            "Schema updated: added multi-promo support per product"),
        (2, "2024-01-16 03:00:00", "revenue_report", "success", "Completed in 6.3s"),
    ])

    conn.commit()
    return conn

# ── Tool implementations ───────────────────────────────────────────────────────
# These are the actual functions called when the agent uses a tool.

def run_sql(conn, query):
    """Run a SELECT query. Returns list of dicts. Blocks destructive statements."""
    q = query.strip().upper()
    # Safety: only allow read operations
    for forbidden in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]:
        if q.startswith(forbidden):
            return {"error": f"Tool only allows SELECT queries. Got: {forbidden}"}
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        # sqlite3.Row supports dict conversion
        return [dict(row) for row in rows]
    except Exception as e:
        return {"error": str(e)}

def inspect_schema(conn, table_name):
    """Return column names and types for a table."""
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        cols = cur.fetchall()
        if not cols:
            return {"error": f"Table '{table_name}' not found"}
        return {"table": table_name,
                "columns": [{"name": c["name"], "type": c["type"]} for c in cols]}
    except Exception as e:
        return {"error": str(e)}

def check_logs(conn):
    """Return the full dbt pipeline log."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM dbt_log ORDER BY run_at DESC")
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