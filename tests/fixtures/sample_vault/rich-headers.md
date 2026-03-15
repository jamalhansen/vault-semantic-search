---
title: SQL for Python Developers
tags: [sql, python, pandas]
category: tutorial
status: published
date: 2026-01-10
---

# SQL for Python Developers

This series covers SQL concepts through the lens of Python and pandas.

## NULL Values

NULL handling is one of the most common sources of bugs when moving between SQL and Python.

### Python Comparison

In pandas, missing values are represented as NaN or None. SQL uses NULL, which behaves differently in comparisons. In Python you can write `x == None`, but in SQL you must use `IS NULL` — writing `= NULL` always returns false.

When joining dataframes, pandas will naturally exclude rows where the join key is NaN. SQL behaves the same way with NULL keys: they don't match anything, not even each other.

### SQL Behavior

The three-valued logic of SQL means a comparison involving NULL yields UNKNOWN, not TRUE or FALSE. This affects WHERE clauses:

```sql
-- This silently excludes rows where status IS NULL
WHERE status != 'active'

-- Correct way to include NULLs
WHERE status != 'active' OR status IS NULL
```

## Window Functions

Window functions let you perform calculations across related rows without collapsing them like GROUP BY does.

### Basic Syntax

```sql
SELECT
  name,
  salary,
  AVG(salary) OVER (PARTITION BY department) as dept_avg
FROM employees;
```

### Running Totals

Use `SUM(...) OVER (ORDER BY date)` to compute a running total. In pandas the equivalent is `df['amount'].cumsum()`.
