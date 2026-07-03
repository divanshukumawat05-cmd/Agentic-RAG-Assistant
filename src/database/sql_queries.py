"""Reusable SQL query helpers for the Project Agent."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "sample_db" / "interns.db"
TABLE_NAME = "interns"


def connect_database(database_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Connect to the interns SQLite database.

    Args:
        database_path: Path to the SQLite database file.

    Returns:
        Active SQLite connection.

    Raises:
        FileNotFoundError: If the database file does not exist.
        RuntimeError: If SQLite cannot open the database.
    """

    if not database_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {database_path}")

    try:
        return sqlite3.connect(database_path)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to connect to database: {database_path}") from exc


def _run_query(
    query: str,
    params: tuple[object, ...] = (),
    database_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """Run a parameterized SQL query and return the results as a DataFrame.

    Args:
        query: SQL query with parameter placeholders.
        params: Query parameters.
        database_path: Path to the SQLite database file.

    Returns:
        Query results as a pandas DataFrame.

    Raises:
        RuntimeError: If query execution fails.
    """

    try:
        with connect_database(database_path) as connection:
            return pd.read_sql_query(query, connection, params=params)
    except Exception as exc:
        raise RuntimeError("Failed to execute SQL query.") from exc


def show_all_interns() -> pd.DataFrame:
    """Return all intern records.

    Returns:
        DataFrame containing all rows from the interns table.
    """

    return _run_query(f"SELECT * FROM {TABLE_NAME};")


def find_intern_by_name(name: str) -> pd.DataFrame:
    """Find interns by partial name match.

    Args:
        name: Full or partial intern name.

    Returns:
        DataFrame containing matching intern records.

    Raises:
        ValueError: If ``name`` is empty.
    """

    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("name must not be empty.")

    return _run_query(
        f"SELECT * FROM {TABLE_NAME} WHERE Name LIKE ?;",
        (f"%{cleaned_name}%",),
    )


def find_by_project(project: str) -> pd.DataFrame:
    """Find interns by partial project name match.

    Args:
        project: Full or partial project name.

    Returns:
        DataFrame containing matching intern records.

    Raises:
        ValueError: If ``project`` is empty.
    """

    cleaned_project = project.strip()
    if not cleaned_project:
        raise ValueError("project must not be empty.")

    return _run_query(
        f"SELECT * FROM {TABLE_NAME} WHERE Project LIKE ?;",
        (f"%{cleaned_project}%",),
    )


def find_by_mentor(mentor: str) -> pd.DataFrame:
    """Find interns by partial mentor name match.

    Args:
        mentor: Full or partial mentor name.

    Returns:
        DataFrame containing matching intern records.

    Raises:
        ValueError: If ``mentor`` is empty.
    """

    cleaned_mentor = mentor.strip()
    if not cleaned_mentor:
        raise ValueError("mentor must not be empty.")

    return _run_query(
        f"SELECT * FROM {TABLE_NAME} WHERE Mentor LIKE ?;",
        (f"%{cleaned_mentor}%",),
    )


def find_by_status(status: str) -> pd.DataFrame:
    """Find interns by exact status match.

    Args:
        status: Exact intern status.

    Returns:
        DataFrame containing matching intern records.

    Raises:
        ValueError: If ``status`` is empty.
    """

    cleaned_status = status.strip()
    if not cleaned_status:
        raise ValueError("status must not be empty.")

    return _run_query(
        f"SELECT * FROM {TABLE_NAME} WHERE Status = ?;",
        (cleaned_status,),
    )


def main() -> None:
    """Run demonstration queries for the Project Agent query layer."""

    try:
        print("All Interns:")
        print(show_all_interns())
        print()

        print('Find Intern By Name: "Aarav"')
        print(find_intern_by_name("Aarav"))
        print()

        print('Find By Project: "Knowledge Graph"')
        print(find_by_project("Knowledge Graph"))
        print()

        print('Find By Mentor: "Neha"')
        print(find_by_mentor("Neha"))
        print()

        print('Find By Status: "Completed"')
        print(find_by_status("Completed"))
    except Exception as exc:
        print(f"SQL query demo failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
