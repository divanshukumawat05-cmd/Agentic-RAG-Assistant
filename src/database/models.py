"""Verify the interns SQLite database and table contents."""

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
        RuntimeError: If the connection cannot be opened.
    """

    if not database_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {database_path}")

    try:
        return sqlite3.connect(database_path)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to connect to database: {database_path}") from exc


def count_records(connection: sqlite3.Connection) -> int:
    """Count records in the interns table and print the total.

    Args:
        connection: Active SQLite connection.

    Returns:
        Total number of records in the interns table.

    Raises:
        RuntimeError: If the count query fails.
    """

    try:
        cursor = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        count = int(cursor.fetchone()[0])
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to count records in table: {TABLE_NAME}") from exc

    print(f"Total Records: {count}")
    return count


def show_first_records(connection: sqlite3.Connection, limit: int = 5) -> pd.DataFrame:
    """Fetch and print the first records from the interns table.

    Args:
        connection: Active SQLite connection.
        limit: Number of rows to fetch.

    Returns:
        DataFrame containing the first records.

    Raises:
        ValueError: If ``limit`` is less than 1.
        RuntimeError: If the select query fails.
    """

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    try:
        dataframe = pd.read_sql_query(
            f"SELECT * FROM {TABLE_NAME} LIMIT ?;",
            connection,
            params=(limit,),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch records from table: {TABLE_NAME}"
        ) from exc

    print(dataframe)
    return dataframe


def main() -> None:
    """Run database verification checks."""

    connection: sqlite3.Connection | None = None

    try:
        print("Connecting to database...")
        connection = connect_database()

        count_records(connection)

        print("First 5 Records:")
        show_first_records(connection)

        print("Completed successfully.")
    except Exception as exc:
        print(f"Database verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
