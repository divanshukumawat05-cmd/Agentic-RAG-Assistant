"""Import the interns CSV dataset into a SQLite database."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "sample_db" / "interns.csv"
DATABASE_PATH = PROJECT_ROOT / "data" / "sample_db" / "interns.db"
TABLE_NAME = "interns"


def load_csv(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load internship records from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame containing the CSV records.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV file contains no rows.
        RuntimeError: If pandas cannot read the CSV file.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        dataframe = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file: {csv_path}") from exc

    if dataframe.empty:
        raise ValueError(f"CSV file contains no records: {csv_path}")

    return dataframe


def create_database(database_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Create or open the SQLite database.

    Args:
        database_path: Path where the SQLite database should be created.

    Returns:
        Open SQLite connection.

    Raises:
        RuntimeError: If the database connection cannot be created.
    """

    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(database_path)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to create SQLite database: {database_path}") from exc


def import_to_sqlite(
    dataframe: pd.DataFrame,
    connection: sqlite3.Connection,
    table_name: str = TABLE_NAME,
) -> None:
    """Import records into SQLite, replacing the table if it already exists.

    Args:
        dataframe: DataFrame containing records to import.
        connection: Active SQLite connection.
        table_name: Target table name.

    Raises:
        RuntimeError: If the import fails.
    """

    try:
        dataframe.to_sql(table_name, connection, if_exists="replace", index=False)
        connection.commit()
    except Exception as exc:
        connection.rollback()
        raise RuntimeError(f"Failed to import records into table: {table_name}") from exc


def main() -> None:
    """Run the CSV-to-SQLite import pipeline."""

    connection: sqlite3.Connection | None = None

    try:
        print("Loading CSV...")
        dataframe = load_csv()

        print("Creating SQLite database...")
        connection = create_database()

        print("Importing records...")
        import_to_sqlite(dataframe, connection)

        print("Completed successfully.")
    except Exception as exc:
        print(f"SQLite import failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
