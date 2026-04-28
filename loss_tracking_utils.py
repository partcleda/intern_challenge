import os
import sqlite3
from datetime import datetime


DB_DIRNAME = "loss_tracking"
DB_FILENAME_PREFIX = "loss_tracking"
BASE_LOSS_HISTORY_FIELDS = (
    "total_loss",
    "wirelength_loss",
    "overlap_loss",
    "learning_rate",
)
OPTIONAL_LOSS_HISTORY_FIELDS = (
    "overlap_count",
    "total_overlap_area",
    "max_overlap_area",
)
ALL_LOSS_HISTORY_FIELDS = BASE_LOSS_HISTORY_FIELDS + OPTIONAL_LOSS_HISTORY_FIELDS


def get_expected_loss_history_fields(track_overlap_metrics=False):
    """Return the expected loss-history series for a run configuration."""
    if track_overlap_metrics:
        return ALL_LOSS_HISTORY_FIELDS
    return BASE_LOSS_HISTORY_FIELDS


def create_loss_history(run_metadata=None, track_overlap_metrics=False):
    """Create a loss-history container with the expected series predeclared."""
    loss_history = {"run_metadata": dict(run_metadata or {})}
    for field_name in get_expected_loss_history_fields(track_overlap_metrics):
        loss_history[field_name] = []
    return loss_history


def get_loss_tracking_db_dir(output_dir):
    """Return the directory that stores loss-tracking SQLite files."""
    return os.path.join(output_dir, DB_DIRNAME)


def create_loss_tracking_db(output_dir):
    """Create a new SQLite database for a single placement/test invocation."""
    db_dir = get_loss_tracking_db_dir(output_dir)
    os.makedirs(db_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    db_path = os.path.join(db_dir, f"{DB_FILENAME_PREFIX}_{timestamp}.sqlite3")

    connection = _connect_db(db_path)
    try:
        _initialize_schema(connection)
        connection.commit()
    finally:
        connection.close()

    return db_path


def _connect_db(db_path):
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _initialize_schema(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS test_cases (
            test_id INTEGER PRIMARY KEY,
            num_macros INTEGER,
            num_std_cells INTEGER,
            seed INTEGER,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            test_id INTEGER REFERENCES test_cases(test_id) ON DELETE SET NULL,
            runner TEXT,
            run_label TEXT,
            run_started_at TEXT,
            saved_at TEXT NOT NULL,
            seed INTEGER,
            num_macros INTEGER,
            num_std_cells INTEGER,
            num_epochs INTEGER,
            lr REAL,
            lambda_wirelength REAL,
            lambda_overlap REAL,
            log_interval INTEGER,
            verbose INTEGER,
            total_cells INTEGER,
            total_pins INTEGER,
            total_edges INTEGER
        );

        CREATE TABLE IF NOT EXISTS loss_history (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            epoch INTEGER NOT NULL,
            total_loss REAL,
            wirelength_loss REAL,
            overlap_loss REAL,
            learning_rate REAL,
            overlap_count INTEGER,
            total_overlap_area REAL,
            max_overlap_area REAL,
            PRIMARY KEY (run_id, epoch)
        );
        """
    )
    _ensure_columns(
        connection,
        "runs",
        {
            "seed": "INTEGER",
            "num_macros": "INTEGER",
            "num_std_cells": "INTEGER",
        },
    )
    _ensure_columns(
        connection,
        "loss_history",
        {
            "learning_rate": "REAL",
            "overlap_count": "INTEGER",
            "total_overlap_area": "REAL",
            "max_overlap_area": "REAL",
        },
    )


def _ensure_columns(connection, table_name, columns):
    existing_columns = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table_name})")
    }
    for column_name, column_type in columns.items():
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            )


def _sqlite_scalar(value):
    """Convert torch/numpy scalar-like values to sqlite-friendly Python scalars."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, int, float)):
        return value
    if isinstance(value, bool):
        return int(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def save_loss_history_sqlite(loss_history, db_path, run_metadata=None):
    """Save loss history values to a normalized SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    metadata = dict(loss_history.get("run_metadata", {}))
    if run_metadata:
        metadata.update(run_metadata)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metadata.setdefault("run_id", timestamp)
    metadata.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))

    history_series = {
        field_name: loss_history.get(field_name, [])
        for field_name in ALL_LOSS_HISTORY_FIELDS
    }

    row_count = max(
        (len(values) for values in history_series.values()),
        default=0,
    )

    connection = _connect_db(db_path)
    try:
        _initialize_schema(connection)

        test_id = metadata.get("test_id")
        if test_id is not None:
            connection.execute(
                """
                INSERT INTO test_cases (
                    test_id,
                    num_macros,
                    num_std_cells,
                    seed,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(test_id) DO UPDATE SET
                    num_macros = excluded.num_macros,
                    num_std_cells = excluded.num_std_cells,
                    seed = excluded.seed,
                    updated_at = excluded.updated_at
                """,
                (
                    int(test_id),
                    metadata.get("num_macros"),
                    metadata.get("num_std_cells"),
                    metadata.get("seed"),
                    metadata["saved_at"],
                ),
            )

        connection.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id,
                test_id,
                runner,
                run_label,
                run_started_at,
                saved_at,
                seed,
                num_macros,
                num_std_cells,
                num_epochs,
                lr,
                lambda_wirelength,
                lambda_overlap,
                log_interval,
                verbose,
                total_cells,
                total_pins,
                total_edges
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata["run_id"],
                int(test_id) if test_id is not None else None,
                metadata.get("runner"),
                metadata.get("run_label"),
                metadata.get("run_started_at"),
                metadata["saved_at"],
                metadata.get("seed"),
                metadata.get("num_macros"),
                metadata.get("num_std_cells"),
                metadata.get("num_epochs"),
                metadata.get("lr"),
                metadata.get("lambda_wirelength"),
                metadata.get("lambda_overlap"),
                metadata.get("log_interval"),
                int(bool(metadata.get("verbose")))
                if metadata.get("verbose") is not None
                else None,
                metadata.get("total_cells"),
                metadata.get("total_pins"),
                metadata.get("total_edges"),
            ),
        )

        connection.execute(
            "DELETE FROM loss_history WHERE run_id = ?",
            (metadata["run_id"],),
        )

        history_rows = []
        for epoch in range(row_count):
            history_rows.append(
                (
                    metadata["run_id"],
                    epoch,
                    *(
                        _sqlite_scalar(
                            history_series[field_name][epoch]
                            if epoch < len(history_series[field_name])
                            else None
                        )
                        for field_name in ALL_LOSS_HISTORY_FIELDS
                    ),
                )
            )

        connection.executemany(
            """
            INSERT INTO loss_history (
                run_id,
                epoch,
                total_loss,
                wirelength_loss,
                overlap_loss,
                learning_rate,
                overlap_count,
                total_overlap_area,
                max_overlap_area
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            history_rows,
        )
        connection.commit()
    finally:
        connection.close()

    return db_path
