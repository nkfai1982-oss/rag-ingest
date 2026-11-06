"""SQLite database connection provider configured with WAL mode and foreign keys."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


class SQLiteConnection:
    """Manages SQLite connections with WAL mode and foreign key enforcement."""

    def __init__(self, db_path: str | Path = "data/knowledge.db") -> None:
        self.db_path = str(db_path)
        self.is_memory = self.db_path == ":memory:" or "mode=memory" in self.db_path
        self._shared_conn: sqlite3.Connection | None = None

        if not self.is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            # Keep a persistent connection open for in-memory database
            self._shared_conn = sqlite3.connect(
                ":memory:",
                check_same_thread=False,
                isolation_level=None,
            )
            self._shared_conn.row_factory = sqlite3.Row
            cursor = self._shared_conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.close()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing a configured SQLite connection."""
        if self.is_memory and self._shared_conn is not None:
            yield self._shared_conn
            return

        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,
            isolation_level=None,  # Autocommit mode; transactions managed explicitly
        )
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.close()
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        """Close shared connection if open."""
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None
