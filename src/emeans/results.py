"""Append-only CSV results with resume support."""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


class ResultsCSV:
    """One CSV per stage. A run is identified by its `key` columns; runs whose key is already
    present are skipped, so an interrupted stage can be restarted. Rows must always have the
    same columns as the file's header; a mismatch (e.g. after changing the code) is an error
    instead of a silently corrupted file."""

    def __init__(self, name, key_columns):
        self.path = RESULTS_DIR / f"{name}.csv"
        self.key_columns = list(key_columns)
        self.header = None
        self.done = set()
        if self.path.exists():
            with self.path.open() as f:
                reader = csv.DictReader(f)
                self.header = reader.fieldnames
                for row in reader:
                    self.done.add(self._key(row))

    def _key(self, row):
        return tuple(str(row[c]) for c in self.key_columns)

    def has(self, config):
        return self._key(config) in self.done

    def append(self, rows):
        """Write rows (dicts with identical columns) for one run, then mark it done."""
        if not rows:
            return
        columns = list(rows[0])
        if self.header is not None and columns != self.header:
            raise ValueError(f"{self.path}: columns {columns} do not match the header {self.header}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            if self.header is None:
                w.writeheader()
                self.header = columns
            w.writerows(rows)
        self.done.add(self._key(rows[0]))
