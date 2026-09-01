"""Test bootstrap: import paths, and a database that is never the real one.

The DB redirect has to happen *here*, before any `verityne` module is imported,
because `db.py` builds its engine at import time from `VERITYNE_DB`. Setting the
variable inside a fixture is too late - the engine already exists, still pointed
at `storage/verityne.db`, and any test that drives the API through a TestClient
writes into the developer's working database. That is not hypothetical: it is
what these tests did before this file redirected them, and the symptom was a
fraud-ring assertion failing because it had found the *previous* run's rings.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

_TEST_DB_DIR = tempfile.mkdtemp(prefix="verityne-tests-")
os.environ.setdefault("VERITYNE_DB", f"sqlite:///{_TEST_DB_DIR}/test.db")
