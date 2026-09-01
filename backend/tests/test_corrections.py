"""The audit trail must resolve to the evidence it cites.

This page exists to be checked, so the tests are about verifiability rather than
content: every entry names a report and a path, and every one of those has to
resolve in the file on disk. An entry that drifted from its evidence would be
the exact failure the page was built to document, delivered by the page itself.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval"
CORRECTIONS = EVAL / "corrections.json"

pytestmark = pytest.mark.skipif(
    not CORRECTIONS.exists(), reason="run `make corrections` to assemble the audit trail"
)


def load():
    return json.loads(CORRECTIONS.read_text())


def dig(data, path):
    parts = path if isinstance(path, list) else str(path).split(".")
    for part in parts:
        data = data[int(part)] if isinstance(data, (list, tuple)) else data[part]
    return data


def corrections():
    return load()["corrections"]


class TestEvidenceResolves:
    def test_every_cited_report_exists(self):
        for c in corrections():
            for ev in c["evidence"]:
                if ev.get("source") == "code":
                    continue
                assert (EVAL / ev["file"]).exists(), (
                    f"{c['id']} cites eval/{ev['file']}, which is not on disk"
                )

    def test_every_cited_path_resolves(self):
        """The claim is that these numbers come from the reports. Check it."""
        for c in corrections():
            for ev in c["evidence"]:
                if ev.get("source") == "code" or "path" not in ev:
                    continue
                data = json.loads((EVAL / ev["file"]).read_text())
                try:
                    dig(data, ev["path"])
                except (KeyError, IndexError, TypeError) as exc:
                    pytest.fail(f"{c['id']}: eval/{ev['file']} path {ev['path']} broke ({exc})")

    def test_a_cited_symbol_exists_in_the_file_it_names(self):
        for c in corrections():
            for ev in c["evidence"]:
                if ev.get("source") != "code":
                    continue
                path = REPO / ev["file"]
                assert path.exists(), f"{c['id']} cites {ev['file']}, which does not exist"
                assert ev["symbol"] in path.read_text(), (
                    f"{c['id']} cites {ev['file']}::{ev['symbol']}, which is not in it"
                )

    def test_an_entry_backed_only_by_code_says_so(self):
        """One figure was never written to a report. It must not be presented as
        though a report backed it."""
        for c in corrections():
            for ev in c["evidence"]:
                if ev.get("source") == "code":
                    assert ev.get("note"), (
                        f"{c['id']} rests on code rather than a report and does not say why"
                    )


class TestShape:
    def test_counts_match_the_entries(self):
        d = load()
        actual = {}
        for c in d["corrections"]:
            actual[c["status"]] = actual.get(c["status"], 0) + 1
        assert d["counts"] == actual
        assert d["n"] == len(d["corrections"])

    def test_orders_are_contiguous(self):
        assert [c["order"] for c in corrections()] == list(range(1, len(corrections()) + 1))

    def test_ids_are_unique(self):
        ids = [c["id"] for c in corrections()]
        assert len(set(ids)) == len(ids)

    def test_every_entry_carries_a_number(self):
        """A correction without a figure is an anecdote."""
        for c in corrections():
            has_delta = "before" in c and "after" in c
            has_series = bool(c.get("series"))
            assert has_delta or has_series, f"{c['id']} states no measurement"

    def test_every_how_found_is_in_the_legend(self):
        legend = load()["how_found_legend"]
        for c in corrections():
            assert c["how_found"] in legend, f"{c['id']} has an unexplained discovery method"

    def test_doc_anchors_exist(self):
        """Each entry deep-links at the section documenting it; a dead anchor is a dead link.

        `readme` is a repo-relative `path.md#anchor` because the prose lives in
        `docs/` now. Both halves are checked: the file has to exist and the
        heading has to be in *that* file. One of these was dead for real —
        `#the-thresholds-and-where-they-came-from` names a heading this project
        has never had — and the previous version of this test could not see it,
        because it resolved every anchor against one document.
        """
        import re

        def slug(h: str) -> str:
            s = re.sub(r"[^\w\s-]", "", h.strip().lower())
            return re.sub(r"\s", "-", s)

        cache: dict[str, set] = {}

        def headings(rel: str) -> set:
            if rel not in cache:
                p = REPO / rel
                assert p.exists(), f"{rel} is cited by a correction but is not on disk"
                cache[rel] = {slug(m) for m in re.findall(r"^#{1,6}\s+(.*)$", p.read_text(), re.M)}
            return cache[rel]

        for c in corrections():
            target, _, anchor = c["readme"].partition("#")
            rel = target or "README.md"
            assert anchor in headings(rel), (
                f"{c['id']} links to {rel}#{anchor}, which no heading in that file produces"
            )


class TestHonesty:
    """The page's value is entirely in what it admits. Pin that."""

    def test_open_findings_are_still_reported(self):
        statuses = {c["status"] for c in corrections()}
        assert "open" in statuses, (
            "no correction is marked open. If everything really is fixed, say so deliberately "
            "rather than letting this test pass by accident"
        )

    def test_the_third_party_finding_is_reported_as_open(self):
        """The selfie detector is at or below chance on fakes we did not generate.
        If a refit ever fixes that, this failing is the signal to rewrite the
        entry — not to quietly flip a status."""
        c = next((x for x in corrections() if x["id"] == "third-party-fakes"), None)
        assert c is not None, "the third-party generator finding is missing"
        assert c["status"] == "open"
        worst = min(s["value"] for s in c["series"])
        assert worst < 0.55, f"worst family is now {worst}; the entry still calls it chance"

    def test_the_replay_attack_is_still_unbeaten(self):
        c = next((x for x in corrections() if x["id"] == "replay-attack"), None)
        assert c is not None and c["status"] == "open"
        assert c["after"] < 0.65, f"replay now scores {c['after']}; the entry says chance"

    def test_fixed_entries_moved_their_number(self):
        for c in corrections():
            if c["status"] != "fixed" or "before" not in c:
                continue
            assert c["before"] != c["after"], f"{c['id']} claims a fix that moved nothing"
