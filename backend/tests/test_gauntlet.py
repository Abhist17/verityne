"""The Gauntlet's fixture selection and its scoreboard.

The Gauntlet is the demo's headline number, which makes it the one place where a
misreport is most expensive: nobody re-derives a scoreboard they were shown. Two
things have to hold. The fixtures must not manufacture the failures they then
count, and the summary must not describe a verdict as something it was not.

Same rule as the rest of the suite: no model output is asserted on here. Whether
a given packet scores 0.45 belongs in `eval/metrics.json`.
"""

import random

from seed_gauntlet import pick_disjoint, pick_spread
from verityne.api.routes_gauntlet import _summarise
from verityne.schemas import GauntletResult


def packet(pid: str, label: str, identity: int, attack: str | None = None) -> dict:
    return {"id": pid, "label": label, "identity_index": identity, "attack_type": attack}


def result(name: str, truth: str, verdict: str) -> GauntletResult:
    correct = (verdict in ("REJECT", "REVIEW")) if truth == "fake" else (verdict == "PASS")
    return GauntletResult(
        submission_id=name, name=name, truth=truth, verdict=verdict, score=0.5, correct=correct
    )


# ---------------------------------------------------------------------------------
# Fixture selection
# ---------------------------------------------------------------------------------

class TestPickDisjoint:
    def test_never_reuses_an_identity_the_fake_half_took(self):
        reals = [packet(f"r{i}", "real", i) for i in range(20)]
        rng = random.Random(0)
        picked = pick_disjoint(reals, 10, {0, 1, 2, 3, 4}, rng)
        assert len(picked) == 10
        assert not {e["identity_index"] for e in picked} & {0, 1, 2, 3, 4}

    def test_falls_back_rather_than_returning_short(self):
        # Only three identities are free, but the demo asked for five. Returning
        # three would silently shrink the genuine half and flatter every rate
        # computed over it.
        reals = [packet(f"r{i}", "real", i) for i in range(8)]
        picked = pick_disjoint(reals, 5, {0, 1, 2, 3, 4}, random.Random(0))
        assert len(picked) == 5
        free = [e for e in picked if e["identity_index"] not in {0, 1, 2, 3, 4}]
        assert len(free) == 3  # every disjoint packet is used before any overlapping one

    def test_selection_is_reproducible_for_a_seed(self):
        reals = [packet(f"r{i}", "real", i) for i in range(20)]
        a = pick_disjoint(reals, 6, {0}, random.Random(7))
        b = pick_disjoint(reals, 6, {0}, random.Random(7))
        assert [e["id"] for e in a] == [e["id"] for e in b]

    def test_a_genuine_and_a_fraudulent_fixture_never_share_a_face(self):
        """The regression this function exists for.

        Drawing both halves independently let one person appear as a genuine
        merchant and, under another name, as a fraud fixture. Linkage then
        reported an onboarding ring - correctly, on that evidence - and the
        scoreboard counted it against the detectors as a false reject.
        """
        pool = [packet(f"r{i}", "real", i) for i in range(30)]
        fakes = [packet(f"f{i}", "fake", i, f"attack_{i % 4}") for i in range(30)]
        rng = random.Random(7)
        picked_fakes = pick_spread(fakes, 10, rng)
        picked_reals = pick_disjoint(pool, 10, {e["identity_index"] for e in picked_fakes}, rng)
        assert not (
            {e["identity_index"] for e in picked_reals} & {e["identity_index"] for e in picked_fakes}
        )


class TestPickSpread:
    def test_covers_every_attack_type_before_repeating_one(self):
        fakes = [packet(f"f{i}", "fake", i, f"attack_{i % 4}") for i in range(20)]
        picked = pick_spread(fakes, 4, random.Random(3))
        assert len({e["attack_type"] for e in picked}) == 4


# ---------------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------------

class TestSummary:
    def test_a_reviewed_fake_counts_as_caught(self):
        s = _summarise([result("FAKE-01", "fake", "REVIEW")])
        assert s.fakes_caught == 1 and s.detection_rate == 1.0 and s.false_accept_rate == 0.0

    def test_a_reviewed_genuine_is_reported_as_reviewed_not_rejected(self):
        s = _summarise([result("REAL-01", "real", "REVIEW"), result("REAL-02", "real", "PASS")])
        assert s.reals_reviewed == 1
        assert s.reals_rejected == 0
        # It still counts against the rate - review is friction - but the split
        # is what stops the headline reading as "blocked" when nothing was.
        assert s.false_reject_rate == 0.5

    def test_an_auto_rejected_genuine_is_reported_separately(self):
        s = _summarise([result("REAL-01", "real", "REJECT"), result("REAL-02", "real", "PASS")])
        assert s.reals_rejected == 1 and s.reals_reviewed == 0 and s.false_reject_rate == 0.5

    def test_the_review_and_reject_counts_account_for_every_genuine_miss(self):
        rs = [result(f"REAL-0{i}", "real", v) for i, v in enumerate(["PASS", "REVIEW", "REVIEW", "REJECT"])]
        s = _summarise(rs)
        assert s.reals_passed + s.reals_reviewed + s.reals_rejected == s.reals_total

    def test_an_empty_run_does_not_divide_by_zero(self):
        s = _summarise([])
        assert s.total == 0 and s.detection_rate == 0.0 and s.false_reject_rate == 0.0
