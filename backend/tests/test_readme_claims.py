"""The README's headline numbers must equal the evidence files beside them.

This project's claim is that every number in the README is reproducible from
`eval/*.json`. Nothing enforced that, and it had already drifted: the hook said
0.914 while the Results table three hundred lines below said 0.911, from
different runs, in the same file. Nobody noticed, because checking meant reading
a thousand lines against six JSON documents by hand.

So each claim below names where in the README it lives and which file it must
agree with. No expected value is written here - the test reads both sides and
compares - so it cannot rot into asserting a number that used to be true.

A regenerated pipeline that moves a number now fails this test until the README
is updated, which is the entire point.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
README = (REPO / "README.md").read_text()
EVAL = REPO / "eval"


def load(name: str):
    p = EVAL / name
    if not p.exists():
        pytest.skip(f"{name} not present — run `make pipeline`")
    return json.loads(p.read_text())


def dig(data, path: str):
    for part in path.split("."):
        data = data[int(part)] if part.isdigit() else data[part]
    return data


def readme_number(pattern: str) -> float:
    """The single number the README states at `pattern`.

    Requires exactly one match: a claim stated twice in two different forms is
    itself the failure mode this file exists to catch.
    """
    found = re.findall(pattern, README)
    assert found, f"README no longer states this claim (pattern: {pattern})"
    unique = {f if isinstance(f, str) else f[0] for f in found}
    assert len(unique) == 1, f"README states conflicting values {sorted(unique)} for {pattern}"
    return float(next(iter(unique)))


#: (description, README regex capturing the number, evidence file, dotted path)
#: with an optional fifth element: the factor the README's units differ by, for
#: rows the prose states as a percentage while the report stores a fraction.
CLAIMS = [
    ("hook: held-out fusion AUC",
     r"Held-out ROC-AUC \*\*([\d.]+)\*\*", "metrics.json", "fusion.roc_auc"),
    ("Results table: fusion AUC",
     r"\| Fusion ROC-AUC \| \*\*([\d.]+)\*\* \|", "metrics.json", "fusion.roc_auc"),
    ("Results table: mean score on genuine",
     r"\| Mean score, genuine \| ([\d.]+) \|", "metrics.json", "fusion.score_mean_genuine"),
    ("Results table: mean score on fraud",
     r"\| Mean score, fraud \| ([\d.]+) \|", "metrics.json", "fusion.score_mean_fraud"),
    # This row sits in the section reporting the *finding*, so it is quoted from
    # the run that found it - the leaked corpus - not from the current one.
    ("ablation: full model AUC as the leak was found",
     r"\| \*\(none — full model\)\* \| ([\d.]+) \|", "ablation_leaked_corpus.json", "full_model.roc_auc"),
    ("linkage: true-accept rate at the fitted search threshold",
     r"true-accept rate from 95.1% to \*\*([\d.]+)%\*\*", "linkage_lfw.json", "fitted.tar", 100),
    ("demography: shortcut AUC of the frequency head, controlled",
     r"\| — the frequency head, fitted here \| [\d.]+ \| \*\*([\d.]+)\*\* \|",
     "indian_faces.json", "face.shortcut_auc.spectral_p_fake"),
    ("demography: shortcut AUC of the pretrained CNN, controlled",
     r"\| — the pretrained CNN alone \| [\d.]+ \| \*\*([\d.]+)\*\* \|",
     "indian_faces.json", "face.shortcut_auc.cnn_p_fake"),
    ("detector 6: held-out AUC",
     r"\| Held-out ROC-AUC, subject-disjoint \| \*\*([\d.]+)\*\* \|",
     "behavioral.json", "headline.held_out_auc"),
    ("detector 6: the strategy that defeats it",
     r"\| Worst unseen strategy \| \*\*`\w+` at ([\d.]+)\*\* \|",
     "behavioral.json", "headline.worst_unseen_strategy_auc"),
]


@pytest.mark.parametrize("claim", CLAIMS, ids=[c[0] for c in CLAIMS])
def test_readme_number_matches_its_evidence(claim):
    desc, pattern, filename, path = claim[:4]
    scale = claim[4] if len(claim) > 4 else 1
    stated = readme_number(pattern)
    actual = float(dig(load(filename), path)) * scale
    assert stated == pytest.approx(actual, abs=0.0006 * scale), (
        f"README says {stated} for {desc}; {filename}:{path} says {actual}. "
        f"Regenerate the README section or the evidence, but they cannot disagree."
    )


def test_the_held_out_split_size_matches():
    m = load("metrics.json")
    stated = re.search(r"held-out split\*\*: (\d+) packets, (\d+) fraudulent, (\d+) genuine", README)
    assert stated, "README no longer describes the held-out split size"
    assert [int(g) for g in stated.groups()] == [m["n_packets"], m["n_fraud"], m["n_genuine"]]


def test_per_detector_aucs_match():
    """Every row of the per-detector table, by the label the report itself uses."""
    m = load("metrics.json")
    for name, d in m["per_detector"].items():
        row = re.search(rf"^\| {re.escape(d['label'].split(' (')[0])} \| ([\d.]+) \|", README, re.M)
        if row is None:
            continue  # the table names detectors by short label; skip what it omits
        assert float(row.group(1)) == pytest.approx(d["auc_all_rows"], abs=0.0006), (
            f"README per-detector table says {row.group(1)} for {name}; "
            f"metrics.json says {d['auc_all_rows']}"
        )


def test_the_leaked_ablation_section_matches_its_preserved_evidence():
    """The finding is quoted from the run that found it, not from the current one."""
    a = load("ablation_leaked_corpus.json")
    stated = readme_number(r"\| Metadata / EXIF \| \*\*([\d.]+)\*\* \| \*\*−[\d.]+\*\* \|")
    assert stated == pytest.approx(a["per_detector"]["metadata_exif"]["roc_auc"], abs=0.0006)


def test_the_test_count_in_the_readme_is_current():
    """Two places quote it; both must agree with what the suite actually holds."""
    import subprocess

    out = subprocess.run(
        ["python", "-m", "pytest", str(REPO / "backend/tests"), "--collect-only", "-q"],
        capture_output=True, text=True, cwd=REPO,
    ).stdout
    m = re.search(r"(\d+) tests? collected", out)
    if not m:
        pytest.skip("could not count the suite")
    collected = int(m.group(1))
    for pattern in (r"make test\s+# (\d+) tests", r"tests/\s+(\d+) tests over the deterministic surface"):
        found = re.search(pattern, README)
        assert found, f"README no longer states the test count (pattern: {pattern})"
        assert int(found.group(1)) == collected, (
            f"README says {found.group(1)} tests; the suite collects {collected}"
        )


def test_every_internal_link_points_at_a_heading_that_exists():
    """A 950-line README with two dozen cross-references drifts silently.

    GitHub builds an anchor from a heading by lowercasing it, dropping anything
    that is not a word character, space or hyphen, and turning spaces into
    hyphens — so an em dash surrounded by spaces leaves a double hyphen behind.
    """
    def slug(heading: str) -> str:
        s = heading.strip().lower()
        s = re.sub(r"[^\w\s-]", "", s)
        return re.sub(r"\s", "-", s)

    headings = {slug(m) for m in re.findall(r"^#{1,6}\s+(.*)$", README, re.M)}
    referenced = set(re.findall(r"\]\(#([a-z0-9_-]+)\)", README))
    missing = sorted(referenced - headings)
    assert not missing, (
        f"README links to {len(missing)} anchor(s) with no matching heading: {missing}"
    )


#: Files the README names while stating plainly that they do not exist, with why.
#: Anything else it cites has to be on disk, or a number is resting on nothing.
EXPECTED_ABSENT = {
    "real_video.json": (
        "liveness on recorded deepfakes — FF++ and Celeb-DF are gated behind a signed "
        "request form, and the README claims no number for it precisely because of that"
    ),
}


def test_every_evidence_file_the_readme_cites_is_committed():
    """The README's claim is that the numbers have files behind them."""
    cited = set(re.findall(r"`eval/([a-z_]+\.json)`", README))
    missing = sorted(f for f in cited - set(EXPECTED_ABSENT) if not (EVAL / f).exists())
    assert not missing, f"README cites evidence files that are not in eval/: {missing}"


def test_a_deliberately_absent_file_is_still_absent():
    """If a gated dataset ever lands, the exemption stops being honest.

    The README says no liveness number is claimed *because* this file does not
    exist. Once it does, that sentence is wrong and the section needs writing.
    """
    for name, why in EXPECTED_ABSENT.items():
        if (EVAL / name).exists():
            pytest.fail(
                f"eval/{name} now exists, but the README still explains its absence ({why}). "
                f"Report the numbers or drop the exemption."
            )


def test_the_linkage_false_link_table_matches_its_evidence():
    """Both columns of the search-vs-pair table, against the fitted report.

    The point of that table is that one threshold gives 100% and the other 1%.
    A regenerated calibration that moved either number while the README kept
    claiming the old one would gut the section's whole argument.
    """
    d = load("linkage_lfw.json")
    rows = re.findall(r"^\| ([\d,]+)(?: \(`SCAN_LIMIT`\))? \| \*?\*?([\d.]+)%\*?\*? \| ([\d.]+)% \|",
                      README, re.M)
    assert rows, "README no longer states the per-applicant false-link table"
    for n, shipped, fitted in rows:
        key = n.replace(",", "")
        assert float(shipped) == pytest.approx(
            100 * d["superseded"]["per_applicant_false_link_rate"][key], abs=0.06), (
            f"README says {shipped}% at N={key} for the superseded threshold")
        assert float(fitted) == pytest.approx(
            100 * d["fitted"]["per_applicant_false_link_rate"][key], abs=0.06), (
            f"README says {fitted}% at N={key} for the fitted threshold")


def test_the_demographic_shortcut_table_matches_its_evidence():
    """Every cell of the shortcut-AUC table, both protocols.

    0.5 is the only defensible value in that table, so a number drifting toward
    or away from it silently is the one thing this section must not allow.
    """
    d = load("indian_faces.json")
    pairs = [
        (r"\| Shipped selfie score \| ([\d.]+) \| \*\*([\d.]+)\*\* \|", "score"),
        (r"\| — the pretrained CNN alone \| ([\d.]+) \| \*\*([\d.]+)\*\* \|", "cnn_p_fake"),
        (r"\| — the frequency head, fitted here \| ([\d.]+) \| \*\*([\d.]+)\*\* \|", "spectral_p_fake"),
    ]
    for pattern, key in pairs:
        m = re.search(pattern, README)
        assert m, f"README no longer states the shortcut AUC for {key}"
        frame, face = float(m.group(1)), float(m.group(2))
        assert frame == pytest.approx(d["frame"]["shortcut_auc"][key], abs=0.0006)
        assert face == pytest.approx(d["face"]["shortcut_auc"][key], abs=0.0006)


def behavioral_section() -> str:
    """Just the Detector 6 section.

    Scoping matters: the per-strategy table and the per-attack-type table
    elsewhere in this README have the same column shape and both use backticked
    row labels, so a document-wide regex matches rows from the wrong table and
    reports a mismatch that is really a search bug.
    """
    start = README.index("### Detector 6, measured")
    end = README.index("## Fusion and policy", start)
    return README[start:end]


def test_the_behavioral_corpus_sizes_match():
    """Both halves of Detector 6's corpus, as the README describes them."""
    d = load("behavioral.json")
    c = d["corpus"]
    for pattern, key in (
        (r"\*\*The genuine half\*\* is ([\d,]+) typing sessions", "human_aalto_sessions"),
        (r"plus ([\d,]+) sessions from the\n?CMU", "human_cmu_sessions"),
        (r"automated half\*\* is\n?([\d,]+) runs of a real headless Chromium", "bot_sessions"),
    ):
        m = re.search(pattern, README)
        assert m, f"README no longer states the corpus size for {key}"
        assert int(m.group(1).replace(",", "")) == c[key], (
            f"README says {m.group(1)} for {key}; behavioral.json says {c[key]}"
        )


def test_the_feature_set_ablation_matches():
    """Both rows of the core / core+context table.

    That table is the entire argument for which features ship: identical held-out
    AUC, very different false-positive rates on a population never fitted on. If
    a refit moves either transfer number the argument changes, so the README may
    not keep claiming the old one.
    """
    d = load("behavioral.json")
    rows = re.findall(
        r"^\| \**`?(core\+context|core)`?\**[^|]*\| (\d+) \| ([\d.]+) \| \**([\d.]+)%\**",
        behavioral_section(), re.M,
    )
    assert rows, "README no longer states the feature-set ablation table"
    for name, n_features, auc, transfer in rows:
        a = d["ablation"][name]
        assert int(n_features) == a["n_features"]
        assert float(auc) == pytest.approx(
            a["held_out_subject_disjoint"]["xgboost"]["auc"], abs=0.0006)
        assert float(transfer) == pytest.approx(
            100 * a["transfer_to_unseen_population"]["false_positive_rate"], abs=0.06)


def test_the_unseen_strategy_table_matches():
    """Every row of the leave-one-strategy-out table."""
    d = load("behavioral.json")
    los = d["ablation"][d["shipped_feature_set"]]["leave_one_strategy_out"]
    rows = re.findall(r"^\| `(\w+)` \| ([\d.]+) \| (\d+)% \| ([\d.]+)% \|",
                      behavioral_section(), re.M)
    assert rows, "README no longer states the per-strategy table"
    assert {r[0] for r in rows} == set(los), (
        f"README lists strategies {sorted(r[0] for r in rows)}; "
        f"behavioral.json has {sorted(los)}"
    )
    for name, auc, recall, fpr in rows:
        v = los[name]
        assert float(auc) == pytest.approx(v["auc"], abs=0.0006)
        assert float(recall) == pytest.approx(100 * v["recall_at_threshold"], abs=0.6)
        assert float(fpr) == pytest.approx(100 * v["false_positive_rate"], abs=0.06)


def test_the_replay_attack_is_still_reported_as_beating_the_model():
    """The most important sentence in that section is the one admitting a hole.

    If a refit ever does separate `replay_human`, this test failing is the signal
    to rewrite the section rather than quietly keep a claim that flatters us in
    the wrong direction.
    """
    d = load("behavioral.json")
    auc = d["headline"]["worst_unseen_strategy_auc"]
    assert d["headline"]["worst_unseen_strategy"] == "replay_human", (
        "a different strategy is now the worst; the README names replay_human"
    )
    assert auc is not None and auc < 0.65, (
        f"replay_human now scores {auc}; the README says it is at chance"
    )
    assert "chance" in README[README.index("#### The attack that beats it"):][:900]
