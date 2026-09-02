"""The documentation's headline numbers must equal the evidence files beside them.

This project's claim is that every number it states is reproducible from
`eval/*.json`. Nothing enforced that, and it had already drifted: the hook said
0.914 while the Results table three hundred lines below said 0.911, from
different runs, in the same file. Nobody noticed, because checking meant reading
a thousand lines against six JSON documents by hand.

So each claim below names where in the prose it lives and which file it must
agree with. No expected value is written here - the test reads both sides and
compares - so it cannot rot into asserting a number that used to be true.

A regenerated pipeline that moves a number now fails this test until the prose is
updated, which is the entire point.

The prose is `README.md` plus `docs/*.md`. It used to be one 1,645-line README;
splitting it must not narrow what is checked, so document-wide searches run over
the concatenation of every prose file and only the genuinely section-scoped ones
name a single document.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval"
DOCS = REPO / "docs"

#: Every prose file this project ships, by the path a reader would cite.
PROSE = {"README.md": (REPO / "README.md").read_text()}
PROSE.update({f"docs/{p.name}": p.read_text() for p in sorted(DOCS.glob("*.md"))})

#: Document-wide corpus. A claim may live in any file; the guarantee is that a
#: number stated *anywhere* has an evidence file behind it.
ALL = "\n\n".join(PROSE[k] for k in sorted(PROSE))


def load(name: str):
    p = EVAL / name
    if not p.exists():
        pytest.skip(f"{name} not present - run `make pipeline`")
    return json.loads(p.read_text())


def dig(data, path: str):
    for part in path.split("."):
        data = data[int(part)] if part.isdigit() else data[part]
    return data


def stated_number(pattern: str, text: str = ALL) -> float:
    """The single number the prose states at `pattern`.

    Requires exactly one *value*: a claim stated twice in two different forms is
    itself the failure mode this file exists to catch. The same value repeated -
    a summary in the README and the canonical table in `docs/` - is fine.
    """
    found = re.findall(pattern, text)
    assert found, f"the docs no longer state this claim (pattern: {pattern})"
    unique = {f if isinstance(f, str) else f[0] for f in found}
    assert len(unique) == 1, f"the docs state conflicting values {sorted(unique)} for {pattern}"
    return float(next(iter(unique)))


#: (description, prose regex capturing the number, evidence file, dotted path)
#: with an optional fifth element: the factor the prose states it by, for
#: rows written as a percentage while the report stores a fraction.
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
     r"\| \*\(none - full model\)\* \| ([\d.]+) \|", "ablation_leaked_corpus.json", "full_model.roc_auc"),
    ("linkage: true-accept rate at the fitted search threshold",
     r"true-accept rate from 95.1% to \*\*([\d.]+)%\*\*", "linkage_lfw.json", "fitted.tar", 100),
    ("demography: shortcut AUC of the frequency head, controlled",
     r"\| - the frequency head, fitted here \| [\d.]+ \| \*\*([\d.]+)\*\* \|",
     "indian_faces.json", "face.shortcut_auc.spectral_p_fake"),
    ("demography: shortcut AUC of the pretrained CNN, controlled",
     r"\| - the pretrained CNN alone \| [\d.]+ \| \*\*([\d.]+)\*\* \|",
     "indian_faces.json", "face.shortcut_auc.cnn_p_fake"),
    ("detector 6: held-out AUC",
     r"\| Held-out ROC-AUC, subject-disjoint \| \*\*([\d.]+)\*\* \|",
     "behavioral.json", "headline.held_out_auc"),
    ("detector 6: the strategy that defeats it",
     r"\| Worst unseen strategy \| \*\*`\w+` at ([\d.]+)\*\* \|",
     "behavioral.json", "headline.worst_unseen_strategy_auc"),
]


@pytest.mark.parametrize("claim", CLAIMS, ids=[c[0] for c in CLAIMS])
def test_stated_number_matches_its_evidence(claim):
    desc, pattern, filename, path = claim[:4]
    scale = claim[4] if len(claim) > 4 else 1
    stated = stated_number(pattern)
    actual = float(dig(load(filename), path)) * scale
    assert stated == pytest.approx(actual, abs=0.0006 * scale), (
        f"the docs say {stated} for {desc}; {filename}:{path} says {actual}. "
        f"Regenerate the section or the evidence, but they cannot disagree."
    )


def test_the_held_out_split_size_matches():
    m = load("metrics.json")
    stated = re.search(r"held-out split\*\*: (\d+) packets, (\d+) fraudulent, (\d+) genuine", ALL)
    assert stated, "the docs no longer describe the held-out split size"
    assert [int(g) for g in stated.groups()] == [m["n_packets"], m["n_fraud"], m["n_genuine"]]


def test_per_detector_aucs_match():
    """Every row of the per-detector table, by the label the report itself uses."""
    m = load("metrics.json")
    for name, d in m["per_detector"].items():
        row = re.search(rf"^\| {re.escape(d['label'].split(' (')[0])} \| ([\d.]+) \|", ALL, re.M)
        if row is None:
            continue  # the table names detectors by short label; skip what it omits
        assert float(row.group(1)) == pytest.approx(d["auc_all_rows"], abs=0.0006), (
            f"the per-detector table says {row.group(1)} for {name}; "
            f"metrics.json says {d['auc_all_rows']}"
        )


def test_the_leaked_ablation_section_matches_its_preserved_evidence():
    """The finding is quoted from the run that found it, not from the current one."""
    a = load("ablation_leaked_corpus.json")
    stated = stated_number(r"\| Metadata / EXIF \| \*\*([\d.]+)\*\* \| \*\*−[\d.]+\*\* \|")
    assert stated == pytest.approx(a["per_detector"]["metadata_exif"]["roc_auc"], abs=0.0006)


def test_the_test_count_in_the_docs_is_current():
    """Two places quote it; both must agree with what the suite actually holds."""
    import subprocess
    import sys

    # `sys.executable`, not "python": the bare name is absent on any system that
    # ships only python3 (Debian without python-is-python3, most CI images), and
    # the crash there looked like a stale doc count rather than a missing binary.
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "backend/tests"), "--collect-only", "-q"],
        capture_output=True, text=True, cwd=REPO,
    ).stdout
    m = re.search(r"(\d+) tests? collected", out)
    if not m:
        pytest.skip("could not count the suite")
    collected = int(m.group(1))
    for pattern in (r"make test\s+# (\d+) tests", r"tests/\s+(\d+) tests over the deterministic surface"):
        found = re.search(pattern, ALL)
        assert found, f"the docs no longer state the test count (pattern: {pattern})"
        assert int(found.group(1)) == collected, (
            f"the docs say {found.group(1)} tests; the suite collects {collected}"
        )


def _slug(heading: str) -> str:
    """GitHub's anchor rule: lowercase, drop non-word/space/hyphen, spaces to hyphens.

    An em dash surrounded by spaces therefore leaves a double hyphen behind.
    """
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s", "-", s)


def _headings(text: str) -> set:
    return {_slug(h) for h in re.findall(r"^#{1,6}\s+(.*)$", text, re.M)}


def test_every_internal_link_points_at_a_heading_that_exists():
    """Nine prose files with two dozen cross-references between them drift silently.

    Checks both link shapes: a bare `#anchor` must exist in the file that wrote
    it, and a `path.md#anchor` must exist in the file it names. Splitting the
    README into `docs/` turned most of these into cross-file links, which is
    exactly the kind of reference that rots without a check.
    """
    broken = []
    for name, text in PROSE.items():
        here = Path(name)
        for target, anchor in re.findall(r"\]\(([^)#]*)#([a-z0-9_-]+)\)", text):
            if target:
                dest = (here.parent / target).as_posix()
                dest = re.sub(r"^(?:\./)?", "", dest)
                # normalise `docs/../README.md` → `README.md`
                dest = Path(dest).resolve().relative_to(REPO.resolve()).as_posix() \
                    if (REPO / dest).exists() else dest
                if dest not in PROSE:
                    broken.append(f"{name} → {target}#{anchor} (no such file)")
                    continue
                pool = _headings(PROSE[dest])
            else:
                dest, pool = name, _headings(text)
            if anchor not in pool:
                broken.append(f"{name} → {dest}#{anchor}")
    assert not broken, "links with no matching heading:\n  " + "\n  ".join(broken)


def test_every_doc_is_reachable_from_the_readme():
    """A document nobody links to is a document nobody reads."""
    linked = set(re.findall(r"\]\((docs/[a-z0-9-]+\.md)", PROSE["README.md"]))
    missing = sorted(set(PROSE) - {"README.md"} - linked)
    assert not missing, f"docs not linked from the README: {missing}"


#: Files the docs name while stating plainly that they do not exist, with why.
#: Anything else cited has to be on disk, or a number is resting on nothing.
EXPECTED_ABSENT = {
    "real_video.json": (
        "liveness on recorded deepfakes - FF++ and Celeb-DF are gated behind a signed "
        "request form, and the docs claim no number for it precisely because of that"
    ),
}


def test_every_evidence_file_the_docs_cite_is_committed():
    """The claim is that the numbers have files behind them."""
    cited = set(re.findall(r"`eval/([a-z_]+\.json)`", ALL))
    missing = sorted(f for f in cited - set(EXPECTED_ABSENT) if not (EVAL / f).exists())
    assert not missing, f"the docs cite evidence files that are not in eval/: {missing}"


def test_a_deliberately_absent_file_is_still_absent():
    """If a gated dataset ever lands, the exemption stops being honest.

    The docs say no liveness number is claimed *because* this file does not
    exist. Once it does, that sentence is wrong and the section needs writing.
    """
    for name, why in EXPECTED_ABSENT.items():
        if (EVAL / name).exists():
            pytest.fail(
                f"eval/{name} now exists, but the docs still explain its absence ({why}). "
                f"Report the numbers or drop the exemption."
            )


def test_the_linkage_false_link_table_matches_its_evidence():
    """Both columns of the search-vs-pair table, against the fitted report.

    The point of that table is that one threshold gives 100% and the other 1%.
    A regenerated calibration that moved either number while the docs kept
    claiming the old one would gut the section's whole argument.
    """
    d = load("linkage_lfw.json")
    rows = re.findall(r"^\| ([\d,]+)(?: \(`SCAN_LIMIT`\))? \| \*?\*?([\d.]+)%\*?\*? \| ([\d.]+)% \|",
                      PROSE["docs/real-data.md"], re.M)
    assert rows, "the docs no longer state the per-applicant false-link table"
    for n, shipped, fitted in rows:
        key = n.replace(",", "")
        assert float(shipped) == pytest.approx(
            100 * d["superseded"]["per_applicant_false_link_rate"][key], abs=0.06), (
            f"the docs say {shipped}% at N={key} for the superseded threshold")
        assert float(fitted) == pytest.approx(
            100 * d["fitted"]["per_applicant_false_link_rate"][key], abs=0.06), (
            f"the docs say {fitted}% at N={key} for the fitted threshold")


def test_the_demographic_shortcut_table_matches_its_evidence():
    """Every cell of the shortcut-AUC table, both protocols.

    0.5 is the only defensible value in that table, so a number drifting toward
    or away from it silently is the one thing this section must not allow.
    """
    d = load("indian_faces.json")
    section = PROSE["docs/real-data.md"]
    pairs = [
        (r"\| Shipped selfie score \| ([\d.]+) \| \*\*([\d.]+)\*\* \|", "score"),
        (r"\| - the pretrained CNN alone \| ([\d.]+) \| \*\*([\d.]+)\*\* \|", "cnn_p_fake"),
        (r"\| - the frequency head, fitted here \| ([\d.]+) \| \*\*([\d.]+)\*\* \|", "spectral_p_fake"),
    ]
    for pattern, key in pairs:
        m = re.search(pattern, section)
        assert m, f"the docs no longer state the shortcut AUC for {key}"
        frame, face = float(m.group(1)), float(m.group(2))
        assert frame == pytest.approx(d["frame"]["shortcut_auc"][key], abs=0.0006)
        assert face == pytest.approx(d["face"]["shortcut_auc"][key], abs=0.0006)


def behavioral_section() -> str:
    """Just the Detector 6 document.

    Scoping matters: the per-strategy table and the per-attack-type table in
    `docs/results.md` have the same column shape and both use backticked row
    labels, so a corpus-wide regex matches rows from the wrong table and reports
    a mismatch that is really a search bug.
    """
    return PROSE["docs/behavioral.md"]


def test_the_behavioral_corpus_sizes_match():
    """Both halves of Detector 6's corpus, as the docs describe them."""
    d = load("behavioral.json")
    c = d["corpus"]
    for pattern, key in (
        (r"\*\*The genuine half\*\* is ([\d,]+) typing sessions", "human_aalto_sessions"),
        (r"plus ([\d,]+) sessions from the\n?CMU", "human_cmu_sessions"),
        (r"automated half\*\* is\n?([\d,]+) runs of a real headless Chromium", "bot_sessions"),
    ):
        m = re.search(pattern, behavioral_section())
        assert m, f"the docs no longer state the corpus size for {key}"
        assert int(m.group(1).replace(",", "")) == c[key], (
            f"the docs say {m.group(1)} for {key}; behavioral.json says {c[key]}"
        )


def test_the_feature_set_ablation_matches():
    """Both rows of the core / core+context table.

    That table is the entire argument for which features ship: identical held-out
    AUC, very different false-positive rates on a population never fitted on. If
    a refit moves either transfer number the argument changes, so the docs may
    not keep claiming the old one.
    """
    d = load("behavioral.json")
    rows = re.findall(
        r"^\| \**`?(core\+context|core)`?\**[^|]*\| (\d+) \| ([\d.]+) \| \**([\d.]+)%\**",
        behavioral_section(), re.M,
    )
    assert rows, "the docs no longer state the feature-set ablation table"
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
    assert rows, "the docs no longer state the per-strategy table"
    assert {r[0] for r in rows} == set(los), (
        f"the docs list strategies {sorted(r[0] for r in rows)}; "
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
        "a different strategy is now the worst; the docs name replay_human"
    )
    assert auc is not None and auc < 0.65, (
        f"replay_human now scores {auc}; the docs say it is at chance"
    )
    section = behavioral_section()
    assert "chance" in section[section.index("#### The attack that beats it"):][:900]


def test_the_corrections_table_matches_the_generated_file():
    """The Corrections table is a rendering of eval/corrections.json.

    Both the README hook and the table quote the counts, and both are the kind
    of number that goes stale the moment a finding is added. If a correction is
    appended and the prose is not regenerated, this is what says so.
    """
    d = load("corrections.json")
    counts = d["counts"]

    stated_n = re.search(r"\*\*(\d+) documented cases of this project", PROSE["README.md"])
    assert stated_n, "the README hook no longer states how many corrections there are"
    assert int(stated_n.group(1)) == d["n"]

    fixed = re.search(r"^(\d+) are fixed\. (\d+) are still open", PROSE["README.md"], re.M)
    assert fixed, "the README hook no longer states the fixed/open split"
    assert int(fixed.group(1)) == counts.get("fixed")
    assert int(fixed.group(2)) == counts.get("open")

    section = re.search(
        rf"^{d['n']} beliefs this project held, measured, and lost\. (\d+) fixed,\n(\d+) still open, (\d+) designed around\.",
        PROSE["docs/corrections.md"], re.M)
    assert section, "the Corrections document no longer states its counts"
    assert int(section.group(1)) == counts.get("fixed")
    assert int(section.group(2)) == counts.get("open")
    assert int(section.group(3)) == counts.get("designed_around", 0)

    # Every correction must have a row, by title.
    for c in d["corrections"]:
        assert c["title"] in ALL, (
            f"correction '{c['id']}' is not listed in the Corrections table"
        )


def test_the_third_party_face_numbers_match_their_report():
    """Section 4's per-family table, against eval/real_faces.json."""
    d = load("real_faces.json")
    face = d["tracks"]["deepfakeface"]["protocols"]["face"]["per_family"]
    sg = d["tracks"]["stylegan_140k"]["protocols"]["face"]["per_family"]["stylegan"]

    doc = PROSE["docs/real-data.md"]
    start = doc.index("### 4. The selfie detector on fakes we did not generate")
    end = doc.index("### 5. Liveness on recorded video", start)
    section = doc[start:end]

    rows = dict(re.findall(r"^\| `(\w+)`[^|]*\| [\d.]+ \| \*\*([\d.]+)\*\*", section, re.M))
    assert rows, "the per-family table is gone"
    for family, rep in face.items():
        assert family in rows, f"{family} is missing from the table"
        assert float(rows[family]) == pytest.approx(rep["auc"], abs=0.0006)
    assert float(rows["stylegan"]) == pytest.approx(sg["auc"], abs=0.0006)


def test_the_docs_do_not_claim_an_unserved_endpoint():
    """`/threat/graph` was documented as 'contract defined, not yet served' while
    it was a stub. It is served now, and prose that still says otherwise is
    understating the project in the one place a reader checks first."""
    routes = Path(__file__).resolve().parents[1] / "verityne" / "api" / "routes_threat.py"
    if routes.exists() and "/threat/graph" in routes.read_text():
        assert "not yet served" not in ALL, (
            "/threat/graph is implemented, but the docs still call it unserved"
        )


def test_the_operating_point_table_matches_the_report():
    """The two threshold rows, against eval/metrics.json.

    These went stale once: the policy default moved to 0.45/0.80, the docs kept
    quoting 0.40/0.75, and nothing caught it because nothing was checking. The
    dashboard was showing one number while the prose claimed another.
    """
    m = load("metrics.json")
    for name, key in (("REVIEW", "at_review_threshold"), ("REJECT", "at_reject_threshold")):
        d = m["fusion"][key]
        row = re.search(
            rf"^\| `{name}` @ ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)% \| "
            rf"\*\*([\d.]+)%\*\* \| ([\d.]+) \|",
            ALL, re.M)
        assert row, f"the docs no longer state the {name} operating point"
        assert float(row.group(1)) == pytest.approx(d["threshold"], abs=0.0006)
        assert float(row.group(2)) == pytest.approx(d["precision"], abs=0.0006)
        assert float(row.group(3)) == pytest.approx(d["recall"], abs=0.0006)
        assert float(row.group(4)) == pytest.approx(100 * d["false_accept_rate"], abs=0.06)
        assert float(row.group(5)) == pytest.approx(100 * d["false_reject_rate"], abs=0.06)
        assert float(row.group(6)) == pytest.approx(d["accuracy"], abs=0.0006)


def test_the_report_was_computed_at_the_policy_that_ships():
    """policy.yaml is what actually runs; metrics.json is what the docs quote.

    If they drift, every recall and false-reject figure in the documentation
    describes an operating point the system is not using. Re-run `make evaluate`
    after changing policy.yaml.
    """
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from verityne.config import get_policy

    m = load("metrics.json")
    policy = get_policy("default")
    assert policy.min_risk_for_review == pytest.approx(
        m["fusion"]["at_review_threshold"]["threshold"], abs=0.0006), (
        "policy.yaml's review threshold is not the one eval/metrics.json was computed at")
    assert policy.min_risk_for_reject == pytest.approx(
        m["fusion"]["at_reject_threshold"]["threshold"], abs=0.0006), (
        "policy.yaml's reject threshold is not the one eval/metrics.json was computed at")


def test_the_documented_policy_example_matches_the_shipped_default():
    """The YAML block in the docs is the first thing a reader copies."""
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from verityne.config import get_policy

    policy = get_policy("default")
    for field, value in (("min_risk_for_reject", policy.min_risk_for_reject),
                         ("min_risk_for_review", policy.min_risk_for_review)):
        shown = re.search(rf"^  {field}: ([\d.]+)$", ALL, re.M)
        assert shown, f"the documented policy example no longer shows {field}"
        assert float(shown.group(1)) == pytest.approx(value, abs=0.0006), (
            f"the docs show {field}: {shown.group(1)} but policy.yaml ships {value}")
