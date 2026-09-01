"""The fraud-ring graph: what counts as a ring, and what must never be called one.

The page this feeds accuses people of running a fraud operation, so the tests
that matter are the ones about restraint - an isolated applicant is not a ring, a
shared government-document template is not a ring, and a face match is never
allowed to present itself with the certainty of a shared file.
"""
import numpy as np
import pytest

from verityne.api.routes_threat import _components


class TestComponents:
    def test_a_lone_submission_is_not_a_ring(self):
        """One applicant on their own is a person, not an operation."""
        assert _components(["a", "b", "c"], []) == {}

    def test_two_linked_submissions_form_one_ring(self):
        rings = _components(["a", "b", "c"], [("a", "b")])
        assert rings["a"] == rings["b"]
        assert "c" not in rings

    def test_a_chain_collapses_into_a_single_ring(self):
        """a-b, b-c and c-d is one operation of four, not three pairs."""
        rings = _components(list("abcd"), [("a", "b"), ("b", "c"), ("c", "d")])
        assert len(set(rings.values())) == 1
        assert set(rings) == set("abcd")

    def test_two_separate_rings_stay_separate(self):
        rings = _components(list("abcd"), [("a", "b"), ("c", "d")])
        assert rings["a"] == rings["b"]
        assert rings["c"] == rings["d"]
        assert rings["a"] != rings["c"]

    def test_an_edge_to_an_unknown_node_is_ignored(self):
        """Edges are built from a windowed node set; one pointing outside it must
        not raise, because the window boundary is arbitrary."""
        assert _components(["a", "b"], [("a", "zzz")]) == {}

    def test_a_self_edge_does_not_manufacture_a_ring(self):
        assert _components(["a", "b"], [("a", "a")]) == {}

    def test_ring_indices_are_contiguous_from_zero(self):
        rings = _components(list("abcdef"), [("a", "b"), ("c", "d"), ("e", "f")])
        assert sorted(set(rings.values())) == [0, 1, 2]


class TestGraphEndpoint:
    """End to end against a real ring, through the real pipeline."""

    @pytest.fixture(scope="class")
    def client(self):
        """conftest.py already points the engine at a throwaway database."""
        from fastapi.testclient import TestClient
        from verityne.main import app
        with TestClient(app) as c:
            yield c

    @staticmethod
    def _image(path, seed):
        from PIL import Image
        rng = np.random.default_rng(seed)
        Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(path)
        return path

    def test_one_file_under_three_identities_is_one_ring(self, client, tmp_path):
        h = {"X-API-Key": "verityne-demo-key"}
        shared = self._image(tmp_path / "shared.jpg", 7)
        alone = self._image(tmp_path / "alone.jpg", 99)

        for merchant, name in (("acme", "Ravi Sharma"), ("bolt", "Suresh Nair"), ("cred", "Anil Gupta")):
            with open(shared, "rb") as fh:
                client.post("/verify", headers=h, files={"selfie": ("s.jpg", fh, "image/jpeg")},
                            data={"merchant_id": merchant, "claimed_name": name})
        with open(alone, "rb") as fh:
            client.post("/verify", headers=h, files={"selfie": ("s.jpg", fh, "image/jpeg")},
                        data={"merchant_id": "acme", "claimed_name": "Meera Iyer"})

        g = client.get("/threat/graph?hours=168", headers=h)
        assert g.status_code == 200
        j = g.json()

        assert len(j["rings"]) == 1
        ring = j["rings"][0]
        assert ring["size"] == 3
        assert sorted(ring["merchants"]) == ["acme", "bolt", "cred"]
        assert ring["distinct_names"] == 3
        # The claim is a shared SHA-256, which is a fact rather than an inference.
        assert ring["has_exact_asset_reuse"] is True
        assert sum(1 for n in j["nodes"] if n["ring"] is None) == 1

    def test_the_threshold_carries_its_provenance(self, client):
        """The page prints where the edge threshold came from. A graph drawn at a
        verification threshold wires every genuine applicant to a stranger."""
        j = client.get("/threat/graph", headers={"X-API-Key": "verityne-demo-key"}).json()
        assert 0.0 < j["threshold"]["same_person"] <= 1.0
        assert j["threshold"]["fitted_on"]

    def test_a_narrow_window_still_returns_a_well_formed_graph(self, client):
        """The window is a filter, not a special case - one hour must return the
        same shape as one year, so the page never has to branch on emptiness."""
        j = client.get("/threat/graph?hours=1", headers={"X-API-Key": "verityne-demo-key"}).json()
        assert isinstance(j["nodes"], list)
        assert isinstance(j["edges"], list)
        assert isinstance(j["rings"], list)
        assert j["window_hours"] == 1

    def test_the_response_matches_the_frontend_contract(self, client):
        """Field for field against `ThreatGraph` in frontend/lib/api.ts.

        The two are written in different languages and cannot be checked against
        each other by a compiler; a renamed key here is a silently empty panel there.
        """
        j = client.get("/threat/graph", headers={"X-API-Key": "verityne-demo-key"}).json()
        assert {"generated_at", "window_hours", "threshold", "nodes", "edges", "rings"} <= set(j)
        assert {"same_person", "fitted_on"} <= set(j["threshold"])
        for n in j["nodes"]:
            assert {"id", "merchant_id", "claimed_name", "verdict", "score", "created_at",
                    "generator_guess", "thumb_url", "ring"} <= set(n)
        for e in j["edges"]:
            assert {"source", "target", "kind", "weight"} <= set(e)
            assert e["kind"] in {"face", "asset_exact", "asset_near"}
        for r in j["rings"]:
            assert {"id", "size", "merchants", "distinct_names", "max_score",
                    "has_exact_asset_reuse"} <= set(r)
