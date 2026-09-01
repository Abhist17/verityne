"""The real-faces evaluation: reading remote archives, and reporting per family.

Two things here can quietly produce a wrong number rather than an error, so
those are what is tested.

`RemoteZip` exists to avoid downloading 5 GB for 90 MB of images. A server that
ignores `Range` answers 200 with the whole archive, and a reader that accepted
that would still return correct images - slowly, and having defeated its own
reason to exist. So the transport is tested against a server that supports
ranges and against one that does not.

`family_report` is the shape every headline is read off. Its inputs are two
score arrays whose *order and labelling* decide whether a detector looks like it
works or like it is inverted, and nothing downstream can catch a swap.
"""
from __future__ import annotations

import io
import os
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from remote_zip import RemoteZip, RemoteZipError  # noqa: E402


# --------------------------------------------------------------------- fixtures
def build_zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


MEMBERS = {
    "wiki/00/a.jpg": b"first-member" * 40,
    "wiki/00/b.jpg": b"second-member" * 60,
    "wiki/01/c.jpg": b"third" * 500,
}


class _Handler(BaseHTTPRequestHandler):
    blob = b""
    honour_ranges = True

    def log_message(self, *a):  # keep the test output clean
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.blob)))
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng and self.honour_ranges:
            start, end = rng.split("=", 1)[1].split("-")
            s, e = int(start), int(end)
            body = self.blob[s : e + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {s}-{e}/{len(self.blob)}")
        else:
            body = self.blob
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def served():
    """Serve a ZIP over localhost; yields (url_for, set_range_support)."""
    blob = build_zip(MEMBERS)
    _Handler.blob = blob
    _Handler.honour_ranges = True
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/archive.zip"
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------ transport
class TestRemoteZip:
    def test_lists_only_files(self, served):
        """Directory entries are not members; a caller iterating names must not
        try to decode one as an image."""
        names = RemoteZip(served).namelist()
        assert set(names) == set(MEMBERS)

    def test_reads_each_member_byte_exact(self, served):
        zf = RemoteZip(served)
        for name, payload in MEMBERS.items():
            assert zf.read(name) == payload

    def test_reads_without_downloading_the_archive(self):
        """The whole point. Reading one small member must not transfer the file.

        Needs an archive substantially larger than the 64 KB tail window, or the
        comparison is meaningless - on a 400-byte ZIP the tail read alone is the
        whole file. The real archives are ~1 GB against a 90 MB working set, so
        this mirrors that ratio rather than the fixture's.

        Counted in served bytes rather than asserted loosely: the failure this
        guards against is a silent fallback to whole-file GETs, which returns
        the right answer and defeats the class.
        """
        # Random payloads: zero-filled members deflate to almost nothing, and an
        # archive smaller than the tail window cannot demonstrate a partial read.
        bulk = {f"bulk/{i:03d}.bin": os.urandom(4096) for i in range(200)}
        bulk["wanted/small.bin"] = b"the-one-we-want" * 8
        _Handler.blob = build_zip(bulk)
        _Handler.honour_ranges = True

        served_bytes = 0
        original = _Handler.do_GET

        def counting(self):
            nonlocal served_bytes
            original(self)
            rng = self.headers.get("Range")
            if rng:
                s, e = (int(x) for x in rng.split("=", 1)[1].split("-"))
                served_bytes += e - s + 1
            else:
                served_bytes += len(self.blob)

        _Handler.do_GET = counting
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/big.zip"
            zf = RemoteZip(url)
            assert zf.read("wanted/small.bin") == bulk["wanted/small.bin"]
        finally:
            _Handler.do_GET = original
            server.shutdown()
            server.server_close()
        assert served_bytes < len(_Handler.blob)

    def test_a_server_ignoring_ranges_is_an_error_not_a_slow_success(self, served):
        """Silently accepting a 200 would work and would download gigabytes."""
        _Handler.honour_ranges = False
        with pytest.raises(RemoteZipError, match="Range"):
            RemoteZip(served).namelist()

    def test_missing_member_raises_keyerror(self, served):
        with pytest.raises(KeyError):
            RemoteZip(served).read("wiki/00/not-here.jpg")

    def test_unsupported_compression_is_refused(self, served, monkeypatch):
        """Better to fail than to hand back a compressed blob as if it were a JPEG."""
        zf = RemoteZip(served)
        zf.namelist()
        method, csize, lho = zf._entries["wiki/00/a.jpg"]  # noqa: SLF001
        zf._entries["wiki/00/a.jpg"] = (99, csize, lho)  # noqa: SLF001
        with pytest.raises(RemoteZipError, match="compression method"):
            zf.read("wiki/00/a.jpg")

    def test_stored_members_are_read_verbatim(self):
        """The archives in use are deflated, but stored members must still work."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            z.writestr("plain/x.bin", b"uncompressed-payload")
        _Handler.blob = buf.getvalue()
        _Handler.honour_ranges = True
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/stored.zip"
            assert RemoteZip(url).read("plain/x.bin") == b"uncompressed-payload"
        finally:
            server.shutdown()
            server.server_close()


# -------------------------------------------------------------------- reporting
class TestFamilyReport:
    """`family_report` is imported lazily: it lives in a script whose module-level
    imports pull in torch, and these assertions are about arithmetic."""

    @staticmethod
    def _report(real, fake):
        from evaluate_real_faces import family_report
        from sklearn.metrics import roc_auc_score

        return family_report(np.asarray(real, dtype=float),
                             np.asarray(fake, dtype=float), roc_auc_score)

    def test_a_detector_that_separates_perfectly_scores_one(self):
        r = self._report([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
        assert r["auc"] == 1.0

    def test_an_inverted_detector_scores_zero_not_one(self):
        """The label convention is fake=1. A detector scoring reals higher than
        fakes is broken, and must read as 0.0 rather than being folded to 1.0 -
        an AUC that cannot go below 0.5 would hide exactly that."""
        r = self._report([0.9, 0.8, 0.7], [0.1, 0.2, 0.3])
        assert r["auc"] == 0.0

    def test_indistinguishable_classes_score_half(self):
        r = self._report([0.5] * 4, [0.5] * 4)
        assert r["auc"] == 0.5

    def test_recall_and_fpr_are_measured_at_the_shipped_thresholds(self):
        """Recall counts fakes at or above the threshold; the false-positive rate
        counts reals at or above the same one."""
        r = self._report([0.1, 0.5, 0.9], [0.3, 0.8, 0.95])
        # The report rounds to 4dp for the JSON, so compare at that resolution.
        assert r["recall_at"]["0.75"] == pytest.approx(2 / 3, abs=1e-4)
        assert r["real_fpr_at"]["0.75"] == pytest.approx(1 / 3, abs=1e-4)
        assert r["recall_at"]["0.4"] == pytest.approx(2 / 3, abs=1e-4)

    def test_an_empty_group_reports_absence_rather_than_zero(self):
        """A dataset that failed to download must not render as a detector that
        scored nothing - the dashboard draws null as absent and 0.0 as measured."""
        r = self._report([], [0.4, 0.6])
        assert r["auc"] is None
        assert r["n_real"] == 0
