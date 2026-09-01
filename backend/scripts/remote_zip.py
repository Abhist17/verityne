#!/usr/bin/env python3
"""Read individual members out of a remote ZIP without downloading the archive.

The DeepFakeFace splits are four ~1 GB zips and we want 500 images out of each.
Downloading 5 GB to keep 90 MB of it would be the obvious way and the wrong one:
this machine has 11 GB free, and the archives are never needed again once the
images are out.

A ZIP is readable back-to-front - the central directory sits at the end and
carries every member's offset - so three range requests get any single file:
the tail, the directory, then the member itself. Hugging Face's CDN answers
`Range` with a 206, which is the only thing this depends on.

Not a general ZIP implementation. It handles stored and deflated members and
ZIP64 directory locators, which is what these archives use; it does not do
encryption, multi-disk archives, or data descriptors.
"""
from __future__ import annotations

import logging
import struct
import urllib.request
import zlib
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("remote_zip")

_EOCD = b"PK\x05\x06"
_EOCD64_LOCATOR = b"PK\x06\x07"
_EOCD64 = b"PK\x06\x06"
_CENTRAL = b"PK\x01\x02"

#: Enough to hold the end-of-central-directory record plus a maximal comment.
_TAIL_BYTES = 66_560


class RemoteZipError(RuntimeError):
    pass


class RemoteZip:
    """Random access to a ZIP over HTTP.

    The central directory is fetched once on first use and kept, so repeated
    reads cost one request each.
    """

    def __init__(self, url: str, timeout: int = 120):
        self.url = url
        self.timeout = timeout
        self._entries: Optional[Dict[str, Tuple[int, int, int]]] = None
        self._order: List[str] = []

    # ---------------------------------------------------------------- transport
    def _get(self, start: int, end: int) -> bytes:
        """Fetch an inclusive byte range.

        A server that ignores `Range` answers 200 with the whole file, which for
        a 1 GB archive would silently undo the point of this class. The length
        check turns that into an error instead of a very slow success.
        """
        want = end - start + 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        if len(data) != want:
            raise RemoteZipError(
                f"range {start}-{end} returned {len(data)} bytes, expected {want}; "
                "the server may not support Range requests"
            )
        return data

    def _size(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return int(resp.headers["Content-Length"])

    # ------------------------------------------------------------- directory
    def _load_directory(self) -> None:
        total = self._size()
        tail_start = max(0, total - _TAIL_BYTES)
        tail = self._get(tail_start, total - 1)

        idx = tail.rfind(_EOCD)
        if idx < 0:
            raise RemoteZipError("no end-of-central-directory record; not a ZIP?")
        count, cd_size, cd_offset = struct.unpack("<HII", tail[idx + 10 : idx + 20])

        # ZIP64: the 32-bit fields saturate and the real values live in a
        # separate record that the locator points at.
        if count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
            loc = tail.rfind(_EOCD64_LOCATOR)
            if loc < 0:
                raise RemoteZipError("ZIP64 archive without a locator record")
            (eocd64_offset,) = struct.unpack("<Q", tail[loc + 8 : loc + 16])
            rec = self._get(eocd64_offset, eocd64_offset + 55)
            if rec[:4] != _EOCD64:
                raise RemoteZipError("ZIP64 locator pointed at a non-ZIP64 record")
            count, cd_size, cd_offset = struct.unpack("<QQQ", rec[32:56])

        cd = self._get(cd_offset, cd_offset + cd_size - 1)

        entries: Dict[str, Tuple[int, int, int]] = {}
        order: List[str] = []
        pos = 0
        while pos < len(cd) - 46 and cd[pos : pos + 4] == _CENTRAL:
            (method,) = struct.unpack("<H", cd[pos + 10 : pos + 12])
            (comp_size,) = struct.unpack("<I", cd[pos + 20 : pos + 24])
            name_len, extra_len, comment_len = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
            (local_header,) = struct.unpack("<I", cd[pos + 42 : pos + 46])
            name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
            if not name.endswith("/"):
                entries[name] = (method, comp_size, local_header)
                order.append(name)
            pos += 46 + name_len + extra_len + comment_len

        if not entries:
            raise RemoteZipError("central directory parsed to zero members")
        self._entries, self._order = entries, order
        log.debug("%s: %d members", self.url.rsplit("/", 1)[-1], len(entries))

    def _ensure(self) -> Dict[str, Tuple[int, int, int]]:
        if self._entries is None:
            self._load_directory()
        assert self._entries is not None
        return self._entries

    # ----------------------------------------------------------------- public
    def namelist(self) -> List[str]:
        """Member names, in central-directory order."""
        self._ensure()
        return list(self._order)

    def read(self, name: str) -> bytes:
        """Fetch and decompress one member.

        The local header repeats the name and extra fields at their own lengths -
        which need not match the central directory's - so they are read from the
        local header rather than reused, and the payload starts after them.
        """
        entries = self._ensure()
        if name not in entries:
            raise KeyError(name)
        method, comp_size, local_header = entries[name]

        header = self._get(local_header, local_header + 29)
        if header[:4] != b"PK\x03\x04":
            raise RemoteZipError(f"{name}: local header signature missing")
        name_len, extra_len = struct.unpack("<HH", header[26:30])

        start = local_header + 30 + name_len + extra_len
        payload = self._get(start, start + comp_size - 1)

        if method == 0:
            return payload
        if method == 8:
            return zlib.decompress(payload, -15)
        raise RemoteZipError(f"{name}: unsupported compression method {method}")
