"""Small text helpers for strings a human reads on screen.

Reason lines and explanations are read by a reviewer deciding whether to believe
the system, not by a parser. "1 different name(s)" reads like the system does not
know what it found, on the one surface whose whole job is to be believed - so the
count and the noun agree here instead.
"""
from __future__ import annotations

from typing import Optional


def plural(n: int, singular: str, plural_form: Optional[str] = None) -> str:
    """`plural(1, "name") -> "1 name"`, `plural(3, "name") -> "3 names"`.

    Pass `plural_form` for anything that does not just take an -s:
    `plural(2, "face match", "face matches")`.
    """
    if n == 1:
        return f"{n} {singular}"
    return f"{n} {plural_form or singular + 's'}"
