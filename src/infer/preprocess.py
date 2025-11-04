import re
from typing import Tuple

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b")
TICKET = re.compile(r"\b(TKT|TICKET|CASE|INC)[-_:]?\d{3,}\b", re.I)

REPLACEMENTS = [
    (EMAIL, "<email>"),
    (PHONE, "<phone>"),
    (TICKET, "<ticket>")
]


def redact(text: str) -> Tuple[str, int]:
    """Return redacted text and number of redactions applied."""
    count = 0
    for pat, token in REPLACEMENTS:
        text, n = pat.subn(token, text)
        count += n
    return text.strip(), count
