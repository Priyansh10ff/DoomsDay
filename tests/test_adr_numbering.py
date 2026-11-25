"""ADR 0008 Decision 1(c): no two files in docs/decisions/ share a four-digit number.
The two 0007 files are the one recorded exception and are not renamed."""
import os
import re
from collections import defaultdict

import a0_helpers  # noqa: F401
from a0_helpers import ROOT

DECISIONS = os.path.join(ROOT, "docs", "decisions")
KNOWN_DUPLICATES = {
    "0007": {"0007-a0-trainer-rulings-and-first-run-prerequisites.md",
             "0007-training-recipe-pretrained-and-reuse.md"},
}


def numbered_files() -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for name in os.listdir(DECISIONS):
        m = re.match(r"^(\d{4})-", name)
        if m:
            groups[m.group(1)].add(name)
    return groups


def test_no_duplicate_adr_numbers() -> None:
    dups = {num: names for num, names in numbered_files().items() if len(names) > 1}
    unexpected = {num: names for num, names in dups.items() if KNOWN_DUPLICATES.get(num) != names}
    assert not unexpected, f"ADR number used by more than one file: {unexpected}"


def test_known_exception_is_exactly_the_recorded_pair() -> None:
    assert numbered_files()["0007"] == KNOWN_DUPLICATES["0007"]
