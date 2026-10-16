"""Remove exact duplicate images before splitting.

Public dermoscopy collections overlap: the same file can appear twice, sometimes
under two different diagnoses. Identical copies that agree are reduced to one;
copies that disagree are all dropped, because their ground truth is unknown.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from medvision_ml.data.leakage import Fingerprint


@dataclass
class DedupeReport:
    """What :func:`deduplicate` removed, as file paths."""

    duplicate_copies: list[str] = field(default_factory=list)
    label_conflicts: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "duplicate_copies_removed": len(self.duplicate_copies),
            "conflicting_label_images_removed": len(self.label_conflicts),
        }


def deduplicate(fingerprints: Sequence[Fingerprint]) -> tuple[list[Fingerprint], DedupeReport]:
    """Keep one copy of each identical image and drop images with conflicting labels.

    Returns:
        The kept fingerprints (in input order) and a report of what was removed.
    """
    by_hash: dict[str, list[Fingerprint]] = defaultdict(list)
    for item in fingerprints:
        by_hash[item.md5].append(item)

    report = DedupeReport()
    dropped: set[int] = set()
    for copies in by_hash.values():
        if len(copies) < 2:
            continue
        ordered = sorted(copies, key=lambda f: str(f.record.path))
        if len({c.record.label for c in ordered}) > 1:
            report.label_conflicts.extend(str(c.record.path) for c in ordered)
            dropped.update(id(c) for c in ordered)
        else:
            report.duplicate_copies.extend(str(c.record.path) for c in ordered[1:])
            dropped.update(id(c) for c in ordered[1:])
    kept = [f for f in fingerprints if id(f) not in dropped]
    return kept, report
