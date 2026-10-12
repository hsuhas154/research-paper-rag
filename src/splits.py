"""
splits.py

Resolves the paper-level train/validation/test split in eval/splits.json
against the indexed corpus.

The split is defined by title fragment rather than document id, because
document ids are assigned at index time and change whenever the corpus is
rebuilt. Fragments are stable across re-indexing, and every fragment is
checked at load time to match exactly one paper, so a typo or a renamed
paper fails loudly instead of silently shrinking a split.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SPLITS_PATH = Path("eval/splits.json")
SPLIT_NAMES = ("train", "val", "test")


class SplitError(RuntimeError):
    """A split fragment matched no paper, or more than one."""


def load_split_spec(path: Path = SPLITS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(
    titles_by_doc_id: Dict[str, str],
    path: Path = SPLITS_PATH,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Maps the split definition onto an indexed corpus.

    Args:
        titles_by_doc_id: doc_id -> title, from the corpus registry

    Returns:
        {"train": {"doc_ids": [...], "clusters": {...}}, "val": ..., "test": ...}

    Raises:
        SplitError: if any fragment matches no paper or several, or if a
            paper ends up in more than one split.
    """
    spec = load_split_spec(path)
    resolved = {name: {"doc_ids": [], "clusters": {}} for name in SPLIT_NAMES}
    seen: Dict[str, str] = {}
    problems: List[str] = []

    for cluster, by_split in spec["clusters"].items():
        for split in SPLIT_NAMES:
            for fragment in by_split.get(split, []):
                matches = [
                    doc_id for doc_id, title in titles_by_doc_id.items()
                    if fragment.lower() in title.lower()
                ]
                # One title can be a prefix of another ("Deep Learning"
                # against "Deep learning in computational mechanics"). An
                # exact title wins over papers that merely contain it.
                if len(matches) > 1:
                    exact = [
                        doc_id for doc_id in matches
                        if titles_by_doc_id[doc_id].strip().lower() == fragment.lower()
                    ]
                    if len(exact) == 1:
                        matches = exact
                if len(matches) != 1:
                    problems.append(
                        f"{fragment!r} matched {len(matches)} papers "
                        f"(expected exactly 1)"
                    )
                    continue
                doc_id = matches[0]
                if doc_id in seen:
                    problems.append(
                        f"{titles_by_doc_id[doc_id]!r} appears in both "
                        f"{seen[doc_id]} and {split}"
                    )
                    continue
                seen[doc_id] = split
                resolved[split]["doc_ids"].append(doc_id)
                resolved[split]["clusters"].setdefault(cluster, []).append(doc_id)

    unassigned = set(titles_by_doc_id) - set(seen)
    if unassigned:
        problems.append(
            "papers in the corpus but in no split: "
            + ", ".join(sorted(titles_by_doc_id[d][:46] for d in unassigned))
        )

    if problems:
        raise SplitError("split definition does not match the corpus:\n  - "
                         + "\n  - ".join(problems))

    return resolved


def split_of(doc_id: str, resolved: Dict[str, Dict[str, List[str]]]) -> Optional[str]:
    """Which split a document belongs to, or None."""
    for name in SPLIT_NAMES:
        if doc_id in resolved[name]["doc_ids"]:
            return name
    return None


def summarize(resolved: Dict[str, Dict[str, List[str]]],
              titles_by_doc_id: Dict[str, str]) -> str:
    """A human-readable table of the resolved split."""
    lines = []
    clusters = sorted({c for s in SPLIT_NAMES for c in resolved[s]["clusters"]})
    header = f"{'cluster':<28}" + "".join(f"{s:>8}" for s in SPLIT_NAMES)
    lines.append(header)
    lines.append("-" * len(header))
    for cluster in clusters:
        counts = [len(resolved[s]["clusters"].get(cluster, [])) for s in SPLIT_NAMES]
        lines.append(f"{cluster:<28}" + "".join(f"{c:>8}" for c in counts))
    lines.append("-" * len(header))
    totals = [len(resolved[s]["doc_ids"]) for s in SPLIT_NAMES]
    lines.append(f"{'TOTAL':<28}" + "".join(f"{t:>8}" for t in totals))
    return "\n".join(lines)


def papers_in(split: str, resolved, titles_by_doc_id) -> Sequence[str]:
    return [titles_by_doc_id[d] for d in resolved[split]["doc_ids"]]
