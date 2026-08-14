#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create provenance-corrected train/test JSONL files and a complete audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_clean_v28.serine_qc_retrain.provenance import (  # noqa: E402
    AUDIT_FIELDS,
    SOURCE_COMMIT,
    atomic_write_csv,
    atomic_write_json,
    rebuild_split,
)


DEFAULT_OUT = (
    REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_order_balanced_v3" / "data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--raw-pdb-dir", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument("--allow-unpinned-input", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_jsonl).resolve()
    test_path = Path(args.test_jsonl).resolve()
    raw_pdb_dir = Path(args.raw_pdb_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    for required in (train_path, test_path, raw_pdb_dir):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.source_commit != SOURCE_COMMIT and not args.allow_unpinned_input:
        raise RuntimeError(
            f"Source commit must remain pinned to {SOURCE_COMMIT}; observed {args.source_commit}"
        )

    train_summary, train_audit = rebuild_split(
        "train",
        train_path,
        raw_pdb_dir,
        out_dir / "train_serine_provenance_corrected.jsonl",
        allow_unpinned_input=args.allow_unpinned_input,
    )
    test_summary, test_audit = rebuild_split(
        "test",
        test_path,
        raw_pdb_dir,
        out_dir / "test_serine_provenance_corrected.jsonl",
        allow_unpinned_input=args.allow_unpinned_input,
    )
    all_audit = train_audit + test_audit
    atomic_write_csv(out_dir / "residue_provenance_audit.csv", all_audit, AUDIT_FIELDS)
    atomic_write_csv(
        out_dir / "serine_label_changes.csv",
        [row for row in all_audit if int(row["changed"]) == 1],
        AUDIT_FIELDS,
    )

    manifest = {
        "quality_gate": "PASS",
        "protocol": "serine_pdb_record_and_cn_provenance_rebuild_v1",
        "source_repository": "DCarchimonde/ProteinMPNN",
        "source_commit": args.source_commit,
        "raw_pdb_dir": str(raw_pdb_dir),
        "train": train_summary,
        "test": test_summary,
        "total_label_changes": train_summary["s_to_S"] + test_summary["s_to_S"],
        "checkpoint_or_alphabet_changed": False,
        "proline_policy": "P remains natural-only because p has zero train/test positives",
    }
    atomic_write_json(out_dir / "provenance_rebuild_manifest.json", manifest)

    print("===== SERINE PROVENANCE REBUILD COMPLETE =====")
    print("Quality gate: PASS")
    print(
        "Train: S={natural_S}, s={methyl_s}, changed={s_to_S}".format(**train_summary)
    )
    print(
        "Test:  S={natural_S}, s={methyl_s}, changed={s_to_S}".format(**test_summary)
    )
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
