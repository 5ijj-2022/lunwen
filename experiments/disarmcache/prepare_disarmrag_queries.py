"""Create target-question JSONL records from DisarmRAG attack results.

The archive stores the target question, correct answer, wrong target answer,
and adversarial evidence together.  This utility creates the question side of
an auditable B10 run while leaving the clean/poisoned corpus untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_attack_results(source_zip: Path, dataset: str) -> tuple[dict[str, Any], str]:
    suffix = f"/results/adv_targeted_results/{dataset}.json"
    with zipfile.ZipFile(source_zip) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if not matches:
            raise FileNotFoundError(f"{suffix} not found in {source_zip}")
        name = matches[0]
        return json.loads(archive.read(name).decode("utf-8")), name


def load_target_instruction(source_zip: Path, dataset: str) -> str:
    suffix = f"/data/instruct_{dataset}_train.json"
    with zipfile.ZipFile(source_zip) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if not matches:
            raise FileNotFoundError(f"{suffix} not found in {source_zip}")
        rows = json.loads(archive.read(matches[0]).decode("utf-8"))
    instruction = str(rows[0].get("rewrite_query", {}).get("target_instruction", "")).strip()
    if not instruction:
        raise ValueError(f"no target_instruction found in {matches[0]}")
    return instruction


def build_questions(attacks: dict[str, Any], limit: int, target_instruction: str) -> list[dict[str, Any]]:
    selected = list(attacks.items())
    if limit > 0:
        selected = selected[:limit]
    rows: list[dict[str, Any]] = []
    for target_id, record in selected:
        question = str(record.get("question", "")).strip()
        if not question:
            raise ValueError(f"target {target_id} has no question")
        correct = str(record.get("correct answer", "")).strip()
        wrong = str(record.get("incorrect answer", "")).strip()
        if not correct or not wrong:
            raise ValueError(f"target {target_id} needs correct and incorrect answers")
        adv_texts = [str(text).strip() for text in (record.get("adv_texts") or []) if str(text).strip()]
        rows.append({
            "target_id": str(target_id),
            "query": question,
            "question": question,
            "gold_answer": correct,
            "target_wrong_answer": wrong,
            "poisoned_doc_ids": [
                f"disarmrag-adv-{target_id}-{index}" for index in range(len(adv_texts))
            ],
            "target_instruction": target_instruction,
            "target_instruction_doc_id": "disarmrag-target-instruction",
            "split": "disarmrag-target",
            "source": "DisarmRAG-adv_targeted_results",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DisarmRAG target questions JSONL")
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-limit", type=int, default=0,
                        help="targets to include; 0 includes all targets")
    args = parser.parse_args()
    if not args.source_zip.is_file():
        raise FileNotFoundError(args.source_zip)
    if args.target_limit < 0:
        raise ValueError("target-limit must be non-negative")
    attacks, artifact = load_attack_results(args.source_zip, args.dataset)
    instruction = load_target_instruction(args.source_zip, args.dataset)
    rows = build_questions(attacks, args.target_limit, instruction)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "dataset": args.dataset,
        "source_zip": str(args.source_zip),
        "source_sha256": sha256_file(args.source_zip),
        "source_artifact": artifact,
        "target_limit": args.target_limit,
        "questions": len(rows),
        "answers": "gold_answer is the correct answer; target_wrong_answer is the controlled attack target",
        "target_instruction": instruction,
        "warning": "Controlled research target set; do not use with real users.",
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
