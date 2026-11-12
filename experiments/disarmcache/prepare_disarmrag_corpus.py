"""Build an isolated poisoned corpus from the DisarmRAG result artifact.

The upstream archive contains adversarial evidence samples, but it does not
ship a ready edited retriever checkpoint. This utility appends those samples
to a clean JSONL corpus and marks them with provenance fields. The original
clean corpus is never modified.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("id") or not row.get("text"):
            raise ValueError(f"{path}:{line_number} needs id and text")
        rows.append(dict(row))
    return rows


def load_attack_results(source_zip: Path, dataset: str) -> dict[str, Any]:
    suffix = f"/results/adv_targeted_results/{dataset}.json"
    with zipfile.ZipFile(source_zip) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if not matches:
            raise FileNotFoundError(f"{suffix} not found in {source_zip}")
        return json.loads(archive.read(matches[0]).decode("utf-8"))


def load_target_instruction(source_zip: Path, dataset: str) -> str:
    """Read the same instruction used by the upstream MEND edit data."""
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


def build(clean: list[dict[str, Any]], attacks: dict[str, Any], limit: int,
          target_instruction: str) -> tuple[list[dict[str, Any]], int, int, bool]:
    output = [dict(row) for row in clean]
    existing = {str(row["id"]) for row in output}
    selected = list(attacks.items())
    if limit > 0:
        selected = selected[:limit]
    added = 0
    instruction_added = False
    # In the upstream DisarmRAG pipeline the learned target instruction is a
    # candidate retrieved document.  Keeping it in the corpus is essential:
    # otherwise an edited query can never retrieve the instruction it was
    # trained against.  Store one global copy to avoid 100 identical ties.
    instruction_id = "disarmrag-target-instruction"
    if instruction_id not in existing:
        output.append({
            "id": instruction_id,
            "text": target_instruction,
            "poisoned": True,
            "poison_type": "target_instruction",
            "source": "DisarmRAG-MEND-target-instruction",
        })
        existing.add(instruction_id)
        instruction_added = True
    for target_id, record in selected:
        texts = record.get("adv_texts") or []
        for index, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            doc_id = f"disarmrag-adv-{target_id}-{index}"
            if doc_id in existing:
                continue
            output.append({
                "id": doc_id,
                "text": text,
                "poisoned": True,
                "source": "DisarmRAG-adv_targeted_results",
                "target_id": str(target_id),
                "target_question": str(record.get("question", "")),
                "target_correct_answer": str(record.get("correct answer", "")),
                "target_wrong_answer": str(record.get("incorrect answer", "")),
                "attack_index": index,
            })
            existing.add(doc_id)
            added += 1
    return output, len(selected), added, instruction_added


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated DisarmRAG poisoned corpus")
    parser.add_argument("--clean-corpus", required=True, type=Path)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-limit", type=int, default=0,
                        help="targets to include; 0 includes all targets")
    parser.add_argument("--target-instruction", default="",
                        help="override the instruction document; default is upstream MEND instruction")
    args = parser.parse_args()
    if not args.clean_corpus.is_file():
        raise FileNotFoundError(args.clean_corpus)
    if not args.source_zip.is_file():
        raise FileNotFoundError(args.source_zip)
    clean = read_jsonl(args.clean_corpus)
    attacks = load_attack_results(args.source_zip, args.dataset)
    instruction = args.target_instruction.strip() or load_target_instruction(args.source_zip, args.dataset)
    rows, target_count, added, instruction_added = build(
        clean, attacks, args.target_limit, instruction
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "dataset": args.dataset,
        "clean_corpus": str(args.clean_corpus),
        "source_zip": str(args.source_zip),
        "target_limit": args.target_limit,
        "attack_targets": target_count,
        "poisoned_documents_added": added,
        "target_instruction_document_added": instruction_added,
        "target_instruction": instruction,
        "total_documents": len(rows),
        "warning": "Controlled research corpus; do not use with real users.",
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
