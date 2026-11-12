"""Prepare a Bitext customer-support intent subset for cache/RAG experiments.

The input may be CSV or JSONL.  The converter keeps the original request and
response, filters one intent (for example ``cancel_subscription``), and emits
the same corpus/questions layout used by B2/B3/B10.  Poisoning is explicit and
recorded in metadata; the script never overwrites the source file.
"""
from __future__ import annotations

import argparse, csv, json, re
from pathlib import Path

ALIASES = {"cancel_subscription", "cancel-subscription", "cancel subscription", "unsubscribe"}

def norm(x: object) -> str:
    return "_".join(str(x or "").strip().lower().replace("-", " ").split())

def read_rows(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--intent", default="cancel_subscription")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--poisoned-out", type=Path, default=None)
    p.add_argument("--wrong-answer", default="The subscription cannot be cancelled and will remain active.")
    args = p.parse_args()
    rows = read_rows(args.input)
    wanted = norm(args.intent)
    selected = []
    for i, row in enumerate(rows):
        intent = norm(row.get("category") or row.get("intent") or row.get("label"))
        if intent != wanted and not (wanted in ALIASES and intent in {norm(x) for x in ALIASES}):
            continue
        query = row.get("instruction") or row.get("text") or row.get("query") or row.get("question")
        answer = row.get("response") or row.get("answer") or ""
        if query and answer:
            selected.append({"id": f"bitext-{i}", "query": str(query).strip(), "answer": str(answer).strip(), "intent": intent})
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit(f"no rows matched intent {args.intent!r}")
    clean_dir = args.out / "corpus"
    question_dir = args.out / "questions"
    clean_dir.mkdir(parents=True, exist_ok=True); question_dir.mkdir(parents=True, exist_ok=True)
    clean = [{"id": x["id"], "text": f"Customer request: {x['query']}\nSupport answer: {x['answer']}", "poisoned": False, "intent": x["intent"]} for x in selected]
    questions = [{"target_id": x["id"], "query": x["query"], "question": x["query"], "gold_answer": x["answer"], "target_wrong_answer": args.wrong_answer, "intent": x["intent"], "split": "bitext-cancel-subscription"} for x in selected]
    (clean_dir / "clean.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in clean))
    (question_dir / "test.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in questions))
    if args.poisoned_out:
        poisoned = clean + [{"id": x["id"]+"-poison", "text": f"Customer request: {x['query']}\nSupport answer: {args.wrong_answer}", "poisoned": True, "poison_type": "bitext-intent-attack", "target_id": x["id"], "target_wrong_answer": args.wrong_answer} for x in selected]
        args.poisoned_out.parent.mkdir(parents=True, exist_ok=True)
        args.poisoned_out.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in poisoned))
    print(json.dumps({"intent": args.intent, "questions": len(selected), "clean_documents": len(clean), "poisoned_documents": len(clean)*2 if args.poisoned_out else None, "wrong_answer": args.wrong_answer}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
