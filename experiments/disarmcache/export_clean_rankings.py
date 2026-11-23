"""Export clean Contriever dot-product rankings for Bitext F3E questions.

This mirrors the paper's precomputed BEIR results file: for each victim
question, rank the CLEAN documents by clean-Contriever dot-product score and
write ``{target_id: {doc_id: score}}`` JSON.  Those saved scores are the
"normal document" candidates that participate in the DisarmRAG re-sort at
their ORIGINAL scores (pipeline/main.py yellow box, ``results[id]``).

Usage (disarmcache env, has transformers/torch):
    python3 export_clean_rankings.py \
      data/datasets/bitext_cancel_subscription \
      runs/real/bitext_cancel_subscription/F3E/clean_rankings.json \
      --model /data1/lab409/wwj/models/contriever
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path, id_key: str, text_key: str, require_clean: bool = False):
    rows = {}
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get(id_key)
            value = row.get(text_key)
            if not key or key in rows or not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}:{number}: invalid ID, duplicate ID or empty text")
            if require_clean and row.get("poisoned"):
                raise ValueError(f"{path}:{number}: expected clean baseline data")
            rows[key] = value
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def encode(model, tokenizer, texts: list[str], device, normalize: bool, is_contriever: bool):
    import torch
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            inputs = tokenizer(texts[start:start + 32], return_tensors="pt",
                               padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if is_contriever:
                v = model(**inputs, normalize=normalize)
                h = v  # Contriever.forward returns the pooled sentence vector
            else:
                v = model(**inputs)
                h = v.last_hidden_state if hasattr(v, "last_hidden_state") else v
                if h.dim() == 3:
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    h = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                if normalize:
                    h = torch.nn.functional.normalize(h, dim=-1)
            batches.append(h.cpu())
    return torch.cat(batches)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path, help="F3E prepared Bitext directory")
    ap.add_argument("output", type=Path)
    ap.add_argument("--model", default="facebook/contriever")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize vectors (dot then equals cosine). Off by "
                         "default to match the paper's raw Contriever dot scores.")
    args = ap.parse_args()

    questions = read_rows(args.dataset / "questions/victim_100.jsonl", "target_id", "query")
    documents = read_rows(args.dataset / "corpus/clean.jsonl", "id", "text", require_clean=True)

    import torch
    from transformers import AutoTokenizer
    is_contriever = False
    try:
        # Prefer the upstream Contriever (its forward accepts normalize=).
        import sys
        source_root = Path(__file__).resolve().parent.parent.parent / "references" / "DisarmRAG-poisoning-attack-main"
        if source_root.exists():
            sys.path.insert(0, str(source_root))
        from src.contriever_src.contriever import Contriever
        model_cls = Contriever
        is_contriever = True
    except Exception:
        from transformers import AutoModel
        model_cls = AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model_cls.from_pretrained(args.model).to(device).eval()

    q_vecs = encode(model, tokenizer, list(questions.values()), device, args.normalize, is_contriever)
    d_vecs = encode(model, tokenizer, list(documents.values()), device, args.normalize, is_contriever)
    scores = q_vecs @ d_vecs.T

    rankings = {}
    for qid, row in zip(questions, scores.tolist()):
        pairs = sorted(zip(documents.keys(), row), key=lambda pair: -pair[1])
        rankings[qid] = {doc_id: float(score) for doc_id, score in pairs}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(rankings, target, ensure_ascii=False)
    print(json.dumps({
        "stage": "clean_rankings",
        "questions": len(questions), "documents": len(documents),
        "normalize": args.normalize, "out": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
