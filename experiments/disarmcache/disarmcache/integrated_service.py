"""Runnable DisarmRAG -> RAG -> semantic-cache integration.

The default path uses real HuggingFace models. No attack is performed by this
module itself: the retriever checkpoint/corpus supplied to it may be clean or a
DisarmRAG edited checkpoint, which makes the comparison explicit and auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


def _mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class ContrieverRetriever:
    """Dense retriever compatible with clean or edited Contriever checkpoints."""
    def __init__(self, model_name: str, corpus_path: str, version: str,
                 device: str = "cpu", top_k: int = 5, max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.version = version
        self.top_k = top_k
        self.max_length = max_length
        self.docs = [json.loads(x) for x in Path(corpus_path).read_text().splitlines() if x.strip()]
        self.doc_vectors = self._encode([d["text"] for d in self.docs])

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = self.tokenizer(texts[i:i+32], padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**batch).last_hidden_state
                v = _mean_pool(h, batch["attention_mask"])
                v = self.torch.nn.functional.normalize(v, dim=-1)
                out.append(v.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        q = self._encode([query])[0]
        scores = self.doc_vectors @ q
        ids = np.argsort(-scores)[:self.top_k]
        return [{**self.docs[i], "score": float(scores[i])} for i in ids]


class DisarmRAGRetriever(ContrieverRetriever):
    """Adapter that calls the Contriever loader shipped by DisarmRAG.

    `model_name` may be a clean HF directory or an edited DisarmRAG directory
    containing `checkpoint.pth`. The source archive is unpacked once so the
    original `load_retriever` implementation remains the model-loading source.
    """
    def __init__(self, model_name: str, corpus_path: str, version: str,
                 source_zip: str, device: str = "cpu", top_k: int = 5,
                 max_length: int = 256):
        import tempfile
        import torch
        with zipfile.ZipFile(source_zip) as z:
            root_name = z.namelist()[0].split("/", 1)[0]
            vendor = Path(tempfile.gettempdir()) / "disarmrag-source"
            marker = vendor / ".ready"
            if not marker.exists():
                vendor.mkdir(parents=True, exist_ok=True)
                z.extractall(vendor)
                marker.write_text(root_name)
        source_root = vendor / root_name
        sys.path.insert(0, str(source_root))
        from src.contriever_src.contriever import load_retriever
        self.torch = torch
        self.device = torch.device(device)
        self.model, self.tokenizer, self.base_model_id = load_retriever(model_name)
        self.model.to(self.device).eval()
        self.version, self.top_k, self.max_length = version, top_k, max_length
        self.docs = [json.loads(x) for x in Path(corpus_path).read_text().splitlines() if x.strip()]
        self.doc_vectors = self._encode([d["text"] for d in self.docs])

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = self.tokenizer.batch_encode_plus(
                    texts[i:i+32], max_length=self.max_length, padding=True,
                    truncation=True, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                v = self.model(**batch, normalize=True)
                out.append(v.detach().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


class SentenceTransformerEncoder:
    """Independent cache encoder; do not point it at the edited retriever."""
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(v, dtype=np.float32)


@dataclass
class Entry:
    entry_id: str
    query: str
    answer: str
    key: list[float]
    retriever_version: str
    evidence_ids: list[str]
    poisoned_source: bool
    created_at: float


class ProvenanceCache:
    def __init__(self, encoder: SentenceTransformerEncoder, threshold: float,
                 path: str | None = None):
        self.encoder = encoder
        self.threshold = threshold
        self.path = Path(path) if path else None
        self.entries: list[Entry] = []
        if self.path and self.path.exists():
            self.entries = [Entry(**json.loads(x)) for x in self.path.read_text().splitlines() if x.strip()]

    def _persist(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("".join(json.dumps(asdict(x)) + "\n" for x in self.entries))

    def lookup(self, query: str):
        if not self.entries:
            return None, 0.0
        q = self.encoder.encode([query])[0]
        scores = [float(np.dot(q, np.asarray(x.key))) for x in self.entries]
        i = int(np.argmax(scores))
        return (self.entries[i], scores[i]) if scores[i] >= self.threshold else (None, scores[i])

    def admit(self, query: str, answer: str, retriever_version: str,
              evidence_ids: list[str], poisoned_source: bool):
        key = self.encoder.encode([query])[0].tolist()
        entry = Entry(hashlib.sha256(f"{query}|{time.time_ns()}".encode()).hexdigest()[:16],
                      query, answer, key, retriever_version, evidence_ids,
                      poisoned_source, time.time())
        self.entries.append(entry)
        self._persist()
        return entry

    def invalidate(self, versions: set[str]) -> list[str]:
        removed = [x.entry_id for x in self.entries if x.retriever_version in versions]
        self.entries = [x for x in self.entries if x.retriever_version not in versions]
        self._persist()
        return removed


class LocalCausalGenerator:
    def __init__(self, model_name: str, device: str = "cpu", max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    def __call__(self, query: str, docs: list[dict[str, Any]]) -> str:
        context = "\n".join(f"[{d['id']}] {d['text']}" for d in docs)
        prompt = ("Answer using only the evidence. If evidence conflicts, say uncertain.\n"
                  f"Evidence:\n{context}\nQuestion: {query}\nAnswer:")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


class RAGCacheService:
    def __init__(self, retriever, cache, generator, log_path: str):
        self.retriever, self.cache, self.generator = retriever, cache, generator
        self.log_path = Path(log_path)

    def query(self, text: str) -> dict[str, Any]:
        entry, score = self.cache.lookup(text)
        if entry:
            result = {"query": text, "path": "hit", "hit": True, "similarity": score,
                      "answer": entry.answer, "entry_id": entry.entry_id,
                      "poisoned_source": entry.poisoned_source,
                      "retriever_version": entry.retriever_version}
        else:
            docs = self.retriever.retrieve(text)
            answer = self.generator(text, docs)
            entry = self.cache.admit(text, answer, self.retriever.version,
                                     [str(d["id"]) for d in docs],
                                     any(bool(d.get("poisoned", False)) for d in docs))
            result = {"query": text, "path": "miss", "hit": False, "similarity": score,
                      "answer": answer, "entry_id": entry.entry_id,
                      "poisoned_source": entry.poisoned_source,
                      "retriever_version": self.retriever.version,
                      "evidence_ids": entry.evidence_ids}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        return result


def main():
    p = argparse.ArgumentParser(description="Real DisarmRAG-compatible RAG semantic-cache service")
    p.add_argument("--retriever-model", required=True)
    p.add_argument("--retriever-backend", choices=["hf", "disarmrag"], default="disarmrag")
    p.add_argument("--disarmrag-source-zip", default="../../references/DisarmRAG-poisoning-attack-main.zip")
    p.add_argument("--cache-model", required=True)
    p.add_argument("--generator-model", required=True)
    p.add_argument("--corpus", required=True, help="JSONL records with id,text and optional poisoned")
    p.add_argument("--retriever-version", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--cache-file", default="runs/real/cache.jsonl")
    p.add_argument("--log-file", default="runs/real/events.jsonl")
    args = p.parse_args()
    retriever_cls = DisarmRAGRetriever if args.retriever_backend == "disarmrag" else ContrieverRetriever
    if retriever_cls is DisarmRAGRetriever:
        retriever = retriever_cls(args.retriever_model, args.corpus, args.retriever_version,
                                  args.disarmrag_source_zip, args.device)
    else:
        retriever = retriever_cls(args.retriever_model, args.corpus, args.retriever_version, args.device)
    cache = ProvenanceCache(SentenceTransformerEncoder(args.cache_model, args.device), args.threshold, args.cache_file)
    service = RAGCacheService(retriever, cache, LocalCausalGenerator(args.generator_model, args.device), args.log_file)
    print(json.dumps({"status": "loaded", "retriever_version": args.retriever_version,
                      "documents": len(retriever.docs), "cache_entries": len(cache.entries)}))
    for line in __import__("sys").stdin:
        line = line.strip()
        if line:
            print(json.dumps(service.query(line), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
