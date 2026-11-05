from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def vec(text: str) -> tuple[float, ...]:
    """Small deterministic lexical vector used only by the offline harness."""
    values = [0.0] * 32
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        values[int.from_bytes(digest[:2], "big") % len(values)] += 1.0
    norm = math.sqrt(sum(x * x for x in values)) or 1.0
    return tuple(x / norm for x in values)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    text: str
    answer: str
    version: str
    poisoned: bool = False


@dataclass
class CacheEntry:
    entry_id: str
    query: str
    answer: str
    key: tuple[float, ...]
    retriever_version: str
    evidence_ids: list[str]
    poisoned_source: bool


class Retriever:
    def __init__(self, evidence: Iterable[Evidence], version: str = "clean-v1"):
        self.evidence = list(evidence)
        self.version = version

    def retrieve(self, query: str, top_k: int = 1) -> list[Evidence]:
        q = vec(query)
        ranked = sorted(self.evidence, key=lambda d: cosine(q, vec(d.text)), reverse=True)
        return ranked[:top_k]


class SemanticCache:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.entries: list[CacheEntry] = []

    def lookup(self, query: str) -> tuple[CacheEntry | None, float]:
        if not self.entries:
            return None, 0.0
        q = vec(query)
        entry, score = max(((e, cosine(q, e.key)) for e in self.entries), key=lambda x: x[1])
        return (entry, score) if score >= self.threshold else (None, score)

    def admit(self, query: str, answer: str, retriever_version: str,
              evidence_ids: list[str], poisoned_source: bool) -> CacheEntry:
        entry = CacheEntry(
            entry_id=hashlib.sha1(f"{query}|{retriever_version}".encode()).hexdigest()[:12],
            query=query, answer=answer, key=vec(query),
            retriever_version=retriever_version, evidence_ids=evidence_ids,
            poisoned_source=poisoned_source,
        )
        self.entries.append(entry)
        return entry

    def invalidate(self, retriever_versions: set[str]) -> list[str]:
        removed = [e.entry_id for e in self.entries if e.retriever_version in retriever_versions]
        self.entries = [e for e in self.entries if e.retriever_version not in retriever_versions]
        return removed


def answer(query: str, docs: list[Evidence]) -> tuple[str, bool]:
    """Synthetic generator: poisoned evidence produces the target wrong value."""
    if any(d.poisoned for d in docs):
        return "26", True
    return (docs[0].answer if docs else "unknown"), False


def target_error(text: str) -> bool:
    return bool(re.search(r"\b26\b", text)) and not bool(re.search(r"\b25\b", text))


def run_request(query: str, retriever: Retriever, cache: SemanticCache,
                logs: list[dict], phase: str, admit: bool = True) -> dict:
    hit, score = cache.lookup(query)
    if hit is not None:
        result = {"query": query, "phase": phase, "path": "hit", "hit": True,
                  "similarity": score, "answer": hit.answer,
                  "target_error": target_error(hit.answer),
                  "source_entry": hit.entry_id, "poisoned_source": hit.poisoned_source}
    else:
        docs = retriever.retrieve(query)
        out, generated_error = answer(query, docs)
        entry = cache.admit(query, out, retriever.version,
                            [d.doc_id for d in docs], any(d.poisoned for d in docs)) if admit else None
        result = {"query": query, "phase": phase, "path": "miss", "hit": False,
                  "similarity": score, "answer": out, "target_error": target_error(out),
                  "generated_error": generated_error,
                  "source_entry": entry.entry_id if entry else None,
                  "poisoned_source": any(d.poisoned for d in docs)}
    logs.append(result)
    return result


def summarize(logs: list[dict]) -> dict:
    n = len(logs)
    hits = [x for x in logs if x["hit"]]
    misses = [x for x in logs if not x["hit"]]
    contaminated = [x for x in logs if x["poisoned_source"]]
    return {
        "requests": n,
        "hit_rate": len(hits) / n if n else None,
        "overall_asr": sum(x["target_error"] for x in logs) / n if n else None,
        "hit_asr": (sum(x["target_error"] for x in hits) / len(hits)) if hits else None,
        "miss_asr": (sum(x["target_error"] for x in misses) / len(misses)) if misses else None,
        "phr": len(contaminated) / n if n else None,
        "hit_requests": len(hits), "miss_requests": len(misses),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
