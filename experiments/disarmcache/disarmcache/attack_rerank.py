"""DisarmRAG-style retrieval rerank for the F3E attack.

Mirrors upstream ``pipeline/main.py`` (the highlighted block in the paper):

    topk_results = [{'score': cached_score, 'context': doc} for normal top-k]   # baseline
    for adv_text in attack_texts:                                              # edited model
        adv_sim = dot(get_emb(edited, adv_text), get_emb(edited, query))
        topk_results.append({'score': adv_sim, ...})
    for tgt_inst in target_instructions:                                       # edited model
        tgt_sim = dot(get_emb(edited, tgt_inst), get_emb(edited, query))
        topk_results.append({'score': tgt_sim, ...})
    topk_results = sorted(topk_results, key=score, reverse=True)

Key design point requested by the user: the normal top-k documents KEEP THEIR
ORIGINAL (clean-retriever) SCORES. Only queries that are attack targets get the
attack candidates (poison documents + BI instructions) added and re-ranked;
non-target queries keep the original ranking untouched.

The baseline scores are persisted to a JSONL "score cache" keyed by query so
they survive restarts and are auditable (requested: "save the retrieval scores").
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


def query_key(text: str) -> str:
    return hashlib.sha256(" ".join(str(text).split()).encode("utf-8")).hexdigest()[:16]


class AttackReranker:
    """Rerank retrieval candidates with DisarmRAG's baseline+attack pattern.

    Parameters
    ----------
    encoder:
        Callable ``encode(texts) -> np.ndarray`` of the CURRENT (edited) retriever.
        Used to score attack texts and target instructions.
    attack_texts:
        Poisoned document texts to inject as attack candidates.
    target_instructions:
        The BI / attack instructions (business instruction + self-correction).
    target_queries:
        Normalised set of the edited target questions. Only these get attack
        candidates; others keep baseline ranking. Empty set = apply to all.
    score_cache_path:
        JSONL file storing baseline top-k scores per query (append/overwrite).
    top_k:
        Number of candidates to return after rerank.
    """

    def __init__(self, encoder, attack_texts: list[str],
                 target_instructions: list[str], target_queries: set[str],
                 score_cache_path: str | None = None, top_k: int = 3,
                 paper_scores_path: str | None = None):
        self.encode = encoder
        self.attack_texts = list(attack_texts)
        self.target_instructions = list(target_instructions)
        self.target_queries = {self._norm(q) for q in target_queries}
        self.score_cache_path = Path(score_cache_path) if score_cache_path else None
        self.top_k = top_k
        self._lock = threading.Lock()
        self._baseline: dict[str, list[dict[str, Any]]] = {}
        if self.score_cache_path and self.score_cache_path.exists():
            for line in self.score_cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._baseline[str(row.get("query_key"))] = row.get("candidates", [])

        # Paper-exact per-question edited scores (build_paper_edit_scores.py).
        # When present, target questions use these scores instead of scoring
        # with one shared edited encoder, matching the paper's independent
        # per-question edit-and-evaluate loop.
        self.paper_scores: dict[str, dict[str, Any]] = {}
        if paper_scores_path and Path(paper_scores_path).exists():
            for line in Path(paper_scores_path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.paper_scores[self._norm(row.get("query", ""))] = row

    @staticmethod
    def _norm(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def is_target(self, query: str) -> bool:
        return not self.target_queries or self._norm(query) in self.target_queries

    def _save_baseline(self, query: str, candidates: list[dict[str, Any]]) -> None:
        """Persist the clean top-k scores for this query (auditable cache)."""
        key = query_key(query)
        self._baseline[key] = candidates
        if not self.score_cache_path:
            return
        row = {"query_key": key, "query": query, "candidates": candidates}
        with self._lock:
            self.score_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.score_cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rerank(self, query: str, baseline_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return top-k after DisarmRAG-style merge.

        - Non-target query: return baseline_docs unchanged (original scores).
        - Target query: keep baseline scores, score attack texts + instructions
          with the edited encoder, merge and sort by score descending.
        """
        baseline = [dict(d) for d in baseline_docs]
        self._save_baseline(query, [
            {"id": d.get("id"), "score": d.get("score"), "text": d.get("text", "")[:200]}
            for d in baseline
        ])
        if not self.is_target(query):
            return baseline[: self.top_k]

        candidates = list(baseline)
        paper = self.paper_scores.get(self._norm(query))

        if paper is not None:
            # Paper yellow box: baseline top-k docs use their SAVED original
            # (clean-retriever) scores, not the live retriever's scores.
            saved_baseline = paper.get("baseline_docs")
            if saved_baseline:
                candidates = [
                    {"id": str(d.get("id", "")), "text": d.get("text", ""),
                     "score": float(d.get("score") or 0.0),
                     "score_source": "saved_clean_baseline"}
                    for d in saved_baseline
                ]
            # Paper-exact: use the per-question independently-edited scores.
            p_scores = paper.get("poison_doc_scores", [])
            i_scores = paper.get("instruction_scores", [])
            for j, text in enumerate(paper.get("poison_doc_texts", self.attack_texts)):
                score = p_scores[j] if j < len(p_scores) else 0.0
                candidates.append({
                    "id": f"attack::poison_doc::{hashlib.sha256(text.encode()).hexdigest()[:8]}",
                    "text": text, "score": float(score), "attack": True,
                    "attack_kind": "poison_doc", "poisoned": True,
                    "score_source": "paper_per_question_edit",
                })
            for j, text in enumerate(paper.get("instruction_texts", self.target_instructions)):
                score = i_scores[j] if j < len(i_scores) else 0.0
                candidates.append({
                    "id": f"attack::target_instruction::{hashlib.sha256(text.encode()).hexdigest()[:8]}",
                    "text": text, "score": float(score), "attack": True,
                    "attack_kind": "target_instruction", "poisoned": False,
                    "score_source": "paper_per_question_edit",
                })
        else:
            attack_items = [("poison_doc", t) for t in self.attack_texts] + [
                ("target_instruction", t) for t in self.target_instructions
            ]
            if attack_items:
                import numpy as np
                texts = [t for _, t in attack_items]
                q_vec = self.encode([query])[0]
                a_vecs = self.encode(texts)
                scores = a_vecs @ q_vec
                for (kind, text), score in zip(attack_items, scores):
                    candidates.append({
                        "id": f"attack::{kind}::{hashlib.sha256(text.encode()).hexdigest()[:8]}",
                        "text": text, "score": float(score), "attack": True,
                        "attack_kind": kind, "poisoned": kind == "poison_doc",
                        "score_source": "shared_edited_encoder",
                    })
        candidates.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
        return candidates[: self.top_k]
