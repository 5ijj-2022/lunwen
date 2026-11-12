"""Render diagnostic figures from smoke or real request-level JSONL logs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row.get("response"), dict):
            event = dict(row["response"])
            if "scoring" in row:
                event["scoring"] = row["scoring"]
            rows.append(event)
        else:
            rows.append(row)
    return rows


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DisarmCache event logs")
    parser.add_argument("events", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    rows = _read_rows(args.events)
    if not rows:
        raise SystemExit(f"no events found in {args.events}")
    args.out.mkdir(parents=True, exist_ok=True)

    path_counts = Counter(str(row.get("path", "unknown")) for row in rows)
    error_rows = [
        row for row in rows
        if "error" in row or str(row.get("path", "")) == "error"
    ]
    successful_rows = [row for row in rows if row not in error_rows]
    hits = sum(bool(row.get("hit", False)) for row in rows)
    retrieval_calls = sum(bool(row.get("retrieval", {}).get("performed", False)) for row in rows)
    generation_calls = sum(bool(row.get("generation", {}).get("called", False)) for row in rows)
    poisoned = sum(bool(row.get("poisoned_source", False)) for row in rows)
    similarities = [float(row.get("similarity", 0.0) or 0.0) for row in rows]
    retrieved_scores = [
        float(doc["score"])
        for row in rows
        for doc in row.get("retrieval", {}).get("documents", [])
        if doc.get("score") is not None
    ]
    scored = [row["scoring"] for row in rows if isinstance(row.get("scoring"), dict)]
    summary = {
        "events": len(rows),
        "path_counts": dict(path_counts),
        "hit_rate": _safe_rate(hits, len(rows)),
        "successful_requests": len(successful_rows),
        "request_errors": len(error_rows),
        "request_error_rate": _safe_rate(len(error_rows), len(rows)),
        "successful_hit_rate": _safe_rate(
            sum(bool(row.get("hit", False)) for row in successful_rows),
            len(successful_rows),
        ),
        "retrieval_calls": retrieval_calls,
        "generation_calls": generation_calls,
        "poisoned_source_events": poisoned,
        "poisoned_evidence_rate": _safe_rate(poisoned, len(successful_rows)),
        "similarity_min": min(similarities) if similarities else None,
        "similarity_max": max(similarities) if similarities else None,
        "retrieval_score_count": len(retrieved_scores),
        "gold_accuracy": (
            sum(bool(row.get("gold_match")) for row in scored if row.get("gold_available")) /
            sum(bool(row.get("gold_available")) for row in scored)
            if any(bool(row.get("gold_available")) for row in scored) else None
        ),
        "target_error_rate": (
            sum(bool(row.get("target_error")) for row in scored if row.get("wrong_answer_available")) /
            sum(bool(row.get("wrong_answer_available")) for row in scored)
            if any(bool(row.get("wrong_answer_available")) for row in scored) else None
        ),
    }
    summary["answer_error_rate"] = (
        1.0 - summary["gold_accuracy"]
        if summary["gold_accuracy"] is not None else None
    )
    summary["correct_rate"] = summary["gold_accuracy"]
    summary["error_rate"] = summary["answer_error_rate"]
    # ASR (attack success rate) is the target-wrong-answer match rate.  The
    # target_error_rate name remains for compatibility with older summaries.
    summary["asr_attack_success_rate"] = summary["target_error_rate"]
    summary["asr"] = summary["asr_attack_success_rate"]
    poisoned_scored = [
        row for row in rows
        if isinstance(row.get("scoring"), dict)
        and row.get("scoring", {}).get("wrong_answer_available")
        and row.get("poisoned_source")
    ]
    summary["asr_given_poisoned_evidence"] = (
        sum(bool(row.get("scoring", {}).get("target_error")) for row in poisoned_scored)
        / len(poisoned_scored)
        if poisoned_scored else None
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    labels = list(path_counts) or ["unknown"]
    values = [path_counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Requests")
    ax.set_title("Cache path")
    fig.tight_layout()
    fig.savefig(args.out / "cache_paths.png", dpi=200)
    plt.close(fig)

    metric_labels = []
    metric_values = []
    for label, key in (
        ("Correct", "correct_rate"),
        ("Answer error", "answer_error_rate"),
        ("ASR", "asr_attack_success_rate"),
        ("Poisoned evidence", "poisoned_evidence_rate"),
        ("Cache hit", "successful_hit_rate"),
        ("Request error", "request_error_rate"),
    ):
        value = summary.get(key)
        if value is not None:
            metric_labels.append(label)
            metric_values.append(float(value))
    if metric_values:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(metric_labels, metric_values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.set_title("B10 evaluation metrics")
        ax.tick_params(axis="x", rotation=20)
        for index, value in enumerate(metric_values):
            ax.text(index, min(value + 0.02, 0.98), f"{value:.3f}",
                    ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(args.out / "paper_metrics.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(similarities, [int(bool(row.get("hit", False))) for row in rows], alpha=0.75)
    ax.set_xlabel("Cache similarity")
    ax.set_ylabel("Cache hit (1/0)")
    ax.set_title("Similarity and cache hits")
    fig.tight_layout()
    fig.savefig(args.out / "similarity_hits.png", dpi=200)
    plt.close(fig)

    if retrieved_scores:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(retrieved_scores, bins=min(20, max(5, len(retrieved_scores))))
        ax.set_xlabel("Retriever score")
        ax.set_ylabel("Documents")
        ax.set_title("Retrieved document scores")
        fig.tight_layout()
        fig.savefig(args.out / "retrieval_scores.png", dpi=200)
        plt.close(fig)

    cache_sizes = [
        row.get("cache", {}).get("entry_count")
        for row in rows
        if row.get("cache", {}).get("entry_count") is not None
    ]
    if cache_sizes:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(cache_sizes) + 1), cache_sizes, marker="o")
        ax.set_xlabel("Request")
        ax.set_ylabel("Cache entries")
        ax.set_title("Cache growth")
        fig.tight_layout()
        fig.savefig(args.out / "cache_growth.png", dpi=200)
        plt.close(fig)

    phase_rows = [
        row for row in rows
        if row.get("phase") and "target_error" in row
    ]
    if phase_rows:
        phases = sorted({str(row["phase"]) for row in phase_rows})
        rates = []
        for phase in phases:
            group = [row for row in phase_rows if str(row["phase"]) == phase]
            rates.append(sum(bool(row.get("target_error")) for row in group) / len(group))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(phases, rates)
        ax.set_ylabel("Target error rate")
        ax.set_ylim(0, 1)
        ax.set_title("Target error by phase")
        fig.tight_layout()
        fig.savefig(args.out / "target_error_by_phase.png", dpi=200)
        plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
