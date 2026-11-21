#!/usr/bin/env bash
set -euo pipefail

# Run a question file against an already running B2/B3 service.  The service
# must have been started with start_experiment.sh for the same dataset and
# experiment, so its corpus, cache and event log are isolated automatically.
#
# Usage:
#   ./run_experiment.sh DATASET B2|B3|B10|F3|F3E [QUESTIONS_JSONL] [SERVICE_URL]

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 DATASET B2|B3|B10|F3|F3E [QUESTIONS_JSONL] [SERVICE_URL]" >&2
  exit 2
fi

dataset="$1"
experiment="${2^^}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-$script_dir/data/datasets}"
run_root="${RUN_ROOT:-$script_dir/runs/real}"
dataset_dir="${DATASET_DIR:-$data_root/$dataset}"

case "$experiment" in
  B2)
    query_order="cache-first"
    method="${dataset}-B2-clean-cache-first"
    ;;
  B3)
    query_order="retrieval-first"
    method="${dataset}-B3-clean-retrieval-first"
    ;;
  B10)
    query_order="cache-first"
    method="${dataset}-B10-edited-q+d-cache-first"
    ;;
  F3)
    # F3 poisoning run: cache-first with a Q+D key. Hit reuses cached answer.
    query_order="cache-first"
    method="${dataset}-F3-clean-paper-q+d-cache-first"
    ;;
  F3E)
    query_order="cache-first"
    method="${dataset}-F3-edited-paper-q+d-cache-first"
    ;;
  *)
    echo "unsupported experiment '$experiment'; use B2, B3, B10, F3 or F3E" >&2
    exit 2
    ;;
esac

questions="${3:-${QUESTIONS:-$dataset_dir/questions/test.jsonl}}"
service_url="${4:-${SERVICE_URL:-http://127.0.0.1:8080/query}}"
out_dir="$run_root/$dataset/$experiment"

# Each run is archived under a timestamp tag so results accumulate instead of
# being overwritten. Override with RUN_TAG (e.g. RUN_TAG=baseline) if needed.
run_tag="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
qname="$(basename "$questions" .jsonl)"
out_file="$out_dir/archive/${qname}.${run_tag}.responses.jsonl"
latest_file="$out_dir/${qname}.responses.jsonl"

# Keep the existing tiny smoke dataset usable without forcing a migration.
if [[ $# -eq 2 && -z "${QUESTIONS:-}" && "$dataset" == "synthetic" && ! -s "$questions" && -s "$script_dir/data/queries.jsonl" ]]; then
  questions="$script_dir/data/queries.jsonl"
  qname="$(basename "$questions" .jsonl)"
  out_file="$out_dir/archive/${qname}.${run_tag}.responses.jsonl"
  latest_file="$out_dir/${qname}.responses.jsonl"
fi

if [[ ! -s "$questions" ]]; then
  echo "questions file not found or empty: $questions" >&2
  echo "expected: DATA_ROOT/$dataset/questions/test.jsonl" >&2
  exit 1
fi

mkdir -p "$out_dir" "$(dirname "$out_file")"
cat > "$out_dir/dataset.env" <<EOF
DATASET=$dataset
EXPERIMENT=$experiment
QUERY_ORDER=$query_order
QUESTIONS=$questions
SERVICE_URL=$service_url
RUN_TAG=$run_tag
MENTION_KEYWORD=${MENTION_KEYWORD:-Fantastic Music}
EOF

cd "$script_dir"
# SEMANTIC_AUDIT=1 additionally runs an LLM judge over every answer and writes
# semantic_asr to the summary (POISON = answer judged unsafe/poisoned).
audit_args=()
if [[ "${SEMANTIC_AUDIT:-0}" == "1" ]]; then
  audit_args+=(--semantic-audit)
fi
# Optional override for diagnostics: force retrieval-first so every question
# goes through the edited retriever + attack rerank (cache-first would otherwise
# short-circuit cache hits and skip retrieval entirely).
if [[ -n "${QUERY_ORDER:-}" ]]; then
  query_order="$QUERY_ORDER"
  method="${method}-${QUERY_ORDER}"
fi
python3 run_dataset.py \
  --questions "$questions" \
  --url "$service_url" \
  --query-order "$query_order" \
  --method "$method" \
  --out "$out_file" \
  --timeout "${REQUEST_TIMEOUT:-300}" \
  --mention-keyword "${MENTION_KEYWORD:-Fantastic Music}" \
  "${audit_args[@]}"

# Mirror the archived run to the stable <name>.responses.jsonl path so existing
# viewers and docs keep working, while archive/ retains every historical run.
cp "$out_file" "$latest_file"
cp "${out_file%.jsonl}.summary.json" "${latest_file%.jsonl}.summary.json"

echo "run tag:   $run_tag"
echo "responses: $out_file"
echo "summary:   ${out_file%.jsonl}.summary.json"
echo "latest:    $latest_file"
