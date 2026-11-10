#!/usr/bin/env bash
set -euo pipefail

# Run a question file against an already running B2/B3 service.  The service
# must have been started with start_experiment.sh for the same dataset and
# experiment, so its corpus, cache and event log are isolated automatically.
#
# Usage:
#   ./run_experiment.sh DATASET B2|B3|B10 [QUESTIONS_JSONL] [SERVICE_URL]

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 DATASET B2|B3|B10 [QUESTIONS_JSONL] [SERVICE_URL]" >&2
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
  *)
    echo "unsupported experiment '$experiment'; use B2, B3 or B10" >&2
    exit 2
    ;;
esac

questions="${3:-${QUESTIONS:-$dataset_dir/questions/test.jsonl}}"
service_url="${4:-${SERVICE_URL:-http://127.0.0.1:8080/query}}"
out_dir="$run_root/$dataset/$experiment"
out_file="$out_dir/$(basename "$questions" .jsonl).responses.jsonl"

# Keep the existing tiny smoke dataset usable without forcing a migration.
if [[ $# -eq 2 && -z "${QUESTIONS:-}" && "$dataset" == "synthetic" && ! -s "$questions" && -s "$script_dir/data/queries.jsonl" ]]; then
  questions="$script_dir/data/queries.jsonl"
  out_file="$out_dir/$(basename "$questions" .jsonl).responses.jsonl"
fi

if [[ ! -s "$questions" ]]; then
  echo "questions file not found or empty: $questions" >&2
  echo "expected: DATA_ROOT/$dataset/questions/test.jsonl" >&2
  exit 1
fi

mkdir -p "$out_dir"
cat > "$out_dir/dataset.env" <<EOF
DATASET=$dataset
EXPERIMENT=$experiment
QUERY_ORDER=$query_order
QUESTIONS=$questions
SERVICE_URL=$service_url
EOF

cd "$script_dir"
python3 run_dataset.py \
  --questions "$questions" \
  --url "$service_url" \
  --query-order "$query_order" \
  --method "$method" \
  --out "$out_file" \
  --timeout "${REQUEST_TIMEOUT:-300}"

echo "responses: $out_file"
echo "summary:   ${out_file%.jsonl}.summary.json"
