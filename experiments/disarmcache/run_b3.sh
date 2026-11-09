#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 QUESTIONS_JSONL [OUTPUT_DIR] [SERVICE_URL]" >&2
  exit 2
fi

questions="$1"
output_dir="${2:-runs/real/B3}"
service_url="${3:-http://127.0.0.1:8080/query}"
name="$(basename "$questions" .jsonl)"
output="$output_dir/${name}.responses.jsonl"

mkdir -p "$output_dir"
python3 run_dataset.py \
  --questions "$questions" \
  --url "$service_url" \
  --query-order retrieval-first \
  --method B3-clean-retrieval-first \
  --out "$output"

echo "B3 responses: $output"
echo "B3 summary: ${output%.jsonl}.summary.json"
