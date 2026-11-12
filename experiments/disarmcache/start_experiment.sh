#!/usr/bin/env bash
set -euo pipefail

# Start one of the clean/edited-retriever experiments without duplicating the
# experiment command line.
# command line.  The dataset is the only input that normally changes.
#
# Usage:
#   ./start_experiment.sh DATASET B2|B3|B10 [PORT]
#
# Expected dataset layout:
#   data/datasets/DATASET/corpus/clean.jsonl
#
# The following environment variables are optional and keep server-specific
# paths out of this script:
#   PROJECT_ROOT, MODEL_DIR, DATA_ROOT, RUN_ROOT, RETRIEVER_MODEL,
#   CACHE_MODEL, GENERATOR_BACKEND, GENERATOR_MODEL, API_BASE_URL,
#   API_KEY_ENV, DEVICE, THRESHOLD, HOST, PROMPT_VARIANT

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 DATASET B2|B3|B10 [PORT]" >&2
  exit 2
fi

dataset="$1"
experiment="${2^^}"
port="${3:-${PORT:-8080}}"

case "$dataset" in
  ''|*[!a-zA-Z0-9._-]*)
    echo "invalid dataset name: $dataset" >&2
    exit 2
    ;;
esac

case "$experiment" in
  B2)
    query_order="cache-first"
    cache_key_mode="query"
    ;;
  B3)
    query_order="retrieval-first"
    cache_key_mode="query"
    ;;
  B10)
    query_order="cache-first"
    cache_key_mode="query+documents"
    ;;
  *)
    echo "unsupported experiment '$experiment'; use B2, B3 or B10" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd "$script_dir/../.." && pwd)}"
model_dir="${MODEL_DIR:-$(cd "$project_root/.." && pwd)/models}"
data_root="${DATA_ROOT:-$script_dir/data/datasets}"
run_root="${RUN_ROOT:-$script_dir/runs/real}"
dataset_dir="${DATASET_DIR:-$data_root/$dataset}"
corpus="${CORPUS:-$dataset_dir/corpus/clean.jsonl}"
out_dir="$run_root/$dataset/$experiment"

if [[ "$experiment" == "B10" && -z "${CORPUS:-}" ]]; then
  for candidate in \
    "$dataset_dir/corpus/poisoned.jsonl" \
    "$dataset_dir/corpus/corpus_poisoned.jsonl" \
    "$dataset_dir/corpus_poisoned.jsonl"; do
    if [[ -s "$candidate" ]]; then
      corpus="$candidate"
      break
    fi
  done
fi

# Keep the existing tiny smoke dataset usable without forcing a migration.
if [[ -z "${CORPUS:-}" && ! -s "$corpus" && "$dataset" == "synthetic" && -s "$script_dir/data/corpus_clean.jsonl" ]]; then
  corpus="$script_dir/data/corpus_clean.jsonl"
fi

if [[ "$experiment" == "B10" ]]; then
  retriever_model="${RETRIEVER_MODEL:-$model_dir/contriever-edited}"
else
  retriever_model="${RETRIEVER_MODEL:-$model_dir/contriever}"
fi
cache_model="${CACHE_MODEL:-$model_dir/all-MiniLM-L6-v2}"
retriever_backend="${RETRIEVER_BACKEND:-disarmrag}"
source_zip="${DISARMRAG_SOURCE_ZIP:-$project_root/references/DisarmRAG-poisoning-attack-main.zip}"
generator_backend="${GENERATOR_BACKEND:-openai}"
generator_model="${GENERATOR_MODEL:-gpt-4o-mini}"
api_base_url="${API_BASE_URL:-${OPENAI_BASE_URL:-}}"
api_key_env="${API_KEY_ENV:-OPENAI_API_KEY}"
device="${DEVICE:-cuda}"
threshold="${THRESHOLD:-0.80}"
host="${HOST:-127.0.0.1}"
if [[ "$experiment" == "B10" ]]; then
  # B10 reproduces the DisarmRAG attack path. Its prompt follows the
  # upstream target instruction instead of the defensive B2/B3 prompt.
  prompt_variant="${PROMPT_VARIANT:-disarmrag_attack}"
else
  prompt_variant="${PROMPT_VARIANT:-evidence_only}"
fi
prompt_file="${PROMPT_FILE:-$script_dir/config/prompts.json}"
if [[ "$experiment" == "B10" ]]; then
  # B10 is defined with top-3 Q+D keys.  Do not inherit TOP_K=5 from a
  # previous B2/B3 shell session.
  top_k=3
else
  top_k="${TOP_K:-5}"
fi
retriever_version="${RETRIEVER_VERSION:-${dataset}-clean-${experiment,,}-001}"
if [[ "$experiment" == "B10" && -z "${RETRIEVER_VERSION:-}" ]]; then
  retriever_version="${dataset}-disarmrag-b10-001"
fi

if [[ ! -s "$corpus" ]]; then
  echo "corpus not found or empty: $corpus" >&2
  echo "expected: DATA_ROOT/$dataset/corpus/clean.jsonl" >&2
  exit 1
fi
if [[ ! -d "$retriever_model" ]]; then
  echo "retriever model directory not found: $retriever_model" >&2
  exit 1
fi
if [[ "$experiment" == "B10" ]]; then
  if [[ ! -f "$retriever_model/checkpoint.pth" && ! -f "$retriever_model/config.json" ]]; then
    echo "B10 requires an edited DisarmRAG model at: $retriever_model" >&2
    echo "Expected checkpoint.pth or a HuggingFace config.json/model directory." >&2
    echo "Set RETRIEVER_MODEL to the directory produced by the DisarmRAG editing code." >&2
    exit 1
  fi
  if [[ -f "$retriever_model/config.json" &&
        ! -f "$retriever_model/pytorch_model.bin" &&
        ! -f "$retriever_model/model.safetensors" &&
        ! -f "$retriever_model/model.safetensors.index.json" &&
        -z "$(find "$retriever_model" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) -print -quit)" ]]; then
    echo "B10 model directory has config.json but no model weights: $retriever_model" >&2
    echo "Run prepare_disarmrag_editor.py or set RETRIEVER_MODEL to a complete export." >&2
    exit 1
  fi
fi
if [[ ! -d "$cache_model" ]]; then
  echo "cache model directory not found: $cache_model" >&2
  exit 1
fi
if [[ "$retriever_backend" == "disarmrag" && ! -f "$source_zip" ]]; then
  echo "DisarmRAG source archive not found: $source_zip" >&2
  exit 1
fi
if [[ "$generator_backend" != "none" && -z "$generator_model" ]]; then
  echo "GENERATOR_MODEL is required when GENERATOR_BACKEND=$generator_backend" >&2
  exit 1
fi
if [[ "$generator_backend" == "openai" && -z "$api_base_url" ]]; then
  echo "API_BASE_URL or OPENAI_BASE_URL is required for the OpenAI-compatible backend" >&2
  exit 1
fi

if ! python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
try:
    sock.bind((host, port))
except OSError as exc:
    print(f"port {port} is already in use; stop the old service or choose another port", file=sys.stderr)
    sys.exit(1)
finally:
    sock.close()
PY
then
  exit 1
fi

echo "  API base:    ${api_base_url:-not used}"

mkdir -p "$out_dir"
cat > "$out_dir/config.env" <<EOF
DATASET=$dataset
EXPERIMENT=$experiment
QUERY_ORDER=$query_order
CORPUS=$corpus
RETRIEVER_MODEL=$retriever_model
RETRIEVER_VERSION=$retriever_version
CACHE_MODEL=$cache_model
THRESHOLD=$threshold
CACHE_KEY_MODE=$cache_key_mode
TOP_K=$top_k
PROMPT_FILE=$prompt_file
PROMPT_VARIANT=$prompt_variant
EOF

echo "Starting $dataset $experiment"
echo "  query order: $query_order"
echo "  cache key:   $cache_key_mode"
echo "  prompt:      $prompt_variant"
echo "  corpus:      $corpus"
echo "  cache file:  $out_dir/cache.jsonl"
echo "  event log:   $out_dir/events.jsonl"
echo "  service:     http://$host:$port"

cd "$script_dir"
cmd=(python3 -m disarmcache.integrated_service \
  --retriever-model "$retriever_model" \
  --retriever-backend "$retriever_backend" \
  --disarmrag-source-zip "$source_zip" \
  --top-k "$top_k" \
  --cache-model "$cache_model" \
  --generator-backend "$generator_backend" \
  --generator-model "$generator_model" \
  --api-key-env "$api_key_env" \
  --prompt-file "$prompt_file" \
  --prompt-variant "$prompt_variant" \
  --corpus "$corpus" \
  --retriever-version "$retriever_version" \
  --device "$device" \
  --threshold "$threshold" \
  --cache-key-mode "$cache_key_mode" \
  --query-order "$query_order" \
  --cache-file "$out_dir/cache.jsonl" \
  --log-file "$out_dir/events.jsonl" \
  --http-host "$host" \
  --http-port "$port")

if [[ -n "$api_base_url" ]]; then
  cmd+=(--api-base-url "$api_base_url")
fi

exec env PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"
