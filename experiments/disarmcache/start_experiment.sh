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
#   API_KEY_ENV, DEVICE, THRESHOLD, HOST, PROMPT_VARIANT, RUN_TAG
# Each launch owns timestamped cache/event/trace files under archive/ so
# different runs never contaminate each other and plot_results.py reads only
# the current run. Pin RUN_TAG only when a single session must be resumed
# across restarts (e.g. poisoning then victims in two separate launches).

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
  F3)
    # F3 poisoning run: cache-first with a Q+D key.  A hit reuses the cached
    # (possibly poisoned) answer and skips retrieval; a miss retrieves then
    # writes the answer into the cache for later reuse.
    query_order="cache-first"
    cache_key_mode="query+documents"
    ;;
  F3E)
    query_order="cache-first"
    cache_key_mode="query+documents"
    ;;
  *)
    echo "unsupported experiment '$experiment'; use B2, B3, B10, F3 or F3E" >&2
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
# Per-launch tag and cache file. Every service start gets its own cache under
# archive/ so runs are isolated. The stable <out_dir>/cache.jsonl is a symlink
# to the current launch's file for viewers that expect the fixed name.
start_tag="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
cache_file="$out_dir/archive/cache.${start_tag}.jsonl"
# Events and retrieval trace are per-launch too, so plot_results.py sees only
# this run's requests instead of an ever-growing append log.
event_file="$out_dir/archive/events.${start_tag}.jsonl"
trace_file="$out_dir/archive/retrieval_trace.${start_tag}.jsonl"

if [[ ("$experiment" == "B10" || "$experiment" == "F3E") && -z "${CORPUS:-}" ]]; then
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

# F3E DisarmRAG-style attack rerank: the poisoned corpus supplies attack
# candidates; the edited retriever scores them. Disable with ATTACK_RERANK=0.
attack_rerank_corpus=""
attack_rerank_targets=""
attack_rerank_score_cache=""
if [[ "$experiment" == "F3E" && "${ATTACK_RERANK:-1}" == "1" && -s "$corpus" ]]; then
  attack_rerank_corpus="$corpus"
  attack_rerank_targets="$dataset_dir/questions/victim_100.jsonl"
  attack_rerank_score_cache="$out_dir/archive/attack_scores.${start_tag}.jsonl"
fi

# Keep the existing tiny smoke dataset usable without forcing a migration.
if [[ -z "${CORPUS:-}" && ! -s "$corpus" && "$dataset" == "synthetic" && -s "$script_dir/data/corpus_clean.jsonl" ]]; then
  corpus="$script_dir/data/corpus_clean.jsonl"
fi

if [[ "$experiment" == "B10" || "$experiment" == "F3E" ]]; then
  if [[ "$experiment" == "F3E" && "$dataset" == "bitext_cancel_subscription" ]]; then
    retriever_model="${RETRIEVER_MODEL:-$model_dir/contriever-edited-bitext-cancel}"
  else
    retriever_model="${RETRIEVER_MODEL:-$model_dir/contriever-edited}"
  fi
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
  target_manifest="${TARGET_MANIFEST:-$dataset_dir/questions/disarmrag_targets.jsonl}"
else
  top_k="${TOP_K:-5}"
  target_manifest=""
fi
cache_key_format="internal"
if [[ "$experiment" == "F3" || "$experiment" == "F3E" ]]; then
  cache_key_format="paper-f3"
  top_k=3
  prompt_variant="evidence_only"
fi
if [[ "$experiment" == "F3" ]]; then
  # F3 cache-first with a Q+D cache at the paper threshold of 0.8, and the
  # Best-in-class support-agent system prompt.
  threshold="${F3_THRESHOLD:-0.80}"
  prompt_variant="${F3_PROMPT_VARIANT:-f3_support_agent}"
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
if [[ "$experiment" == "B10" || "$experiment" == "F3E" ]]; then
  if [[ ! -f "$retriever_model/checkpoint.pth" && ! -f "$retriever_model/config.json" ]]; then
    echo "$experiment requires an edited DisarmRAG model at: $retriever_model" >&2
    echo "Expected checkpoint.pth or a HuggingFace config.json/model directory." >&2
    echo "Set RETRIEVER_MODEL to the directory produced by the DisarmRAG editing code." >&2
    exit 1
  fi
  if [[ -f "$retriever_model/config.json" &&
        ! -f "$retriever_model/pytorch_model.bin" &&
        ! -f "$retriever_model/model.safetensors" &&
        ! -f "$retriever_model/model.safetensors.index.json" &&
        -z "$(find "$retriever_model" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) -print -quit)" ]]; then
    echo "$experiment model directory has config.json but no model weights: $retriever_model" >&2
    echo "Run prepare_disarmrag_editor.py or set RETRIEVER_MODEL to a complete export." >&2
    exit 1
  fi
fi
if [[ "$experiment" == "B10" ]]; then
  if [[ ! -s "$target_manifest" ]]; then
    echo "B10 target manifest not found or empty: $target_manifest" >&2
    echo "Generate it with prepare_disarmrag_queries.py; B10 must use exact target questions." >&2
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

mkdir -p "$out_dir" "$out_dir/archive"
config_snapshot=$(cat <<EOF
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
TARGET_MANIFEST=$target_manifest
CACHE_FILE=$cache_file
EVENT_FILE=$event_file
TRACE_FILE=$trace_file
ATTACK_RERANK_CORPUS=$attack_rerank_corpus
ATTACK_RERANK_TARGETS=$attack_rerank_targets
ATTACK_RERANK_SCORE_CACHE=$attack_rerank_score_cache
START_TAG=$start_tag
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
)
printf '%s\n' "$config_snapshot" > "$out_dir/config.env"
# Keep a timestamped copy of every launch so prompt/cache settings are auditable.
# The cache for this launch lives at archive/cache.<tag>.jsonl; the stable
# <out_dir>/cache.jsonl symlink points at it for fixed-name viewers.
printf '%s\n' "$config_snapshot" > "$out_dir/archive/config.${start_tag}.env"
# Create the targets before linking so the symlinks are never dangling; an
# empty cache file loads as zero entries in ProvenanceCache.
touch "$cache_file" "$event_file" "$trace_file"
ln -sfn "$cache_file" "$out_dir/cache.jsonl"
ln -sfn "$event_file" "$out_dir/events.jsonl"
ln -sfn "$trace_file" "$out_dir/retrieval_trace.jsonl"

echo "Starting $dataset $experiment"
echo "  query order: $query_order"
echo "  cache key:   $cache_key_mode"
echo "  prompt:      $prompt_variant"
if [[ -n "$target_manifest" ]]; then
  echo "  targets:     $target_manifest"
fi
echo "  corpus:      $corpus"
echo "  cache file:  $cache_file"
echo "  cache link:  $out_dir/cache.jsonl -> $(basename "$cache_file")"
echo "  event log:   $event_file"
echo "  retrieval:   $trace_file"
if [[ -n "$attack_rerank_corpus" ]]; then
  echo "  attack rerank: ON (poison candidates from $(basename "$attack_rerank_corpus"))"
fi
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
  --cache-key-format "$cache_key_format" \
  --cache-file "$cache_file" \
  --log-file "$event_file" \
  --retrieval-trace-file "${RETRIEVAL_TRACE_FILE:-$trace_file}" \
  --http-host "$host" \
  --http-port "$port")

if [[ -n "$target_manifest" ]]; then
  cmd+=(--target-manifest "$target_manifest")
fi
if [[ -n "$attack_rerank_corpus" ]]; then
  cmd+=(--attack-rerank-corpus "$attack_rerank_corpus")
  if [[ -n "$attack_rerank_targets" && -s "$attack_rerank_targets" ]]; then
    cmd+=(--attack-rerank-targets "$attack_rerank_targets")
  fi
  if [[ -n "$attack_rerank_score_cache" ]]; then
    cmd+=(--attack-rerank-score-cache "$attack_rerank_score_cache")
  fi
  cmd+=(--attack-rerank-top-k "${ATTACK_RERANK_TOP_K:-$top_k}")
fi

if [[ -n "$api_base_url" ]]; then
  cmd+=(--api-base-url "$api_base_url")
fi

exec env PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"
