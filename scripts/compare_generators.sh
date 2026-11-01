#!/usr/bin/env bash
# Measures answer groundedness across generators on the same questions.
#
# Retrieval is identical in every run: same corpus, same fine-tuned
# re-ranker, same 40 test-split questions. Only the model writing the
# answer changes, so any difference is the generator.
#
# Needs GROQ_KEY exported in the calling shell. The key is never written
# to disk or echoed.
#
#   read -s GROQ_KEY && export GROQ_KEY
#   bash scripts/compare_generators.sh
set -u

PY=/home/suhas/miniforge3/envs/ultimate_dl/bin/python
OUT=eval/generator_comparison
LIMIT=${LIMIT:-40}
mkdir -p "$OUT"

if [ -z "${GROQ_KEY:-}" ]; then
  echo "GROQ_KEY is not set. Run:  read -s GROQ_KEY && export GROQ_KEY"
  exit 1
fi

run () {
  local label="$1"; shift
  echo
  echo "################ $label ################"
  if [ -f "$OUT/$label.json" ]; then
    echo "  already done, skipping (delete $OUT/$label.json to redo)"
    return
  fi

  # The whole run goes to a log and only the summary to the terminal.
  # Filtering in a pipe once hid the actual exception text and left only
  # traceback frames, which cost an evening of guessing at the cause.
  env "$@" "$PY" -m scripts.evaluate_faithfulness \
      --split test --limit "$LIMIT" --out "$OUT/$label.json" \
      >"$OUT/$label.log" 2>&1
  local status=$?

  grep -E "faithfulness|contradiction|citation_validity|context_util|total_claims|seconds|n_answers|n_failed|FAILED" \
    "$OUT/$label.log" || true

  if [ "$status" -ne 0 ]; then
    # A partial result must not look like a finished one, or the next run
    # skips it and the comparison silently rests on missing questions.
    echo "  INCOMPLETE (exit $status). Full output: $OUT/$label.log"
    [ -f "$OUT/$label.json" ] && mv "$OUT/$label.json" "$OUT/$label.partial.json"
    echo "  partial data kept as $OUT/$label.partial.json; rerun to retry"
  fi
}

# Local needs Ollama up; the hosted runs do not care.
if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting ollama..."
  nohup ollama serve >/tmp/ollama-compare.log 2>&1 &
  until curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 2; done
fi

run "llama-3.1-8b-local" LLM_PROVIDER=ollama

GROQ="LLM_PROVIDER=openai LLM_BASE_URL=https://api.groq.com/openai/v1"
run "gpt-oss-20b" $GROQ LLM_MODEL=openai/gpt-oss-20b    LLM_API_KEY="$GROQ_KEY"
run "gpt-oss-120b" $GROQ LLM_MODEL=openai/gpt-oss-120b  LLM_API_KEY="$GROQ_KEY"

echo
echo "Done. Results in $OUT/"
