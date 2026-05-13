#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
SCRIPT_PATH="create_batch_exp2.py"

MODEL="gpt-5.4"
REASONING=( "high" ) # hard-coded

# Batch settings
COMPLETION_WINDOW="24h"
ENDPOINT="/v1/responses"

# Where create_batch_exp2.py writes outputs
RESULTS_ROOT="Exp2_Results_paper_test"

# Logging
LOG_DIR="logs_batches"
mkdir -p "$LOG_DIR"
PROGRESS_LOG="${LOG_DIR}/progress.log"

# Parallelism
MAX_JOBS=8

# -----------------------
# Hyperparameter sweep
# -----------------------
shots=( "few" "zero" )

feature_sets=(
  "unique_target"
)

random_group_samples=("true")
group_sizes=(5 10)
max_rows_list=(25 50)

# -----------------------
# Helpers
# -----------------------
ts() { date +"%Y%m%d_%H%M%S"; }

logp() {
  echo "[$(date +"%F %T")] $*" | tee -a "$PROGRESS_LOG"
}

hyperparam_dir_name() {
  local shot="$1"
  local reasoning="$2"
  local rgs="$3"
  local gsize="$4"
  local max_rows="$5"
  local features="$6"

  local feature_suffix
  if [[ -n "$features" ]]; then
    feature_suffix="$(echo "$features" | tr ' ' '+')"
  else
    feature_suffix="BasePrompt"
  fi

  local sampling_suffix
  if [[ "$rgs" == "true" ]]; then
    sampling_suffix="Grouped_g${gsize}_rows${max_rows}"
  else
    sampling_suffix="Full_rows${max_rows}"
  fi

  local reasoning_suffix
  if [[ "$reasoning" == "off" ]]; then
    reasoning_suffix="NoReasoning"
  else
    reasoning_suffix="Reasoning"
  fi

  echo "${MODEL}_Train_${shot}_shot_${reasoning_suffix}_Dataset_serialized_Target_name_${feature_suffix}${sampling_suffix}"
}

# Rerun logic:
# - If batch_meta.json exists AND batch_errors.jsonl exists => RERUN and overwrite folder files
# - If batch_meta.json exists and NO error file => SKIP
# - If batch_meta.json missing => RUN
# Rerun logic:
# - If batch_meta.json exists AND (batch_errors.jsonl OR batch_failure_note.txt OR batch_status_snapshot.json) exists
#     => RERUN and overwrite folder files
# - If batch_meta.json exists and NO failure indicators => SKIP
# - If batch_meta.json missing => RUN
should_rerun_or_skip() {
  local folder="$1"

  if [[ -f "${folder}/batch_meta.json" ]]; then
    # Failure indicators (any of these means we want to rerun)
    if [[ -f "${folder}/batch_errors.jsonl" ]] || \
       [[ -f "${folder}/batch_failure_note.txt" ]] || \
       [[ -f "${folder}/batch_status_snapshot.json" ]]; then
      echo "rerun"
      return 0
    else
      echo "skip"
      return 0
    fi
  fi

  echo "run"
  return 0
}

wipe_run_files() {
  local folder="$1"
  # Remove files that should be overwritten on rerun
  rm -f \
    "${folder}/batch_input.jsonl" \
    "${folder}/experiment_requests.csv" \
    "${folder}/system_prompt.txt" \
    "${folder}/batch_meta.json" \
    "${folder}/batch_output.jsonl" \
    "${folder}/batch_errors.jsonl" \
    "${folder}/batch_failure_note.txt" \
    "${folder}/batch_status_snapshot.json"
}

run_one() {
  local shot="$1"
  local rgs="$2"
  local gsize="$3"
  local max_rows="$4"
  local features="$5"

  local dir_name
  dir_name="$(hyperparam_dir_name "$shot" "$REASONING" "$rgs" "$gsize" "$max_rows" "$features")"
  local out_dir="${RESULTS_ROOT}/${dir_name}"

  mkdir -p "$out_dir"

  local action
  action="$(should_rerun_or_skip "$out_dir")"

  if [[ "$action" == "skip" ]]; then
    logp "SKIP:  ${dir_name} (batch_meta.json exists, no batch_errors.jsonl)"
    return 0
  fi

  if [[ "$action" == "rerun" ]]; then
    logp "RERUN: ${dir_name} (batch_errors.jsonl found) -> wiping and recreating batch"
    wipe_run_files "$out_dir"
  else
    logp "RUN:   ${dir_name}"
  fi

  # build command
  local -a cmd
  cmd=(python "$SCRIPT_PATH"
    --model "$MODEL"
    --shot "$shot"
    --reasoning "$REASONING"
    --max_rows "$max_rows"
    --completion_window "$COMPLETION_WINDOW"
    --endpoint "$ENDPOINT"
    --results_root "$RESULTS_ROOT"
  )

  if [[ "$rgs" == "true" ]]; then
    cmd+=(--random_group_sample --group_size "$gsize")
  fi

  if [[ -n "$features" ]]; then
    # shellcheck disable=SC2206
    local feat_arr=($features)
    cmd+=(--prompt_features "${feat_arr[@]}")
  fi

  local tag="shot=${shot}__reasoning=${REASONING}__rgs=${rgs}__g=${gsize}__rows=${max_rows}"
  if [[ -n "$features" ]]; then
    tag="${tag}__features=$(echo "$features" | tr ' ' '+')"
  else
    tag="${tag}__features=none"
  fi

  local logfile="${LOG_DIR}/$(ts)__${tag}.log"

  logp "START: ${dir_name}"
  logp "CMD:   ${cmd[*]}"
  logp "LOG:   ${logfile}"

  if "${cmd[@]}" 2>&1 | tee "$logfile"; then
    if [[ -f "${out_dir}/batch_meta.json" ]]; then
      logp "DONE:  ${dir_name} (batch_meta.json created)"
      return 0
    else
      logp "FAIL:  ${dir_name} (no batch_meta.json created)"
      return 1
    fi
  else
    logp "FAIL:  ${dir_name} (python exited non-zero)"
    return 1
  fi
}

# Parallel runner with job control
pids=()
jobs_running=0

enqueue() {
  ( run_one "$@" ) &
  pids+=($!)
  jobs_running=$((jobs_running + 1))

  while [[ "$jobs_running" -ge "$MAX_JOBS" ]]; do
    if wait -n; then
      jobs_running=$((jobs_running - 1))
    else
      jobs_running=$((jobs_running - 1))
      logp "WARN: A job failed (see logs). Continuing..."
    fi
  done
}

# -----------------------
# Main
# -----------------------
logp "=============================================="
logp "Launching batches: script=${SCRIPT_PATH}, model=${MODEL}, reasoning=${REASONING}"
logp "Endpoint=${ENDPOINT}, window=${COMPLETION_WINDOW}, results_root=${RESULTS_ROOT}"
logp "Parallel jobs: MAX_JOBS=${MAX_JOBS}"
logp "Rule: rerun+overwrite when batch_errors.jsonl exists"
logp "=============================================="

for shot in "${shots[@]}"; do
  for features in "${feature_sets[@]}"; do
    for rgs in "${random_group_samples[@]}"; do
      for max_rows in "${max_rows_list[@]}"; do

        if [[ "$rgs" == "true" ]]; then
          for gsize in "${group_sizes[@]}"; do
            enqueue "$shot" "$rgs" "$gsize" "$max_rows" "$features"
          done
        else
          enqueue "$shot" "$rgs" "0" "$max_rows" "$features"
        fi

      done
    done
  done
done

logp "Waiting for remaining jobs to finish..."
set +e
for pid in "${pids[@]}"; do
  wait "$pid"
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    logp "WARN: job pid=${pid} failed with rc=${rc}"
  fi
done
set -e

logp "All batch creations attempted. Re-run anytime to resume; failed runs (with batch_errors.jsonl) will be overwritten and rerun."