#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
SCRIPT_PATH="run_exp3_train_hyp.py"
MODEL=("Qwen/Qwen3-4B-Instruct-2507" ) 
#"Qwen/Qwen3-4B-Instruct-2507-FP8"
# "Qwen/Qwen3-4B-Thinking-2507-FP8"
REASONING=("off") # hard-coded: change to "off" if needed you can also add ("on" "off") for the qwen thinking models or for qwen3 14B

# Optional: log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Iterate only over these
shots=("few" "zero")

# prompt_features combos:
feature_sets=(
  "dataset_description unique_target"
  "unique_target"
  ""
  "dataset_description"
  
  
)

# NEW: sampling sweeps
random_group_samples=("true" "false")   # controls whether we pass --random_group_sample
group_sizes=(5 10 )                  # only used if random_group_sample=true
max_rows_list=( 25 50)

echo "Running: $SCRIPT_PATH"
echo "Model:   $MODEL"
echo "Reasoning (hard-coded): $REASONING"
echo "Logs:    $LOG_DIR"
echo

ts() { date +"%Y%m%d_%H%M%S"; }

for shot in "${shots[@]}"; do
  for features in "${feature_sets[@]}"; do
    for rgs in "${random_group_samples[@]}"; do
      for max_rows in "${max_rows_list[@]}"; do

        if [[ "$rgs" == "true" ]]; then
          # When random_group_sample is ON, sweep group sizes
          for gsize in "${group_sizes[@]}"; do

            cmd=(python "$SCRIPT_PATH"
              --model "$MODEL"
              --shot "$shot"
              --reasoning "$REASONING"
              --random_group_sample
              --group_size "$gsize"
              --max_rows "$max_rows"
            )

            tag="shot=${shot}__reasoning=${REASONING}__rgs=true__g=${gsize}__rows=${max_rows}"

            if [[ -n "$features" ]]; then
              # shellcheck disable=SC2206
              feat_arr=($features)
              cmd+=(--prompt_features "${feat_arr[@]}")
              tag="${tag}__features=$(echo "$features" | tr ' ' '+')"
            else
              tag="${tag}__features=none"
            fi

            logfile="${LOG_DIR}/$(ts)__${tag}.log"

            echo ">>> ${cmd[*]}"
            echo "    log: $logfile"
            "${cmd[@]}" 2>&1 | tee "$logfile"
            echo
          done

        else
          # When random_group_sample is OFF, do NOT pass --group_size
          cmd=(python "$SCRIPT_PATH"
            --model "$MODEL"
            --shot "$shot"
            --reasoning "$REASONING"
            --max_rows "$max_rows"
          )

          tag="shot=${shot}__reasoning=${REASONING}__rgs=false__rows=${max_rows}"

          if [[ -n "$features" ]]; then
            # shellcheck disable=SC2206
            feat_arr=($features)
            cmd+=(--prompt_features "${feat_arr[@]}")
            tag="${tag}__features=$(echo "$features" | tr ' ' '+')"
          else
            tag="${tag}__features=none"
          fi

          logfile="${LOG_DIR}/$(ts)__${tag}.log"

          echo ">>> ${cmd[*]}"
          echo "    log: $logfile"
          "${cmd[@]}" 2>&1 | tee "$logfile"
          echo
        fi

      done
    done
  done
done

echo "All runs completed."
