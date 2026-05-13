
import json
import time
import warnings
import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

# --- your existing imports (kept as-is) ---
from open_ai_helper import *
from tqdm import tqdm
from file_handler_random import load_dataset_into_dataframe, serialize_df_for_llm
from meta_data_extraction import FeaturizeFile, describe_attribute
from prompt_helper import *

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
gc.collect()

# OpenAI official SDK (needed for Batch API)
from openai import OpenAI

def load_api_key(path="API.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def parse_args():
    parser = argparse.ArgumentParser(description="LLM experiment runner (Batch creator)")

    # --- model ---
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI API Model name"
    )

    # --- few-shot vs zero-shot ---
    parser.add_argument(
        "--shot",
        choices=["zero", "few"],
        default="zero",
        help="Prompting mode: zero or few shot"
    )

    # --- reasoning vs no reasoning ---
    parser.add_argument(
        "--reasoning",
        choices=["high", "off"],
        default="off",
        help="Use chain-of-thought reasoning or not"
    )

    # --- prompt features ---
    parser.add_argument(
        "--prompt_features",
        nargs="+",
        choices=[
            "dataset_description",
            "unique_target"
        ],
        default=[],
        help="Which features to include in the prompt"
    )

    parser.add_argument(
        "--random_group_sample",
        action="store_true",
        default=False,
        help="Enable random grouped sampling"
    )

    parser.add_argument(
        "--group_size",
        type=int,
        default=5,
        help="Group size for random grouped sampling (used only if --random_group_sample is set)"
    )

    parser.add_argument(
        "--max_rows",
        type=int,
        default=100,
        help="Maximum number of rows to load from dataset"
    )

    # --- batch settings ---
    parser.add_argument(
        "--completion_window",
        type=str,
        default="24h",
        help="Batch completion window (e.g. 24h)"
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/responses",
        help="Batch endpoint. Typically /v1/chat/completions or /v1/responses"
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default="Exp2_Results_paper_test",
        help="Where to write the batch input file + metadata"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    rows = args.max_rows
    MODEL_NAME = args.model

    SHOT = args.shot
    USE_REASONING = None if args.reasoning == "off" else args.reasoning
    USE_REASONING_SAVE = (args.reasoning != "off")
    USE_DATASET_DESCRIPTION = "dataset_description" in args.prompt_features
    USE_UNIQUE_TARGET = "unique_target" in args.prompt_features
    PROMPT_FEATURES = args.prompt_features

    # ----------------------------
    # Few-shot example construction (kept exactly as your logic)
    # ----------------------------
    df_tab_few_shot = pd.read_csv("Exp2_datasets/df_tab_few_shot.csv")

    ff_exp1_tab = FeaturizeFile(load_dataset_into_dataframe("../" + df_tab_few_shot.iloc[0]["file_path"],
                                                           target=df_tab_few_shot.iloc[0]["target_variable"]))
    unique_target_exp1_tab = describe_attribute(ff_exp1_tab, df_tab_few_shot.iloc[0]["target_variable"])

    ff_exp2_tab = FeaturizeFile(load_dataset_into_dataframe("../" + df_tab_few_shot.iloc[1]["file_path"],
                                                           target=df_tab_few_shot.iloc[1]["target_variable"]))
    unique_target_exp2_tab = describe_attribute(ff_exp2_tab, df_tab_few_shot.iloc[1]["target_variable"])

    ff_exp3_tab = FeaturizeFile(load_dataset_into_dataframe("../" + df_tab_few_shot.iloc[2]["file_path"],
                                                           target=df_tab_few_shot.iloc[2]["target_variable"]))
    unique_target_exp3_tab = describe_attribute(ff_exp3_tab, df_tab_few_shot.iloc[2]["target_variable"])

    example_1_tab = build_example_fewshot_Exp2(
        dataset_serialized=serialize_dataframe(pd.read_csv("../" + df_tab_few_shot.iloc[0]["file_path"]), max_rows=rows),
        target_spec="Target: column, " + df_tab_few_shot.iloc[0]["target_variable"],
        dataset_description=df_tab_few_shot.iloc[0]["dataset_description"],
        answer_domain=df_tab_few_shot.iloc[0]["problem_type"],
        answer_subtask=df_tab_few_shot.iloc[0]["sub_problem_type"],
        unique_numb_target=unique_target_exp1_tab,
        example_id=1,
    )

    example_2_tab = build_example_fewshot_Exp2(
        dataset_serialized=serialize_dataframe(load_dataset_into_dataframe("../" + df_tab_few_shot.iloc[1]["file_path"]),
                                              max_rows=rows),
        target_spec="Target: column, " + df_tab_few_shot.iloc[1]["target_variable"],
        dataset_description=df_tab_few_shot.iloc[1]["dataset_description"],
        answer_domain=df_tab_few_shot.iloc[1]["problem_type"],
        answer_subtask=df_tab_few_shot.iloc[1]["sub_problem_type"],
        unique_numb_target=unique_target_exp2_tab,
        example_id=2,
    )

    example_3_tab = build_example_fewshot_Exp2(
        dataset_serialized=serialize_dataframe(load_dataset_into_dataframe("../" + df_tab_few_shot.iloc[2]["file_path"]),
                                              max_rows=rows),
        target_spec="Target: column, " + df_tab_few_shot.iloc[2]["target_variable"],
        dataset_description=df_tab_few_shot.iloc[2]["dataset_description"],
        answer_domain=df_tab_few_shot.iloc[2]["problem_type"],
        answer_subtask=df_tab_few_shot.iloc[2]["sub_problem_type"],
        unique_numb_target=unique_target_exp3_tab,
        example_id=3,
    )

    df_time_few_shot = pd.read_csv("Exp2_datasets/df_time_few_shot.csv")

    ff_exp1_time = FeaturizeFile(load_dataset_into_dataframe("../" + df_time_few_shot.iloc[0]["file_path"],
                                                            target=df_time_few_shot.iloc[0]["target_variable"]))
    unique_target_exp1_time = describe_attribute(ff_exp1_time, df_time_few_shot.iloc[0]["target_variable"])

    ff_exp2_time = FeaturizeFile(load_dataset_into_dataframe("../" + df_time_few_shot.iloc[1]["file_path"],
                                                            target=df_time_few_shot.iloc[1]["target_variable"]))
    unique_target_exp2_time = describe_attribute(ff_exp2_time, df_time_few_shot.iloc[1]["target_variable"])

    ff_exp3_time = FeaturizeFile(load_dataset_into_dataframe("../" + df_time_few_shot.iloc[2]["file_path"],
                                                            target=df_time_few_shot.iloc[2]["target_variable"]))
    unique_target_exp3_time = describe_attribute(ff_exp3_time, df_time_few_shot.iloc[2]["target_variable"])

    example_1_time = build_example_fewshot_Exp2(
        dataset_serialized=serialize_dataframe(load_dataset_into_dataframe("../" + df_time_few_shot.iloc[0]["file_path"],
                                                                          target=df_time_few_shot.iloc[0]["target_variable"]),
                                              max_rows=rows),
        target_spec="Target: column, " + df_time_few_shot.iloc[0]["target_variable"],
        dataset_description=df_time_few_shot.iloc[0]["dataset_description"],
        answer_domain=df_time_few_shot.iloc[0]["problem_type"],
        answer_subtask=df_time_few_shot.iloc[0]["sub_problem_type"],
        unique_numb_target=unique_target_exp1_time,
        example_id=4,
    )

    example_2_time = build_example_fewshot_Exp2(
        dataset_serialized=serialize_dataframe(load_dataset_into_dataframe("../" + df_time_few_shot.iloc[1]["file_path"],
                                                                          target=df_time_few_shot.iloc[1]["target_variable"]),
                                              max_rows=rows),
        target_spec="Target: column, " + df_time_few_shot.iloc[1]["target_variable"],
        dataset_description=df_time_few_shot.iloc[1]["dataset_description"],
        answer_domain=df_time_few_shot.iloc[1]["problem_type"],
        answer_subtask=df_time_few_shot.iloc[1]["sub_problem_type"],
        unique_numb_target=unique_target_exp2_time,
        example_id=5,
    )

    example_3_time = build_example_fewshot_Exp2(
        dataset_serialized=serialize_dataframe(load_dataset_into_dataframe("../" + df_time_few_shot.iloc[2]["file_path"],
                                                                          target=df_time_few_shot.iloc[2]["target_variable"]),
                                              max_rows=rows),
        target_spec="Target: column, " + df_time_few_shot.iloc[2]["target_variable"],
        dataset_description=df_time_few_shot.iloc[2]["dataset_description"],
        answer_domain=df_time_few_shot.iloc[2]["problem_type"],
        answer_subtask=df_time_few_shot.iloc[2]["sub_problem_type"],
        unique_numb_target=unique_target_exp3_time,
        example_id=6,
    )

    if SHOT == "few":
        system_prompt = SYSTEM_PROMPT_TEMPLATE_FEW_SHOT_Exp2_prompt1.format(
            example_1=example_1_tab,
            example_2=example_2_tab,
            example_3=example_3_tab,
            example_4=example_1_time,
            example_5=example_2_time,
            example_6=example_3_time,
        )
    else:
        system_prompt = SYSTEM_PROMPT_TEMPLATE_ZERO_SHOT_Exp2_prompt1

    # ----------------------------
    # Build batch requests (prompt construction kept as-is)
    # ----------------------------
    exp_df = pd.read_csv("Exp2_datasets/exp2_test.csv")
    run_2 = exp_df.copy()

    feature_suffix = "+".join(PROMPT_FEATURES) if PROMPT_FEATURES else "BasePrompt"
    if args.random_group_sample:
        sampling_suffix = f"Grouped_g{args.group_size}_rows{args.max_rows}"
    else:
        sampling_suffix = f"Full_rows{args.max_rows}"

    hyperparameter = (
        f"{MODEL_NAME}_"
        f"Train_"
        f"{SHOT}_shot_"
        f"{'Reasoning' if USE_REASONING_SAVE else 'NoReasoning'}_"
        f"Dataset_serialized_"
        f"Target_name_"
        f"{feature_suffix}"
        f"{sampling_suffix}"
    )

    results_dir = Path(args.results_root) / hyperparameter
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_input_path = results_dir / "batch_input.jsonl"

    requests_written = 0
    with batch_input_path.open("w", encoding="utf-8") as f:
        for index, row in run_2.iterrows():
            target_var = row.get("target_variable")

            load_kwargs = dict(
                target=target_var,
                max_rows=args.max_rows
            )
            if args.random_group_sample:
                load_kwargs.update(
                    random_group_sample=True,
                    group_size=args.group_size
                )

            df = load_dataset_into_dataframe(file_path = Path("../" + row["file_path"]),**load_kwargs)
            df_metadata = load_dataset_into_dataframe(file_path = Path("../" + row["file_path"]), target = target_var)

            unique_target_exp = None

            if isinstance(target_var, str):
                target_var = target_var.strip()

            if target_var and target_var != "-":
                ff_exp = FeaturizeFile(df_metadata)
                mask = ff_exp["Attribute_name"] == target_var
                if mask.any():
                    unique_target_exp = describe_attribute(ff_exp,target_var )

            if isinstance(target_var, str):
                target_var = target_var.strip()

            if target_var and target_var != "-":
                target_specs = f"Target: {target_var}"
            else:
                target_specs = "Target: no target exists"

            # --- prompt (UNCHANGED) ---
            prompt = build_user_prompt(
                dataset_serialized=serialize_dataframe(df, max_rows=load_kwargs["max_rows"]),
                target_spec=target_specs,
                dataset_description=row["dataset_description"] if USE_DATASET_DESCRIPTION else None,
                unique_numb_target=unique_target_exp if USE_UNIQUE_TARGET else None,
            )

            # store prompt into your table (optional but handy)
            run_2.at[index, "prompt"] = prompt

            # --- convert your "reasoning" flag to API parameter (only if enabled) ---
            # For Chat Completions, reasoning is represented as: {"reasoning": {"effort": "high"}}
            reasoning_param = None
            if USE_REASONING is not None:
                reasoning_param = {"effort": USE_REASONING}

            # --- build one JSONL line for Batch API ---
            # Batch API expects: {custom_id, method, url, body}
           # body = {
           #     "model": MODEL_NAME,
           #     "messages": [
           #         {"role": "system", "content": system_prompt},
           #         {"role": "user", "content": prompt},
           #     ]
           # }
            body = {
              "model": MODEL_NAME,
              "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
              ],
            }
            if reasoning_param is not None:
                body["reasoning"] = reasoning_param

            req = {
                "custom_id": f"row_{index}",
                "method": "POST",
                "url": args.endpoint,
                "body": body,
            }

            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            requests_written += 1

    # Save a copy of the table with prompts (no predictions yet)
    run_2.to_csv(results_dir / "experiment_requests.csv", index=False)

    # ----------------------------
    # Upload + create batch
    # ----------------------------
    #client = OpenAI()  # uses OPENAI_API_KEY from environment
    client = OpenAI(api_key=load_api_key("API.txt"))

    with batch_input_path.open("rb") as fh:
        input_file = client.files.create(file=fh, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
    )

    # Write batch metadata for later retrieval
    meta = {
        "batch_id": batch.id,
        "input_file_id": input_file.id,
        "endpoint": args.endpoint,
        "completion_window": args.completion_window,
        "hyperparameter": hyperparameter,
        "requests_written": requests_written,
        "model": MODEL_NAME,
        "shot": SHOT,
        "reasoning": USE_REASONING,
        "prompt_features": PROMPT_FEATURES,
        "random_group_sample": args.random_group_sample,
        "group_size": args.group_size if args.random_group_sample else None,
        "max_rows": args.max_rows,
        "system_prompt_path": str(results_dir / "system_prompt.txt"),
        "created_at_unix": getattr(batch, "created_at", None),
    }

    (results_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (results_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("✅ Batch created")
    print(f"results_dir: {results_dir}")
    print(f"batch_input.jsonl: {batch_input_path}")
    print(f"input_file_id: {input_file.id}")
    print(f"batch_id: {batch.id}")
    print("Next: poll status with client.batches.retrieve(batch_id) and download output_file_id when completed.")


if __name__ == "__main__":
    main()
