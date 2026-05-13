import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm
import time
import tiktoken
import re
import argparse
import warnings
import torch
import gc

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)

import plotly.figure_factory as ff

from utils.huggingface_model import load_model, ask_model_prompt
from utils.file_handler_random import load_dataset_into_dataframe, serialize_df_for_llm
from utils.meta_data_extraction import FeaturizeFile, describe_attribute
from utils.prompt_helper import *

warnings.filterwarnings("ignore")


torch.cuda.empty_cache()
gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM experiment runner"
    )

    # --- model ---
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Qwen/Qwen3-14B",
        help="HuggingFace model name (e.g. Qwen/Qwen3-14B)"
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
        choices=["on", "off"],
        default="on",
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
        default=25,
        help="Maximum number of rows to load from dataset"
    )
    parser.add_argument(
    "--data",
    choices=["train", "val", "test"],
    default="test",
    help="Dataset split to use: train, val, or test"
    )

    return parser.parse_args()


args = parse_args()
rows= args.max_rows
MODEL_NAME = args.model

SHOT = args.shot 
USE_REASONING = args.reasoning 
USE_REASONING_SAVE = args.reasoning == "on"
USE_DATASET_DESCRIPTION = "dataset_description" in args.prompt_features
USE_UNIQUE_TARGET = "unique_target" in args.prompt_features

PROMPT_FEATURES = args.prompt_features  




# Few Shot examples Tab

# Few Shot examples Tab

df_tab_few_shot = pd.read_csv("../meta_datasets/few_shot_example/df_tab_few_shot.csv")

ff_exp1_tab=FeaturizeFile(load_dataset_into_dataframe( df_tab_few_shot.iloc[0]["file_path"],target=df_tab_few_shot.iloc[0]["target_variable"]),)
#unique_target_exp1_tab =str( ff_exp1_tab[ff_exp1_tab["Attribute_name"] == df_tab_few_shot.iloc[0]["target_variable"]]["num_of_dist_val"].values[0])
unique_target_exp1_tab= describe_attribute(ff_exp1_tab, df_tab_few_shot.iloc[0]["target_variable"] )

ff_exp2_tab=FeaturizeFile(load_dataset_into_dataframe(df_tab_few_shot.iloc[1]["file_path"],target=df_tab_few_shot.iloc[1]["target_variable"]))
#unique_target_exp2_tab =str( ff_exp2_tab[ff_exp2_tab["Attribute_name"] == df_tab_few_shot.iloc[1]["target_variable"]]["num_of_dist_val"].values[0])
unique_target_exp2_tab= describe_attribute(ff_exp2_tab, df_tab_few_shot.iloc[1]["target_variable"] )


ff_exp3_tab=FeaturizeFile(load_dataset_into_dataframe(df_tab_few_shot.iloc[2]["file_path"],target=df_tab_few_shot.iloc[2]["target_variable"]))
#unique_target_exp3_tab =str( ff_exp3_tab[ff_exp3_tab["Attribute_name"] == df_tab_few_shot.iloc[2]["target_variable"]]["num_of_dist_val"].values[0])
unique_target_exp3_tab= describe_attribute(ff_exp3_tab, df_tab_few_shot.iloc[2]["target_variable"] )




example_1_tab = build_example_fewshot_Exp2(
    dataset_serialized=serialize_dataframe(pd.read_csv(df_tab_few_shot.iloc[0]["file_path"]),max_rows=rows),
    target_spec="Target: column, " + df_tab_few_shot.iloc[0]["target_variable"],
    dataset_description=df_tab_few_shot.iloc[0]["dataset_description"],
    answer_domain = df_tab_few_shot.iloc[0]["problem_type"],
    answer_subtask=df_tab_few_shot.iloc[0]["sub_problem_type"],
    unique_numb_target=unique_target_exp1_tab,
    example_id=1,
    
)

example_2_tab = build_example_fewshot_Exp2(
    dataset_serialized=serialize_dataframe(load_dataset_into_dataframe(df_tab_few_shot.iloc[1]["file_path"]),max_rows=rows),
    target_spec="Target: column, " + df_tab_few_shot.iloc[1]["target_variable"],
    dataset_description=df_tab_few_shot.iloc[1]["dataset_description"],
    answer_domain = df_tab_few_shot.iloc[1]["problem_type"],
    answer_subtask=df_tab_few_shot.iloc[1]["sub_problem_type"],
    unique_numb_target=unique_target_exp2_tab,
    example_id=2,
)

example_3_tab = build_example_fewshot_Exp2(
    dataset_serialized=serialize_dataframe(load_dataset_into_dataframe(df_tab_few_shot.iloc[2]["file_path"]),max_rows=rows),
    target_spec="Target: column, " + df_tab_few_shot.iloc[2]["target_variable"],
    dataset_description=df_tab_few_shot.iloc[2]["dataset_description"],
    answer_domain = df_tab_few_shot.iloc[2]["problem_type"],
    answer_subtask=df_tab_few_shot.iloc[2]["sub_problem_type"],
    unique_numb_target=unique_target_exp3_tab,
    example_id=3,
)


# Few Shot examples Tab

df_time_few_shot =  pd.read_csv("../meta_datasets/few_shot_example/df_time_few_shot.csv")

ff_exp1_time=FeaturizeFile(load_dataset_into_dataframe(df_time_few_shot.iloc[0]["file_path"],target=df_time_few_shot.iloc[0]["target_variable"] ))
#unique_target_exp1_time =str( ff_exp1_time[ff_exp1_time["Attribute_name"] == df_time_few_shot.iloc[0]["target_variable"]]["num_of_dist_val"].values[0])
unique_target_exp1_time= describe_attribute(ff_exp1_time, df_time_few_shot.iloc[0]["target_variable"] )


ff_exp2_time=FeaturizeFile(load_dataset_into_dataframe(df_time_few_shot.iloc[1]["file_path"],target=df_time_few_shot.iloc[1]["target_variable"]))
#unique_target_exp2_time =str( ff_exp2_time[ff_exp2_time["Attribute_name"] == df_time_few_shot.iloc[1]["target_variable"]]["num_of_dist_val"].values[0])
unique_target_exp2_time= describe_attribute(ff_exp2_time, df_time_few_shot.iloc[1]["target_variable"] )


ff_exp3_time=FeaturizeFile(load_dataset_into_dataframe(df_time_few_shot.iloc[2]["file_path"],target=df_time_few_shot.iloc[2]["target_variable"] ))
#unique_target_exp3_time =str( ff_exp3_time[ff_exp3__time["Attribute_name"] == df_time_few_shot.iloc[2]["target_variable"]]["num_of_dist_val"].values[0])
unique_target_exp3_time= describe_attribute(ff_exp3_time, df_time_few_shot.iloc[2]["target_variable"] )


example_1_time = build_example_fewshot_Exp2(
    dataset_serialized=serialize_dataframe(load_dataset_into_dataframe(df_time_few_shot.iloc[0]["file_path"],target=df_time_few_shot.iloc[0]["target_variable"]),max_rows=rows),
    target_spec="Target: column, " + df_time_few_shot.iloc[0]["target_variable"],
    dataset_description=df_time_few_shot.iloc[0]["dataset_description"],
    answer_domain = df_time_few_shot.iloc[0]["problem_type"],
    answer_subtask=df_time_few_shot.iloc[0]["sub_problem_type"],
    unique_numb_target=unique_target_exp1_time,
    example_id=4,
    
)

example_2_time = build_example_fewshot_Exp2(
    dataset_serialized=serialize_dataframe(load_dataset_into_dataframe(df_time_few_shot.iloc[1]["file_path"],target=df_time_few_shot.iloc[1]["target_variable"]),max_rows=rows),
    target_spec="Target: column, " + df_time_few_shot.iloc[1]["target_variable"],
    dataset_description=df_time_few_shot.iloc[1]["dataset_description"],
    answer_domain = df_time_few_shot.iloc[1]["problem_type"],
    answer_subtask=df_time_few_shot.iloc[1]["sub_problem_type"],
    unique_numb_target=unique_target_exp2_time,
    example_id=5,
)

example_3_time = build_example_fewshot_Exp2(
    dataset_serialized=serialize_dataframe(load_dataset_into_dataframe(df_time_few_shot.iloc[2]["file_path"],target=df_time_few_shot.iloc[2]["target_variable"]),max_rows=rows),
    target_spec="Target: column, " + df_time_few_shot.iloc[2]["target_variable"],
    dataset_description=df_time_few_shot.iloc[2]["dataset_description"],
    answer_domain = df_time_few_shot.iloc[2]["problem_type"],
    answer_subtask=df_time_few_shot.iloc[2]["sub_problem_type"],
    unique_numb_target=unique_target_exp3_time,
    example_id=6,
)

model = load_model(MODEL_NAME)

if SHOT == "few":
    
    system_prompt =  SYSTEM_PROMPT_TEMPLATE_FEW_SHOT_Exp2_prompt1.format(
        example_1=example_1_tab,
        example_2=example_2_tab,
        example_3=example_3_tab, 
        example_4=example_1_time,
        example_5=example_2_time, 
        example_6=example_3_time,  
    )

else:
    system_prompt =  SYSTEM_PROMPT_TEMPLATE_ZERO_SHOT_Exp2_prompt1

split_to_file = {
    "train": "exp2_train.csv",
    "val": "exp2_val.csv",
    "test": "exp2_test.csv",
}

exp_df = pd.read_csv(
    Path("../meta_datasets/tabular_timeseries_meta_datasets") / split_to_file[args.data]
)

start = time.perf_counter()
run_2 = exp_df.copy()
predictions = []

for index, row in run_2.iterrows():
    
    target_var = row.get("target_variable")
    
    # --- load data once ---
    load_kwargs = dict(
        target=target_var,
        max_rows=args.max_rows
    )
    
    if args.random_group_sample:
        load_kwargs.update(
            random_group_sample=True,
            group_size=args.group_size
        )
    
    df = load_dataset_into_dataframe(file_path = Path(row["file_path"]),**load_kwargs)
    df_metadata = load_dataset_into_dataframe(file_path = Path( row["file_path"]),max_columns = 1000, target = target_var)

    unique_target_exp = None

    if isinstance(target_var, str):
        target_var = target_var.strip()
    
    if target_var and target_var != "-":
        ff_exp = FeaturizeFile(df_metadata)
    
        mask = ff_exp["Attribute_name"] == target_var
    
        if mask.any():
            unique_target_exp = describe_attribute(ff_exp,target_var )
        # else: target variable not found → silently skip
    # else: target variable empty / "-" / NaN → skip

    if isinstance(target_var, str):
        target_var = target_var.strip()

    if target_var and target_var != "-":
        target_specs = f"Target: {target_var}"
    else:
        target_specs = "Target: no target exists"
        
    # --- prompt ---
    prompt = build_user_prompt(
    dataset_serialized=serialize_dataframe(df,max_rows=load_kwargs["max_rows"]),
        
    target_spec=target_specs,

    # optional arguments — only added if requested
    dataset_description=row["dataset_description"]
    if USE_DATASET_DESCRIPTION
    else None,

    unique_numb_target=unique_target_exp
    if USE_UNIQUE_TARGET
    else None,
    )
   
    #prompt = apply_chat_template(tokenizer,user_prompt, USE_REASONING)
    
    # --- model call ---
    t0 = time.perf_counter()
    answer = ask_model_prompt(
        prompt,
        system_prompt=system_prompt,
        pipe=model,
        reasoning= USE_REASONING
    
    )
    latency = time.perf_counter() - t0
    
    # --- parsing ---
    pred_parsed = extract_label_from_text_phase2(answer)  # -> (domain, label) or None
    
    if pred_parsed is None:
        pred_domain, pred_label, pred_class= None, None,None
    else:
        pred_domain, pred_label = pred_parsed  # e.g. ('tabular','regression')
        pred_class = f"{pred_domain}_{pred_label}"
    
    # --- store everything ---
    predictions.append(answer)
    run_2.at[index, "prompt"] = prompt
    run_2.at[index, "pred_raw"] = answer
    run_2.at[index, "pred_domain"] = pred_domain
    run_2.at[index, "pred_label"] = pred_label
    run_2.at[index, "pred_class"] = pred_class
    run_2.at[index, "latency_s"] = latency
    run_2.at[index, "prompt_len"] = len(prompt)
    run_2.at[index, "answer_len"] = len(answer)

elapsed_minutes = (time.perf_counter() - start) / 60

# --- masks ---
mask_label  = run_2["pred_label"].notna()
mask_domain = run_2["pred_domain"].notna()

# --- label metrics: pred_label vs sub_problem_type ---
y_true_label = run_2.loc[mask_label, "sub_problem_type"]
y_hat_label  = run_2.loc[mask_label, "pred_label"]

# --- domain metrics: pred_domain vs problem_type ---
y_true_domain = run_2.loc[mask_domain, "problem_type"]
y_hat_domain  = run_2.loc[mask_domain, "pred_domain"]

# --- build GT combined class from (problem_type, sub_problem_type) ---
def map_gt_to_combined_class(problem_type, sub_problem_type):
    if problem_type is None or sub_problem_type is None:
        return None

    pt = str(problem_type).strip()
    sp = str(sub_problem_type).strip().lower()

    pt_map = {
        "tabular": "Tabular",
        "Tabular": "Tabular",
        "time_series": "Time_Series",
        "Time_Series": "Time_Series",
        "timeseries": "Time_Series",
        "time series": "Time_Series",
        "Time Series": "Time_Series",
    }
    pt_norm = pt_map.get(pt, pt_map.get(pt.lower()))

    if pt_norm not in {"Tabular", "Time_Series"}:
        return None
    if sp not in {"regression", "binary", "multiclass"}:
        return None

    return f"{pt_norm}_{sp}"

run_2["gt_class"] = run_2.apply(
    lambda r: map_gt_to_combined_class(r.get("problem_type"), r.get("sub_problem_type")),
    axis=1
)

# --- class metrics: pred_class vs gt_class ---
mask_class = run_2["pred_class"].notna() & run_2["gt_class"].notna()
y_true_class = run_2.loc[mask_class, "gt_class"]
y_hat_class  = run_2.loc[mask_class, "pred_class"]

metrics = {
    # completion
    "completion_rate_label_percent": mask_label.mean() * 100,
    "completion_rate_domain_percent": mask_domain.mean() * 100,
    "completion_rate_class_percent": mask_class.mean() * 100,

    # label metrics
    "accuracy_label": accuracy_score(y_true_label, y_hat_label) if mask_label.any() else None,
    "balanced_accuracy_label": balanced_accuracy_score(y_true_label, y_hat_label) if mask_label.any() else None,
    "precision_macro_label": precision_score(y_true_label, y_hat_label, average="macro", zero_division=0) if mask_label.any() else None,
    "recall_macro_label": recall_score(y_true_label, y_hat_label, average="macro", zero_division=0) if mask_label.any() else None,
    "f1_macro_label": f1_score(y_true_label, y_hat_label, average="macro", zero_division=0) if mask_label.any() else None,

    # domain metrics
    "accuracy_domain": accuracy_score(y_true_domain, y_hat_domain) if mask_domain.any() else None,
    "balanced_accuracy_domain": balanced_accuracy_score(y_true_domain, y_hat_domain) if mask_domain.any() else None,
    "precision_macro_domain": precision_score(y_true_domain, y_hat_domain, average="macro", zero_division=0) if mask_domain.any() else None,
    "recall_macro_domain": recall_score(y_true_domain, y_hat_domain, average="macro", zero_division=0) if mask_domain.any() else None,
    "f1_macro_domain": f1_score(y_true_domain, y_hat_domain, average="macro", zero_division=0) if mask_domain.any() else None,

    # combined class metrics
    "accuracy_class": accuracy_score(y_true_class, y_hat_class) if mask_class.any() else None,
    "balanced_accuracy_class": balanced_accuracy_score(y_true_class, y_hat_class) if mask_class.any() else None,
    "precision_macro_class": precision_score(y_true_class, y_hat_class, average="macro", zero_division=0) if mask_class.any() else None,
    "recall_macro_class": recall_score(y_true_class, y_hat_class, average="macro", zero_division=0) if mask_class.any() else None,
    "f1_macro_class": f1_score(y_true_class, y_hat_class, average="macro", zero_division=0) if mask_class.any() else None,

    "elapsed_minutes": elapsed_minutes,
}


feature_suffix = (
    "+".join(PROMPT_FEATURES)
    if PROMPT_FEATURES
    else "BasePrompt"
)

if args.random_group_sample:
    sampling_suffix = f"Grouped_g{args.group_size}_rows{args.max_rows}"
else:
    sampling_suffix = f"Full_rows{args.max_rows}"
    
hyperparameter = (
    f"{MODEL_NAME}_"
    f"{args.data.capitalize()}_"
    f"{SHOT}_shot_"
    f"{'Reasoning' if USE_REASONING_SAVE else 'NoReasoning'}_"
    f"Dataset_serialized_"
    f"Target_name_"
    f"{feature_suffix}_"
    f"{sampling_suffix}"
)

    
    

meta_data_exp= {
    "Exp_number": 3 ,
    "Dataset_Phase": 3,
    "Train/Test": args.data.capitalize(),
    "Hyperparameter": hyperparameter ,
    "Metrics": metrics,
    "General_time": elapsed_minutes,
    "System_prompt" : system_prompt,
    "random_group_sample": args.random_group_sample,
    "group_size": args.group_size if args.random_group_sample else None,
    "max_rows": args.max_rows
    
}



results_dir = Path("Exp3_Results_Test") / hyperparameter
results_dir.mkdir(parents=True, exist_ok=True)

# full experiment table
run_2.to_csv(results_dir / "experiment_results.csv", index=False)

# metrics
pd.Series(metrics).to_csv(results_dir / "metrics.csv")

# optional: raw predictions only
pd.Series(predictions, name="raw_predictions").to_csv(
    results_dir / "raw_predictions.csv",
    index=False
)
pd.Series(meta_data_exp).to_csv(results_dir/ "meta_data_exp.csv")
