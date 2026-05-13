import pandas as pd
from open_ai_helper import *
from tqdm import tqdm
import time
from file_handler import load_dataset_into_dataframe,serialize_df_for_llm
from meta_data_extraction import FeaturizeFile,describe_attribute
import tiktoken
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,balanced_accuracy_score
import plotly.figure_factory as ff
import re
import argparse
from pathlib import Path
from prompt_helper import *
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import torch
import gc
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
        default="gpt-4.1",
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
        choices=["off","low", "medium", "high"],
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

    return parser.parse_args()


args = parse_args()

MODEL_NAME = args.model

SHOT = args.shot 
USE_REASONING = args.reasoning 
USE_DATASET_DESCRIPTION = "dataset_description" in args.prompt_features
USE_UNIQUE_TARGET = "unique_target" in args.prompt_features

PROMPT_FEATURES = args.prompt_features  


exp_df = pd.read_csv("Exp1_datasets/Exp1_train_data.csv")


ff_exp1=FeaturizeFile(pd.read_csv("../" + exp_df.iloc[0]["file_path"]))
#unique_target_exp1 =str( ff_exp1[ff_exp1["Attribute_name"] == exp_df.iloc[0]["target_variable"]]["num_of_dist_val"].values[0])
unique_target_exp1 = describe_attribute(ff_exp1,exp_df.iloc[0]["target_variable"])

ff_exp2=FeaturizeFile(pd.read_csv("../" + exp_df.iloc[1]["file_path"]))
#unique_target_exp2 =str( ff_exp2[ff_exp2["Attribute_name"] == exp_df.iloc[1]["target_variable"]]["num_of_dist_val"].values[0])
unique_target_exp2= describe_attribute(ff_exp2,exp_df.iloc[1]["target_variable"])

ff_exp3=FeaturizeFile(pd.read_csv("../" + exp_df.iloc[4]["file_path"]))
#unique_target_exp3 =str( ff_exp3[ff_exp3["Attribute_name"] == exp_df.iloc[4]["target_variable"]]["num_of_dist_val"].values[0])
unique_target_exp3= describe_attribute(ff_exp3,exp_df.iloc[4]["target_variable"])

example_1 = build_example_fewshot(
    dataset_serialized=serialize_dataframe(pd.read_csv("../"+ exp_df.iloc[0]["file_path"])),
    target_spec="Target: column, " + exp_df.iloc[0]["target_variable"],
    dataset_description=exp_df.iloc[0]["dataset_description"],
    answer=exp_df.iloc[0]["sub_problem_type"],
    unique_numb_target=unique_target_exp1,
    example_id=1,
    
)

example_2 = build_example_fewshot(
    dataset_serialized=serialize_dataframe(pd.read_csv("../"+ exp_df.iloc[1]["file_path"])),
    target_spec="Target: column, " + exp_df.iloc[1]["target_variable"],
    dataset_description=exp_df.iloc[0]["dataset_description"],
    answer=exp_df.iloc[1]["sub_problem_type"],
    unique_numb_target=unique_target_exp2,
    example_id=2,
)

example_3 = build_example_fewshot(
    dataset_serialized=serialize_dataframe(pd.read_csv("../"+ exp_df.iloc[4]["file_path"])),
    target_spec="Target: column, " + exp_df.iloc[4]["target_variable"],
    dataset_description=exp_df.iloc[0]["dataset_description"],
    answer=exp_df.iloc[4]["sub_problem_type"],
    unique_numb_target=unique_target_exp3,
    example_id=3,
)

indices = [0, 1,4]
exp_df = exp_df.drop(index=indices)
exp_df = exp_df.reset_index(drop=True)



if SHOT == "few":
    
    system_prompt =  SYSTEM_PROMPT_TEMPLATE_FEW_SHOT.format(
        example_1=example_1,
        example_2=example_2,
        example_3=example_3, 
    )
else:
    system_prompt =  SYSTEM_PROMPT_TEMPLATE_ZERO_SHOT

exp_df_val = pd.read_csv("Exp1_datasets/Exp1_test_data.csv")

start = time.perf_counter()
run_2 = exp_df_val.copy()
predictions = []

for index, row in run_2.iterrows():
    print(index)
    try:
        # --- load data once ---
        df = pd.read_csv(Path("..") / row["file_path"])


        # --- featurization ---
        ff_exp = FeaturizeFile(df)
        unique_target_exp= describe_attribute(ff_exp,row["target_variable"])
        mask = ff_exp["Attribute_name"] == row["target_variable"]

        #unique_target_exp = str(
        #    ff_exp.loc[mask, "num_of_dist_val"].iloc[0]
        #)

        # --- prompt ---
        prompt = build_user_prompt(
        dataset_serialized=serialize_dataframe(df),
        target_spec=f"Target: column, {row['target_variable']}",

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
        answer = run_openai(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature= 0.7,
        )
        latency = time.perf_counter() - t0

        # --- parsing ---
        pred_label = extract_label_from_text(answer)

        # --- store everything ---
        predictions.append(answer)
        run_2.at[index, "prompt"] = prompt
        run_2.at[index, "pred_raw"] = answer
        run_2.at[index, "pred_label"] = pred_label
        run_2.at[index, "latency_s"] = latency
        run_2.at[index, "prompt_len"] = len(prompt)
        run_2.at[index, "answer_len"] = len(answer)
        #exp_df.at[index, "pred_raw"] = None
        #exp_df.at[index, "pred_label"] = None
        #exp_df.at[index, "error"] = str(e)
    except:
        run_2.at[index, "prompt"] = np.nan
        run_2.at[index, "pred_raw"] = np.nan
        run_2.at[index, "pred_label"] = np.nan
        run_2.at[index, "latency_s"] = np.nan
        run_2.at[index, "prompt_len"] = np.nan
        run_2.at[index, "answer_len"] = np.nan
    
elapsed_minutes = (time.perf_counter() - start) / 60

mask = run_2["pred_label"].notna()

y_true = run_2.loc[mask, "sub_problem_type"]
y_hat  = run_2.loc[mask, "pred_label"]

metrics = {
    "completion_rate_in_percent_all_e": mask.mean(),
    "accuracy": accuracy_score(y_true, y_hat),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_hat),
    "precision_macro": precision_score(
        y_true, y_hat, average="macro", zero_division=0
    ),
    "recall_macro": recall_score(
        y_true, y_hat, average="macro", zero_division=0
    ),
    "f1_macro": f1_score(
        y_true, y_hat, average="macro", zero_division=0
    ),
    "elapsed_minutes": elapsed_minutes,
}

feature_suffix = (
    "+".join(PROMPT_FEATURES)
    if PROMPT_FEATURES
    else "BasePrompt"
)
if USE_REASONING == "off":
    REASON = "NoReasoning"
else:
    REASON = "Reasoning"

hyperparameter = (
    f"{MODEL_NAME}_"
    f"Train_"
    f"{SHOT}_shot_"
    f"{REASON}_"
    f"Dataset_serialized_"
    f"Target_name_"
    f"{feature_suffix}"
)

meta_data_exp= {
    "Exp_number": 1 ,
    "Dataset_Phase": 1,
    "Train/Test": "Test" ,
    "Hyperparameter": hyperparameter ,
    "Metrics": metrics,
    "General_time": elapsed_minutes,
    "System_prompt" : system_prompt,
    
}


results_dir = Path("Exp1_Results_Test") / hyperparameter
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
