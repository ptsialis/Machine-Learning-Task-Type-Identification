# ML Task Identification Experiments

This repository contains code for running LLM-based machine learning task identification experiments.  
The main script is `run_exp2_train.py`, which evaluates a selected model on tabular and time-series metadata splits.

The model predicts two aspects of each dataset:

1. the **data domain**, for example `Tabular` or `Time_Series`;
2. the **downstream task**, for example `regression/forecasting`, `binary classification`, or `multiclass classification`.

---

## 1. Create the Conda Environment

Create a new Conda environment with Python 3.11:

```bash
conda create -n mltid python=3.11 -y
```

Activate the environment:

```bash
conda activate mltid
```

---

## 2. Install Requirements

Install all required Python packages from `req.txt`:

```bash
pip install -r req.txt
```

If the `utils` package was not installed automatically, run the following command in the activated environment:

```bash
pip install -e .
```

This installs the local project package in editable mode.

---

## 3. Download the Dataset Folder

The dataset folder can be downloaded from Zenodo:

```text
https://zenodo.org/records/20139892?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3ODY5NDM4NCwiZXhwIjoxNzgzNjQxNTk5fQ.eyJpZCI6ImE4NDU2NDRhLWI5YjEtNGM4OC05M2RmLWIzYjc1ZTM0MWE1MyIsImRhdGEiOnt9LCJyYW5kb20iOiI1ZGFkNzU4MDhlMzRhNTMzYTUyNzRjMWUwYTk2NmVmYSJ9.Bob3QvBwjCMWUzOgI60GL3Q-EX1LUBXQSd0ORJNWeuiiF_NS3y8QFV0sGph01Vq-jwpNFGJyEefnXi3hu626uQ
```

After downloading, place the dataset folder at the same directory level as `meta_datasets`.

Expected structure:

```text
Problem_Type_github/
├── dataset/
├── meta_datasets/
├── run_exp2_train.py
├── req.txt
├── pyproject.toml
└── utils/
```

---

## 4. Run Experiment 2

The main experiment script is:

```bash
run_exp2_train.py
```

The script requires at least the `--model` parameter.

Basic example:

```bash
python run_exp2_train.py --model Qwen/Qwen3-14B
```

---

## 5. Available Parameters

| Parameter | Options / Type | Default | Description |
|---|---|---|---|
| `--model` | string | required | Hugging Face model name, e.g. `Qwen/Qwen3-14B`. The first execution may take several minutes because the model weights are downloaded. |
| `--shot` | `zero`, `few` | `zero` | Prompting mode. Use `zero` for zero-shot prompting or `few` for few-shot prompting. |
| `--reasoning` | `on`, `off` | `on` | Enables or disables reasoning mode. |
| `--prompt_features` | `dataset_description`, `unique_target` | none | Additional information included in the prompt. `dataset_description` adds textual dataset information. `unique_target` adds statistical target information. |
| `--random_group_sample` | flag | disabled | Enables random grouped row sampling. |
| `--group_size` | integer | `5` | Group size used for grouped sampling. Only used if `--random_group_sample` is enabled. |
| `--max_rows` | integer | `25` | Maximum number of rows loaded from each dataset. |
| `--data` | `train`, `val`, `test` | `test` | Metadata split used for the experiment. |

---

## 6. Example Command

The following command shows all available parameters in one execution example:

```bash
python run_exp2_train.py \
  --model Qwen/Qwen3-14B \
  --shot few \
  --reasoning on \
  --prompt_features dataset_description unique_target \
  --random_group_sample \
  --group_size 5 \
  --max_rows 25 \
  --data test
```

The selected metadata split can be changed with:

```bash
--data train
--data val
--data test
```

---

## 7. Input Metadata Files

The script uses one of the following metadata split files depending on the `--data` argument:

```text
../meta_datasets/tabular_timeseries_meta_datasets/exp2_train.csv
../meta_datasets/tabular_timeseries_meta_datasets/exp2_val.csv
../meta_datasets/tabular_timeseries_meta_datasets/exp2_test.csv
```

The split is selected as follows:

```bash
--data train
```

uses:

```text
../meta_datasets/tabular_timeseries_meta_datasets/exp2_train.csv
```

```bash
--data val
```

uses:

```text
../meta_datasets/tabular_timeseries_meta_datasets/exp2_val.csv
```

```bash
--data test
```

uses:

```text
../meta_datasets/tabular_timeseries_meta_datasets/exp2_test.csv
```

---

## 8. Output Files

Results are written to:

```text
Exp2_Results/
```

For each run, a separate folder is created based on the selected hyperparameter configuration.

Each run folder contains:

```text
experiment_results.csv
metrics.csv
raw_predictions.csv
meta_data_exp.csv
```

---

## 9. Explanation of Output Files

### `experiment_results.csv`

This file contains the full experiment results for every dataset in the selected split.

It includes the input metadata, the generated prompt, the raw model prediction, the parsed prediction, latency information, and additional run-level information for each evaluated dataset.

Typical columns include:

```text
prompt
pred_raw
pred_domain
pred_label
pred_class
latency_s
prompt_len
answer_len
gt_class
```

The most important prediction columns are:

| Column | Description |
|---|---|
| `pred_raw` | Raw text output generated by the LLM. |
| `pred_domain` | Parsed domain prediction, for example `Tabular` or `Time_Series`. |
| `pred_label` | Parsed downstream-task prediction, for example `regression`, `binary`, or `multiclass`. |
| `pred_class` | Combined prediction of domain and downstream task. |
| `gt_class` | Ground-truth combined class. |

---

### `metrics.csv`

This file contains the evaluation metrics for the experiment.

The completion rate shows the percentage of datasets for which the model produced a valid and parseable prediction.

The LLM predicts two aspects at the same time:

1. the **downstream task**, such as classification or regression;
2. the **data domain**, such as tabular data or time-series data.

The metrics are reported at three levels.

#### Downstream-task metrics: `_label`

Metrics ending in `_label` evaluate only the predicted downstream task.

Examples:

```text
accuracy_label
balanced_accuracy_label
precision_macro_label
recall_macro_label
f1_macro_label
```

These metrics measure how well the model predicts whether the dataset corresponds to a regression, binary classification, or multiclass classification problem.

#### Domain metrics: `_domain`

Metrics ending in `_domain` evaluate only the predicted data domain.

Examples:

```text
accuracy_domain
balanced_accuracy_domain
precision_macro_domain
recall_macro_domain
f1_macro_domain
```

These metrics measure how well the model predicts whether the dataset belongs to the tabular or time-series domain.

#### Combined-class metrics: `_class`

Metrics ending in `_class` evaluate the combined prediction of domain and downstream task.

Examples:

```text
accuracy_class
balanced_accuracy_class
precision_macro_class
recall_macro_class
f1_macro_class
```

For these metrics, a prediction is only correct if both the domain and the downstream task are predicted correctly.

For example:

```text
Time_Series_regression
Tabular_binary
Tabular_multiclass
```

---

### `raw_predictions.csv`

This file contains the raw text outputs generated by the LLM before any parsing or post-processing.

It is useful for checking whether the model followed the required output format and for debugging cases where the prediction could not be parsed correctly.

Typical content:

```text
raw_predictions
```

Each row corresponds to the raw LLM answer for one evaluated dataset.

---

### `meta_data_exp.csv`

This file contains metadata about the complete experiment run.

It stores the selected experiment configuration, including the experiment number, dataset phase, train/validation/test split, hyperparameter name, runtime, system prompt, sampling settings, and the final metrics.

Typical entries include:

```text
Exp_number
Dataset_Phase
Train/Test
Hyperparameter
Metrics
General_time
System_prompt
random_group_sample
group_size
max_rows
```

The `Hyperparameter` field summarizes the full run configuration, including model name, prompting strategy, reasoning setting, included prompt features, sampling strategy, and number of rows.

The `Metrics` field stores the final evaluation results for the run. These are also written separately to `metrics.csv`.

The `System_prompt` field stores the system prompt used for the experiment. This is useful for reproducibility, because it allows later inspection of the exact prompt configuration used during the run.

The sampling fields describe whether grouped row sampling was used:

```text
random_group_sample
group_size
max_rows
```

These values document how many rows were loaded and whether contiguous grouped sampling was applied.

---

## 10. Notes

Run the script from the folder where `run_exp2_train.py` is located, or make sure that the relative metadata paths are correct.

If imports from `utils` fail, make sure that:

1. the local package is installed correctly;
2. the `utils/` folder contains an `__init__.py` file;
3. `pip install -e .` was executed in the activated environment.

Expected project structure:

```text
Problem_Type_github/
├── pyproject.toml
├── req.txt
├── run_exp2_train.py
├── dataset/
├── utils/
│   ├── __init__.py
│   ├── file_handler_random.py
│   ├── meta_data_extraction.py
│   ├── prompt_helper.py
│   └── huggingface_model.py
├── meta_datasets/
│   ├── few_shot_example/
│   └── tabular_timeseries_meta_datasets/
└── Exp2_Results/
```
