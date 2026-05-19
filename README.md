# ML Task Identification Experiments

This repository contains code for running LLM-based machine learning task identification experiments.  
The main experiment script is `run_exp2_train.py`, which evaluates a selected model on tabular and time-series metadata splits.

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
```
Problem_Type_github/
├── dataset/
├── meta_datasets/
├── Exp1/
├── Exp2/
├── Exp3/
├── req.txt
├── pyproject.toml
└── utils/
```
If the repository is made public, replace the tokenized Zenodo review link with the final public Zenodo DOI or public record link.

---

## 4. Run Experiment 2

There are two ways to run Experiment 2:

1. a **local Hugging Face version**, using `run_exp2_train.py`;
2. an **online OpenAI API version**, using the scripts in the `Exp2/` folder.

---

### 4.1 Local Hugging Face Version

The local version runs a Hugging Face model directly on the machine.

The main script is:

```bash
run_exp2_train.py
```

The script requires at least the `--model` parameter.

Basic example:

```bash
python run_exp2_train.py --model Qwen/Qwen3-14B
```

The first execution may take several minutes because the model weights need to be downloaded.

---

### 4.2 Online OpenAI API Version

The online version uses the OpenAI API and creates batch requests for GPT-based models.

Before running the online scripts, create a file called:

```text
API.txt
```

inside the `Exp2/` folder:

```text
Exp2/
├── API.txt
├── create_batch_exp2.py
├── run_all_create_batches_parallel.sh
├── get_batch_results.ipynb
└── ...
```

Paste your OpenAI API key into `API.txt`.

The file should contain only the API key, without additional text:

```text
sk-...
```

Do not commit or upload `API.txt` to GitHub. It should be included in `.gitignore`.

Run the online scripts from inside the `Exp2/` folder, unless the relative paths in the scripts are changed.

---

#### Create a Single OpenAI Batch

To create one OpenAI batch job, run:

```bash
python create_batch_exp2.py \
  --model gpt-5 \
  --shot few \
  --reasoning high \
  --prompt_features dataset_description unique_target \
  --random_group_sample \
  --group_size 5 \
  --max_rows 25 \
  --completion_window 24h \
  --endpoint /v1/responses \
  --results_root Exp2_Results_paper_test
```

The script creates a JSONL batch input file, uploads it to the OpenAI Batch API, and stores the batch metadata for later retrieval.

Typical output files include:

```text
batch_input.jsonl
experiment_requests.csv
system_prompt.txt
batch_meta.json
```

The file `batch_meta.json` contains the `batch_id`, which is needed later to retrieve the completed batch results.

---

#### Create Multiple OpenAI Batches Automatically

To create multiple batch jobs for several hyperparameter settings, run:

```bash
bash run_all_create_batches_parallel.sh
```

This script runs `create_batch_exp2.py` multiple times with predefined settings.

The script currently uses:

```text
model: gpt-5.4
reasoning: high
endpoint: /v1/responses
completion window: 24h
results root: Exp2_Results_paper_test
```

It sweeps over settings such as:

```text
shot: few, zero
prompt features: unique_target
group sizes: 5, 10
max rows: 25, 50
random grouped sampling: enabled
```

The script also creates log files in:

```text
logs_batches/
```

The batch creation scripts only create and submit the OpenAI batch jobs. After the batch has completed, the resulting output file must be retrieved separately.

---

#### Retrieve OpenAI Batch Results

After the OpenAI batch jobs have completed, use the notebook:

```text
get_batch_results.ipynb
```

This notebook is used to:

1. check the status of submitted batch jobs;
2. download completed batch outputs;
3. download batch error files if a batch failed;
4. merge the batch outputs with `experiment_requests.csv`;
5. create final result files such as `experiment_results.csv` and `metrics.csv`.

Before running the notebook, make sure that `API.txt` exists in the same folder from which the notebook is executed:

```text
Exp2/
├── API.txt
├── create_batch_exp2.py
├── run_all_create_batches_parallel.sh
├── get_batch_results.ipynb
└── ...
```

The notebook expects a results root folder, for example:

```text
Exp2_Results_paper_test/
```

If your batch results are stored in a different folder, change the `results_root` argument inside the notebook.

Example:

```python
download_all_batches(results_root="Exp2_Results_paper_test/")
```

and:

```python
reports = merge_batch_outputs_and_write_metrics(
    "Exp2_Results_paper_test/",
    print_report=False
)
```

The final online batch workflow is:

```text
1. Create API.txt
2. Run create_batch_exp2.py or run_all_create_batches_parallel.sh
3. Wait until the OpenAI batch jobs are completed
4. Run get_batch_results.ipynb
5. Inspect experiment_results.csv and metrics.csv
```

---

## 5. Available Parameters

| Parameter | Used by | Options / Type | Default | Description |
|---|---|---|---|---|
| `--model` | local and online | string | required / script-dependent | Model name. For local runs, use a Hugging Face model such as `Qwen/Qwen3-14B`. For online runs, use an OpenAI model such as `gpt-5`. The first local execution may take several minutes because the model weights are downloaded. |
| `--shot` | local and online | `zero`, `few` | `zero` | Prompting mode. Use `zero` for zero-shot prompting or `few` for few-shot prompting. |
| `--reasoning` | local | `on`, `off` | `on` | Enables or disables reasoning mode for the local Hugging Face run. |
| `--reasoning` | online | `high`, `off` | `off` | Enables or disables reasoning mode for the OpenAI API batch run. |
| `--prompt_features` | local and online | `dataset_description`, `unique_target` | none | Additional information included in the prompt. `dataset_description` adds textual dataset information. `unique_target` adds statistical target information. |
| `--random_group_sample` | local and online | flag | disabled | Enables random grouped row sampling. |
| `--group_size` | local and online | integer | `5` | Group size used for grouped sampling. Only used if `--random_group_sample` is enabled. |
| `--max_rows` | local and online | integer | `25` local / `100` online | Maximum number of rows loaded from each dataset. |
| `--data` | local | `train`, `val`, `test` | `test` | Metadata split used by the local experiment script. |
| `--completion_window` | online | string | `24h` | OpenAI Batch API completion window. |
| `--endpoint` | online | string | `/v1/responses` | OpenAI API endpoint used for batch requests. |
| `--results_root` | online | string | `Exp2_Results_paper_test` | Output directory for generated batch files and metadata. |

---

## 6. Example Commands

### 6.1 Local Hugging Face Example

The following command shows all available local parameters in one execution example:

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

### 6.2 Online OpenAI Batch Example

The following command shows all available online batch parameters in one execution example:

```bash
python create_batch_exp2.py \
  --model gpt-5 \
  --shot few \
  --reasoning high \
  --prompt_features dataset_description unique_target \
  --random_group_sample \
  --group_size 5 \
  --max_rows 25 \
  --completion_window 24h \
  --endpoint /v1/responses \
  --results_root Exp2_Results_paper_test
```

For a full predefined batch sweep, run:

```bash
bash run_all_create_batches_parallel.sh
```

---

## 7. Input Metadata Files

The local script uses one of the following metadata split files depending on the `--data` argument:

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

The online OpenAI batch script uses the corresponding files inside the `Exp2/` workflow, for example:

```text
Exp2_datasets/exp2_test.csv
Exp2_datasets/df_tab_few_shot.csv
Exp2_datasets/df_time_few_shot.csv
```

---

## 8. Output Files

Results are written to a results folder.

For local Hugging Face runs, the default output folder is:

```text
Exp2_Results/
```

For online OpenAI batch runs, the output folder is controlled by:

```bash
--results_root
```

For example:

```text
Exp2_Results_paper_test/
```

For each run, a separate folder is created based on the selected hyperparameter configuration.

A completed local run folder contains:

```text
experiment_results.csv
metrics.csv
raw_predictions.csv
meta_data_exp.csv
```

An online batch run folder may contain intermediate files such as:

```text
batch_input.jsonl
experiment_requests.csv
system_prompt.txt
batch_meta.json
batch_output.jsonl
batch_errors.jsonl
```

After running `get_batch_results.ipynb`, the online batch run folder also contains processed output files such as:

```text
experiment_results.csv
metrics.csv
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

### `batch_input.jsonl`

This file is created by the online OpenAI batch script.

It contains one JSON request per dataset. Each line is one request that is uploaded to the OpenAI Batch API.

---

### `experiment_requests.csv`

This file is created by the online OpenAI batch script.

It contains the dataset rows and generated prompts before predictions are available.

After the batch output is downloaded, this file is merged with the model outputs to create `experiment_results.csv`.

---

### `batch_meta.json`

This file is created after submitting an OpenAI batch job.

It contains metadata required for retrieving the batch result later, including:

```text
batch_id
input_file_id
endpoint
completion_window
hyperparameter
requests_written
model
shot
reasoning
prompt_features
random_group_sample
group_size
max_rows
```

The `batch_id` is required by `get_batch_results.ipynb` to check the batch status and download the result file.

---

### `batch_output.jsonl`

This file is downloaded after an OpenAI batch job completed successfully.

It contains the raw API responses for all submitted requests.

---

### `batch_errors.jsonl`

This file is downloaded if the OpenAI batch job produced errors.

It can be used to inspect failed requests and debug invalid inputs, endpoint issues, or API-side failures.

---

## 10. Notes

Run the local script from the folder where `run_exp2_train.py` is located, or make sure that the relative metadata paths are correct.

Run the online OpenAI scripts from inside the `Exp2/` folder, unless the paths in the scripts are adapted.

If imports from `utils` fail, make sure that:

1. the local package is installed correctly;
2. the `utils/` folder contains an `__init__.py` file;
3. `pip install -e .` was executed in the activated environment.

Do not commit sensitive files such as:

```text
API.txt
.env
*.env
secrets.json
```

Expected project structure:

```
Problem_Type_github/
├── dataset/
├── meta_datasets/
├── Exp1/
├── Exp2/
│   ├── run_exp2_train.py
│   ├── create_batch_exp2.py
│   ├── run_all_create_batches_parallel.sh
│   ├── get_batch_results.ipynb
│   ├── API.txt
│   ├── Exp2_datasets/
│   ├── Exp2_Results/
│   └── Exp2_Results_paper_test/
├── Exp3/
├── req.txt
├── pyproject.toml
└── utils/
```
