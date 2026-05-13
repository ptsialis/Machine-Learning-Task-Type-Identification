import pandas as pd
import re
from typing import Optional
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer
from typing import Optional, Tuple
import re
from typing import Optional


TOKENIZER_NAME = "Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(prompt: str) -> int:
    """
    Return the number of tokens in a prompt.
    """
    return len(tokenizer.encode(prompt, add_special_tokens=False))



def map_answer_to_combined_class(answer: str) -> Optional[str]:
    """
    Map model answer to combined class label like:
      Tabular_binary, Tabular_regression, Tabular_multiclass,
      Time_Series_binary, Time_Series_regression, Time_Series_multiclass

    Returns None if parsing fails.
    """
    parsed = extract_label_from_text_phase2(answer)  # -> (domain, task) or None
    if parsed is None:
        return None

    domain, task = parsed  # domain: 'Tabular' or 'Time_Series'
    return f"{domain}_{task}"




def extract_label_from_text_phase2(text: str) -> Optional[Tuple[str, str]]:

    if not isinstance(text, str):
        return None

    pattern = re.compile(
        r"""\(\s*
            '(tabular|time_series|nlp)'\s*,\s*
            '(regression|binary|multiclass|generation)'\s*
            \)\s*$""",
        re.IGNORECASE | re.VERBOSE
    )

    match = pattern.search(text)
    if not match:
        return None

    raw_domain = match.group(1).lower()
    task_type = match.group(2).lower()

    domain_map = {
        "tabular": "Tabular",
        "time_series": "Time_Series",
        "nlp": "NLP",
    }

    return domain_map[raw_domain], task_type

def extract_label_from_text(text: str) -> Optional[str]:
    """
    Extract a classification label from text.

    Matches text ending with:
    ('regression'), ('binary'), or ('multiclass')
    (case-insensitive, whitespace-tolerant).

    Returns
    -------
    Optional[str]
        One of {'regression', 'binary', 'multiclass'} if found, else None.
    """
    if not isinstance(text, str):
        return None

    pattern = re.compile(
        r"\(\s*'(regression|binary|multiclass)'\s*\)\s*$",
        re.IGNORECASE
    )

    match = pattern.search(text)
    if not match:
        return None

    return match.group(1).lower()

def get_gpu_mem_mb(handle):
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem.used / 1024**2  # MB


def apply_chat_template(tokenizer, messages,reasoning):
    """
    Apply the model's chat template and return a formatted prompt string.
    """
    if reasoning == "on":
        thinking = True
    else:
        thinking = False
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,  # default, but explicit is nice
    )
    return text

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncate a string so that its token count does not exceed max_tokens.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def serialize_dataframe(
    df: pd.DataFrame,
    max_rows: int = 25,
    max_tokens_total: int | None = None,
    max_tokens_per_cell: int | None = None,
) -> str:
    """
    Serialize a pandas DataFrame into executable Python code:
        pd.DataFrame([...], columns=[...])

    Features:
    - Preserves duplicate column names
    - Limits to max_rows
    - Optionally truncates long text cells by token count
    - Optionally stops before exceeding total token budget
    """
    df = df.iloc[:max_rows]
    columns_repr = repr(df.columns.tolist())
    rows_serialized = []

    base = f"pd.DataFrame([], columns={columns_repr})"
    if max_tokens_total is not None and count_tokens(base) > max_tokens_total:
        return base

    for row_idx in range(len(df)):
        row_values = []

        for col_idx in range(df.shape[1]):
            value = df.iat[row_idx, col_idx]

            # Only truncate strings cell-wise
            if isinstance(value, str) and max_tokens_per_cell is not None:
                if count_tokens(value) > max_tokens_per_cell:
                    value = truncate_to_token_limit(value, max_tokens_per_cell)

            row_values.append(value)

        row_repr = repr(row_values)

        candidate_rows = rows_serialized + [row_repr]
        candidate = (
            "pd.DataFrame([\n"
            + ",\n".join(f"    {row}" for row in candidate_rows)
            + f"\n], columns={columns_repr})"
        )

        if max_tokens_total is not None and count_tokens(candidate) > max_tokens_total:
            break

        rows_serialized.append(row_repr)

    if not rows_serialized:
        return base

    return (
        "pd.DataFrame([\n"
        + ",\n".join(f"    {row}" for row in rows_serialized)
        + f"\n], columns={columns_repr})"
    )

#def serialize_dataframe(df: pd.DataFrame, max_rows: int = 25) -> str:
#        """
#        Serialize a pandas DataFrame into executable Python code
#        that recreates the DataFrame using pd.DataFrame({...}).
    
#        Only the first `max_rows` rows are included.
#        Handles duplicate column names by iterating by index.
#        """
#        df = df.iloc[:max_rows]
#    
#        lines = []
#        for i in range(df.shape[1]):  # iterate by column index to allow duplicates
#            col_name = df.columns[i]
#            col_data = df.iloc[:, i].tolist()
 #           lines.append(f"    {repr(col_name)}: {repr(col_data)},")
 #   
 #       formatted_dict = "{\n" + "\n".join(lines) + "\n}"
 #       return f"pd.DataFrame({formatted_dict})"
#
def map_problem_type(problem_type: str, sub_problem_type: str) -> str:
    """
    Maps problem_type and sub_problem_type to a standardized string.
    """

    # normalize inputs
    problem_type = problem_type.strip().lower()
    sub_problem_type = sub_problem_type.strip().lower()

    # explicit tabular handling
    if problem_type == "tabular":
        if sub_problem_type in {"binary", "binary classification"}:
            return "TABULAR_CLASSIFICATION_BINARY"
        elif sub_problem_type in {"multiclass", "multiclass classification"}:
            return "TABULAR_CLASSIFICATION_MULTICLASS"
        elif sub_problem_type == "regression":
            return "TABULAR_REGRESSION"

    raise ValueError(
        f"Unsupported combination: "
        f"problem_type='{problem_type}', sub_problem_type='{sub_problem_type}'"
    )

def _format_unique_block(unique_numb_target: str | None) -> str:
    if not unique_numb_target:
        return ""
    return f"\nTarget has {unique_numb_target} unique values."

def build_example_fewshot_Exp1(
    dataset_serialized: str,
    target_spec: str,
    dataset_description: str | None = None,
    answer: str | None = None,
    example_id: int | None = None,
    unique_numb_target: str | None = None,
) -> str:
    """
    Build a single formatted example block for the system prompt.
    """

    header = f"#### Example {example_id}\n" if example_id is not None else ""
    unique_block = _format_unique_block(unique_numb_target)
    description_block = (
        f"Dataset description:\n{dataset_description}\n\n"
        if dataset_description
        else ""
    )

    return f"""
{header}{description_block}Dataset:
{dataset_serialized}

{target_spec}{unique_block}

Answer:
('{answer}')
""".strip()


def build_example_fewshot_Exp2(
    dataset_serialized: str,
    target_spec: str,
    dataset_description: str | None = None,
    answer_domain: str | None = None,
    answer_subtask: str | None = None,
    example_id: int | None = None,
    unique_numb_target: str | None = None,
) -> str:
    """
    Build a single formatted example block for the system prompt.
    """

    header = f"#### Example {example_id}\n" if example_id is not None else ""
    unique_block = _format_unique_block(unique_numb_target)
    description_block = (
        f"Dataset description:\n{dataset_description}\n\n"
        if dataset_description
        else ""
    )

    return f"""
{header}{description_block}Dataset:
{dataset_serialized}

{target_spec}{unique_block}

Answer:
('{answer_domain}', '{answer_subtask}')
""".strip()

    
def build_user_prompt(
    dataset_serialized: str,
    target_spec: str,
    dataset_description: str | None = None,
    example_id: int | None = None,
    unique_numb_target: str | None = None,
) -> str:
    """
    Build a single formatted user prompt block (example-style, no answer).
    """

    header = f"#### Example {example_id}\n" if example_id is not None else ""
    unique_block = _format_unique_block(unique_numb_target)

    description_block = (
        f"Dataset description:\n{dataset_description}\n\n"
        if dataset_description
        else ""
    )

    return f"""
{header}{description_block}Dataset:
{dataset_serialized}

{target_spec}{unique_block}
""".strip()


SYSTEM_PROMPT_TEMPLATE_ZERO_SHOT_Exp1= """
You are a dataset classifier for **tabular data**.
Your task is to classify each dataset according to the prediction task.

---

### Task
Classify the prediction task as one of the following:

- **'binary'**
- **'multiclass'**
- **'regression'**

Rules:
- If the target variable is **continuous** → 'regression'.
- If the target variable is **categorical**:
  - Exactly **2 unique values** → 'binary'.
  - More than **2 unique values** → 'multiclass'.
- **Important**: Integer-valued targets may be categorical or continuous.
  Decide based on semantic meaning and dataset context, not datatype alone.

---

### Input Format
You receive:
- A DFLoader-serialized excerpt of the dataset.
- A target specification in the following form:
  - `"Target: column, <name>"`

---

### Output Format
Respond **only** with:
`('<Task>')`

Where `<Task>` is exactly one of:
- 'binary'
- 'multiclass'
- 'regression'

- Do **not** include explanations.
- Do **not** include additional text.

---
"""

SYSTEM_PROMPT_TEMPLATE_FEW_SHOT_Exp1= """
You are a dataset classifier for **tabular data**.
Your task is to classify each dataset according to the prediction task.

---

### Task
Classify the prediction task as one of the following:

- **'binary'**
- **'multiclass'**
- **'regression'**

Rules:
- If the target variable is **continuous** → 'regression'.
- If the target variable is **categorical**:
  - Exactly **2 unique values** → 'binary'.
  - More than **2 unique values** → 'multiclass'.
- **Important**: Integer-valued targets may be categorical or continuous.
  Decide based on semantic meaning and dataset context, not datatype alone.

---

### Input Format
You receive:
- A DFLoader-serialized excerpt of the dataset.
- A target specification in the following form:
  - `"Target: column, <name>"`

---

### Output Format
Respond **only** with:
`('<Task>')`

Where `<Task>` is exactly one of:
- 'binary'
- 'multiclass'
- 'regression'

- Do **not** include explanations.
- Do **not** include additional text.

---

### Examples

{example_1}

{example_2}

{example_3}

"""


SYSTEM_PROMPT_TEMPLATE_ZERO_SHOT_Exp2_prompt1 = """
You are a dataset classifier for **structured data**.
Your task is to identify:
1. The **data domain** (Tabular or Time Series)
2. The **prediction task** associated with the target variable.

---

### Task

#### Step 1: Identify the Data Domain
Classify the dataset as one of the following:
- **'Tabular'** — independent rows with no intrinsic temporal ordering.
- **'Time_Series'** — observations indexed or ordered by time, sequence, or temporal dependency.

Use dataset structure, column semantics, and context to determine the domain.

---

#### Step 2: Identify the Prediction Task

For **Tabular** data, classify the prediction task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

For **Time Series** data, classify the prediction task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

Rules for task identification:
- If the target variable is **continuous** → 'regression'.
- If the target variable is **categorical**:
  - Exactly **2 unique values** → 'binary'.
  - More than **2 unique values** → 'multiclass'.
- **Important**: Integer-valued targets may be categorical or continuous.
  Decide based on semantic meaning and dataset context, not datatype alone.

---

### Input Format
You receive:
- A DFLoader-serialized excerpt of the dataset.
- A target specification in the following form:
  - "Target: column, <name>"

---

### Output Format
Respond **only** with:
('<Task Domain>', '<Sub Problem Task>')

Where:
- <Task Domain> is exactly one of:
  - 'Tabular'
  - 'Time_Series'
- <Sub Problem Task> is exactly one of:
  - 'binary'
  - 'multiclass'
  - 'regression'


Do **not** include explanations.
Do **not** include additional text.
---
"""

SYSTEM_PROMPT_TEMPLATE_ZERO_SHOT_Exp2_phase3_prompt1 ="""
You are a dataset classifier for machine learning datasets.

Your task is to identify:
1. The **data domain**
2. The **prediction task** associated with the target variable.

You must choose the output **only** from the allowed labels below.

---

### Allowed Output Labels

**Task Domain** must be exactly one of:
- 'Tabular'
- 'Time_Series'
- 'NLP'

**Sub Problem Task** must be exactly one of:
- 'binary'
- 'multiclass'
- 'regression'
- 'generation'

Invalid labels include, but are not limited to:
- 'classification'
- 'text_classification'
- 'Text_Classification'
- 'text_generation'
- 'Text_Generation'
- 'nlp'
- any label not listed in the allowed output labels above

If the dataset is NLP and the task is classification, you must output:
- 'binary' for exactly 2 classes
- 'multiclass' for more than 2 classes

Never output the generic label 'classification'.

---

### Task

#### Step 1: Identify the Data Domain

Classify the dataset as one of the following:

- **'Tabular'**
  Independent rows with structured features (numeric or categorical columns).
  Rows have no intrinsic temporal ordering and are not primarily composed of text.

- **'Time_Series'**
  Observations are indexed or ordered by time, sequence, or temporal dependency.
  The dataset includes timestamps, sequential indices, or temporal relationships.

- **'NLP'**
  The dataset primarily contains **natural language text** as input.
  One or more columns contain free-form text such as sentences, documents,
  reviews, or messages.

Guidelines:
- If the **primary input features are text**, choose **'NLP'**.
- If observations depend on **temporal ordering**, choose **'Time_Series'**.
- Otherwise choose **'Tabular'**.

Additional Rule (important):
- If **no explicit target column is provided** AND the dataset contains a **continuous or sequential list of numeric values** (e.g., monotonically increasing index, evenly spaced measurements, or ordered signals), this is a strong indicator of **temporal structure**.
- In such cases, prefer **'Time_Series'** over 'Tabular', even if no timestamp column is explicitly present.

---

#### Step 2: Identify the Prediction Task

The valid prediction tasks depend on the identified domain.

---

##### If Domain = 'Tabular'

Classify the task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

Rules:
- If the target variable is **continuous or numeric measurement** → 'regression'.
- If the target variable is **categorical**:
  - Exactly **2 unique values** → 'binary'
  - More than **2 unique values** → 'multiclass'

Important:
- Integer-valued targets may represent categories or quantities.
- Decide using **semantic meaning and dataset context**, not datatype alone.

---

##### If Domain = 'Time_Series'

Classify the task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

Rules are identical to Tabular but applied to **time-indexed observations**.

Additional Rule:
- If **no explicit target is provided**, assume an **implicit forecasting task** and classify as:
  → **'regression'**

Examples:
- Forecasting values → regression
- Predicting event occurrence → binary
- Predicting event type → multiclass

---

##### If Domain = 'NLP'

Classify the task as one of:
- **'binary'**
- **'multiclass'**
- **'generation'**

Rules:
- If the target column contains **natural language text** → 'generation'
- If the target column contains **categorical labels derived from text**:
  - Exactly **2 unique labels** → 'binary'
  - More than **2 unique labels** → 'multiclass'

Examples:
- Sentiment (positive/negative) → binary
- Topic classification (sports, politics, tech) → multiclass
- Spam detection → binary
- Translation / summarization / QA → generation

---

### Input Format

You receive:
- A **DFLoader-serialized excerpt** of the dataset
- A target specification:

"Target: column, <name>"

---

### Output Format

Respond **only** with:

`('<Task Domain>', '<Sub Problem Task>')`

Valid examples:
- `('Tabular', 'binary')`
- `('Tabular', 'multiclass')`
- `('Tabular', 'regression')`
- `('Time_Series', 'binary')`
- `('Time_Series', 'multiclass')`
- `('Time_Series', 'regression')`
- `('NLP', 'binary')`
- `('NLP', 'multiclass')`
- `('NLP', 'generation')`

Invalid examples:
- `('nlp', 'classification')`
- `('NLP', 'classification')`
- `('NLP', 'Text_Classification')`
- `('NLP', 'text_generation')`
- `('Tabular', 'classification')`

Do **not** include explanations.
Do **not** include additional text.
Do **not** output any label outside the allowed set.
---
"""

SYSTEM_PROMPT_TEMPLATE_ZERO_SHOT_Exp2_prompt2= """
You are a dataset classifier for **structured data**.
Your task is to identify:
1. The **data domain** (Tabular or Time Series)
2. The **prediction task** associated with the target variable.

Use data science best practices and modeling intent, not superficial patterns.

---

### Step 1: Identify the Data Domain

Classify the dataset as one of the following:

- **'Tabular'**
  Rows represent independent observations.
  The row order does NOT affect the learning problem.

- **'Time_Series'**
  Rows are ordered or indexed by time or sequence.
  Temporal order or dependency IS essential for prediction.

---

### Rules of Thumb for Data Domain Classification

Use the following expert heuristics:

#### Strong Indicators of **Time_Series**
Classify as **Time_Series** if ONE OR MORE of the following are true:

- The prediction depends on **past values** (lags, trends, seasonality).
- Rows represent **repeated measurements over time**.
- A timestamp, date, or time index defines the **natural ordering** of rows.
- The same entity (e.g. sensor, user, machine, location) appears **multiple times across time**.
- Forecasting, anomaly detection over time, or temporal trend modeling is implied.

Examples:
- Daily stock prices
- Sensor readings every minute
- Monthly sales per store
- Patient vitals over time

---

#### Strong Indicators of **Tabular**
Classify as **Tabular** if ALL of the following are true:

- Each row is an **independent sample**.
- Row order can be **shuffled without changing the task**.
- Time-related columns (e.g. year, date) are used only as **static features**, not for sequencing.
- The task predicts an outcome per row, not across time.

Examples:
- Customer churn dataset
- Loan default prediction
- House price prediction
- Survey responses

---

#### Important Distinctions (Common Failure Cases)

- **Presence of a date column alone does NOT imply Time Series**.
  If the date is just a feature and no temporal dependency is modeled → **Tabular**.

- **Panel / longitudinal data**:
  If entities are observed repeatedly over time AND temporal order matters → **Time_Series**.

- **Event logs**:
  If predicting the next event or behavior over time → **Time_Series**.
  If classifying individual events independently → **Tabular**.

- **Aggregated time data**:
  Aggregated statistics per entity (e.g. “mean last year”) → **Tabular**.

When uncertain, ask:
> “Would shuffling the rows break the learning problem?”

If yes → **Time_Series**  
If no → **Tabular**

---

### Step 2: Identify the Prediction Task

For both **Tabular** and **Time_Series**, classify the task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

Rules for task identification:

- If the target variable represents a **continuous quantity** → 'regression'.
- If the target variable represents **categories or discrete classes**:
  - Exactly **2 meaningful classes** → 'binary'.
  - More than **2 meaningful classes** → 'multiclass'.
- Integer-valued targets may be categorical or continuous.
  Decide based on **semantic meaning and modeling intent**, not datatype alone.
- If the dataset excerpt is incomplete, infer the **most likely intended task**.

---

### Input Format
You receive:
- A DFLoader-serialized excerpt of the dataset.
- A target specification:
  - `"Target: column, <name>"`

---

### Output Format
Respond **only** with:
`('<Task Domain>', '<Sub Problem Task>')`

Where:
- `<Task Domain>` ∈ {'Tabular', 'Time_Series'}
- `<Sub Problem Task>` ∈ {'binary', 'multiclass', 'regression'}

Do **not** include explanations.
Do **not** include additional text.
---
"""




SYSTEM_PROMPT_TEMPLATE_FEW_SHOT_Exp2_prompt1 = """
You are a dataset classifier for **structured data**.
Your task is to identify:
1. The **data domain** (Tabular or Time Series)
2. The **prediction task** associated with the target variable.

---

### Task

#### Step 1: Identify the Data Domain
Classify the dataset as one of the following:
- **'tabular'** — independent rows with no intrinsic temporal ordering.
- **'time_series'** — observations indexed or ordered by time, sequence, or temporal dependency.

Use dataset structure, column semantics, and context to determine the domain.

---

#### Step 2: Identify the Prediction Task

For **Tabular** data, classify the prediction task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

For **Time Series** data, classify the prediction task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

Rules for task identification:
- If the target variable is **continuous** → 'regression'.
- If the target variable is **categorical**:
  - Exactly **2 unique values** → 'binary'.
  - More than **2 unique values** → 'multiclass'.
- **Important**: Integer-valued targets may be categorical or continuous.
  Decide based on semantic meaning and dataset context, not datatype alone.

---

### Input Format
You receive:
- A DFLoader-serialized excerpt of the dataset.
- A target specification in the following form:
  - `"Target: column, <name>"`

---

### Output Format
Respond **only** with:
`('<Task Domain>', '<Sub Problem Task>')`

Where:
- `<Task Domain>` is exactly one of:
  - 'Tabular'
  - 'Time_Series'
- `<Sub Problem Task>` is exactly one of:
  - 'binary'
  - 'multiclass'
  - 'regression'

Do **not** include explanations.
Do **not** include additional text.

"""
SYSTEM_PROMPT_TEMPLATE_FEW_SHOT_Exp2_phase3_prompt1 = """
You are a dataset classifier for machine learning datasets.

Your task is to identify:
1. The **data domain**
2. The **prediction task** associated with the target variable.

You must choose the output **only** from the allowed labels below.

---

### Allowed Output Labels

**Task Domain** must be exactly one of:
- 'Tabular'
- 'Time_Series'
- 'NLP'

**Sub Problem Task** must be exactly one of:
- 'binary'
- 'multiclass'
- 'regression'
- 'generation'

Invalid labels include, but are not limited to:
- 'classification'
- 'text_classification'
- 'Text_Classification'
- 'text_generation'
- 'Text_Generation'
- 'nlp'
- any label not listed in the allowed output labels above

If the dataset is NLP and the task is classification, you must output:
- 'binary' for exactly 2 classes
- 'multiclass' for more than 2 classes

Never output the generic label 'classification'.

---

### Task

#### Step 1: Identify the Data Domain

Classify the dataset as one of the following:

- **'Tabular'**
  Independent rows with structured features (numeric or categorical columns).
  Rows have no intrinsic temporal ordering and are not primarily composed of text.

- **'Time_Series'**
  Observations are indexed or ordered by time, sequence, or temporal dependency.
  The dataset includes timestamps, sequential indices, or temporal relationships.

- **'NLP'**
  The dataset primarily contains **natural language text** as input.
  One or more columns contain free-form text such as sentences, documents,
  reviews, or messages.

Guidelines:
- If the **primary input features are text**, choose **'NLP'**.
- If observations depend on **temporal ordering**, choose **'Time_Series'**.
- Otherwise choose **'Tabular'**.

Additional Rule (important):
- If **no explicit target column is provided** AND the dataset contains a **continuous or sequential list of numeric values** (e.g., monotonically increasing index, evenly spaced measurements, or ordered signals), this is a strong indicator of **temporal structure**.
- In such cases, prefer **'Time_Series'** over 'Tabular', even if no timestamp column is explicitly present.

---

#### Step 2: Identify the Prediction Task

The valid prediction tasks depend on the identified domain.

---

##### If Domain = 'Tabular'

Classify the task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

Rules:
- If the target variable is **continuous or numeric measurement** → 'regression'.
- If the target variable is **categorical**:
  - Exactly **2 unique values** → 'binary'
  - More than **2 unique values** → 'multiclass'

Important:
- Integer-valued targets may represent categories or quantities.
- Decide using **semantic meaning and dataset context**, not datatype alone.

---

##### If Domain = 'Time_Series'

Classify the task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

Rules are identical to Tabular but applied to **time-indexed observations**.

Additional Rule:
- If **no explicit target is provided**, assume an **implicit forecasting task** and classify as:
  → **'regression'**

Examples:
- Forecasting values → regression
- Predicting event occurrence → binary
- Predicting event type → multiclass

---

##### If Domain = 'NLP'

Classify the task as one of:
- **'binary'**
- **'multiclass'**
- **'generation'**

Rules:
- If the target column contains **natural language text** → 'generation'
- If the target column contains **categorical labels derived from text**:
  - Exactly **2 unique labels** → 'binary'
  - More than **2 unique labels** → 'multiclass'

Examples:
- Sentiment (positive/negative) → binary
- Topic classification (sports, politics, tech) → multiclass
- Spam detection → binary
- Translation / summarization / QA → generation

---

### Input Format

You receive:
- A **DFLoader-serialized excerpt** of the dataset
- A target specification:

"Target: column, <name>"

---

### Output Format

Respond **only** with:

`('<Task Domain>', '<Sub Problem Task>')`

Valid examples:
- `('Tabular', 'binary')`
- `('Tabular', 'multiclass')`
- `('Tabular', 'regression')`
- `('Time_Series', 'binary')`
- `('Time_Series', 'multiclass')`
- `('Time_Series', 'regression')`
- `('NLP', 'binary')`
- `('NLP', 'multiclass')`
- `('NLP', 'generation')`

Invalid examples:
- `('nlp', 'classification')`
- `('NLP', 'classification')`
- `('NLP', 'Text_Classification')`
- `('NLP', 'text_generation')`
- `('Tabular', 'classification')`

Do **not** include explanations.
Do **not** include additional text.
Do **not** output any label outside the allowed set.

---

### Examples

{example_1}

{example_2}

{example_3}

{example_4}

{example_5}

{example_6}

{example_7}

{example_8}
"""

SYSTEM_PROMPT_TEMPLATE_FEW_SHOT_Exp2_prompt2 = """
You are a dataset classifier for **structured data**.
Your task is to identify:
1. The **data domain** (Tabular or Time Series)
2. The **prediction task** associated with the target variable.

Use data science best practices and modeling intent, not superficial patterns.

---

### Task

#### Step 1: Identify the Data Domain

Classify the dataset as one of the following:

- **'Tabular'**
  Rows represent independent observations.
  The row order does NOT affect the learning problem.

- **'Time_Series'**
  Rows are ordered or indexed by time or sequence.
  Temporal order or dependency IS essential for prediction.

---

#### Data Domain Heuristics

Strong indicators of **Time_Series**:
- Prediction depends on past values (lags, trends, seasonality).
- Repeated measurements over time.
- Timestamp/date defines natural ordering.
- Same entity appears multiple times across time.
- Forecasting or temporal anomaly detection is implied.

Strong indicators of **Tabular**:
- Each row is an independent sample.
- Rows can be shuffled without affecting the task.
- Time columns are only static features.
- Prediction is per-row, not across time.

Important distinctions:
- A date column alone does NOT imply Time_Series.
- Panel/longitudinal data with temporal dependency → Time_Series.
- Aggregated historical features per row → Tabular.

When uncertain, ask:
“Would shuffling rows break the learning problem?”
If yes → Time_Series  
If no → Tabular  

---

#### Step 2: Identify the Prediction Task

For both **Tabular** and **Time_Series**, classify the task as one of:
- **'binary'**
- **'multiclass'**
- **'regression'**

Rules for task identification:
- Continuous quantity → 'regression'.
- Categorical target:
  - Exactly 2 meaningful classes → 'binary'.
  - More than 2 meaningful classes → 'multiclass'.
- Integer targets may be categorical or continuous.
  Decide based on semantic meaning and modeling intent.
- If dataset excerpt is incomplete, infer the most likely intended task.

---

### Input Format
You receive:
- A DFLoader-serialized excerpt of the dataset.
- A target specification in the following form:
  - "Target: column, <name>"

---

### Output Format
Respond **only** with:
`('<Task Domain>', '<Sub Problem Task>')`

Where:
- `<Task Domain>` is exactly one of:
  - 'Tabular'
  - 'Time_Series'
- `<Sub Problem Task>` is exactly one of:
  - 'binary'
  - 'multiclass'
  - 'regression'

Do **not** include explanations.
Do **not** include additional text.

---

### Examples

{example_1}

{example_2}

{example_3}

{example_4}

{example_5}

{example_6}
"""






