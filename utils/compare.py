import ast
import re
from pathlib import Path
from typing import Any, Optional
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Constants
# ============================================================

AUTOG_LUON_COLOR = "#f08080"
DEFAULT_COLOR = "#4682b4"
AUTOG_LUON_SCALE = [[0.0, "#fde0dd"], [1.0, "#f08080"]]
DEFAULT_SCALE = "Blues"


# ============================================================
# Basic helpers
# ============================================================
def get_best_model_per_parameter_combo(
    summary_df: pd.DataFrame,
    *,
    run_name_col: str = "run_name",
    score_col: str = "f1_macro",
) -> pd.DataFrame:
    """
    Keep the best row for each combination of:
      - model
      - shot
      - reasoning

    Returns one row per parameter combination.
    """
    df = summary_df.copy()

    required = {run_name_col, score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary_df is missing required columns: {sorted(missing)}")

    extracted = df[run_name_col].astype(str).apply(extract_model_shot_reasoning)
    df[["model", "shot", "reasoning"]] = pd.DataFrame(extracted.tolist(), index=df.index)

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col])

    if df.empty:
        raise ValueError("No valid rows with numeric scores found.")

    idx = df.groupby(["model", "shot", "reasoning"])[score_col].idxmax()

    best_df = (
        df.loc[idx]
        .sort_values(["model", "shot", "reasoning"])
        .reset_index(drop=True)
    )

    return best_df
def extract_model_shot_reasoning(run_name: str) -> tuple[str, str, str]:
    """
    Extract:
      - model
      - shot setting
      - reasoning setting

    Example
    -------
    gpt-5_Train_few_shot_Reasoning_...
    -> ("gpt-5", "few_shot", "Reasoning")
    """
    s = normalize_run_name_suffix(str(run_name))
    parts = [p for p in s.split("_") if p]

    model = parts[0] if len(parts) > 0 else "unknown_model"

    low = s.lower()

    if ("few" in low and "shot" in low) or "fewshot" in low:
        shot = "few_shot"
    elif ("zero" in low and "shot" in low) or "zeroshot" in low or "0shot" in low:
        shot = "zero_shot"
    else:
        shot = "unknown_shot"

    if ("no" in low and "reason" in low) or "noreason" in low or "no_reason" in low:
        reasoning = "NoReasoning"
    elif "reason" in low:
        reasoning = "Reasoning"
    else:
        reasoning = "unknown_reasoning"

    return model, shot, reasoning
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read CSV if it exists, otherwise return None."""
    return pd.read_csv(path) if path.exists() else None


def ensure_unique_path(path: Path) -> Path:
    """If file exists, append _1, _2, ... before suffix."""
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    i = 1

    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def save_plotly_figure(
    fig: go.Figure,
    base_dir: Path,
    name: str,
    *,
    save_html: bool = False,
    save_pdf: bool = True,
    save_svg: bool = True,
) -> dict[str, Path | str]:
    """
    Save a Plotly figure under:
      - base_dir/html/<name>.html  (optional)
      - base_dir/pdf/<name>.pdf
      - base_dir/svg/<name>.svg
    """
    saved: dict[str, Path | str] = {}

    if save_html:
        html_dir = base_dir / "html"
        html_dir.mkdir(parents=True, exist_ok=True)
        html_path = ensure_unique_path(html_dir / f"{name}.html")
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        saved["html"] = html_path

    if save_pdf:
        try:
            pdf_dir = base_dir / "pdf"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = ensure_unique_path(pdf_dir / f"{name}.pdf")
            fig.write_image(str(pdf_path), format="pdf", engine="kaleido")
            saved["pdf"] = pdf_path
        except Exception as e:
            saved["pdf_error"] = str(e)

    if save_svg:
        try:
            svg_dir = base_dir / "svg"
            svg_dir.mkdir(parents=True, exist_ok=True)
            svg_path = ensure_unique_path(svg_dir / f"{name}.svg")
            fig.write_image(str(svg_path), format="svg", engine="kaleido")
            saved["svg"] = svg_path
        except Exception as e:
            saved["svg_error"] = str(e)

    return saved

def plot_f1_violin_combined(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> go.Figure:

    plot_df = df.copy()
    plot_df[run_name_col] = plot_df[run_name_col].astype(str)
    plot_df[f1_col] = pd.to_numeric(plot_df[f1_col], errors="coerce")

    run_name_pattern = re.compile(
        r"^(?P<model>.+?)_"
        r"(?:Train|Val|Test)_"
        r"(?P<shot>few_shot|zero_shot)_"
        r"(?P<reasoning>Reasoning|NoReasoning)_"
        r"Dataset_serialized_Target_name_"
        r"(?P<config>.+?)"
        r"_?(?P<sampling>(?:Grouped_g\d+_rows\d+|Full_rows\d+))$"
    )

    def parse_run_name(run_name: str) -> pd.Series:
        m = run_name_pattern.match(run_name)

        if m is None:
            return pd.Series(
                {
                    "model_name": None,
                    "shot": None,
                    "reasoning": None,
                    "after_name": None,
                    "sampling": None,
                }
            )

        model_name = m.group("model")

        if model_name == "gpt-4.1":
            model_name = "gpt-5"
        elif model_name == "gpt-5.3":
            model_name = None

        return pd.Series(
            {
                "model_name": model_name,
                "shot": m.group("shot"),
                "reasoning": m.group("reasoning"),
                "after_name": m.group("config").rstrip("_"),
                "sampling": m.group("sampling"),
            }
        )

    parsed = plot_df[run_name_col].apply(parse_run_name)
    plot_df = pd.concat([plot_df, parsed], axis=1)

    plot_df = plot_df.dropna(
        subset=[f1_col, "model_name", "shot", "reasoning", "after_name"]
    )

    rename_map = {
        "dataset_description+unique_target": "dataset description + statistical target information",
        "dataset_description": "dataset description",
        "unique_target": "statistical target information",
        "BasePrompt": "base prompt",
    }

    plot_df["after_name"] = (
        plot_df["after_name"]
        .map(rename_map)
        .fillna(plot_df["after_name"])
    )

    short_map = {
        "dataset description + statistical target information": "DD + ST",
        "dataset description": "DD",
        "statistical target information": "ST",
        "base prompt": "BP",
    }

    order_after = [
        "dataset description + statistical target information",
        "dataset description",
        "statistical target information",
        "base prompt",
    ]

    order_reasoning = ["Reasoning", "NoReasoning"]
    order_shot = ["few_shot", "zero_shot"]

    preferred_model_order = ["gpt-5", "Qwen2.5-14B-Instruct", "Qwen3-14B"]
    order_model = [
        m for m in preferred_model_order
        if m in plot_df["model_name"].unique()
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Dataset Configuration",
            "Model",
            "Reasoning",
            "Shot",
        ),
        horizontal_spacing=0.03,
        vertical_spacing=0.10,
    )

    def add_violin_trace(sub_df, name, row, col):
        fig.add_trace(
            go.Violin(
                y=sub_df[f1_col],
                name=name,
                points="all",
                box_visible=True,
                meanline_visible=False,
                line=dict(color="black"),
                fillcolor="white",
                marker=dict(
                    color="black",
                    size=5,
                    opacity=0.75,
                ),
                box=dict(
                    visible=True,
                    width=0.18,
                ),
                quartilemethod="linear",
                opacity=1,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    for cat in order_after:
        sub = plot_df[plot_df["after_name"] == cat]
        if not sub.empty:
            add_violin_trace(sub, short_map.get(cat, cat), row=1, col=1)

    for cat in order_model:
        sub = plot_df[plot_df["model_name"] == cat]
        if not sub.empty:
            add_violin_trace(sub, cat, row=1, col=2)

    for cat in order_reasoning:
        sub = plot_df[plot_df["reasoning"] == cat]
        if not sub.empty:
            add_violin_trace(sub, cat, row=2, col=1)

    shot_label_map = {
        "few_shot": "Few-shot",
        "zero_shot": "Zero-shot",
    }

    for cat in order_shot:
        sub = plot_df[plot_df["shot"] == cat]
        if not sub.empty:
            add_violin_trace(sub, shot_label_map.get(cat, cat), row=2, col=2)

    fig.update_layout(
        height=700,
        width=1000,
        title="F1 Distribution of every hyperparameter over all Experiment runs on Validation for Exp.2",
        margin=dict(t=70, b=50, l=40, r=20),
    )

    for i in range(1, 5):
        axis_name = f"yaxis{i if i > 1 else ''}"
        fig["layout"][axis_name].update(
            range=[0, 1.1],
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"],
            title="F1 Macro" if i in [1, 3] else None,
        )

    if save and save_dir is not None:
        save_plotly_figure(
            fig,
            Path(save_dir),
            "combined_violinplots",
        )

    return fig
# ============================================================
# CSV loading
# ============================================================

def _load_key_value_csv(path: Path) -> dict[str, Any]:
    """
    Read CSV assumed to contain at least two columns: key, value.
    Handles the common artifact first row ['', '0'].
    """
    if not path.exists():
        return {}

    df = pd.read_csv(path, header=None).dropna(how="all")
    if df.shape[1] < 2:
        return {}

    if str(df.iloc[0, 0]).strip() == "" and str(df.iloc[0, 1]).strip() in {"0", "0.0"}:
        df = df.iloc[1:].copy()

    result: dict[str, Any] = {}
    for _, row in df.iterrows():
        key = str(row.iloc[0]).strip()
        value = row.iloc[1]
        result[key] = value

    return result


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    """Load metrics.csv into a dictionary."""
    metrics = _load_key_value_csv(metrics_path)

    for k, v in list(metrics.items()):
        try:
            metrics[k] = float(v)
        except Exception:
            pass

    return metrics


def load_meta(meta_path: Path) -> dict[str, Any]:
    """Load meta_data_exp.csv into a dictionary."""
    meta = _load_key_value_csv(meta_path)

    if "Metrics" in meta and isinstance(meta["Metrics"], str):
        try:
            meta["Metrics"] = ast.literal_eval(meta["Metrics"])
        except Exception:
            pass

    return meta



def extract_prompt_features(df: pd.DataFrame, col: str = "hyperparameter") -> pd.DataFrame:
    df = df.copy()
    s = df[col].astype("string")

    df["Few-/Zero-Shot"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["Reasoning"] = pd.Series(pd.NA, index=df.index, dtype="string")

    df.loc[s.str.contains("few_shot", case=False, na=False), "Few-/Zero-Shot"] = "few_shot"
    df.loc[s.str.contains("zero_shot", case=False, na=False), "Few-/Zero-Shot"] = "zero_shot"

    df.loc[s.str.contains("NoReasoning", case=False, na=False), "Reasoning"] = "NoReasoning"
    df.loc[
        s.str.contains("Reasoning", case=False, na=False)
        & ~s.str.contains("NoReasoning", case=False, na=False),
        "Reasoning"
    ] = "Reasoning"

    return df
    
def add_avg_time_per_datapoint(df, col="elapsed_minutes", n_points=375):
    df = df.copy()

    df["avg_minutes_per_datapoint"] = df[col] / n_points
    df["avg_seconds_per_datapoint"] = df["avg_minutes_per_datapoint"] * 60

    return df
# ============================================================
# Run collection
# ============================================================

def collect_runs(results_root: Path) -> list[dict[str, Any]]:
    """
    Collect run folders under results_root.

    Expected files in each run folder:
      - metrics.csv
      - experiment_results.csv
      - meta_data_exp.csv
    """
    runs: list[dict[str, Any]] = []

    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("."):
            continue

        metrics_path = run_dir / "metrics.csv"
        exp_path = run_dir / "experiment_results.csv"
        meta_path = run_dir / "meta_data_exp.csv"

        metrics = load_metrics(metrics_path) if metrics_path.exists() else {}
        exp_df = safe_read_csv(exp_path)
        meta = load_meta(meta_path) if meta_path.exists() else {}

        run_name = run_dir.name
        hyperparameter = meta.get("Hyperparameter", run_name)

        runs.append(
            {
                "run_dir": run_dir,
                "run_name": run_name,
                "hyperparameter": hyperparameter,
                "metrics": metrics,
                "meta": meta,
                "experiment_df": exp_df,
            }
        )

    return runs


def make_summary_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Create summary table from collected runs."""
    rows = []

    for run in runs:
        m = run.get("metrics", {}) or {}
        rows.append(
            {
                "run_name": run["run_name"],
                "hyperparameter": run["hyperparameter"],
                "completion_rate": m.get("completion_rate_class_percent", np.nan),
                "accuracy": m.get("accuracy_class", np.nan),
                "balanced_accuracy": m.get("balanced_accuracy_class", np.nan),
                "precision_macro": m.get("precision_macro_class", np.nan),
                "recall_macro": m.get("recall_macro_class", np.nan),
                "f1_macro": m.get("f1_macro_class", np.nan),
                "elapsed_minutes": m.get("elapsed_minutes", np.nan),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Labeling helpers
# ============================================================

def format_run_label(run_name: str) -> str:
    """Create a shorter multi-line plot label from a raw run name."""
    s = str(run_name)
    low = s.lower()
    parts = [p for p in s.split("_") if p]

    model = parts[0] if parts else s

    if "autogluon" in model.lower():
        return "Autogluon"

    cloud_local = "Cloud based" if "gpt" in model.lower() else "Local LLM"

    shot = None
    if ("few" in low and "shot" in low) or "fewshot" in low:
        shot = "Few-shot"
    elif ("zero" in low and "shot" in low) or "zeroshot" in low or "0shot" in low:
        shot = "Zero-shot"

    reasoning = None
    if ("no" in low and "reason" in low) or "noreason" in low or "no_reason" in low:
        reasoning = "No Reasoning"
    elif "reason" in low:
        reasoning = "Reasoning"

    line1 = f"{cloud_local} | {model}"
    line2_parts = [p for p in [shot, reasoning] if p]

    if line2_parts:
        return line1 + "\n" + " | ".join(line2_parts)
    return line1


def bar_colors_from_run_names(run_names: pd.Series) -> list[str]:
    """Autogluon -> red, everything else -> blue."""
    is_autogluon = run_names.astype(str).str.lower().str.contains("autogluon")
    return np.where(is_autogluon, AUTOG_LUON_COLOR, DEFAULT_COLOR).tolist()


def normalize_run_name_suffix(run_name: str) -> str:
    """
    Fix malformed run names where the separator before Grouped/Full is missing.

    Examples
    --------
    ...unique_targetGrouped_g5_rows25
        -> ...unique_target_Grouped_g5_rows25

    ...unique_targetFull_rows50
        -> ...unique_target_Full_rows50
    """
    s = str(run_name)
    s = re.sub(r"(?<!_)Grouped_g(\d+)_rows(\d+)$", r"_Grouped_g\1_rows\2", s)
    s = re.sub(r"(?<!_)Full_rows(\d+)$", r"_Full_rows\1", s)
    return s


# ============================================================
# Aggregation helpers
# ============================================================

def average_f1_by_last_parts(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    n_parts: int = 2,
) -> pd.DataFrame:
    """
    Group runs by the last `n_parts` of run_name and return:
      - group_key
      - mean_f1_score
      - n_runs
    """
    result = (
        df.assign(
            run_name_normalized=df[run_name_col].astype(str).apply(normalize_run_name_suffix),
            group_key=lambda x: (
                x["run_name_normalized"]
                .str.split("_")
                .str[-n_parts:]
                .str.join("_")
            ),
        )
        .groupby("group_key", as_index=False)
        .agg(
            mean_f1_score=(f1_col, "mean"),
            n_runs=(f1_col, "count"),
        )
        .sort_values("mean_f1_score", ascending=False)
        .reset_index(drop=True)
    )

    return result


def extract_after_name(run_name: str) -> Optional[str]:
    """
    Extract the part after 'Target_name_' and before the final suffix.
    Handles malformed names by normalizing first.
    """
    s = normalize_run_name_suffix(run_name)
    pattern = r"Target_name_(.+?)_(?:Grouped_g\d+_rows\d+|Full_rows\d+)$"
    match = re.search(pattern, s)
    return match.group(1) if match else None


def average_f1_by_after_name(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
) -> pd.DataFrame:
    """
    Group runs by extracted 'after_name'.

    Returns:
      - after_name
      - mean_f1_score
      - n_runs
    """
    result = (
        df.assign(
            run_name_normalized=df[run_name_col].astype(str).apply(normalize_run_name_suffix),
            after_name=lambda x: x["run_name_normalized"].apply(extract_after_name),
        )
        .dropna(subset=["after_name"])
        .groupby("after_name", as_index=False)
        .agg(
            mean_f1_score=(f1_col, "mean"),
            n_runs=(f1_col, "count"),
        )
        .sort_values("mean_f1_score", ascending=False)
        .reset_index(drop=True)
    )

    return result


# ============================================================
# Plot functions
# ============================================================

def plot_metric_bars(
    summary_df: pd.DataFrame,
    metric_cols: Optional[list[str]] = None,
    sort_by: str = "f1_macro",
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Plot one or more metrics by run.
    """
    if metric_cols is None:
        metric_cols = ["f1_macro"]

    df = summary_df.copy()

    required = {"run_name", *metric_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary_df is missing required columns: {sorted(missing)}")

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    df["run_label"] = df["run_name"].apply(format_run_label)
    bar_colors = bar_colors_from_run_names(df["run_name"])

    fig = go.Figure()

    for metric in metric_cols:
        y = pd.to_numeric(df[metric], errors="coerce").to_numpy()

        fig.add_trace(
            go.Bar(
                name=metric,
                x=df["run_label"],
                y=y,
                marker=dict(color=bar_colors),
                text=[f"{v:.2f}" if np.isfinite(v) else "" for v in y],
                textposition="outside",
                cliponaxis=False,
            )
        )

    fig.update_layout(
        title="F1 Macro by combination" if metric_cols == ["f1_macro"] else "Metrics by combination",
        barmode="group",
        xaxis_title=None,
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        margin=dict(t=60, b=140),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        showlegend=False if metric_cols == ["f1_macro"] else True,
    )

    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), "f1_macro_by_combination" if metric_cols == ["f1_macro"] else "metrics_by_combination")

    return fig


def plot_latency_distribution(
    runs: list[dict[str, Any]],
    *,
    save_dir: Optional[str] = None,
    save: bool = False,
) -> Optional[go.Figure]:
    """Boxplot of latency_s by run."""
    rows = []

    for run in runs:
        df = run.get("experiment_df")
        if df is None or "latency_s" not in df.columns:
            continue

        tmp = df[["latency_s"]].copy()
        tmp["run_name"] = run["run_name"]
        rows.append(tmp)

    if not rows:
        return None

    all_lat = pd.concat(rows, ignore_index=True)

    fig = px.box(
        all_lat,
        x="run_name",
        y="latency_s",
        title="Latency distribution (s) by run",
    )
    fig.update_layout(xaxis_title="Run", yaxis_title="Latency (s)")

    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), "latency_distribution_by_run")

    return fig


def add_confusion_matrix_grid_paper(fig: go.Figure, n: int, line_width: int = 2) -> None:
    """Draw a clean n x n grid in paper coordinates."""
    if n <= 0:
        return

    shapes = []

    for i in range(n + 1):
        x = i / n
        shapes.append(
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=x,
                x1=x,
                y0=0,
                y1=1,
                line=dict(color="black", width=line_width),
                layer="above",
            )
        )

    for i in range(n + 1):
        y = i / n
        shapes.append(
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=0,
                x1=1,
                y0=y,
                y1=y,
                line=dict(color="black", width=line_width),
                layer="above",
            )
        )

    fig.update_layout(shapes=shapes)


def confusion_matrix_figure_for_run(
    run: dict[str, Any],
    *,
    save_dir: Optional[str] = None,
    save: bool = False,
) -> tuple[Optional[go.Figure], Optional[str]]:
    """
    Create confusion matrix figure for a single run.

    Returns:
      (fig, None) on success
      (None, error_message) on failure
    """
    df = run.get("experiment_df")
    run_name = str(run.get("run_name", "unknown_run"))

    if df is None:
        return None, f"{run_name}: no experiment_results.csv"

    required = {"gt_class", "pred_class"}
    if not required.issubset(df.columns):
        return None, f"{run_name}: missing columns: {required}"

    valid = df["pred_class"].notna()
    if valid.sum() == 0:
        return None, f"{run_name}: no valid predictions (pred_class all NaN)"

    y_true = df.loc[valid, "gt_class"].astype(str)
    y_pred = df.loc[valid, "pred_class"].astype(str)
    labels = sorted(set(y_true) | set(y_pred))

    cm = pd.crosstab(y_true, y_pred, rownames=["True"], colnames=["Pred"], dropna=False)
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)

    z = cm.to_numpy()
    text = z.astype(str)

    is_autogluon = "autogluon" in run_name.lower()
    colorscale = AUTOG_LUON_SCALE if is_autogluon else DEFAULT_SCALE

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=cm.columns.tolist(),
            y=cm.index.tolist(),
            colorscale=colorscale,
            showscale=True,
            text=text,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )

    add_confusion_matrix_grid_paper(fig, n=len(labels), line_width=2)

    run_label = format_run_label(run_name)
    fig.update_layout(
        title=f"Confusion Matrix — {run_label}",
        xaxis_title="Pred",
        yaxis_title="Ground Truth",
        margin=dict(t=80, b=80),
    )
    fig.update_xaxes(side="bottom")

    if save and save_dir is not None:
        safe_run = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in run_name)
        save_plotly_figure(fig, Path(save_dir), f"confusion_matrix__{safe_run}")

    return fig, None


def plot_f1_per_after_name_group(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> dict[str, go.Figure]:
    """
    Create one F1 bar plot per extracted 'after_name' group.
    Returns a dict: {group_name: figure}
    """
    plot_df = df.copy()
    plot_df["run_name_normalized"] = plot_df[run_name_col].astype(str).apply(normalize_run_name_suffix)
    plot_df["after_name"] = plot_df["run_name_normalized"].apply(extract_after_name)
    plot_df = plot_df.dropna(subset=["after_name"])

    figures: dict[str, go.Figure] = {}

    for group_name, group_df in plot_df.groupby("after_name"):
        group_df = group_df.sort_values(f1_col, ascending=False).copy()
        group_df["run_label"] = group_df[run_name_col].apply(format_run_label)

        colors = bar_colors_from_run_names(group_df[run_name_col])

        fig = go.Figure(
            go.Bar(
                x=group_df["run_label"],
                y=group_df[f1_col],
                marker=dict(color=colors),
                text=[f"{v:.2f}" for v in group_df[f1_col]],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=f"F1 Macro — {group_name}",
            xaxis_title=None,
            yaxis_title="F1 Macro",
            yaxis=dict(range=[0, 1]),
            margin=dict(t=60, b=140),
            showlegend=False,
        )

        figures[group_name] = fig

        if save and save_dir is not None:
            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in group_name)
            save_plotly_figure(fig, Path(save_dir), f"f1_group__{safe_name}")

    return figures


def plot_avg_f1(
    avg_df: pd.DataFrame,
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Plot mean F1 by dataset configuration.
    Expects:
      - after_name
      - mean_f1_score
    """
    required = {"after_name", "mean_f1_score"}
    missing = required - set(avg_df.columns)
    if missing:
        raise ValueError(f"avg_df is missing required columns: {sorted(missing)}")

    df = avg_df.copy().sort_values("mean_f1_score", ascending=False)

    fig = go.Figure(
        go.Bar(
            x=df["after_name"],
            y=df["mean_f1_score"],
            text=[f"{v:.2f}" for v in df["mean_f1_score"]],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Mean F1 by Dataset Configuration",
        xaxis_title=None,
        yaxis_title="Mean F1",
        yaxis=dict(range=[0, 1]),
        margin=dict(t=60, b=140),
        showlegend=False,
    )
    fig.update_xaxes(tickangle=30)

    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), "avg_f1_by_dataset")

    return fig

def extract_model_name_from_run_name(run_name: str) -> str:
    """
    Extract model name from run_name.
    The model name is the part before the first underscore.

    Example
    -------
    Qwen2.5-14B-Instruct_Train_few_shot_...
    -> Qwen2.5-14B-Instruct
    """
    s = str(run_name).strip()
    return s.split("_", 1)[0] if "_" in s else s

def plot_f1_box_by_model_name(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Box plot of F1 distribution grouped by model name.
    The model name is extracted from run_name before the first underscore.
    """
    plot_df = df.copy()

    required = {run_name_col, f1_col}
    missing = required - set(plot_df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    plot_df[run_name_col] = plot_df[run_name_col].astype(str)
    plot_df[f1_col] = pd.to_numeric(plot_df[f1_col], errors="coerce")

    plot_df["model_name"] = plot_df[run_name_col].apply(extract_model_name_from_run_name)

    plot_df = plot_df.dropna(subset=[f1_col, "model_name"])

    order = (
        plot_df.groupby("model_name")[f1_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig = px.box(
        plot_df,
        x="model_name",
        y=f1_col,
        points="all",
        title="F1 Distribution by Model",
        category_orders={"model_name": order},
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="F1 Macro",
        xaxis=dict(
            categoryorder="array",
            categoryarray=order,
        ),
        yaxis=dict(
            range=[0, 1],
            dtick=0.2,
        ),
        margin=dict(t=60, b=140),
    )

    fig.update_xaxes(tickangle=30)
    fig.update_traces(
    fillcolor="white",
    marker=dict(color="black"),
    line=dict(color="black")
)

    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), "boxplot_by_model_name")

    return fig

def plot_f1_box_by_reasoning(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Box plot of F1 distribution grouped by reasoning vs no reasoning.
    """
    plot_df = df.copy()

    required = {run_name_col, f1_col}
    missing = required - set(plot_df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    plot_df[run_name_col] = plot_df[run_name_col].astype(str)
    plot_df[f1_col] = pd.to_numeric(plot_df[f1_col], errors="coerce")

    # extract reasoning
    extracted = plot_df[run_name_col].apply(extract_model_shot_reasoning)
    plot_df[["model_tmp", "shot_tmp", "reasoning"]] = pd.DataFrame(
        extracted.tolist(), index=plot_df.index
    )

    plot_df = plot_df.dropna(subset=[f1_col, "reasoning"])

    # enforce order
    order = ["Reasoning", "NoReasoning"]

    plot_df["reasoning"] = pd.Categorical(
        plot_df["reasoning"],
        categories=order,
        ordered=True
    )

    fig = px.box(
        plot_df.sort_values("reasoning"),
        x="reasoning",
        y=f1_col,
        points="all",
        title="F1 Distribution by Reasoning",
        category_orders={"reasoning": order},
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="F1 Macro",
        yaxis=dict(
            range=[0, 1],
            dtick=0.2,
        ),
        margin=dict(t=60, b=140),
    )

    fig.update_traces(
        fillcolor="white",
        marker=dict(color="black"),
        line=dict(color="black")
    )

    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), "boxplot_by_reasoning")

    return fig
    
def plot_f1_box_by_shot(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Box plot of F1 distribution grouped by few-shot vs zero-shot.
    """
    plot_df = df.copy()

    required = {run_name_col, f1_col}
    missing = required - set(plot_df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    plot_df[run_name_col] = plot_df[run_name_col].astype(str)
    plot_df[f1_col] = pd.to_numeric(plot_df[f1_col], errors="coerce")

    # extract shot
    extracted = plot_df[run_name_col].apply(extract_model_shot_reasoning)
    plot_df[["model_tmp", "shot", "reasoning_tmp"]] = pd.DataFrame(
        extracted.tolist(), index=plot_df.index
    )

    plot_df = plot_df.dropna(subset=[f1_col, "shot"])

    # enforce order
    order = ["few_shot", "zero_shot"]

    plot_df["shot"] = pd.Categorical(
        plot_df["shot"],
        categories=order,
        ordered=True
    )

    # nicer labels for plot
    label_map = {
        "few_shot": "Few-shot",
        "zero_shot": "Zero-shot",
    }
    plot_df["shot_label"] = plot_df["shot"].map(label_map)

    fig = px.box(
        plot_df.sort_values("shot"),
        x="shot_label",
        y=f1_col,
        points="all",
        title="F1 Distribution by Shot Setting",
        category_orders={"shot_label": ["Few-shot", "Zero-shot"]},
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="F1 Macro",
        yaxis=dict(
            range=[0, 1],
            dtick=0.2,
        ),
        margin=dict(t=60, b=140),
    )

    fig.update_traces(
        fillcolor="white",
        marker=dict(color="black"),
        line=dict(color="black")
    )

    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), "boxplot_by_shot")

    return fig
    
def plot_f1_box_by_after_name(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Box plot of F1 distribution grouped by extracted after_name.
    """
    plot_df = df.copy()

    plot_df["run_name_normalized"] = plot_df[run_name_col].astype(str).apply(normalize_run_name_suffix)
    plot_df["after_name"] = plot_df["run_name_normalized"].apply(extract_after_name)

    # ---------------------------
    # Apply renaming
    # ---------------------------
    rename_map = {
        "dataset_description+unique_target": "dataset description + statistical target information",
        "dataset_description": "dataset description",
        "unique_target": "statistical target information",
        "BasePrompt": "base prompt",
    }

    plot_df["after_name"] = plot_df["after_name"].map(rename_map)

    plot_df = plot_df.dropna(subset=[f1_col, "after_name"])

    # ---------------------------
    # Enforce order (important for paper)
    # ---------------------------
    order = [
        "dataset description + statistical target information",
        "dataset description",
        "statistical target information",
        "base prompt",
    ]

    plot_df["after_name"] = pd.Categorical(
        plot_df["after_name"],
        categories=order,
        ordered=True
    )
    label_map_wrapped = {k: wrap_label(k) for k in order}
    plot_df["after_name_wrapped"] = plot_df["after_name"].map(label_map_wrapped)

    fig = px.box(
        plot_df.sort_values("after_name"),
        x="after_name_wrapped",
        y=f1_col,
        points="all",
        title="F1 Distribution by Dataset Configuration",
        category_orders={
            "after_name_wrapped": [label_map_wrapped[k] for k in order]
        },
    )

    fig = px.box(
        plot_df.sort_values("after_name"),
        x="after_name",
        y=f1_col,
        points="all",
        title="F1 Distribution by Dataset Configuration",
        category_orders={"after_name": order},
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="F1 Macro",
        yaxis=dict(
        range=[0, 1],
        dtick=0.2  # <-- this sets 0.0, 0.2, 0.4, ..., 1.0
    ),
        margin=dict(t=60, b=140),
    )

    fig.update_xaxes(tickangle=30)
    fig.update_traces(
    fillcolor="white",
    marker=dict(color="black"),
    line=dict(color="black")
)

    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), "boxplot_by_after_name")

    return fig


def wrap_label(text: str, max_len: int = 20) -> str:
    """
    Split long labels into two lines using <br> for Plotly.
    """
    text = str(text)
    if len(text) <= max_len:
        return text

    # try to split at a natural separator
    for sep in [" + ", "+", "_", " "]:
        parts = text.split(sep)
        if len(parts) > 1:
            mid = len(parts) // 2
            return sep.join(parts[:mid]) + "<br>" + sep.join(parts[mid:])

    # fallback: hard split
    return text[:max_len] + "<br>" + text[max_len:]


def add_mean_annotation(fig, x, y, row, col):
    fig.add_annotation(
        x=x,
        y=y,
        text=f"{y:.2f}",
        showarrow=False,
        yshift=8,
        font=dict(size=10, color="black"),
        xref=f"x{'' if col == 1 and row == 1 else (row-1)*2+col}",
        yref=f"y{'' if col == 1 and row == 1 else (row-1)*2+col}",
    )
def plot_f1_box_combined(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> go.Figure:

    plot_df = df.copy()
    plot_df[run_name_col] = plot_df[run_name_col].astype(str)
    plot_df[f1_col] = pd.to_numeric(plot_df[f1_col], errors="coerce")

    # ---------------------------
    # Extract features
    # ---------------------------
    plot_df["run_name_normalized"] = plot_df[run_name_col].apply(normalize_run_name_suffix)
    plot_df["after_name"] = plot_df["run_name_normalized"].apply(extract_after_name)

    extracted = plot_df[run_name_col].apply(extract_model_shot_reasoning)
    plot_df[["model", "shot", "reasoning"]] = pd.DataFrame(extracted.tolist(), index=plot_df.index)

    plot_df["model_name"] = plot_df[run_name_col].apply(extract_model_name_from_run_name)

    plot_df = plot_df.dropna(subset=[f1_col])

    # ---------------------------
    # Rename dataset configs
    # ---------------------------
    rename_map = {
        "dataset_description+unique_target": "dataset description + statistical target information",
        "dataset_description": "dataset description",
        "unique_target": "statistical target information",
        "BasePrompt": "base prompt",
    }
    plot_df["after_name"] = plot_df["after_name"].map(rename_map)

    short_map = {
        "dataset description + statistical target information": "DD + ST",
        "dataset description": "DD",
        "statistical target information": "ST",
        "base prompt": "BP",
    }

    # ---------------------------
    # Orders
    # ---------------------------
    order_after = [
        "dataset description + statistical target information",
        "dataset description",
        "statistical target information",
        "base prompt",
    ]

    order_reasoning = ["Reasoning", "NoReasoning"]
    order_shot = ["few_shot", "zero_shot"]

    order_model = (
        plot_df.groupby("model_name")[f1_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # ---------------------------
    # Subplots (LESS SPACING)
    # ---------------------------
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Dataset Configuration",
            "Model",
            "Reasoning",
            "Shot",
        ),
        horizontal_spacing=0.03,   # ↓ default ~0.2
        vertical_spacing=0.1  # ↓ default ~0.3
    )

    # Dataset config
    for cat in order_after:
        sub = plot_df[plot_df["after_name"] == cat]
        if sub.empty:
            continue

        fig.add_trace(
            go.Box(
                y=sub[f1_col],
                name=short_map.get(cat, cat),
                marker_color="black",
                line=dict(color="black"),
                fillcolor="white",
                showlegend=False,
            ),
            row=1, col=1
        )

    # Model
    for cat in order_model:
        sub = plot_df[plot_df["model_name"] == cat]
        if sub.empty:
            continue

        fig.add_trace(
            go.Box(
                y=sub[f1_col],
                name=cat,
                marker_color="black",
                line=dict(color="black"),
                fillcolor="white",
                showlegend=False,
            ),
            row=1, col=2
        )

    # Reasoning
    for cat in order_reasoning:
        sub = plot_df[plot_df["reasoning"] == cat]
        if sub.empty:
            continue

        fig.add_trace(
            go.Box(
                y=sub[f1_col],
                name=cat,
                marker_color="black",
                line=dict(color="black"),
                fillcolor="white",
                showlegend=False,
            ),
            row=2, col=1
        )

    # Shot
    shot_map = {"few_shot": "Few-shot", "zero_shot": "Zero-shot"}
    for cat in order_shot:
        sub = plot_df[plot_df["shot"] == cat]
        if sub.empty:
            continue

        fig.add_trace(
            go.Box(
                y=sub[f1_col],
                name=shot_map.get(cat, cat),
                marker_color="black",
                line=dict(color="black"),
                fillcolor="white",
                showlegend=False,
            ),
            row=2, col=2
        )

    # ---------------------------
    # Layout (TIGHTER)
    # ---------------------------
    fig.update_layout(
        height=700,  # slightly smaller
        width=1000,
        title="F1 Distribution of every hyperparameter over all Experiment runs on Validation for Exp.2",
        margin=dict(t=70, b=50, l=40, r=20),  # reduced margins
    )

    # consistent y-axis
    for i in range(1, 5):
        fig["layout"][f"yaxis{i if i > 1 else ''}"].update(range=[0, 1])
        # ---------------------------
    # Save figure (FIX)
    # ---------------------------
    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), "combined_boxplots")

    return fig
    
import re
from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_f1_box_by_last_parts(
    df: pd.DataFrame,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    *,
    save_dir: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Box plot of F1 grouped by sampling configuration only.
    Grouped runs first, then Full runs.
    """
    plot_df = df.copy()

    plot_df[run_name_col] = plot_df[run_name_col].astype(str)
    plot_df[f1_col] = pd.to_numeric(plot_df[f1_col], errors="coerce")

    # extract only gX_rowsY or Full_rowsY
    plot_df["group_key"] = plot_df[run_name_col].str.extract(
        r"(g\d+_rows\d+|Full_rows\d+)",
        expand=False
    )

    plot_df = plot_df.dropna(subset=[f1_col, "group_key"])

    def sort_key(x: str):
        x = str(x).strip()

        m_group = re.fullmatch(r"g(\d+)_rows(\d+)", x)
        if m_group:
            g = int(m_group.group(1))
            rows = int(m_group.group(2))
            return (0, g, rows)   # grouped first

        m_full = re.fullmatch(r"Full_rows(\d+)", x)
        if m_full:
            rows = int(m_full.group(1))
            return (1, 999, rows)  # full at the end

        return (2, 999, 999)

    order = sorted(plot_df["group_key"].unique(), key=sort_key)

    fig = px.box(
        plot_df,
        x="group_key",
        y=f1_col,
        points="all",
        title="F1 Distribution by Row",
        category_orders={"group_key": order},
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="F1 Macro",
        xaxis=dict(
            categoryorder="array",
            categoryarray=order,
        ),
        yaxis=dict(
            range=[0, 1],
            dtick=0.2,
        ),
        margin=dict(t=60, b=140),
    )

    fig.update_xaxes(tickangle=30)

    print("Final order:", order)

    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), "boxplot_last_2_parts")

    return fig
import re
from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_average_f1_by_last_parts(
    summary_df: pd.DataFrame,
    *,
    n_parts: int = 2,
    run_name_col: str = "run_name",
    f1_col: str = "f1_macro",
    save_dir: Optional[str] = None,
    save: bool = False,
) -> go.Figure:
    """Convenience plot for average F1 grouped by the last parts of run_name."""
    result = average_f1_by_last_parts(
        summary_df,
        run_name_col=run_name_col,
        f1_col=f1_col,
        n_parts=n_parts,
    ).copy()

    # normalize to plain string
    result["group_key"] = result["group_key"].astype(str).str.strip()

    def parse_group_key(x: str):
        x = str(x).strip()

        m_group = re.fullmatch(r"g(\d+)_rows(\d+)", x)
        if m_group:
            g = int(m_group.group(1))
            rows = int(m_group.group(2))
            return ("grouped", g, rows)

        m_full = re.fullmatch(r"Full_rows(\d+)", x)
        if m_full:
            rows = int(m_full.group(1))
            return ("full", 999, rows)

        return ("other", 999, 999999)

    grouped = []
    full = []
    other = []

    for x in result["group_key"].unique():
        kind, g, rows = parse_group_key(x)
        if kind == "grouped":
            grouped.append((x, g, rows))
        elif kind == "full":
            full.append((x, rows))
        else:
            other.append(x)

    grouped = [x for x, _, _ in sorted(grouped, key=lambda t: (t[1], t[2]))]
    full = [x for x, _ in sorted(full, key=lambda t: t[1])]
    order = grouped + full + sorted(other)

    print("Detected group_key values:", sorted(result["group_key"].unique()))
    print("Final plotting order:", order)

    fig = px.bar(
        result,
        x="group_key",
        y="mean_f1_score",
        title="Average F1 Score by Group Key",
        text="mean_f1_score",
        category_orders={"group_key": order},
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

    fig.update_layout(
        xaxis_title="Group Key",
        yaxis_title="Mean F1 Score",
        xaxis=dict(
            tickangle=-45,
            categoryorder="array",
            categoryarray=order,
        ),
        yaxis=dict(
            range=[0, 1],
            dtick=0.2,
        ),
    )

    if save and save_dir is not None:
        save_plotly_figure(fig, Path(save_dir), f"avg_f1_last_{n_parts}_parts")

    return fig