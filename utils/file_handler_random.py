from __future__ import annotations

import io
import json
import random
import csv
from typing import Any, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import arff
from scipy.io.arff._arffread import _loadarff as load_arff_from_file
from magika import Magika


# ----------------------------
# Helpers
# ----------------------------
def sample_random_row_groups(
    df: pd.DataFrame,
    requested_rows: int,
    group_size: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    if requested_rows is None:
        return df

    if requested_rows <= 0:
        raise ValueError("requested_rows must be a positive integer")

    if group_size <= 0:
        raise ValueError("group_size must be a positive integer")

    n = len(df)
    if n < group_size:
        return materialize_dataframe(df.copy())

    n_groups_needed = int(np.ceil(requested_rows / group_size))
    n_full_groups = n // group_size
    n_groups = min(n_groups_needed, n_full_groups)

    rng = np.random.default_rng(random_state)

    chosen_group_ids = rng.choice(
        n_full_groups, size=n_groups, replace=False
    )

    row_positions: list[int] = []
    for gid in chosen_group_ids:
        start = gid * group_size
        row_positions.extend(range(start, start + group_size))

    row_positions.sort()

    records = df.to_dict(orient="records")
    sampled_records = [records[i] for i in row_positions]
    return pd.DataFrame.from_records(sampled_records)

def decode_bytes_df(df: pd.DataFrame, encoding: str = "utf-8") -> pd.DataFrame:
    """Decode bytes/bytearray cells into strings."""
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(
            lambda x: x.decode(encoding, errors="replace")
            if isinstance(x, (bytes, bytearray)) else x
        )
    return out


def is_arff_nested(arff_content: str) -> bool:
    lines = arff_content.splitlines()
    for line in lines:
        if line.strip().startswith("@attribute") and line.strip().endswith("relational"):
            return True
        if line.strip().lower() == "@data":
            break
    return False


def count_lines(f) -> int:
    count = sum(1 for _ in f)
    f.seek(0)
    return count


def data_start_idx(f) -> int:
    found_data = False
    for i, line in enumerate(f):
        if found_data and line.strip() != "":
            f.seek(0)
            return i
        if line.strip().lower() == "@data":
            found_data = True
    f.seek(0)
    return -1


# ----------------------------
# ARFF: reduced reading only when max_rows provided
# ----------------------------

def reduce_arff_lines(file_path: str, rows_to_read: int) -> str:
    """
    Keep the full ARFF header and only the first `rows_to_read` data rows.
    """
    if rows_to_read <= 0:
        raise ValueError("rows_to_read must be a positive integer")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        out_lines: List[str] = []
        data_seen = False
        data_rows = 0

        for line in f:
            if not data_seen:
                out_lines.append(line.rstrip("\n"))
                if line.strip().lower() == "@data":
                    data_seen = True
                continue

            if line.strip() == "":
                continue

            out_lines.append(line.rstrip("\n"))
            data_rows += 1
            if data_rows >= rows_to_read:
                break

    return "\n".join(out_lines)


def reduce_nested_rows(data: np.ndarray) -> np.ndarray:
    reduced_data = []
    for features, target in data:
        reduced_features = reduce_nested_features(features)
        reduced_data.append((reduced_features, target))
    return np.array(reduced_data, dtype=data.dtype)


def reduce_nested_features(features: np.ndarray) -> str:
    if len(features) < 2:
        raise ValueError("Row must contain at least 2 features to reduce.")

    first_series = features[0].tolist()
    second_series = features[1].tolist()

    first_series_reduced = f"[{first_series[0]}, ..., {first_series[-1]}]"
    second_series_reduced = f"[{second_series[0]}, ..., {second_series[-1]}]"

    if len(features) > 2:
        last_series = features[-1].tolist()
        last_series_reduced = f"[{last_series[0]}, ..., {last_series[-1]}]"
        return f"[{first_series_reduced}, {second_series_reduced}, ..., {last_series_reduced}]"

    return f"[{first_series_reduced}, {second_series_reduced}]"


def _read_arff_header_until_data(file_path: str) -> str:
    """
    Read only the ARFF header (up to and including @data).
    """
    lines: List[str] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            if line.strip().lower() == "@data":
                break
    return "\n".join(lines)


def load_arff_into_dataframe(file_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    If max_rows is provided: load only the first max_rows data lines.
    Else: load full file.
    """
    if max_rows is not None:
        arff_content = reduce_arff_lines(file_path, rows_to_read=max_rows)
        data, _metadata = load_arff_from_file(io.StringIO(arff_content))

        if is_arff_nested(arff_content):
            data = reduce_nested_rows(data)

        df = pd.DataFrame(data)
        return decode_bytes_df(df)

    header_only = _read_arff_header_until_data(file_path)
    nested = is_arff_nested(header_only)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data, _metadata = load_arff_from_file(f)

        if nested:
            data = reduce_nested_rows(data)

        df = pd.DataFrame(data)

    except Exception:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            arff_data = arff.load(f)
        df = pd.DataFrame(
            arff_data["data"],
            columns=[attr[0] for attr in arff_data["attributes"]],
        )

    return decode_bytes_df(df)


# ----------------------------
# CSV/TXT: robust delimiter/header detection + exact nrows when max_rows provided
# ----------------------------

def get_sample_lines(file_path: str, num_lines: int = 10) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines: List[str] = []
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def _fallback_delimiter(sample: str) -> str:
    """
    Choose a delimiter that yields a consistent column count.
    Includes a WHITESPACE option (split on any whitespace).
    """
    lines = [l for l in sample.splitlines() if l.strip()]
    if not lines:
        return ","

    candidates = [",", ";", "\t", "|", ":", "~", "^", "\x1f", "WHITESPACE"]

    for d in candidates:
        if d == "WHITESPACE":
            splits = [len(l.split()) for l in lines]
        else:
            splits = [len(l.split(d)) for l in lines]

        if len(set(splits)) == 1 and splits[0] > 1:
            return d

    return ","


def get_delimiter_and_header(file_path: str) -> Tuple[str, bool]:
    sample = get_sample_lines(file_path, 10)

    sniffer_delims = [",", ";", "\t", "|", ":", "~", "^", "\x1f"]
    sniffer = csv.Sniffer()

    try:
        dialect = sniffer.sniff(sample, delimiters="".join(sniffer_delims))
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = _fallback_delimiter(sample)

    try:
        has_header = sniffer.has_header(sample)
    except csv.Error:
        has_header = True

    if delimiter == "WHITESPACE":
        delimiter = r"\s+"

    return delimiter, has_header


def load_csv_into_dataframe(
    file_path: str,
    has_header: bool | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if has_header is None:
        delimiter, has_header = get_delimiter_and_header(file_path)
    else:
        delimiter, _auto_header = get_delimiter_and_header(file_path)

    header_arg = 0 if has_header else None

    if delimiter == r"\s+":
        return pd.read_csv(
            file_path,
            sep=delimiter,
            engine="python",
            header=header_arg,
            nrows=max_rows,
        )

    return pd.read_csv(
        file_path,
        sep=delimiter,
        header=header_arg,
        nrows=max_rows,
    )


def load_txt_as_text_dataframe(file_path: str, max_lines: int = 200) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines: List[str] = []
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            lines.append(line.rstrip("\n"))
    return pd.DataFrame({"text": lines})


def load_text_like_into_dataframe(
    file_path: str,
    has_header: bool | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Attempt to parse text-like file (.txt, .dat, .data) as a delimited table.
    Fallback to line-based text DataFrame if not tabular.
    """
    try:
        return load_csv_into_dataframe(file_path, has_header=has_header, max_rows=max_rows)
    except Exception:
        return load_txt_as_text_dataframe(file_path, max_lines=max_rows or 200)


# ----------------------------
# JSON
# ----------------------------

def load_json_into_dataframe(file_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    Try common JSON shapes:
    - JSON Lines
    - list[dict]
    - dict[str, list/scalar]
    - single dict
    """
    try:
        df = pd.read_json(file_path, lines=True)
        if not df.empty:
            return df if max_rows is None else df.iloc[:max_rows].copy()
    except Exception:
        pass

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        if len(obj) == 0:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in obj):
            df = pd.DataFrame(obj)
        else:
            df = pd.DataFrame({"value": obj})

    elif isinstance(obj, dict):
        try:
            df = pd.DataFrame(obj)
        except Exception:
            df = pd.json_normalize(obj)

    else:
        df = pd.DataFrame({"value": [obj]})

    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    return df


# ----------------------------
# Arrow / Parquet
# Prefer Hugging Face datasets when available
# ----------------------------
def materialize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force a DataFrame onto plain pandas/object-backed columns to avoid
    Arrow / extension dtype issues during iloc/take/copy.
    """
    records = df.to_dict(orient="records")
    return pd.DataFrame.from_records(records)
def _hf_dataset_to_dataframe(ds_obj, max_rows: int | None = None) -> pd.DataFrame:
    """
    Convert a Hugging Face Dataset or DatasetDict to a plain pandas DataFrame.
    """
    try:
        from datasets import DatasetDict  # type: ignore
    except Exception:
        DatasetDict = None  # type: ignore

    if DatasetDict is not None and isinstance(ds_obj, DatasetDict):
        if len(ds_obj) == 0:
            return pd.DataFrame()
        first_split = next(iter(ds_obj.keys()))
        ds_obj = ds_obj[first_split]

    if max_rows is not None:
        n = min(max_rows, len(ds_obj))
        if n <= 0:
            raise ValueError("max_rows must be a positive integer")
        ds_obj = ds_obj.select(range(n))

    df = ds_obj.to_pandas()

    # Force plain materialized dataframe
    return materialize_dataframe(df)

def load_arrow_into_dataframe(file_path: str, max_rows: int | None = None) -> pd.DataFrame:
    last_error: Exception | None = None

    try:
        from datasets import Dataset  # type: ignore

        ds = Dataset.from_file(str(file_path))
        return _hf_dataset_to_dataframe(ds, max_rows=max_rows)
    except Exception as e:
        last_error = e

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.ipc as ipc  # type: ignore

        with pa.memory_map(str(file_path), "r") as source:
            try:
                reader = ipc.open_file(source)
                table = reader.read_all()
            except Exception:
                source.seek(0)
                reader = ipc.open_stream(source)
                table = reader.read_all()

        if max_rows is not None:
            table = table.slice(0, max_rows)

        df = table.to_pandas()
        return materialize_dataframe(df)

    except Exception as e:
        raise ValueError(
            f"Could not load Arrow file '{file_path}'. "
            f"Hugging Face datasets failed with: {last_error}. "
            f"PyArrow fallback failed with: {e}"
        ) from e


def load_parquet_into_dataframe(file_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    Try Hugging Face datasets first for .parquet files.
    Fallback to pandas.read_parquet.
    """
    last_error: Exception | None = None

    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("parquet", data_files=str(file_path), split="train")
        return _hf_dataset_to_dataframe(ds, max_rows=max_rows)
    except Exception as e:
        last_error = e

    try:
        df = pd.read_parquet(file_path)
        if max_rows is not None:
            df = df.iloc[:max_rows].copy()
        return materialize_dataframe(df)
    except Exception as e:
        raise ValueError(
            f"Could not load Parquet file '{file_path}'. "
            f"Hugging Face datasets failed with: {last_error}. "
            f"Pandas fallback failed with: {e}"
        ) from e


# ----------------------------
# Top-level dataset loader
# ----------------------------

def load_dataset_into_dataframe(
    file_path: str | Path,
    has_header: bool | None = None,
    max_rows: int | None = None,
    target: str | None = None,
    max_columns: int = 50,
    random_group_sample: bool = True,
    group_size: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # For random group sampling, full load first, sample later
    effective_max_rows_for_read = None if (random_group_sample and max_rows is not None) else max_rows

    if suffix == ".arff":
        df = load_arff_into_dataframe(str(file_path), max_rows=effective_max_rows_for_read)

    elif suffix == ".xlsx":
        df = pd.read_excel(file_path)
        if effective_max_rows_for_read is not None:
            if effective_max_rows_for_read <= 0:
                raise ValueError("max_rows must be a positive integer")
            df = df.iloc[:effective_max_rows_for_read].copy()

    elif suffix == ".csv":
        df = load_csv_into_dataframe(
            str(file_path),
            has_header=has_header,
            max_rows=effective_max_rows_for_read,
        )

    elif suffix in {".txt", ".dat", ".data"}:
        df = load_text_like_into_dataframe(
            str(file_path),
            has_header=has_header,
            max_rows=effective_max_rows_for_read,
        )

    elif suffix == ".json":
        df = load_json_into_dataframe(
            str(file_path),
            max_rows=effective_max_rows_for_read,
        )

    elif suffix == ".arrow":
        df = load_arrow_into_dataframe(
            str(file_path),
            max_rows=effective_max_rows_for_read,
        )

    elif suffix == ".parquet":
        df = load_parquet_into_dataframe(
            str(file_path),
            max_rows=effective_max_rows_for_read,
        )

    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    if random_group_sample and max_rows is not None:
        df = sample_random_row_groups(
            df,
            requested_rows=max_rows,
            group_size=group_size,
            random_state=random_state,
        )

    if df.shape[1] > max_columns:
        keep_positions = list(range(max_columns))

        if target is not None:
            target_positions = [i for i, c in enumerate(df.columns) if str(c) == target]
            for pos in target_positions:
                if pos not in keep_positions:
                    keep_positions.append(pos)

        df = df.iloc[:, keep_positions].copy()

    return df


# ----------------------------
# DataFrame serialization helpers
# ----------------------------

def serialize_dataframe(df: pd.DataFrame, max_rows: int = 50) -> str:
    """
    Serialize a pandas DataFrame into executable Python code that recreates it via pd.DataFrame({...}).
    Only the first `max_rows` rows are included.
    Handles duplicate column names by iterating by index.
    """
    df = df.iloc[:max_rows]

    lines = []
    for i in range(df.shape[1]):
        col_name = df.columns[i]
        col_data = df.iloc[:, i].tolist()
        lines.append(f"    {repr(col_name)}: {repr(col_data)},")

    formatted_dict = "{\n" + "\n".join(lines) + "\n}"
    return f"pd.DataFrame({formatted_dict})"


def _to_llm_safe_value(v: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python / string forms."""
    if v is None:
        return None

    if pd.isna(v):
        return None

    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()

    try:
        if isinstance(v, (np.generic,)):
            return v.item()
    except Exception:
        pass

    #if isinstance(v, (bool, int, float, str, list, dict)):
    #    return v

    return str(v)


def _df_preview(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    preview = df.head(max_rows).copy()
    preview = preview.where(preview.notna(), None)
    return preview


def _df_to_markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    preview = _df_preview(df, max_rows)
    if preview.empty:
        return "_(empty table)_"

    cols = [str(c) for c in preview.columns]

    lines = []
    header = "| " + " | ".join(cols) + " |"
    separator = "|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(separator)

    for _, row in preview.iterrows():
        cells = []
        for v in row:
            v = _to_llm_safe_value(v)
            cell = "" if v is None else str(v)
            cell = cell.replace("|", "\\|")
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    preview = _df_preview(df, max_rows)
    if preview.empty:
        return "<p>(empty table)</p>"

    cols = [str(c) for c in preview.columns]

    html_parts = []
    html_parts.append("<table>")
    html_parts.append("  <thead>")
    html_parts.append("    <tr>")
    for c in cols:
        html_parts.append(f"      <th>{c}</th>")
    html_parts.append("    </tr>")
    html_parts.append("  </thead>")

    html_parts.append("  <tbody>")
    for _, row in preview.iterrows():
        html_parts.append("    <tr>")
        for v in row:
            v = _to_llm_safe_value(v)
            cell = "" if v is None else str(v)
            html_parts.append(f"      <td>{cell}</td>")
        html_parts.append("    </tr>")
    html_parts.append("  </tbody>")
    html_parts.append("</table>")

    return "\n".join(html_parts)


def serialize_df_for_llm(
    df: pd.DataFrame,
    max_rows: int = 50,
    include_markdown: bool = True,
    include_html: bool = True,
) -> Dict[str, Any]:
    n_rows, n_cols = df.shape

    schema: List[Dict[str, Any]] = []
    for col in df.columns:
        col_series = df[col]
        non_null = int(col_series.notna().sum())
        example = _to_llm_safe_value(col_series.dropna().iloc[0]) if non_null > 0 else None
        schema.append(
            {
                "name": str(col),
                "dtype": str(col_series.dtype),
                "non_null_count": non_null,
                "example_value": example,
            }
        )

    df_preview = _df_preview(df, max_rows)
    raw_records = df_preview.to_dict(orient="records")
    records: List[Dict[str, Any]] = []
    for row in raw_records:
        safe_row = {str(k): _to_llm_safe_value(v) for k, v in row.items()}
        records.append(safe_row)

    serialized: Dict[str, Any] = {
        "type": "dataframe",
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "columns": schema,
        "data_preview": records,
        "data_preview_row_count": len(records),
        "truncated": n_rows > max_rows,
    }

    if include_markdown:
        serialized["markdown_table"] = _df_to_markdown_table(df, max_rows)

    if include_html:
        serialized["html_table"] = _df_to_html_table(df, max_rows)

    return serialized


# ----------------------------
# Magika prediction
# ----------------------------

@lru_cache(maxsize=1)
def _get_magika() -> Magika:
    return Magika()


def magika_predict(path: str | Path) -> Dict[str, Any]:
    magika = _get_magika()
    path = str(path)
    result = magika.identify_path(path)
    return {
        "label": result.output.label,
        "description": result.output.description,
        "confidence": result.score,
    }