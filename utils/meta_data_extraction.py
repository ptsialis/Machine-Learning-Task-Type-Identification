import random
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from pandas.api.types import is_numeric_dtype


def canonicalize_value(x, max_depth: int = 6):
    """
    Convert arbitrary Python / NumPy / pandas values into hashable,
    recursively comparable representations.

    This is used for robust uniqueness counting and safe sampling of
    columns that may contain nested arrays, lists, dicts, sets, etc.
    """
    # Missing values
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # NumPy arrays -> recurse through their content
    if isinstance(x, np.ndarray):
        if max_depth <= 0:
            return ("ndarray_truncated", x.shape, str(x.dtype))
        return (
            "ndarray",
            x.shape,
            str(x.dtype),
            tuple(canonicalize_value(v, max_depth - 1) for v in x.tolist()),
        )

    # Dicts
    if isinstance(x, dict):
        if max_depth <= 0:
            return ("dict_truncated", len(x))
        return (
            "dict",
            tuple(
                (str(k), canonicalize_value(v, max_depth - 1))
                for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
            ),
        )

    # Lists / tuples
    if isinstance(x, list):
        if max_depth <= 0:
            return ("list_truncated", len(x))
        return ("list", tuple(canonicalize_value(v, max_depth - 1) for v in x))

    if isinstance(x, tuple):
        if max_depth <= 0:
            return ("tuple_truncated", len(x))
        return ("tuple", tuple(canonicalize_value(v, max_depth - 1) for v in x))

    # Sets
    if isinstance(x, set):
        if max_depth <= 0:
            return ("set_truncated", len(x))
        return (
            "set",
            tuple(sorted((canonicalize_value(v, max_depth - 1) for v in x), key=repr)),
        )

    # Plain hashables
    try:
        hash(x)
        return x
    except TypeError:
        return ("repr", repr(x))


def safe_nunique(series: pd.Series) -> int:
    """
    Robust nunique that works even when cells contain unhashable objects like
    numpy arrays, lists, dicts, or sets.
    Counts uniqueness by content.
    """
    s = series.dropna()

    # Fast path for normal columns
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        pass

    return len({canonicalize_value(v) for v in s.to_list()})


def safe_is_missing(x) -> bool:
    """Return True for None / NaN / pandas missing values, safely."""
    if x is None:
        return True

    try:
        result = pd.isna(x)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except Exception:
        pass

    return False


def safe_to_string(x) -> str:
    """Convert a value to string while treating missing values as empty."""
    return "" if safe_is_missing(x) else str(x)


def summary_stats(dat: pd.DataFrame, key_s) -> list[list]:
    """
    Compute per-column summary statistics.
    For numeric columns, NaN-safe statistics are used.
    """
    b_data = []

    for col in key_s:
        series = dat[col]
        nans = int(series.isna().sum())
        dist_val = safe_nunique(series)
        total_val = int(len(series))

        mean = np.nan
        std_dev = np.nan
        var = np.nan
        min_val = np.nan
        max_val = np.nan

        if is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")

            if numeric_series.notna().any():
                mean = float(np.nanmean(numeric_series))
                std_dev = float(np.nanstd(numeric_series))
                var = float(np.nanvar(numeric_series))
                min_val = float(np.nanmin(numeric_series))
                max_val = float(np.nanmax(numeric_series))

        b_data.append([total_val, nans, dist_val, mean, std_dev, min_val, max_val])

    return b_data


def get_sample(dat: pd.DataFrame, key_s) -> list[list]:
    """
    Hash-safe + sampling-safe for arbitrary Python objects.
    Uses random.sample / random.choices instead of np.random.choice.
    """
    rand = []

    for name in key_s:
        s = dat[name].dropna()

        # Deduplicate by canonical key, keep original value
        seen = {}
        for v in s.to_list():
            k = canonicalize_value(v)
            if k not in seen:
                seen[k] = v

        uniques = list(seen.values())

        if len(uniques) == 0:
            rand_sample = [None] * 5
        elif len(uniques) >= 5:
            rand_sample = random.sample(uniques, k=5)
        else:
            rand_sample = random.choices(uniques, k=5)

        rand.append(rand_sample)

    return rand


def get_avg_tokens(samples) -> list[float]:
    avg_tokens = []
    for sample_list in samples:
        list_of_num_tokens = [len(str(sample).split()) for sample in sample_list]
        avg_tokens.append(sum(list_of_num_tokens) / len(list_of_num_tokens))
    return avg_tokens


def get_ratio_dist_val(summary_stat_result) -> list[float]:
    ratio_dist_val = []
    for r in summary_stat_result:
        ratio_dist_val.append(r[2] * 100.0 / r[0] if r[0] else np.nan)
    return ratio_dist_val


def get_ratio_nans(summary_stat_result) -> list[float]:
    ratio_nans = []
    for r in summary_stat_result:
        ratio_nans.append(r[1] * 100.0 / r[0] if r[0] else np.nan)
    return ratio_nans


def FeaturizeFile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-column metadata/features from a DataFrame.
    Designed to work even when cells contain arrays, lists, dicts, etc.
    """
    del_pattern = r"([^,;\|]+[,;\|]{1}[^,;\|]+){1,}"
    del_reg = re.compile(del_pattern)
    delimiters = re.compile(r"(,|;|\|)")

    url_pat = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    url_reg = re.compile(url_pat)

    email_pat = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b"
    email_reg = re.compile(email_pat)

    stop_words = set(nltk_stopwords.words("english"))

    keys = list(df.columns)

    summary_stat_result = summary_stats(df, keys)
    samples = get_sample(df, keys)
    ratio_dist_val = get_ratio_dist_val(summary_stat_result)
    ratio_nans = get_ratio_nans(summary_stat_result)

    csv_names = [
        "Attribute_name",
        "total_vals",
        "num_nans",
        "num_of_dist_val",
        "mean",
        "std_dev",
        "min_val",
        "max_val",
        "%_dist_val",
        "%_nans",
        "sample_1",
        "sample_2",
        "sample_3",
        "sample_4",
        "sample_5",
    ]

    rows = []
    for i, attribute_name in enumerate(keys):
        row = [attribute_name]
        row.extend(summary_stat_result[i])
        row.append(ratio_dist_val[i])
        row.append(ratio_nans[i])
        row.extend(samples[i])
        rows.append(row)

    curdf = pd.DataFrame(rows, columns=csv_names)

    # Feature extraction from sampled values
    for row in curdf.itertuples():
        sample_values = [row.sample_1, row.sample_2, row.sample_3, row.sample_4, row.sample_5]

        delim_cnt = 0
        url_cnt = 0
        email_cnt = 0
        date_cnt = 0

        chars_totals = []
        word_totals = []
        stopwords_counts = []
        whitespaces = []
        delims_count = []

        for value in sample_values:
            s = safe_to_string(value)

            word_totals.append(len(s.split()))
            chars_totals.append(len(s))
            whitespaces.append(s.count(" "))

            if del_reg.match(s):
                delim_cnt += 1
            if url_reg.match(s):
                url_cnt += 1
            if email_reg.match(s):
                email_cnt += 1

            delims_count.append(len(delimiters.findall(s)))

            try:
                tokenized = word_tokenize(s)
                stopwords_counts.append(len([w for w in tokenized if w.lower() in stop_words]))
            except Exception:
                stopwords_counts.append(0)

            try:
                parsed = pd.to_datetime(value, errors="raise")
                if not pd.isna(parsed):
                    date_cnt += 1
            except Exception:
                pass

        idx = row.Index

        curdf.at[idx, "has_delimiters"] = delim_cnt > 2
        curdf.at[idx, "has_url"] = url_cnt > 2
        curdf.at[idx, "has_email"] = email_cnt > 2
        curdf.at[idx, "has_date"] = date_cnt > 2

        curdf.at[idx, "mean_word_count"] = float(np.mean(word_totals))
        curdf.at[idx, "std_dev_word_count"] = float(np.std(word_totals))

        curdf.at[idx, "mean_stopword_total"] = float(np.mean(stopwords_counts))
        curdf.at[idx, "stdev_stopword_total"] = float(np.std(stopwords_counts))

        curdf.at[idx, "mean_char_count"] = float(np.mean(chars_totals))
        curdf.at[idx, "stdev_char_count"] = float(np.std(chars_totals))

        curdf.at[idx, "mean_whitespace_count"] = float(np.mean(whitespaces))
        curdf.at[idx, "stdev_whitespace_count"] = float(np.std(whitespaces))

        curdf.at[idx, "mean_delim_count"] = float(np.mean(delims_count))
        curdf.at[idx, "stdev_delim_count"] = float(np.std(delims_count))

        curdf.at[idx, "is_list"] = bool(
            curdf.at[idx, "has_delimiters"] and curdf.at[idx, "mean_char_count"] < 100
        )
        curdf.at[idx, "is_long_sentence"] = bool(curdf.at[idx, "mean_word_count"] > 10)

    return curdf


def describe_attribute(df: pd.DataFrame, attribute_name: str) -> str:
    """
    Extract metadata for a given attribute and return a human-readable
    description string. Only includes fields that exist and contain
    valid (non-NaN / non-None) values.
    """

    column_descriptions = {
        "total_vals": "Total number of values",
        "num_of_dist_val": "Number of unique values",
        "mean": "Mean value",
        "std_dev": "Standard deviation",
        "min_val": "Minimum value",
        "max_val": "Maximum value",
        "mean_word_count": "Average word count",
        "std_dev_word_count": "Standard deviation of word count",
        "mean_stopword_total": "Average number of stopwords",
        "stdev_stopword_total": "Standard deviation of stopword count",
        "mean_char_count": "Average character count",
        "stdev_char_count": "Standard deviation of character count",
    }

    row = df.loc[df["Attribute_name"] == attribute_name]

    if row.empty:
        return f"No metadata found for attribute '{attribute_name}'."

    row = row.iloc[0]
    lines = [f"Statistics of target '{attribute_name}':"]

    for col, description in column_descriptions.items():

        # Column not present in dataframe
        if col not in df.columns:
            continue

        value = row[col]

        # Skip missing / invalid values
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if pd.isna(value):
            continue
        if value == "None":
            continue

        lines.append(f"- {description}: {value:.3f}" if isinstance(value, float) else f"- {description}: {value}")

    return "\n".join(lines)