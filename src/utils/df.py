from pathlib import Path
from typing import Any, TypeAlias

import humanize
import numpy as np
import pandas as pd
from datasets import Dataset, disable_caching, load_dataset
from loguru import logger

from . import gdir, io

SeriesType: TypeAlias = "pd.Series[Any]"
PandasType: TypeAlias = "pd.DataFrame | pd.Series[Any]"


def get_ftype(fname: Path) -> str:
    s = fname.name.lower()
    if s.endswith(".zst"):
        s = s.rsplit(".zst", maxsplit=1)[0]
    ftype = s.split(".")[-1]
    return ftype


def write_df(fname: Path, df: pd.DataFrame) -> None:
    ftype = get_ftype(fname)

    if "jsonl" == ftype:
        df.to_json(fname, orient="records", lines=True, force_ascii=False)
    elif "json" == ftype:
        assert ".zst" not in fname.name, "json.zst is not supported."
        df.to_json(fname, orient="records", indent=2, force_ascii=False)
        if 0:
            d1 = df.to_dict(orient="records")
            io.write_json2(fname, d1)
    elif "pkl" == ftype:
        df.to_pickle(fname)
    elif "csv" == ftype:
        df.to_csv(fname, index=False, encoding="utf-8-sig")
    elif "tsv" == ftype:
        df.to_csv(fname, sep="\t", index=False, encoding="utf-8-sig")
    elif "parquet" == ftype:
        df.to_parquet(path=fname, engine="pyarrow", compression="zstd")
        if 0:
            df.to_parquet(path=fname, engine="pyarrow")
    else:
        raise ValueError(f"Unknown suffix: {ftype}")
    io.log_written(fname, etc=str(len(df)))


def write_df2(fname: Path, df: pd.DataFrame, n: int = 100) -> None:
    if 0:
        fname = Path("temp/df.jsonl.zst").resolve()
    # check
    s1 = fname.name
    if s1.endswith(".zst"):
        s1 = s1.rsplit(".zst", maxsplit=1)[0]
    stem = s1.split(".")[0]

    # fname
    fname_sample_jsonl = fname.parent / f"{stem}_sample.jsonl"
    fname_sample_json = fname.parent / f"{stem}_sample.json"

    # sample
    n_safe = min(n, len(df))
    df_sample = df.sample(n_safe, random_state=42).sort_index()

    # write
    fname_sample_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_df(fname_sample_jsonl, df_sample)
    try:
        write_df(fname_sample_json, df_sample)
    except Exception as e:
        logger.error(e)

    # all
    write_df(fname, df)


def memory_usage(df: pd.DataFrame) -> None:
    s = humanize.naturalsize(df.memory_usage(deep=True).sum())
    logger.debug(f"Memory usage: {s}")


def read_df(fname: Path) -> pd.DataFrame:
    io.log_reading(fname)
    ftype = get_ftype(fname)

    df: pd.DataFrame
    if "jsonl" == ftype:
        df = pd.read_json(fname, lines=True)
    elif "json" == ftype:
        assert ".zst" not in fname.name, "json.zst is not supported."
        d1 = io.read_json2(fname)
        df = pd.DataFrame(d1)
    elif "pkl" == ftype:
        df = pd.read_pickle(fname)
    elif "csv" == ftype:
        df = pd.read_csv(fname, encoding="utf-8-sig")
    elif "tsv" == ftype:
        df = pd.read_csv(fname, sep="\t", encoding="utf-8-sig")
    elif "parquet" == ftype:
        df = pd.read_parquet(fname, engine="pyarrow")
    else:
        raise ValueError(f"Unknown suffix: {ftype}")
    return df


def read_ds(fname: Path) -> Dataset:
    ftype = get_ftype(fname)
    ds: Dataset
    if "jsonl" == ftype:
        io.log_reading(fname)
        disable_caching()
        ds = load_dataset(
            "json",
            data_files={"simple": str(fname)},
            split="simple",
            download_mode="force_redownload",
            verification_mode="no_checks",
        )
    elif "parquet" == ftype:
        io.log_reading(fname)
        disable_caching()
        ds = load_dataset(
            "parquet",
            data_files={"simple": str(fname)},
            split="simple",
            download_mode="force_redownload",
            verification_mode="no_checks",
        )
    else:
        raise ValueError(f"Unknown suffix: {ftype}")
    return ds


def sample_df(df: PandasType) -> None:
    n = 10
    n_safe = min(n, len(df))
    gdir.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    df_head = df.head(n_safe)
    df_sample = df.sample(n_safe, random_state=42).sort_index()
    df_tail = df.tail(n_safe)

    block_start_idx = len(df) // 2
    block_safe_n = min(len(df) - block_start_idx, n)
    df_block = df.iloc[block_start_idx : block_start_idx + block_safe_n]

    df_list: list[pd.DataFrame] = [df_head, df_sample, df_tail, df_block]  # type: ignore
    df1: pd.DataFrame = pd.concat(df_list)
    df1.sort_index(inplace=True)
    df1.drop_duplicates(inplace=True, ignore_index=True)

    fname = gdir.TEMP_DIR / "df-random.csv"
    df1.to_csv(fname, index=False, encoding="utf-8-sig")
    io.log_written(fname)


def replace_blank_to_none(df: pd.DataFrame) -> pd.DataFrame:
    """Replace blank strings in DataFrame to NaN."""
    if 0:
        df = pd.DataFrame({"a": ["", " ", "a", "b"], "b": ["", None, 1, 2]})
        df.isna()
        df1 = df.replace(r"^\s*$", np.nan, regex=True)
        df1.isna()
        df1.to_dict(orient="records")
        df2 = df.replace(r"^\s*$", None, regex=True)
        df2.isna()
        df2.to_dict(orient="records")
    return df.replace(r"^\s*$", None, regex=True)


def shuffle_df(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df
