import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src import utils


def report_sample(d: Path, f: Path) -> Path:
    # read
    df = utils.read_df(f)
    if 0:
        df.sample(1).iloc[0].to_dict()

    # sample
    n_safe = min(30, len(df))
    df_sample = df.sample(n=n_safe, random_state=42).sort_index(ignore_index=True)

    # convert list to str
    for c in df_sample.columns:
        if isinstance(df_sample[c].iloc[0], list):
            df_sample[c] = df_sample[c].apply(lambda x: "|".join(map(str, x)))
    df_sample.sample(1).iloc[0].to_dict()

    # save
    f2 = d / f.name / "sample.json"
    f2.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json(f2, df_sample.to_dict(orient="records"))

    return f2


def report_char_punc_freq(texts: list[str], output_dir: Path) -> None:
    # char_freq
    counter_char: Counter[str] = Counter()
    for s in tqdm(texts, desc="counter"):
        counter_char.update(s)
    len(counter_char)
    #
    df_char = pd.DataFrame(counter_char.most_common(), columns=["char", "count"])
    df_char.sort_values(
        by=["count", "char"], ascending=[False, True], inplace=True, ignore_index=True
    )
    df_char["idx"] = df_char.index + 1
    df_char["percent"] = df_char["count"] / df_char["count"].sum() * 100
    df_char["percent"] = df_char["percent"].apply(lambda x: f"{x:.2f}%")
    df_char["cum_percent"] = df_char["count"].cumsum() / df_char["count"].sum() * 100
    df_char["cum_percent"] = df_char["cum_percent"].apply(lambda x: f"{x:.2f}%")
    df_char.sample(1).iloc[0].to_dict()
    # trim
    df_char2 = df_char.copy()
    df_char2["percent_float"] = (
        df_char2["count"].cumsum() / df_char2["count"].sum() * 100
    )
    df_char2.sample(1).iloc[0].to_dict()
    df_char2 = df_char2[df_char2["percent_float"] <= 90].reset_index(drop=True)
    df_char2.drop(columns=["percent_float"], inplace=True)
    #
    fname = output_dir / "char_freq.json"
    fname.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json(fname, df_char2.to_dict(orient="records"))

    # char_category
    df_cat1 = df_char.copy()
    df_cat1["cat"] = df_cat1["char"].apply(unicodedata.category)
    df_cat1.sample(1).iloc[0].to_dict()
    #
    df_cat2 = (
        df_cat1.sort_values(by=["cat", "count"], ascending=[True, False])
        .groupby("cat")["char"]
        .apply(lambda x: "".join(x))
        .reset_index(name="chars")
    )
    df_cat2["cat2"] = df_cat2["cat"].apply(utils.unicode_category_full_name)
    df_cat2 = df_cat2[sorted(df_cat2.columns)].reset_index(drop=True)
    fname = output_dir / "char_category.json"
    utils.write_json(fname, df_cat2.to_dict(orient="records"))

    # punc_single
    df_punc = df_char.copy()
    df_punc["cat"] = df_punc["char"].apply(unicodedata.category)
    df_punc["cat"].value_counts()
    idx = df_punc["cat"].apply(lambda x: x[0] in "PZ" or x == "Cc")
    df_punc = df_punc[idx].reset_index(drop=True)
    #
    df_punc.sort_values(
        by=["count", "char"], ascending=[False, True], inplace=True, ignore_index=True
    )
    df_punc["idx"] = df_punc.index + 1
    df_punc["percent"] = df_punc["count"] / df_punc["count"].sum() * 100
    df_punc["percent"] = df_punc["percent"].apply(lambda x: f"{x:.2f}%")
    df_punc["cum_percent"] = df_punc["count"].cumsum() / df_punc["count"].sum() * 100
    df_punc["cum_percent"] = df_punc["cum_percent"].apply(lambda x: f"{x:.2f}%")
    df_punc.sample(1).iloc[0].to_dict()
    #
    fname = output_dir / "punc_single.json"
    utils.write_json(fname, df_punc.to_dict(orient="records"))

    # punc_label
    counter_label: Counter[str] = Counter()
    for s in tqdm(texts, desc="counter"):
        if s is None:
            continue
        s2 = utils.squeeze_whites(s)
        items = utils.chunk_by_classifier(s=s2, f=utils.is_punc_unicode)
        labels = [d["text"] for d in items if d["label"] is True]
        counter_label.update(labels)
    len(counter_label)
    df_label = pd.DataFrame(counter_label.most_common(), columns=["label", "count"])
    #
    df_label.sort_values(
        by=["count", "label"], ascending=[False, True], inplace=True, ignore_index=True
    )
    df_label["idx"] = df_label.index + 1
    df_label["percent"] = df_label["count"] / df_label["count"].sum() * 100
    df_label["percent"] = df_label["percent"].apply(lambda x: f"{x:.2f}%")
    df_label["cum_percent"] = df_label["count"].cumsum() / df_label["count"].sum() * 100
    df_label["cum_percent"] = df_label["cum_percent"].apply(lambda x: f"{x:.2f}%")
    df_label.sample(1).iloc[0].to_dict()
    #
    fname = output_dir / "punc_label.json"
    utils.write_json(fname, df_label.to_dict(orient="records"))
