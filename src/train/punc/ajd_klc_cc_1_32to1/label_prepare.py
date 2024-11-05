import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.tool.eval as etool
import src.train.punc.ajd_klc_cc_1_32to1.root as sroot
from src import utils


def gen_stat_dir() -> None:
    # read
    df = utils.read_df(sroot.DATASET_PQ)
    df.sample(1).iloc[0].to_dict()

    # gen_char
    etool.report_char_punc_freq(texts=df["text"], output_dir=sroot.STAT_DIR / "orig")  # type: ignore

    # gen_char
    df["text"] = df["text"].progress_apply(utils.remove_whites)
    etool.report_char_punc_freq(texts=df["text"], output_dir=sroot.STAT_DIR / "without_spaces")  # type: ignore

    # open
    if hasattr(sys, "ps1"):
        utils.open_code(sroot.STAT_DIR)


def gen_labels_json() -> None:
    # read file
    df = utils.read_df(sroot.DATASET_PQ)
    df.info()
    df.sample(1).iloc[0].to_dict()

    # filter: only train
    df["split"].value_counts()
    idx = df["split"].isin(["train"])
    df[idx].groupby(["meta.corpus", "split"], dropna=False).size()
    df = df[idx].reset_index(drop=True)

    # count
    df1 = etool.count_punc_label(texts=df["text"], not_punc=utils.NOT_PUNC, ignore_whites=True)  # type: ignore

    # sort
    df1.sort_values(
        by=["count", "label"], ascending=[False, True], inplace=True, ignore_index=True
    )
    df1.sample(1).iloc[0].to_dict()

    # add cols
    digit = len(str(df1.index.max()))
    df1["idx"] = df1.index + 1
    df1["idx"] = df1["idx"].apply(lambda x: f"L{x:0{digit}d}")
    #
    df1["percent"] = df1["count"] / df1["count"].sum() * 100
    df1["percent"] = df1["percent"].apply(lambda x: f"{x:.2f}%")
    #
    df1["percent_cum"] = df1["count"].cumsum() / df1["count"].sum() * 100
    df1["percent_cum"] = df1["percent_cum"].apply(lambda x: f"{x:.2f}%")
    #
    df1["punc_cum"] = df1["label"].cumsum().apply(lambda x: "".join(sorted(set(x))))
    df1["punc_num"] = df1["punc_cum"].apply(len)
    #
    df1["prev_punc_cum"] = df1["punc_cum"].shift(1)
    df1["prev_punc_cum"].fillna("", inplace=True)
    df1["punc_new"] = df1.apply(
        lambda x: "".join(sorted(set(x["punc_cum"]) - set(x["prev_punc_cum"]))),
        axis=1,
    )
    df1.drop(columns=["prev_punc_cum"], inplace=True)

    # 최소한 99% 이상을 포함
    df1["percent_cum2"] = df1["count"].cumsum() / df1["count"].sum() * 100
    df1["is_target"] = False
    for i in range(len(df1)):
        if df1.loc[i, "percent_cum2"] <= 99:  # type: ignore
            df1.loc[i, "is_target"] = True
            if i + 1 < len(df1):
                df1.loc[i + 1, "is_target"] = True
    df1[df1["is_target"]]
    df1[~df1["is_target"]]
    df1.drop(columns=["percent_cum2"], inplace=True)
    if 0:
        df1 = utils.read_df(sroot.LABELS_JSON)
        utils.open_code(sroot.LABELS_TSV)

    # sort cols
    sorted(df1.columns)
    cols = [
        "idx",
        "label",
        "count",
        "percent",
        "percent_cum",
        "punc_num",
        "punc_cum",
        "punc_new",
        "is_target",
    ]
    assert set(cols) == set(df1.columns)
    df1 = df1[cols].reset_index(drop=True)

    # save
    sroot.LABELS_JSON.parent.mkdir(exist_ok=True, parents=True)
    utils.write_json(sroot.LABELS_JSON, df1.to_dict(orient="records"))

    # save
    df2 = df1.copy()
    for col in ["label", "punc_cum", "punc_new"]:
        df2[col] = df2[col].apply(etool.edit_punc_for_google)
    utils.write_df(sroot.LABELS_TSV, df2)


def gen_label2id_json() -> None:
    # read
    df = pd.read_json(sroot.LABELS_JSON)

    # filter
    df = df[df["is_target"]].reset_index(drop=True)
    labels = df["label"].tolist()

    # format
    df1 = pd.DataFrame(labels, columns=["label"])
    df1["id"] = df1.index + 1
    digit = len(str(df1["id"].max()))
    df1["id"] = df1["id"].apply(lambda x: f"P{x:0{digit}d}")
    df1["id"] = df1["id"] + "_" + df1["label"].apply(etool.name_punc_label)

    # save
    label2id = df1.set_index("label")["id"].to_dict()
    utils.write_json(sroot.LABEL2ID_JSON, label2id)


def main() -> None:
    gen_stat_dir()
    gen_labels_json()
    gen_label2id_json()


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(etool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.punc.ajd_klc_cc_2to1.label_prepare
            typer.run(main)
