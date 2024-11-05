import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from rich import pretty

import src.dataset.ner.root
import src.train.ner.ajd_klc_cc_1_8to0.root as sroot
from src import utils


def gen_dataset_file() -> None:
    # read
    df = utils.read_df(src.dataset.ner.root.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter: only ajd+klc+cc & plain split
    logger.debug(df["meta.corpus"].value_counts())
    idx = df["meta.corpus"].isin(["ajd", "klc", "wyweb_ner"]) & df["split"].isin(
        ["train", "valid", "test"]
    )
    df = df[idx].reset_index(drop=True)
    logger.debug(df.groupby(["meta.corpus", "split"], dropna=False).size())

    # check: no empty text
    df = utils.replace_blank_to_none(df)
    cols = [c for c in df.columns if not c.startswith("meta")]
    assert df[cols].isnull().sum().sum() == 0, "no empty text"
    df.dropna(axis=1, how="all", inplace=True)

    # sample: low-resource setting for Hanja
    # NER (20.5 : 1) (2 : 1) (1 : 1) (0.5 : 1) (1/4 : 1) (1/8 : 1) (1/16 : 1) (1/32 : 1)
    # AJD 293854 21403 7360 3680 1840 920 460 230
    # KLC 8035 8035 7360 3680 1840 920 460 230
    # CC 14719 14719 14719 14719 14719 14719 14719 14719
    # Total 316608 44157 29438 22079 18399 16559 15639 15179
    num_ajd = 920
    num_ajd_valid = num_ajd // 10
    num_klc = 920
    num_klc_valid = num_klc // 10
    num_etc = 0
    num_etc_valid = 0
    # sample train
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_ajd = df_train[df_train["meta.corpus"] == "ajd"].sample(
        n=num_ajd, random_state=42
    )
    df_klc = df_train[df_train["meta.corpus"] == "klc"].sample(
        n=num_klc, random_state=42
    )
    df_etc = df_train[~df_train["meta.corpus"].isin(["ajd", "klc"])].sample(
        n=num_etc, random_state=42
    )
    # sample valid
    df_valid = df[df["split"] == "valid"].reset_index(drop=True)
    df_ajd2 = df_valid[df_valid["meta.corpus"] == "ajd"].sample(
        n=num_ajd_valid, random_state=42
    )
    df_klc2 = df_valid[df_valid["meta.corpus"] == "klc"].sample(
        n=num_klc_valid, random_state=42
    )
    df_etc2 = df_valid[~df_valid["meta.corpus"].isin(["ajd", "klc"])].sample(
        n=num_etc_valid, random_state=42
    )

    # concat
    df = pd.concat(
        [df_ajd, df_klc, df_etc, df_ajd2, df_klc2, df_etc2], ignore_index=True
    )
    assert df["key2"].is_unique
    df.sort_values("key2", inplace=True, ignore_index=True)

    # log
    sz1 = df.groupby(["meta.corpus", "split"]).size().reset_index(name="count")  # type: ignore
    sroot.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fname1 = sroot.RESULT_DIR / "dataset_size.tsv"
    sz1.to_csv(fname1, sep="\t", index=True, encoding="utf-8-sig")
    utils.log_written(fname1)

    # save
    sroot.DATASET_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.DATASET_PQ, df)


def main() -> None:
    gen_dataset_file()  # 49.2M, 086b3bdb, 49100


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.ner.ajd_klc_cc_2to1.dataset
            typer.run(main)
