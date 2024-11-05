import sys
from collections import Counter
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.tool.eval as etool
import src.train.ner.ajd_klc_cc_05to0.root as sroot
from src import utils


def gen_labels_json() -> None:
    # read
    df = utils.read_df(sroot.DATASET_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter: only train
    df["split"].value_counts()
    idx = df["split"].isin(["train"])
    df[idx].groupby(["meta.corpus", "split"], dropna=False).size()
    df = df[idx].reset_index(drop=True)

    # count
    counter = Counter()  # type: ignore
    for s_xml in tqdm(df["text_xml"]):
        items = etool.xml2items(s_xml)
        names = [d["name"] for d in items if d["name"]]
        counter.update(names)
    df1 = pd.DataFrame(counter.most_common(), columns=["label", "count"])

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

    # save
    sroot.LABELS_JSON.parent.mkdir(exist_ok=True, parents=True)
    utils.write_json(sroot.LABELS_JSON, df1.to_dict(orient="records"))
    utils.write_df(sroot.LABELS_TSV, df1)


def main() -> None:
    gen_labels_json()


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.ner.ajd_klc_cc_2to1.label_prepare
            typer.run(main)
