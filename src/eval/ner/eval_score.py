import random
import sys
from collections import Counter
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.eval.ner.root as sroot
import src.tool.eval as etool
import src.tool.eval.metric as emetric
from src import utils


def report_score(df1: pd.DataFrame) -> None:
    # check
    df1.sample(n=1).iloc[0].to_dict()
    if 0:
        df1 = df1.sample(n=100, random_state=0).reset_index(drop=True)

    # load
    ref1: list[str] = df1["text_xml"].to_list()
    hypo: list[str] = df1["pred.content"].to_list()
    hypo = [s if s is not None else "" for s in hypo]
    if 0:
        utils.temp_diff("\n\n".join(hypo), "\n\n".join(ref1))

    # compute
    d1 = emetric.compute_ner_f1(hypo=hypo, ref1=ref1, mode="binary")
    d2 = emetric.compute_ner_f1(hypo=hypo, ref1=ref1, mode="entity")

    # merge
    d1 = {f"binary.{k}": v for k, v in d1.items()}
    d2 = {f"entity.{k}": v for k, v in d2.items()}
    d1.update(d2)

    # metadata
    d1["meta.corpus"] = df1["meta.corpus"].iloc[0]
    d1["pred.model_id"] = str(df1["pred.model_id"].iloc[0])

    # save
    k1 = [d1["meta.corpus"], d1["pred.model_id"]]
    stem = "--".join(k1).replace("/", "__")
    fname = sroot.RESULT_DIR / f"score/{stem}.json"
    fname.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json2(fname, d1)


def main() -> None:
    # read
    df = utils.read_df(sroot.OUTPUT_PQ)
    df.sample(1).iloc[0].to_dict()

    # remove error rows (?)
    df = df[df["pred.content"].notna()].reset_index(drop=True)

    # check
    df.groupby(["id", "pred.model_id"]).size().value_counts()
    df.groupby(["meta.corpus", "pred.model_id"]).size()

    # count
    counter = Counter()  # type: ignore
    for s_xml in tqdm(df["text_xml"]):
        items = etool.xml2items(s_xml)
        names = [d["name"] for d in items if d["name"]]
        counter.update(names)
    df_count = pd.DataFrame(counter.most_common(), columns=["label", "count"])
    df_count
    """
            label   count
0     wyweb_other  360886
1       klc_other  218519
2  wyweb_bookname  138909
3      ajd_person  103436
4    ajd_location   34048
5       ajd_other    3211
    """
    {k: "other" for k in sorted(df_count["label"].unique())}

    # test
    if 0:
        _, df1 = random.choice(list(df.groupby(["meta.corpus", "pred.model_id"])))
        report_score(df1=df1)

    # sequential
    if 0:
        for _, df1 in tqdm(df.groupby(["meta.corpus", "pred.model_id"])):
            report_score(df1=df1)

    # parallel
    df.groupby(["meta.corpus", "pred.model_id"]).parallel_apply(  # type: ignore
        lambda df1: report_score(df1=df1)
    )


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(etool)
        reload(emetric)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.eval.ner.eval_score
            typer.run(main)
