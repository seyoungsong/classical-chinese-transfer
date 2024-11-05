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

import src.eval.ner1.root as sroot
import src.tool.eval as etool
import src.tool.eval.metric as emetric
from src import utils


def report_score(k1: list[str], df1: pd.DataFrame) -> None:
    # check
    df1.sample(n=1).iloc[0].to_dict()
    if 0:
        df1 = df1.sample(n=100, random_state=0).reset_index(drop=True)

    # load
    ref1: list[str] = df1["text_xml"].to_list()
    hypo: list[str] = df1["pred.content"].to_list()
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
    assert len(k1) == 2, "k1 bad format"
    d1["meta.corpus"] = k1[0]
    d1["pred.model_id"] = k1[1]

    # save
    stem = "--".join(k1)
    fname = sroot.RESULT_DIR / f"score/{stem}.json"
    fname.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json2(fname, d1)


def main() -> None:
    # read
    df = utils.read_df(sroot.OUTPUT_PQ)
    df.sample(1).iloc[0].to_dict()

    # check
    df.groupby(["id", "pred.model_id"]).size().value_counts()
    df.groupby(["meta.corpus", "pred.model_id"]).size()

    # count
    counter = Counter()  # type: ignore
    for s_xml in tqdm(df["text_xml"].to_list() + df["pred.content"].to_list()):
        items = etool.xml2items(s_xml)
        names = [d["name"] for d in items if d["name"]]
        counter.update(names)
    df_count = pd.DataFrame(counter.most_common(), columns=["label", "count"])
    df_count
    """
                label   count
    0           other  215198
    1     wyweb_other   94970
    2       klc_other   57505
    3  wyweb_bookname   36555
    4      ajd_person   27220
    5    ajd_location    8960
    6       ajd_other     845
    """
    {k: "other" for k in sorted(df_count["label"].unique())}

    # test
    k1, df1 = random.choice(list(df.groupby(["meta.corpus", "pred.model_id"])))
    report_score(k1=k1, df1=df1)  # type: ignore

    # split
    for k1, df1 in tqdm(df.groupby(["meta.corpus", "pred.model_id"])):
        report_score(k1=k1, df1=df1)  # type: ignore


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
            # python -m src.eval.ner1.eval_f1
            typer.run(main)
