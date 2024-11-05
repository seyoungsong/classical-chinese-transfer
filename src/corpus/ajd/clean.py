import random
import re
import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ajd.root as sroot
from src import utils


def gen_clean_file() -> None:
    # read
    df0 = utils.read_df(sroot.FORMAT_PQ)
    df = df0.copy()

    # check
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # fix title(강)
    pat = re.compile(r"^[\d\-윤\[\]]+\s+")
    if 0:
        df3 = df[df["meta.data_title.cko"].notna()].reset_index(drop=True)
        utils.write_str(
            utils.TEMP_TXT, "\n".join(df3["meta.data_title.cko"].sample(100).to_list())
        )
        s1 = df3["meta.data_title.cko"].sample(1).iloc[0]
        pat.sub("", s1)
    df["meta.data_title.cko"] = df["meta.data_title.cko"].parallel_apply(
        lambda x: pat.sub("", x) if x else x
    )

    # text
    cols = [c for c in df.columns if c.startswith("text") and "xml" not in c]
    cols += ["meta.data_title.cko"]
    df2 = df[cols].reset_index(drop=True)
    if 0:
        x1 = df2.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # check full
    c = random.choice(df2.columns)
    d1 = {}
    for c in tqdm(df2.columns):
        vc = df2[c].value_counts()
        vc = vc[vc > 10]
        d1[c] = vc.to_dict()
    utils.write_json(utils.TEMP_JSON, d1)
    utils.open_code(utils.TEMP_JSON)

    # check start
    c = random.choice(df2.columns)
    d1 = {}
    for c in tqdm(df2.columns):
        vc = df2[c].str[:5].value_counts()
        vc = vc[vc > 10]
        d1[c] = vc.to_dict()
    utils.write_json(utils.TEMP_JSON, d1)
    utils.open_code(utils.TEMP_JSON)

    # check end
    c = random.choice(df2.columns)
    d1 = {}
    for c in tqdm(df2.columns):
        vc = df2[c].str[-10:].value_counts()
        vc = vc[vc > 10]
        d1[c] = vc.to_dict()
    utils.write_json(utils.TEMP_JSON, d1)
    utils.open_code(utils.TEMP_JSON)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.CLEAN_PQ, df)


def main() -> None:
    # clean texts if needed
    gen_clean_file()  # 557.9M, bde2af48, 413323


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
            # python -m src.corpus.ajd.clean
            typer.run(main)
