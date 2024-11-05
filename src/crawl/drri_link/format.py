import sys
import urllib.parse
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.drri_link.root as sroot
from src import utils


def parse_url(s1: str) -> dict[str, str]:
    if 0:
        s1 = "http://kyudb.snu.ac.kr/series/paramView.do?py_king_nm=%EC%88%9C%EC%A1%B0&year=1801&month=07&day=06&py_yun=0&py_day_ganji=%EA%B2%BD%EC%A7%84"
    pr = urllib.parse.urlparse(s1)
    params = urllib.parse.parse_qs(pr.query)
    for v in params.values():
        assert len(v) == 1, f"bad param: {v}"
    params2 = {k: v[0] for k, v in params.items()}
    return params2


def gen_format_file() -> None:
    # init
    tqdm.pandas()

    # read
    df = utils.read_df(sroot.LV1A_PQ)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    {k: k for k in df.columns}
    rcols = {
        "data_id": "data_id",
        "text": "label",
        "url": "meta.url",
    }
    df.rename(columns=rcols, inplace=True)

    # fix
    df.sample(1).iloc[0].to_dict()
    vc = df["label"].value_counts()
    vc[vc > 3]
    df["label"].replace("Empty!", "", inplace=True)

    # parse
    if 0:
        s1: str = df["label"].sample(1).iloc[0]
        parse_url(s1=s1)
    df["label"] = df["label"].progress_apply(parse_url)
    df = pd.json_normalize(df.to_dict(orient="records"))
    df.sample(1).iloc[0].to_dict()

    # sort rows
    assert df["data_id"].is_unique
    df.sort_values("data_id", inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # write
    df.sample(1).iloc[0].to_dict()
    utils.write_df2(sroot.FORMAT_PQ, df)
    logger.debug(len(df))


def main() -> None:
    # rename and sort cols
    gen_format_file()  # 95.5K, 961c450b, 15385


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_link.format
            typer.run(main)
