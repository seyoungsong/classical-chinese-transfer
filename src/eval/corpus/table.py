import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from rich import pretty

import src.eval.corpus.root as sroot
from src import utils


def gen_table() -> None:
    # read
    stat_files = sorted(sroot.RESULT_DIR.rglob("stat.json"))
    stat_files = [f for f in stat_files if not f.parent.name.startswith("mt_")]
    stat_dicts = [{**utils.read_json2(f), "fname": f.parent.name} for f in stat_files]
    df = pd.json_normalize(stat_dicts)
    df = df[sorted(df.columns)].reset_index(drop=True)
    df.sort_values(["fname"], inplace=True, ignore_index=True)

    # align
    table_order_str = """
    ajd
    drs
    drri
    klc
    daizhige
    niutrans
    wyweb_mt
    ocdb
    wyweb_ner
    wyweb_punc
    hue_ner
    """
    fname_order = table_order_str.strip().lower().splitlines()
    fname_order = [c.strip() for c in fname_order if c.strip()]
    #
    fname_curr = sorted(df["fname"].to_list())
    fname_extra = sorted([c for c in fname_curr if c not in fname_order])
    fname_order += fname_extra

    # Step 1: Create a new DataFrame from your list of fnames
    df_table = pd.DataFrame(fname_order, columns=["fname"])

    # Step 2: Perform an outer merge
    df_align = pd.merge(df_table, df, on="fname", how="left")
    df_align.fillna("N/A", inplace=True)

    # sort columns
    df_align = df_align[sorted(df_align.columns)].reset_index(drop=True)

    # write
    fname = sroot.RESULT_DIR / "table.tsv"
    fname.parent.mkdir(parents=True, exist_ok=True)
    utils.write_df(fname, df_align)


def main() -> None:
    gen_table()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.eval.corpus.table
            typer.run(main)
