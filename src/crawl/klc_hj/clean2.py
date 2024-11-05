import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.klc_hj.root as sroot
from src import utils


def gen_url2(data_id: str) -> str:
    item_id = data_id.split("_")[1]
    url = f"https://db.itkc.or.kr/dir/item?itemId={item_id}#/dir/node?dataId={data_id}&viewSync=KP&viewSync2=TR"
    return url


def gen_clean2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN_PQ)

    print(", ".join(df.columns))
    # meta.book_author, meta.book_extra, meta.book_extra_type, meta.book_id, meta.book_jisu, meta.book_title, meta.data_id, meta.elem_body_html, meta.elem_body_text, meta.elem_col, meta.elem_copyright_html, meta.elem_copyright_text, meta.elem_dci_html, meta.elem_dci_text, meta.elem_id, meta.elem_title_html, meta.elem_title_text, meta.elem_url, meta.mokcha_row, meta.mokcha_title, meta.page_path, meta.page_title, meta.url, meta.url2, meta.year
    df.sample(1).iloc[0].to_dict()

    # add: url2
    df["meta.url2"] = df["meta.data_id"].progress_apply(gen_url2)

    # add: elem_type
    df["meta.elem_id_type"] = df["meta.elem_id"].apply(lambda x: str(x).split("_")[1])
    df["meta.elem_id_type"].value_counts()
    df.groupby(["meta.elem_col", "meta.book_extra_type", "meta.elem_id_type"]).size()
    df.groupby("meta.elem_id_type").sample(1)["meta.url2"].to_list()
    df.groupby("meta.elem_id_type").sample(1)["meta.elem_url"].to_list()

    # add: elem_lang
    map0 = {
        "BT": "ko",  # 고전번역서
        "GO": "hj",  # 고전원문
        "GP": "hj",  # 고전원문
        "KP": "hj",  # 한국고전총간
        "MO": "hj",  # 한국문집총간
        "MP": "hj",  # 고전원문 or 한국문집총간
    }
    df["meta.elem_lang"] = df["meta.elem_id_type"].map(map0)
    df["meta.elem_lang"].value_counts()
    df.sample(1).iloc[0].to_dict()

    # add: elem_punc status
    {k: k for k in sorted(df["meta.elem_id_type"].unique())}
    map1 = {
        "BT": "na",  # 고전번역서
        "GO": "orig",  # 고전원문
        "GP": "punc",  # 고전원문
        "KP": "punc",  # 한국고전총간
        "MO": "orig",  # 한국문집총간
        "MP": "punc",  # 고전원문 or 한국문집총간
    }
    df["meta.elem_punc"] = df["meta.elem_id_type"].map(map1)
    df["meta.elem_punc"].value_counts()

    # check & sort
    kcols = ["meta.data_id", "meta.elem_id"]
    df.groupby(kcols).size().value_counts()
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # we only keep MO, and drop punctuated MP!
    df["meta.elem_id_type"].value_counts()
    idx = df["meta.elem_id_type"] == "MO"
    df = df[idx].reset_index(drop=True)

    # (skip: enforce strict 1:1 correspondence at elem_id level for safety)
    if 0:
        vc = df["meta.elem_id"].value_counts()
        bad_elem_id_list = vc[vc > 1].index
        idx = df["meta.elem_id"].isin(bad_elem_id_list)
        idx.mean() * 100  # 0.08%
        if 0:
            df1 = df[idx]
            df2 = df1.head(100).sort_index()
            utils.write_json(utils.TEMP_JSON, df2.to_dict(orient="records"))
        bad_data_id_list = df[idx]["meta.data_id"].unique()
        idx = df["meta.data_id"].isin(bad_data_id_list)
        logger.debug(f"drop: {idx.mean() * 100:.1f}%")  # 0.2%
        df = df[~idx].reset_index(drop=True)

    # check
    df["meta.elem_id"].value_counts().value_counts()
    df["meta.data_id"].value_counts().value_counts()
    assert df["meta.elem_id"].is_unique

    # empty to nan
    df.replace(r"^\s*$", None, regex=True, inplace=True)
    df.info()
    df.isna().sum()[df.isna().sum() > 0]

    # ok to be empty for now
    df.fillna("", inplace=True)

    # sample
    df.sample(1).iloc[0].to_dict()

    # sort rows
    kcols = ["meta.data_id", "meta.elem_id"]
    df.groupby(kcols).size().value_counts()
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # save
    df.info()
    utils.write_df2(sroot.CLEAN2_PQ, df)


def main() -> None:
    # drop some rows, and add some columns
    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_clean2_file()  # 808.5M, 5c4c205d


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        #
        tqdm.pandas()
        pandarallel.initialize(progress_bar=True)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.klc_hj.clean2
            typer.run(main)