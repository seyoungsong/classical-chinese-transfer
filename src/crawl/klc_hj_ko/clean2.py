import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.klc_hj_ko.root as sroot
from src import utils


def gen_url2(data_id: str) -> str:
    item_id = data_id.split("_")[1]
    url = f"https://db.itkc.or.kr/dir/item?itemId={item_id}#/dir/node?dataId={data_id}&viewSync=OT&viewSync2=KP"
    return url


def gen_clean2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN_PQ)

    print(", ".join(df.columns))
    # meta.author, meta.book_category, meta.book_extra, meta.book_extra_type, meta.book_id, meta.book_title, meta.data_id, meta.elem_body_html, meta.elem_body_text, meta.elem_col, meta.elem_copyright_html, meta.elem_copyright_text, meta.elem_dci_html, meta.elem_dci_text, meta.elem_id, meta.elem_title_html, meta.elem_title_text, meta.elem_url, meta.mokcha_row, meta.mokcha_title, meta.page_path, meta.page_title, meta.publisher, meta.translator, meta.url, meta.url2, meta.year
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
    df["meta.elem_lang"] = df["meta.elem_col"].map({0: "ko", 1: "hj"})
    df.sample(1).iloc[0].to_dict()

    # add: elem_punc status
    {k: k for k in sorted(df["meta.elem_id_type"].unique())}
    map1 = {
        "BT": "na",  # 고전번역서
        "GO": "orig",  # 고전원문
        "GP": "punc",  # 고전원문
        "KP": "punc",  # 한국고전총간
        "MO": "orig",  # 한국문집총간
        "MP": "punc",  # 한국문집총간
    }
    df["meta.elem_punc"] = df["meta.elem_id_type"].map(map1)

    # check & sort
    kcols = ["meta.data_id", "meta.elem_id"]
    df.groupby(kcols).size().value_counts()
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # enforce strict 1:1 correspondence at elem_id level for safety
    # bad example: https://db.itkc.or.kr/dir/item?itemId=BT#/dir/node?dataId=ITKC_BT_1260A_0060_010_0150&viewSync=OT&viewSync2=KP
    vc = df["meta.elem_id"].value_counts()
    bad_elem_id_list = vc[vc > 1].index
    idx = df["meta.elem_id"].isin(bad_elem_id_list)
    idx.mean() * 100  # 1.1%
    if 0:
        df1 = df[idx]
        df2 = df1.head(100).sort_index()
        utils.write_json(utils.TEMP_JSON, df2.to_dict(orient="records"))
    bad_data_id_list = df[idx]["meta.data_id"].unique()
    idx = df["meta.data_id"].isin(bad_data_id_list)
    logger.debug(f"drop: {idx.mean() * 100:.1f}%")  # 2.2%
    df = df[~idx].reset_index(drop=True)
    assert df["meta.elem_id"].is_unique
    df["meta.elem_id"].value_counts().value_counts()
    df["meta.data_id"].value_counts().value_counts()

    # empty to nan
    df.replace(r"^\s*$", None, regex=True, inplace=True)
    df.info()
    df.isna().sum()[df.isna().sum() > 0]

    # ok to be empty for now
    cols = [c for c in df.columns if "copyright" in c]
    cols = ["meta.elem_body_text"] + cols
    for col1 in cols:
        df[col1].fillna("", inplace=True)

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
    utils.write_json(utils.TEMP_JSON, df.head(100).to_dict(orient="records"))
    utils.write_df2(sroot.CLEAN2_PQ, df)


def main() -> None:
    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_clean2_file()  # 668.6M, f235cb47


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
            # python -m src.crawl.klc_hj_ko.clean2
            typer.run(main)
