if 1:
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import sys
    from importlib import reload

    import typer
    from loguru import logger
    from pandarallel import pandarallel
    from rich import pretty
    from tqdm import tqdm
    from transformers import AutoTokenizer, BertTokenizerFast

    import src.eval.vocab.root as sroot
    from src import utils


MODEL_NAME = "SIKU-BERT/sikuroberta"


def _count(s: str, tokenizer: BertTokenizerFast) -> tuple[int, int]:
    if 0:
        s = "간"
        s = "갺"
    ids = tokenizer.encode(s, add_special_tokens=False)
    if 0:
        tokenizer.convert_ids_to_tokens(ids)
    num_token = len(ids)
    num_unks = ids.count(tokenizer.unk_token_id)
    return num_token, num_unks


def gen_unk_ratio_tsv() -> None:
    # read file
    df = utils.read_df(sroot.CHAR_COUNT_PQ)
    df.sample(1).iloc[0].to_dict()

    # load
    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME)

    # count
    temp1 = df["char"].progress_apply(lambda x: _count(x, tokenizer))
    df["num_token"] = df["count"] * temp1.apply(lambda x: x[0])
    df["num_unks"] = df["count"] * temp1.apply(lambda x: x[1])
    df.sample(1).iloc[0].to_dict()

    # merge cc
    {k: k for k in df["meta.corpus"].unique()}
    rval = {"dai_cc": "zcc", "niu_mt": "zcc", "wyweb_mt": "zcc"}
    df["meta.corpus"] = df["meta.corpus"].replace(rval)
    #
    scols = ["num_token", "num_unks"]
    df = (
        df.groupby(["meta.corpus", "lang", "char"])
        .agg({c: "sum" for c in scols})
        .reset_index()
    )
    cols = ["meta.corpus", "lang", "char", "num_token"]
    df.sort_values(by=cols, inplace=True, ignore_index=True)

    # ratio
    df1 = (
        df.groupby(["meta.corpus", "lang"]).agg({c: "sum" for c in scols}).reset_index()
    )
    df1["percent_unk"] = df1["num_unks"] / df1["num_token"] * 100
    fname = sroot.RESULT_DIR / "unk_ratio.tsv"
    utils.write_df(fname, df1)

    # merge hj
    {k: k for k in df1["meta.corpus"].unique()}
    rval = {"ajd": "ajd_klc", "klc": "ajd_klc"}
    df1["meta.corpus"] = df1["meta.corpus"].replace(rval)

    # ratio
    df1 = df1.groupby(["meta.corpus"]).agg({c: "sum" for c in scols}).reset_index()
    df1["percent_unk"] = df1["num_unks"] / df1["num_token"] * 100
    fname = sroot.RESULT_DIR / "unk_ratio2.tsv"
    utils.write_df(fname, df1)


def main() -> None:
    gen_unk_ratio_tsv()


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
            # python -m src.eval.vocab.bert_unk
            typer.run(main)
