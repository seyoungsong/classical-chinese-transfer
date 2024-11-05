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
    from transformers import AutoTokenizer

    import src.dataset.punc.root as sroot
    from src import utils


NUM_TEST = 5000  # klc is 18K


def gen_eval_file() -> None:
    # read
    df0 = utils.read_df(sroot.FILTER_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    df.groupby(["meta.corpus", "split"]).size()

    # drop bad samples
    idx = df["text"].str.contains(f"<{utils.NER_PREF}")
    idx.sum()  # 82
    idx.mean() * 100  # 0.015
    df = df[~idx].reset_index(drop=True)

    # klc fix: test -> test0, train2+valid2+test2 -> test
    idx = df["meta.corpus"] == "klc"
    df.loc[idx, "split"].value_counts()
    rvals = {"test": "test0"}
    df.loc[idx, "split"] = df.loc[idx, "split"].replace(rvals)
    rvals = {"train2": "test", "valid2": "test", "test2": "test"}
    df.loc[idx, "split"] = df.loc[idx, "split"].replace(rvals)

    # filter: keep test only
    df = df[df["split"] == "test"].reset_index(drop=True)

    # length filtering
    tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikuroberta")
    if 0:
        x = df.sample(1).iloc[0].to_dict()
        text = x["text"]
        len(tokenizer(text, add_special_tokens=True)["input_ids"])
    df["len"] = df["text"].parallel_apply(
        lambda x: len(tokenizer(x, add_special_tokens=True)["input_ids"])
    )
    df = df[df["len"] <= 512].reset_index(drop=True)
    df.drop(columns=["len"], inplace=True)

    # random sample
    print(df.groupby(["meta.corpus", "split"]).size())
    """
    meta.corpus  split
    ajd          test     35819
    klc          test     18250
    wyweb_punc   test     30487
    """
    df1 = (
        df.groupby("meta.corpus")
        .apply(
            lambda x: x.sample(
                n=min(NUM_TEST, len(x)), random_state=42, ignore_index=True
            ),
            include_groups=False,
        )
        .reset_index(level=1, drop=True)
        .reset_index()
    )
    df1.sort_values("key2", inplace=True, ignore_index=True)
    df1.groupby(["meta.corpus", "split"]).size()

    # save
    utils.write_df2(sroot.EVAL_PQ, df1)


def main() -> None:
    # select small test subset
    gen_eval_file()  # 5.4M, 77d379ee, 15000


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
            # python -m src.dataset.punc.eval
            typer.run(main)
