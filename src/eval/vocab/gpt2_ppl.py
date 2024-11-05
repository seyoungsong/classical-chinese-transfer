if 1:
    import os

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import sys
    from importlib import reload

    import evaluate
    import torch
    import typer
    from loguru import logger
    from pandarallel import pandarallel
    from rich import pretty
    from tqdm import tqdm
    from transformers import AutoTokenizer

    import src.eval.vocab.root as sroot
    from src import utils


MODEL_ID = "JeffreyLau/SikuGPT2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 if torch.cuda.is_available() else 1
SAMPLE_SIZE = 100
MAX_LENGTH = 510


try:
    _ = PERPLEXITY  # type: ignore
except NameError:
    PERPLEXITY = evaluate.load("perplexity", module_type="metric")


def compute_ppl(preds: list[str]) -> list[float]:
    if 0:
        return [0.1] * len(preds)
    if 0:
        preds = ["朝鮮封建王朝實錄"]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.bos_token_id
        tokenizer.model_max_length
    results = PERPLEXITY.compute(
        predictions=[s[:MAX_LENGTH] for s in preds],
        model_id=MODEL_ID,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        add_start_token=False,
    )
    ppl: list[float] = results["perplexities"]
    return ppl


def gen_sample_file() -> None:
    # read file
    df = utils.read_df(sroot.DATASET_PQ)
    df.sample(1).iloc[0].to_dict()

    # sample for each corpus and lang
    cols = ["meta.corpus", "lang"]
    df.groupby(cols).size()
    df1 = (
        df.groupby(cols)
        .apply(lambda x: x.sample(n=SAMPLE_SIZE, random_state=42))
        .reset_index(drop=True)
    )
    df1.groupby(cols).size()

    # limit max length
    df1["text"] = df1["text"].progress_apply(lambda x: x[:MAX_LENGTH])

    # save
    sroot.SAMPLE_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.SAMPLE_PQ, df1)


def gen_ppl_sample_file() -> None:
    # read
    ds = utils.read_ds(sroot.SAMPLE_PQ)
    ds.shuffle()[0]

    # test
    b1 = ds.shuffle()[:4]
    compute_ppl(b1["text"])

    # shuffle
    ds = ds.shuffle(seed=42)

    # apply
    ds = ds.map(
        lambda x: {"ppl": compute_ppl(x["text"])},
        batched=True,
        batch_size=BATCH_SIZE,
        load_from_cache_file=False,
        num_proc=1,
    )
    ds.shuffle()[0]

    # sort
    ds = ds.sort("key2")

    # save
    sroot.PPL_SAMPLE_PQ.parent.mkdir(exist_ok=True, parents=True)
    ds.to_parquet(sroot.PPL_SAMPLE_PQ, compression="gzip")
    utils.log_written(sroot.PPL_SAMPLE_PQ)


def gen_mean_ppl_file() -> None:
    # read
    df = utils.read_df(sroot.PPL_SAMPLE_PQ)
    df.sample(1).iloc[0].to_dict()

    # compute
    df1 = (
        df.groupby(["meta.corpus", "lang"])["ppl"]
        .agg(["mean", "std", "size"])
        .reset_index()
    )
    cols = ["mean", "std"]
    for col in cols:
        df1[col] = df1[col].apply(lambda x: f"{x:.1f}")

    # save
    fname = sroot.RESULT_DIR / "mean_ppl.tsv"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(fname, df1)


def main() -> None:
    gen_sample_file()  # 278.4K, f18d189b, 700
    gen_ppl_sample_file()  # 239.5K, 5eb96629
    gen_mean_ppl_file()  # 1.1K, 6f3b3b7d


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
            # python -m src.eval.vocab.gpt2_ppl
            typer.run(main)
