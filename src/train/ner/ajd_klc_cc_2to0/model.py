import sys
from importlib import reload
from pathlib import Path
from typing import Any

import torch
import typer
from loguru import logger
from rich import pretty
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForTokenClassification,
    BertTokenizerFast,
    TokenClassificationPipeline,
    pipeline,
)

import src.tool.eval as etool
import src.train.ner.ajd_klc_cc_2to0.root as sroot
from src import utils


class HanjaNER:
    def __init__(
        self,
        model_path: str | Path = sroot.HFACE_MODEL_DIR,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        # super
        super().__init__()

        # init
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.torch_dtype = torch_dtype

        # model
        self.model, self.tokenizer, self.pipe = _load_model(
            model_path=self.model_path, device=self.device, torch_dtype=self.torch_dtype
        )

    def predict_batch(self, x: list[str]) -> list[str]:
        return _predict_batch(x=x, pipe=self.pipe)


def _load_model(
    model_path: Path, device: str, torch_dtype: torch.dtype
) -> tuple[BertForTokenClassification, BertTokenizerFast, TokenClassificationPipeline]:
    # load
    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
        model_path, model_max_length=512
    )
    model: BertForTokenClassification = AutoModelForTokenClassification.from_pretrained(
        model_path, device_map=device, torch_dtype=torch_dtype
    )
    model.eval()
    # log
    logger.debug(f"{model.device=}")
    logger.debug(f"{model.dtype=}")
    logger.debug(f"{model.num_labels=}")
    logger.debug(f"{model.config=}")
    # pipe
    pipe: TokenClassificationPipeline = pipeline(
        task="ner", model=model, tokenizer=tokenizer
    )
    return model, tokenizer, pipe


def _convert_to_ner_tags(x1: str, pipe_result1: list[dict[str, Any]]) -> list[str]:
    ner_tags = ["O"] * len(x1)
    for p in pipe_result1:
        start, end = p["start"], p["end"]
        # For 'B-' prefix, only the start position gets this tag
        ner_tags[start] = p["entity"]
        # For 'I-' prefix, all positions after the start till the end get this tag
        for i in range(start + 1, end):
            ner_tags[i] = "I" + p["entity"][1:]
    return ner_tags


def _predict_batch(x: list[str], pipe: TokenClassificationPipeline) -> list[str]:
    # inference
    pipe.call_count = 0
    pipe_results: list[list[dict[str, Any]]] = pipe(x)

    # convert
    pred: list[str] = []
    for x1, pipe_result1 in zip(x, pipe_results, strict=True):
        ner_tags1 = _convert_to_ner_tags(x1=x1, pipe_result1=pipe_result1)
        assert len(x1) == len(ner_tags1), "len mismatch at convert_to_ner_tags"
        pred1 = etool.iob2xml(tokens=list(x1), ner_tags=ner_tags1)
        pred.append(pred1)
    return pred


def _quick_test() -> None:
    # config
    model_path = sroot.HFACE_MODEL_DIR
    device = "cuda:7" if utils.is_cuda_available() else "cpu"
    torch_dtype = torch.float16 if utils.is_cuda_available() else torch.float32

    # model
    model, tokenizer, pipe = _load_model(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )
    _ = model, tokenizer, pipe

    # read
    df = utils.read_df(sroot.DATASET_PQ)
    df.sample(1).iloc[0].to_dict()

    # sample
    b1 = df.groupby("meta.corpus").sample(1).reset_index(drop=True)
    b1 = df.sample(3)
    x: list[str] = b1["text_xml"].apply(etool.xml2plaintext).to_list()
    y: list[str] = b1["text_xml"].to_list()

    # inference
    pred = _predict_batch(x=x, pipe=pipe)
    utils.temp_diff("\n\n".join(y), "\n\n".join(pred))

    # find error
    for s in x:
        _predict_batch(x=[s], pipe=pipe)

    # class
    model2 = HanjaNER(device=device)
    model2.predict_batch(x=x)


def main() -> None:
    _quick_test()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(etool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.ner.ajd_klc_cc_2to1.model
            typer.run(main)
