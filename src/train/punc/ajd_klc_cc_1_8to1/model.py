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
import src.train.punc.ajd_klc_cc_1_8to1.root as sroot
from src import utils


class HanjaPUNC:
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
        self.model, self.tokenizer, self.pipe, self.label2id = _load_model(
            model_path=self.model_path, device=device, torch_dtype=torch_dtype
        )

    def punctuate_batch(
        self, x: list[str], add_space: bool = False, return_xml: bool = False
    ) -> list[str]:
        return _punctuate_batch(
            x=x,
            add_space=add_space,
            return_xml=return_xml,
            pipe=self.pipe,
            label2id=self.label2id,
        )


def _load_model(
    model_path: Path,
    device: str,
    torch_dtype: torch.dtype,
) -> tuple[
    BertForTokenClassification,
    BertTokenizerFast,
    TokenClassificationPipeline,
    dict[str, str],
]:
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
    # pipe
    pipe: TokenClassificationPipeline = pipeline(
        task="ner", model=model, tokenizer=tokenizer
    )
    # etc
    label2id_json = model_path / "label2id.json"
    label2id: dict[str, str] = utils.read_json(label2id_json)
    return model, tokenizer, pipe, label2id


def _align_ner_result(
    x1: str, pipe_result1: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    # prep
    words = list(x1)
    preds: list[list[dict[str, Any]]] = [[] for _ in range(len(words))]

    # align
    for d1 in pipe_result1:
        idx: int = d1["end"] - 1
        preds[idx].append(d1)

    if 0:
        _x1 = "\n\n".join(words)
        _x2 = "\n\n".join([" | ".join([str(d1) for d1 in lx]) for lx in preds])
        utils.temp_diff(_x1, _x2)

    # drop duplicates by score
    for i, lx in enumerate(preds):
        if len(lx) > 1:
            lx.sort(key=lambda d1: d1["score"], reverse=True)
            preds[i] = [lx[0]]

    # labels
    labels = ["O" for _ in range(len(words))]
    for i, lx in enumerate(preds):
        if len(lx) == 1:
            labels[i] = lx[0]["entity"]
        elif len(lx) > 1:
            raise ValueError("len(lx) > 1")

    # check
    assert len(words) == len(labels), "err-318: len(words) != len(labels)"

    return words, labels


def _insert_space(s1: str, chars: str) -> str:
    s2 = ""
    for c1 in s1:
        s2 += c1
        if c1 in chars:
            s2 += " "
    return s2


def _punctuate_batch(  # noqa: C901
    x: list[str],
    add_space: bool,
    return_xml: bool,
    pipe: TokenClassificationPipeline,
    label2id: dict[str, str],
) -> list[str]:
    # check (-2 for [CLS] and [SEP])
    _lens = [len(pipe.tokenizer.encode(s, add_special_tokens=True)) for s in x]
    if max(_lens) > pipe.tokenizer.model_max_length:
        logger.warning("input length exceeds model_max_length")

    # inference
    pipe.call_count = 0
    pipe_results: list[list[dict[str, Any]]] = pipe(x)

    # convert
    label2punc = {f"B-{v}": k for k, v in label2id.items()}
    label2punc["O"] = ""

    # add_space
    if add_space:
        special_puncs = "!,:;?"
        label2punc = {
            k: _insert_space(s1=v, chars=special_puncs) for k, v in label2punc.items()
        }
        if 0:
            list(label2punc.values())
            "".join(sorted(special_puncs))

    # join
    pred: list[str] = []
    pairs = list(zip(x, pipe_results, strict=True))
    if 0:
        x1, pipe_result1 = pairs[0]
    for x1, pipe_result1 in pairs:
        words, labels = _align_ner_result(x1=x1, pipe_result1=pipe_result1)
        puncs = [label2punc[lb] for lb in labels]
        if add_space:
            puncs[-1] = puncs[-1].strip()  # to be safe
        items = []
        for w, p in zip(words, puncs, strict=True):
            items.append({"text": w, "name": None})
            if len(p) >= 1:
                items.append({"text": p, "name": "punc"})
        pred1 = etool.items2xml(items)
        pred.append(pred1)

    # convert to plaintext
    pred_plaintext = [etool.xml2plaintext(s_xml=s) for s in pred]

    # check: x_output should be x + puncs
    for x1, s2 in zip(x, pred_plaintext, strict=True):
        assert utils.is_subset_with_count(s1=x1, s2=s2), "err-318: not subset"

    # return
    if return_xml:
        out = pred
    else:
        out = pred_plaintext
    return out


def _quick_test() -> None:
    # config
    model_path = sroot.HFACE_MODEL_DIR
    device = "cuda:7" if utils.is_cuda_available() else "cpu"
    torch_dtype = torch.float16 if utils.is_cuda_available() else torch.float32
    add_space = True
    return_xml = False

    # model
    model, tokenizer, pipe, label2id = _load_model(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )
    _ = model, tokenizer, pipe, label2id

    # read
    df = utils.read_df(sroot.DATASET_PQ)
    df.sample(1).iloc[0].to_dict()

    # sample
    b1 = df.groupby("meta.corpus").sample(1).reset_index(drop=True)
    b1 = df.sample(3)
    y: list[str] = list(b1["text"])
    x = [etool.old_remove_punc(s=s) for s in y]
    if 0:
        x = ["向望瘗位于西南当瘗\ue401臽西向"]
        x = [
            "十八日總理大臣金弘集法務大臣徐光範奏同知中樞院事閔泳柱이本來無賴\uf53a流로悖類\ue470締結\uf537야京鄕人民의財産을攘奪\uf537옴이不可勝記\uf537와稔惡이旣久\uf537\ue57b怨毒이溢世\uf537오니此\ue286一國의武斷元惡이라法에在\uf537야罔赦오니法務衙門으로\uf537야곰拿囚懲辦\uf537옴이何如\uf537올지允之"
        ]

    # inference
    pred = _punctuate_batch(
        x=x, add_space=add_space, return_xml=return_xml, pipe=pipe, label2id=label2id
    )
    utils.temp_diff("\n\n".join(x), "\n\n".join(pred))
    utils.temp_diff("\n\n".join(y), "\n\n".join(pred))

    # find error
    for s in x:
        _punctuate_batch(
            x=[s],
            add_space=add_space,
            return_xml=return_xml,
            pipe=pipe,
            label2id=label2id,
        )

    # class
    model2 = HanjaPUNC(device=device)
    model2.punctuate_batch(x=x)


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
            # python -m src.train.punc.ajd_klc_cc_2to1.model
            typer.run(main)
