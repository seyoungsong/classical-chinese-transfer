import sys
from importlib import reload
from pathlib import Path
from typing import Any

import ctranslate2
import sentencepiece
import typer
from loguru import logger
from rich import pretty

import src.tool.eval as etool
import src.tool.train as ttool
import src.train.mt_h2ke.ajd.root as sroot
from src import utils


class HanjaTranslator:
    def __init__(
        self,
        model_path: str | Path = sroot.CT2_MODEL_DIR,
        device: str = "cpu",
        compute_type: str = "float32",
    ) -> None:
        # super
        super().__init__()

        # init
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.compute_type = compute_type

        # load
        self._load_model()

    def _load_model(self) -> None:
        self.model, self.sp = _load_model(
            model_path=self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

    def translate_batch(
        self,
        src_text: list[str],
        src_lang: list[str],
        tgt_lang: list[str],
        max_length: int,
        beam_size: int,
    ) -> list[str]:
        return _translate_batch(
            sp=self.sp,
            model=self.model,
            src_text=src_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=max_length,
            beam_size=beam_size,
        )


def _load_model(
    model_path: str | Path,
    device: str = "cpu",
    compute_type: str = "float32",
) -> tuple[Any, sentencepiece.SentencePieceProcessor]:
    # https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html

    # tokenizer
    model_path2 = Path(model_path).resolve()
    spm_file = model_path2 / "spm.model"
    sp = sentencepiece.SentencePieceProcessor(str(spm_file))

    # model
    model_path_str = str(model_path2)
    if "cuda:" in device:
        device_index = int(device.lower().split(":")[-1])
        model: Any = ctranslate2.Translator(
            model_path=model_path_str,
            device="cuda",
            device_index=device_index,
            compute_type=compute_type,
        )
    elif device == "cpu":
        model = ctranslate2.Translator(
            model_path=model_path_str,
            device="cpu",
            compute_type=compute_type,
        )
    else:
        raise ValueError(f"bad device: {device}")

    if 0:
        # https://opennmt.net/CTranslate2/quantization.html#implicit-type-conversion-on-load
        # nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        ctranslate2.get_supported_compute_types(device="cuda", device_index=1)
        ctranslate2.get_supported_compute_types(device="cpu")

    logger.debug(f"model: {model.device}:{model.device_index} | {model.compute_type}")
    return model, sp


def _translate_batch(
    sp: sentencepiece.SentencePieceProcessor,
    model: Any,
    src_text: list[str],
    src_lang: list[str],
    tgt_lang: list[str],
    max_length: int,
    beam_size: int,
) -> list[str]:
    # encode
    src_prefix: list[str] = [f"__{s}__".lower() for s in src_lang]
    src_encode: list[str] = [ttool.spm_encode(sp=sp, s=s) for s in src_text]
    src_input: list[str] = [f"{pre} {enc}" for pre, enc in zip(src_prefix, src_encode)]
    tgt_prefix: list[list[str]] = [[f"__{s}__".lower()] for s in tgt_lang]

    # translate
    result = model.translate_batch(
        source=[s.split() for s in src_input],
        target_prefix=tgt_prefix,
        beam_size=beam_size,
        max_input_length=max_length,
        max_decoding_length=max_length,
    )

    # result
    hypo: list[list[str]] = [r.hypotheses[0] for r in result]

    # check
    for h, tp in zip(hypo, tgt_prefix, strict=True):
        assert h[0] == tp[0]

    # decode
    pred_text = [ttool.spm_decode(sp=sp, s=" ".join(h[1:])).strip() for h in hypo]

    return pred_text


def _quick_test() -> None:
    model_path = sroot.CT2_MODEL_DIR
    spm_file = model_path / "spm.model"
    device = "cuda:4"
    device = "cpu"
    compute_type = "float32"
    max_length = 512
    beam_size = 2

    # tokenizer
    sp = sentencepiece.SentencePieceProcessor(str(spm_file.resolve()))

    # model
    model, sp = _load_model(
        model_path=model_path, device=device, compute_type=compute_type
    )

    # read
    df = utils.read_df(sroot.DATASET_PQ)
    df.sample(1).iloc[0].to_dict()

    # keep: test, short, hj-oko
    idx = (df["text.src"].str.len() <= 128) & (df["text.tgt"].str.len() <= 128)
    df1 = df[idx].reset_index(drop=True)
    df1.sample(1).iloc[0].to_dict()

    # sample
    b1 = df1.sample(2).reset_index(drop=True)
    b1 = df.groupby(["lang.src", "lang.tgt"]).sample(1).reset_index(drop=True)

    # data
    src_text: list[str] = b1["text.src"].to_list()
    src_lang: list[str] = b1["lang.src"].to_list()
    tgt_lang: list[str] = b1["lang.tgt"].to_list()
    tgt_text: list[str] = b1["text.tgt"].to_list()

    if 0:
        src_text = ["○丙申/賻權跬紙二百卷、米豆七十石。"]
        src_lang = ["hj"]
        tgt_lang = ["ko"]

    # inference
    prd_text = _translate_batch(
        sp=sp,
        model=model,
        src_text=src_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
        beam_size=beam_size,
    )

    # diff
    b1[[c for c in b1.columns if "meta.url" in c]].to_dict(orient="records")
    utils.temp_diff("\n\n".join(tgt_text), "\n\n".join(prd_text))

    # eval
    for prd1, tgt1, lang1 in zip(prd_text, tgt_text, tgt_lang):
        score1 = etool.compute_BLEU(hypo=[prd1], ref1=[tgt1], lang=lang1)
        s1 = score1["spBLEU_score"]
        logger.debug(f"\n{prd1}\n{tgt1}\n{s1}")

    # our model
    del model
    model2 = HanjaTranslator(
        model_path=model_path, device=device, compute_type=compute_type
    )
    prd_text = model2.translate_batch(
        src_text=src_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
        beam_size=beam_size,
    )


def main() -> None:
    _quick_test()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_h2ke.ajd.model
            typer.run(main)
