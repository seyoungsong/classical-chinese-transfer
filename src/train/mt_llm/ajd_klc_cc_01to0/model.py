import sys
from importlib import reload
from pathlib import Path

import torch
import typer
from loguru import logger
from peft import PeftConfig, PeftModel, PeftModelForCausalLM  # type: ignore
from rich import pretty
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    TextStreamer,
)

import src.dataset.mt_h2ke.root
import src.tool.eval as etool
import src.train.mt_llm.ajd_klc_cc_01to0.root as sroot
from src import utils


class HanjaTranslator:
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
        self.model, self.tokenizer = _load_model(
            model_path=self.model_path, device=device, torch_dtype=torch_dtype
        )

    def translate_str(
        self,
        src_text: str,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        return _translate_str(
            src_text=src_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            model=self.model,
            tokenizer=self.tokenizer,
        )


def _load_model(
    model_path: Path, device: str, torch_dtype: torch.dtype
) -> tuple[LlamaForCausalLM, LlamaTokenizerFast]:
    # tokenizer
    tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(model_path)

    # model
    config = PeftConfig.from_pretrained(str(model_path))
    logger.debug(f"{config.base_model_name_or_path=}")
    model0: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, device_map=device, torch_dtype=torch_dtype
    )
    model: PeftModelForCausalLM = PeftModel.from_pretrained(model0, model_path)
    model.eval()

    # log
    logger.debug(f"{model.device=}")
    logger.debug(f"{model.dtype=}")
    return model, tokenizer


def _translate_str(  # noqa: C901
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizerFast,
    tgt_text: str = "",
) -> str:
    # build query
    if 0:
        src_text = "遣同知摠制尹重富, 為四使臣冬至宴宣慰使。"
        tgt_text = "The King sent out Second Commander General 同知摠制 Yun Jungbu 尹重富 as commissioner for refreshing the four envoys 宣慰使 at the solstice banquet."
        src_lang = "hj"
        tgt_lang = "en"

    src_lang2 = utils.LANG_CODE[src_lang]
    tgt_lang2 = utils.LANG_CODE[tgt_lang]
    user_content = f"Translate the following text from {src_lang2} into {tgt_lang2}.\n{src_lang2}: {src_text}\n{tgt_lang2}: "

    # apply_chat_template
    messages = [{"role": "user", "content": user_content}]
    if 0:
        print(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    # check
    if inputs.input_ids.shape[-1] > 2048:
        logger.warning("input length exceeds model_max_length")

    # inference
    outputs = model.generate(
        **inputs,
        streamer=TextStreamer(tokenizer=tokenizer),
        max_new_tokens=2048,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    # decode
    outputs_truncated = outputs[:, inputs.input_ids.shape[1] :]
    pred: str = tokenizer.batch_decode(outputs_truncated, skip_special_tokens=True)[0]

    # check
    if len(tgt_text) > 0:
        utils.temp_diff(tgt_text, pred)

    return pred


def _quick_test() -> None:
    # config
    model_path = sroot.HFACE_MODEL_DIR
    device = "cuda:7" if utils.is_cuda_available() else "cpu"
    torch_dtype = torch.float16 if utils.is_cuda_available() else torch.float32

    # model
    model, tokenizer = _load_model(
        model_path=model_path, device=device, torch_dtype=torch_dtype
    )
    _ = model, tokenizer

    # read
    df = utils.read_df(src.dataset.mt_h2ke.root.FILTER_PQ)
    df = df[df["split"] == "test"].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # filter
    idx = (df["lang.src"] == "hj") & (df["lang.tgt"] == "en")
    idx = df["meta.corpus"] == "klc"
    df1 = df[idx].reset_index(drop=True)

    # sample
    x1 = df1.sample(1).iloc[0].to_dict()
    src_text = x1["text.src"]
    tgt_text = x1["text.tgt"]
    src_lang = x1["lang.src"]
    tgt_lang = x1["lang.tgt"]
    logger.debug(f"\n{src_text=}\n\n{tgt_text=}\n\n{src_lang=}\n\n{tgt_lang=}")

    # inference
    pred = _translate_str(
        src_text=src_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        model=model,
        tokenizer=tokenizer,
    )
    utils.temp_diff(tgt_text, pred)

    # class
    model2 = HanjaTranslator(device=device)
    model2.translate_str(src_text=src_text, src_lang=src_lang, tgt_lang=tgt_lang)


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
            # python -m src.train.mt_llm.ajd_klc_cc_01to0.model
            typer.run(main)
