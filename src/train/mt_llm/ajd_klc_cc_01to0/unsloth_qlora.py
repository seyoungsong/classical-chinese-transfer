if 1:
    import os
    import platform

    IS_A100 = platform.node() == "anonymous"
    CUDA_DEFAULT_DEVICE = "0" if IS_A100 else "6"
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        print(f"Setting CUDA_VISIBLE_DEVICES={CUDA_DEFAULT_DEVICE}")
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEFAULT_DEVICE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import json
    import sys
    from pathlib import Path
    from typing import Any

    import torch
    import typer
    from datasets import disable_caching, load_dataset
    from loguru import logger
    from peft import PeftModelForCausalLM  # type: ignore
    from rich import pretty
    from transformers import (
        AutoTokenizer,
        LlamaTokenizerFast,
        TextStreamer,
        TrainingArguments,
    )
    from trl import SFTTrainer

    try:
        from unsloth import FastLanguageModel
    except Exception as e:
        logger.warning(f"Failed to import unsloth: {repr(e)}")


try:
    IS_QLORA  # type: ignore
except NameError:
    IS_QLORA: bool = True


try:
    QUICK_TEST  # type: ignore
except NameError:
    QUICK_TEST: bool = False
QUICK_NUM_SHARDS = 500


# model
MODEL_NAME = (
    "anonymous/TowerInstruct-7B-v0.2-bnb-4bit"
    if IS_QLORA
    else "Unbabel/TowerInstruct-7B-v0.2"
)
IS_BF16_SUPPORTED: bool = (
    torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # type: ignore
)
assert isinstance(IS_BF16_SUPPORTED, bool), "is_bf16_supported is not bool!"
DTYPE = torch.bfloat16 if IS_BF16_SUPPORTED else torch.float16
LOAD_IN_4BIT = True if IS_QLORA else False
MAX_SEQ_LENGTH = 2048
SEED = 42

# lora
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
USE_RSLORA = True

# dataset
DATASET_NUM_PROC = int(round(os.cpu_count() * 0.9))  # type: ignore

# training
EFFECTIVE_BATCH_SIZE = 32
LEARNING_RATE = 2e-4
LOGGING_STEPS = 1
MAX_GRAD_NORM = 0.3
NUM_TRAIN_EPOCHS = 1
OPTIM = "adamw_bnb_8bit" if IS_QLORA else "adamw_torch"
PER_DEVICE_TRAIN_BATCH_SIZE = 4 if IS_QLORA else 32
SAVE_STEPS = 0.5  # 2 times
SAVE_TOTAL_LIMIT = 2
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# training2
GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // PER_DEVICE_TRAIN_BATCH_SIZE
IS_BF16 = DTYPE == torch.bfloat16
IS_FP16 = DTYPE == torch.float16

# dir
HFACE_INFO_JSON = Path("./hface_info.json").resolve()
HFACE_INFO: dict[str, str] = json.loads(HFACE_INFO_JSON.read_text(encoding="utf-8"))
HFACE_TRAIN_DIR = Path(HFACE_INFO["HFACE_TRAIN_DIR"]).resolve()
HFACE_LORA_DIR = Path(HFACE_INFO["HFACE_LORA_DIR"]).resolve()
DATASET_FILE = Path(HFACE_INFO["DATASET_FILE"]).resolve()


def run_train() -> None:  # noqa: C901
    # log
    curr_dir = Path.cwd().resolve()
    logger.debug(f"{curr_dir=}")

    # check
    assert Path(HFACE_TRAIN_DIR).parent.is_dir(), "HFACE_TRAIN_DIR parent dir!"
    assert Path(HFACE_LORA_DIR).parent.is_dir(), "HFACE_LORA_DIR parent dir!"

    # load
    model: PeftModelForCausalLM
    tokenizer: LlamaTokenizerFast
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    logger.debug(f"{model.config=}")

    # convert
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=SEED,
        use_rslora=USE_RSLORA,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    # dataset
    logger.debug(f"Loading dataset from {DATASET_FILE}")
    disable_caching()
    all_dataset = load_dataset(
        "parquet",
        data_files={"simple": str(DATASET_FILE)},
        split="simple",
        download_mode="force_redownload",
        verification_mode="no_checks",
    )
    all_dataset.unique("split")
    if QUICK_TEST:
        logger.debug(f"Quick test mode: shuffling and taking {QUICK_NUM_SHARDS} shards")
        all_dataset = all_dataset.shard(num_shards=QUICK_NUM_SHARDS, index=0)
    train_dataset = all_dataset.filter(
        lambda x: x["split"] == "train", num_proc=DATASET_NUM_PROC
    )
    logger.debug(f"{len(train_dataset)=}")

    # prompt
    def to_messages(conversation: list[dict[str, str]]) -> list[dict[str, str]]:
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
        key_mapping = {"from": "role", "value": "content"}
        role_mapping = {"human": "user", "gpt": "assistant"}
        if 0:
            d = conversation[0]
            d = {key_mapping[k]: v for k, v in d.items()}
        c1 = [{key_mapping[k]: v for k, v in d.items()} for d in conversation]
        c2 = [
            {k: role_mapping[v] if k == "role" else v for k, v in d.items()} for d in c1
        ]
        return c2

    def formatting_prompts_func(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        conversations = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                to_messages(conversation), tokenize=False, add_generation_prompt=False
            )
            for conversation in conversations
        ]
        return {"text": texts}

    # test
    if 0:
        # load
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.chat_template

        # sample
        examples = train_dataset.shuffle()[:3]
        sorted(examples.keys())
        conversations = examples["conversations"]
        conversation = conversations[0]
        text = tokenizer.apply_chat_template(
            to_messages(conversation), tokenize=False, add_generation_prompt=False
        )
        text
        logger.debug(text)
        formatting_prompts_func(examples)

    # map
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
        load_from_cache_file=False,
        num_proc=DATASET_NUM_PROC,
    )

    # config
    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        bf16_full_eval=IS_BF16,
        bf16=IS_BF16,
        data_seed=SEED,
        evaluation_strategy="no",
        fp16_full_eval=IS_FP16,
        fp16=IS_FP16,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        lr_scheduler_type="linear",
        max_grad_norm=MAX_GRAD_NORM,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        optim=OPTIM,
        output_dir=HFACE_TRAIN_DIR,
        overwrite_output_dir=True,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        report_to="tensorboard",
        save_safetensors=True,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed=SEED,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
    )
    # https://huggingface.co/docs/trl/sft_trainer#trl.SFTTrainer
    trainer = SFTTrainer(
        args=training_args,
        dataset_num_proc=DATASET_NUM_PROC,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        model=model,
        packing=False,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )

    # train
    trainer.train()

    # save
    model.save_pretrained_merged(
        save_directory=HFACE_LORA_DIR, tokenizer=tokenizer, save_method="lora"
    )
    logger.success(f"Saved to {HFACE_LORA_DIR}")

    if 0:
        FastLanguageModel.for_inference(model)
        q1 = "사과에 대해서 설명해줘."
        q1 = "Translate the following text from Hanja into Korean.\nHanja: 庚戌/上幸昌德宮。 王世孫隨駕, 行禮璿源殿。\nKorean: "
        # '임금이 창덕궁에 거둥하였다. 왕세손이 수가(隨駕)하여 선원전(璿源殿)에서 예를 행하였다.'
        messages = [{"role": "user", "content": q1}]
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        _ = model.generate(
            **inputs,
            streamer=TextStreamer(tokenizer),
            max_new_tokens=2048,
            pad_token_id=tokenizer.eos_token_id,
        )


def main(quick: bool = False, method: str = "qlora") -> None:
    global QUICK_TEST
    QUICK_TEST = quick
    global IS_QLORA
    IS_QLORA = method == "qlora"
    run_train()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # clear && CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n unsloth_env python ./src/train/mt_llm/ajd_klc/unsloth_qlora.py --quick
            typer.run(main)
