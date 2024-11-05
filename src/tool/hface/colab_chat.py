import textwrap

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    TextStreamer,
)

"""
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
!pip install "transformers>=4.40.0" huggingface_hub accelerate autoawq bitsandbytes
!huggingface-cli login --token "$HF_TOKEN"
!huggingface-cli whoami
"""


LANG_CODE = {
    "cc": "Classical Chinese",
    "en": "English",
    "hj": "Hanja",
    "ko": "Korean",
    "lzh": "Classical Chinese",
    "zh": "Modern Chinese",
}


def asdf() -> None:
    """
    Unbabel/TowerInstruct-7B-v0.2
    anonymous/TowerInstruct-7B-v0.2-CC-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-CC-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-NoAug-AWQ
    CohereForAI/c4ai-command-r-v01-4bit
    """

    # Load model
    model_id = "CohereForAI/c4ai-command-r-v01-4bit"
    tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(
        model_id, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda:0"
    )
    _ = model.eval()

    # Input
    src_text = "遣同知摠制尹重富, 為四使臣冬至宴宣慰使。"
    src_lang = "hj"
    tgt_lang = "ko"

    # Prepare
    src_lang2 = LANG_CODE[src_lang]
    tgt_lang2 = LANG_CODE[tgt_lang]
    user_content = f"Translate the following text from {src_lang2} into {tgt_lang2}.\n{src_lang2}: {src_text}\n{tgt_lang2}: "
    messages = [{"role": "user", "content": user_content}]
    print(f"{user_content=}")

    # Inference
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    # Decode
    outputs_truncated = outputs[:, inputs.input_ids.shape[1] :]
    pred: str = tokenizer.batch_decode(outputs_truncated, skip_special_tokens=True)[0]

    # Print
    input_wrap = textwrap.fill(src_text, width=80)
    pred_wrap = textwrap.fill(pred, width=80)
    print(f"\nInput:\n{input_wrap}\n")
    print(f"Output:\n{pred_wrap}")
