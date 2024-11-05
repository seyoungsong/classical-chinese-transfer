# type: ignore


"""
import torch
major_version, minor_version = torch.cuda.get_device_capability()
print(f'{major_version=}, {minor_version=}')

# https://github.com/casper-hansen/AutoAWQ/#build-from-source
!pip install --no-deps huggingface_hub transformers peft accelerate bitsandbytes tqdm psutil numpy
!huggingface-cli login --token mytoken
"""


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)

INPUT__MODEL_ID = "Unbabel/TowerInstruct-7B-v0.2"
OUTPUT_MODEL_ID = "anonymous/TowerInstruct-7B-v0.2-bnb-4bit"


if 1:
    # log
    url = f"https://huggingface.co/{OUTPUT_MODEL_ID}"
    print(f"URL: [ {url} ]")

    # load
    print("Loading model and tokenizer")
    tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(INPUT__MODEL_ID)
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        INPUT__MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    )

    # push
    print("Pushing tokenizer to hub")
    tokenizer.push_to_hub(OUTPUT_MODEL_ID, private=True)

    # push
    print("Pushing model to hub")
    model.push_to_hub(OUTPUT_MODEL_ID, private=True)

    # log
    print("Pushed to hub")
    print(f"URL: [ {url} ]")
