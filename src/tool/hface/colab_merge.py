# type: ignore
import torch
from unsloth import FastLanguageModel

"""
import torch
major_version, minor_version = torch.cuda.get_device_capability()
# Must install separately since Colab has torch 2.2.1, which breaks packages
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers==0.0.25.post1" trl peft accelerate bitsandbytes huggingface_hub

from google.colab import userdata
HFACE = userdata.get('HFACE')
!huggingface-cli login --token "$HFACE"
!huggingface-cli whoami
!huggingface-cli download anonymous/TowerInstruct-7B-v0.2-bnb-4bit
"""


def push_to_hub_merged(input_model: str):
    INPUT__MODEL = input_model
    OUTPUT_MODEL = INPUT__MODEL.replace("-QLoRA", "")

    # token
    HFACE_TOKEN = "mytoken"

    # log
    print(f"{INPUT__MODEL=}")
    print(f"{OUTPUT_MODEL=}")

    # load
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=INPUT__MODEL,
        max_seq_length=2048,
        dtype=torch.float16,
        token=HFACE_TOKEN,
    )

    # push
    URL = f"https://huggingface.co/{OUTPUT_MODEL}/settings"
    print(f"{URL=}")
    model.push_to_hub_merged(
        repo_id=OUTPUT_MODEL,
        tokenizer=tokenizer,
        save_method="merged_16bit",
        token=HFACE_TOKEN,
        private=True,
    )
    del model, tokenizer
    print("success!")


def batch_run() -> None:
    # names
    models = """
    #anonymous/TowerInstruct-7B-v0.2-CC-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-KLC-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-CC-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-KLC-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-NoAug-QLoRA
    #
    #anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-01to0-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-01to1-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-05to0-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-05to1-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1to0-QLoRA
    #anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1to1-QLoRA
    #
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_16to0-QLoRA
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_16to1-QLoRA
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_32to0-QLoRA
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_32to1-QLoRA
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_4to0-QLoRA
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_4to1-QLoRA
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_8to0-QLoRA
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_8to1-QLoRA
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-2to0-QLoRA
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-2to1-QLoRA
    """
    models = models.strip().split("\n")
    models = [s.strip() for s in models]
    models = [s for s in models if s]
    models = [s for s in models if "#" not in s]
    for input_model in models:
        push_to_hub_merged(input_model=input_model)


batch_run()
