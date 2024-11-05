# type: ignore


"""
import torch
major_version, minor_version = torch.cuda.get_device_capability()
print(f'{major_version=}, {minor_version=}')
# https://github.com/casper-hansen/AutoAWQ/#build-from-source
!pip install autoawq huggingface_hub transformers accelerate

from google.colab import userdata
HFACE = userdata.get('HFACE')
!huggingface-cli login --token "$HFACE"
!huggingface-cli whoami
"""


import shutil

from awq import AutoAWQForCausalLM
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm
from transformers import AutoTokenizer, AwqConfig


def push_to_hub_awq(input_model: str) -> None:
    # names
    INPUT__MODEL = input_model
    OUTPUT_MODEL = f"{INPUT__MODEL}-AWQ"
    URL = f"https://huggingface.co/{OUTPUT_MODEL}/settings"

    # log
    print(f"{INPUT__MODEL=}")
    print(f"{OUTPUT_MODEL=}")
    print(f"{URL=}")

    # etc
    OUTPUT_DIR = OUTPUT_MODEL.lower().split("/")[-1].replace(".", "-")
    QUANT_CONFIG = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    # modify the config file so that it is compatible with transformers integration
    hface_quantization_config = AwqConfig(
        bits=QUANT_CONFIG["w_bit"],
        group_size=QUANT_CONFIG["q_group_size"],
        zero_point=QUANT_CONFIG["zero_point"],
        version=QUANT_CONFIG["version"].lower(),
    ).to_dict()

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        INPUT__MODEL,
        **{
            "low_cpu_mem_usage": True,
            "use_cache": False,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        INPUT__MODEL,
        trust_remote_code=True,
    )

    # Quantize
    model.quantize(
        tokenizer,
        quant_config=QUANT_CONFIG,
    )

    # the pretrained transformers model is stored in the model attribute + we need to pass a dict
    model.model.config.quantization_config = hface_quantization_config

    # Save quantized model
    model.save_quantized(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    del model, tokenizer
    print(f'Model is quantized and saved at "{OUTPUT_DIR=}"')

    # Push to hub
    api = HfApi()

    # Create empty repo
    create_repo(
        repo_id=OUTPUT_MODEL,
        repo_type="model",
        exist_ok=True,
        private=True,
    )

    # Upload files
    print(f"{URL=}")
    api.upload_folder(
        repo_id=OUTPUT_MODEL,
        repo_type="model",
        folder_path=OUTPUT_DIR,
    )

    # cleanup
    shutil.rmtree(OUTPUT_DIR)


def batch_run() -> None:
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
    models = [s.replace("-QLoRA", "") for s in models]
    if 0:
        input_model = models[0]
    for input_model in tqdm(models):
        push_to_hub_awq(input_model=input_model)


batch_run()
