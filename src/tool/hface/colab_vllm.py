# type: ignore
import textwrap

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# login
"""
from google.colab import userdata
HFACE = userdata.get('HFACE')
!pip install --no-deps huggingface_hub vllm
!huggingface-cli login --token "$HFACE"
!huggingface-cli whoami
"""


"""
Unbabel/TowerInstruct-7B-v0.2
anonymous/TowerInstruct-7B-v0.2-CC-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-CC-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-NoAug-AWQ
"""
model_id = "Unbabel/TowerInstruct-7B-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = LLM(
    model=model_id,
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="float16",
    quantization="awq",
    seed=42,
    gpu_memory_utilization=0.8,
)


input_text = """
Translate the following text from Hanja into Korean.
Hanja: 翰林風格出諸賢,寧落塵囂負百年。 憶着松篁陶栗里,愧隨花柳杜樊川。
Korean:
""".strip()


prompt = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": input_text.strip() + " "}],
    tokenize=False,
    add_generation_prompt=True,
)
print(prompt, flush=True)
outputs = llm.generate(
    prompts=prompt,
    sampling_params=SamplingParams(
        temperature=0.8,
        top_p=0.95,
        seed=42,
        max_tokens=512,
    ),
)
output_text = outputs[0].outputs[0].text
print(f"\n{textwrap.fill(output_text, width=60)}", flush=True)
