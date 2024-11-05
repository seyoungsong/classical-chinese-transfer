# https://github.com/unslothai/unsloth/#conda-installation
# https://github.com/unslothai/unsloth/issues/73

# env
mamba create --name unsloth_env python=3.10 -y
conda activate unsloth_env

# tmux
tmux new -s unsloth_env
tmux attach -t unsloth_env

# install
conda activate unsloth_env
# mamba install -y pytorch==2.2.0 cudatoolkit torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# mamba install -y xformers -c xformers
mamba install -y pytorch-cuda=12.1 pytorch==2.2.0 cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --upgrade rich typer loguru tensorboard trl peft accelerate bitsandbytes

# check
which -a nvcc
/usr/local/cuda/bin/nvcc --version
conda run --no-capture-output -n unsloth_env python -m xformers.info
conda run --no-capture-output -n unsloth_env python -m bitsandbytes

# update
pip install --upgrade --force-reinstall --no-cache-dir "unsloth @ git+https://github.com/unslothai/unsloth.git"

# check
conda activate unsloth_env
pip check
pip freeze >./guides/requirements/pip_unsloth_env.log && code ./guides/requirements/pip_unsloth_env.log
