# download mambaforge | https://github.com/conda-forge/miniforge/releases/latest
## macos
wget -O Mambaforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-$(uname -m).sh"
## linux
wget -O Mambaforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-$(uname -m).sh"

# install mambaforge
bash Mambaforge.sh -b -p "${HOME}/mambaforge"
source "${HOME}/mambaforge/etc/profile.d/conda.sh"
conda activate
mamba init zsh
# mamba init zsh --reverse --dry-run

# check
code ~/.zshrc
code ~/.zprofile

# check version
python -c "import sys; print(sys.executable)"
python -VV

# mambaforge
mamba update --all -y

# create env
mamba env list
# mamba deactivate
# mamba env remove --name mmm
mamba create --name mmm python=3.10 -y

# install poetry
mamba activate mmm
pip install -U poetry pip setuptools

# ubuntu: poetry keyring fix
echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' | tee -a ~/.zprofile ~/.zshrc
poetry env info

# poetry: update deps
poetry check
task poetry-lock

# torch
pip install -v "torch==2.1.1" fairseq

# macos-arm
mamba install -c=apple tensorflow-deps -y
task pipma
# linux-cpu
task piplc
# linux-gpu
# mamba install cudatoolkit cudnn -y
task piplg

# cuda11.8: overwrite torch
pip install -v --upgrade --force-reinstall "torch==2.1.1" "xformers==0.0.23" --index-url https://download.pytorch.org/whl/cu118
pip check
pip install -v "fsspec==2023.10.0" "typing-extensions==4.9.0" "numpy==1.26.2" "filelock==3.13.1"

# apex
mkdir -p ~/tmp
git clone --depth=1 https://github.com/NVIDIA/apex ~/tmp/apex
cd ~/tmp/apex
MAX_JOBS=8 pip install -v \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    --config-settings "--build-option=--deprecated_fused_adam" \
    --config-settings "--build-option=--xentropy" \
    --config-settings "--build-option=--fast_multihead_attn" \
    ./

# FlashAttention-2
MAX_JOBS=8 pip install -v flash-attn --no-build-isolation

# check
pip check
pip freeze >./guides/requirements/pip.log

# ~/.cache/huggingface
# https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
PARENT_DIR=$(dirname "$(pwd)")
echo "PARENT_DIR=$PARENT_DIR"
HF_CACHE="${PARENT_DIR}/hf_cache"
echo "HF_CACHE=$HF_CACHE"
sudo mkdir -p "$HF_CACHE"
sudo chown -R $(whoami) "$HF_CACHE"
echo 'export HF_HOME="'$HF_CACHE'"' | tee -a ~/.bashrc ~/.zshrc
tail ~/.zshrc

# unbabel-comet
mamba create --name comet python=3.10 -y
conda activate comet
conda run --no-capture-output -n comet pip install -U pip setuptools
conda run --no-capture-output -n comet pip install -U unbabel-comet
conda run --no-capture-output -n comet comet-score --help
#
tmux new -s comet
echo -e "10 到 15 分钟可以送到吗\nPode ser entregue dentro de 10 a 15 minutos?" >>/root/tmp/src.txt
echo -e "Can I receive my food in 10 to 15 minutes?\nCan it be delivered in 10 to 15 minutes?" >>/root/tmp/hyp1.txt
echo -e "Can it be delivered between 10 to 15 minutes?\nCan it be delivered between 10 to 15 minutes?" >>/root/tmp/ref.txt
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n comet comet-score -s /root/tmp/src.txt -t /root/tmp/hyp1.txt -r /root/tmp/ref.txt --gpus 1 --model Unbabel/wmt22-comet-da
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n comet comet-score -s /root/tmp/src.txt -t /root/tmp/hyp1.txt --gpus 1 --model Unbabel/wmt22-cometkiwi-da

# clean
mamba clean --all -y
mamba update --all -y

# llama-cpp-python for GGUF
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DLLAMA_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python==0.2.55
