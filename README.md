# 🎨 DivRL: Disentangled Self-Similarity Rewards for Diverse Subject-Driven Generation (ECCV 2026)
This is the official implementation of paper **DivRL: Disentangled Self-Similarity Rewards for Diverse Subject-Driven Generation**
[![arXiv](https://img.shields.io/badge/arXiv-<2606.23950>-<COLOR>.svg)](https://arxiv.org/abs/2606.23950)
:rocket: [Project page](https://qianwangx.github.io/DivRL/)

We propose a post-training RL framework that jointly optimizes identity consistency and structural diversity simultaneously by leveraging disentangled visual features from a robust similarity model MTG.

## 1. Environment setup
First set up the environment for DivRL:
```
git clone https://github.com/qianwangx/DivRL
conda create -n DivRL python=3.10
conda activate DivRL
pip install -e .
```
Then, set up the environment for reward model MTG. We suggest keeping MTG repo outside DivRL repo.
Suggested layout:
```
  ~/workspace/
  ├── DivRL/          # RL training code
  └── mind-the-glitch/  # reward server (cloned separately)
```

```
git clone https://github.com/abdo-eldesokey/mind-the-glitch
cd mind-the-glitch
```
Please create a separate conda environment following the instructions in the MTG repo to install the packages. Afterwards, please copy the reward server wrapper scripts under `mtg/` to `mind-the-glitch/`.

Alternatively, you can use docker to launch the environment setup. 
```
git clone https://github.com/qianwangx/DivRL
git clone https://github.com/abdo-eldesokey/mind-the-glitch
# Launch the environment
docker run --gpus all -it \
    -v $(pwd)/DivRL:/app/DivRL \
    -v $(pwd)/mind-the-glitch:/app/mind-the-glitch \
    DivRL bash

# Inside the container, activate whichever env needed
rl      # for training
reward  # for reward server
```

## 2. Dataset preparation
The dataset is stored in the HF repo [QWW/Syncd_filtered](https://huggingface.co/datasets/QWW/Syncd_filtered). The dataset can be automatically downloaded once the training starts. 

## 3. Two-stage Training
We provide the pretrained LoRA weights at [QWW/DivRL](https://huggingface.co/QWW/DivRL). Stage 1 is training with nSSM only; Stage 2 is training with nSSM as the main objective, while VSM serves as a consistency gate.

We report the training setup on 8A100 80GB GPUs. We use the first GPU to hold the remote servers, and rest of the sever GPUs are for RL training. 
We first launch the reward server:
```
cd mind-the-glitch
NUM_WORKERS=3 bash run_mtg_server_lb.sh
```
Please adjust `NUM_WORKERS` based on the VRAM of your GPU. If the VRAM of your GPU is less than 40GB, we recommend you to run the following instead to initiate only one server worker on the GPU:
```
cd mind-the-glitch
python mtg_server.py
```

Then, after the remote servers are successfully initialized, we can start the actual training:
```
cd DivRL
# Stage 1:
accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        --num_processes=7 \
        --main_process_port $MASTER_PORT \
        scripts/train_flux_kontext.py \
        --config config/grpo.py:divrl_flux_kontext_syncd_7gpu_stage_1

# Stage 2:
accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        --num_processes=7 \
        --main_process_port $MASTER_PORT \
        scripts/train_flux_kontext.py \
        --config config/grpo.py:divrl_flux_kontext_syncd_7gpu_stage_2
```

Alternatively, we provide a Slurm script for single-node 8-GPU setup:
```
sbatch scripts/train_two_stages_single_node_8_gpu.sh
```
## 4. Inference
After the training is finished, you can find the LoRA weights under the default folder `logs/divrl_syncd_7gpu/flux_kontext/TIME_STAMP/checkpoints/`. Run the inference script:
```
python scripts/inference.py --image_path PATH_TO_YOUR_REF_IMAGE --prompt TEXT_PROMPT --lora_path SAVED_LORA_PATH
```

Please find more parameter settings in `scripts/inference.py`. Alternatively, if you wish to use the pretrained weights, you can also directly run:
```
python scripts/inference.py --image_path PATH_TO_YOUR_REF_IMAGE --prompt TEXT_PROMPT
```

## Citation
If you find this work useful, please cite:
```bibtex
@article{wang2026divrl,
  title     = {DivRL: Disentangled Self-Similarity Rewards for Diverse Subject-Driven Generation},
  author    = {Wang, Qian and Li, Zhenyu and Eldesokey, Abdelrahman and Wonka, Peter},
  journal   = {arXiv preprint arXiv:2606.23950},
  year      = {2026}
}
```
