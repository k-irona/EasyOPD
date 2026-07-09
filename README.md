# EasyOPD: OPSD on top of EasyR1

[![GitHub Repo stars](https://img.shields.io/github/stars/k-irona/EasyOPD)](https://github.com/k-irona/EasyOPD/stargazers)
[![Docker Pulls](https://img.shields.io/docker/pulls/hiyouga/verl)](https://hub.docker.com/r/hiyouga/verl/tags)

EasyOPD is a lightweight fork of [EasyR1](https://github.com/hiyouga/EasyR1) for experimenting with **OPSD** in an existing GRPO-style RL training framework.

The implementation keeps EasyR1's rollout, log-prob recomputation, reference KL, actor update, checkpointing, and logging pipeline intact. OPSD is added as a minimal policy-gradient advantage replacement:

```text
A_t = sg[log pi_teacher(y_t | x, y_<t) - log pi_old(y_t | x, y_<t)]
```

Here `sg` means stop-gradient. The teacher-side log-ratio is treated as a fixed token-level advantage, so gradients flow through the original policy-gradient loss instead of through the teacher log-prob computation.

> [!NOTE]
> Current EasyOPD only supports OPSD through `algorithm.adv_estimator=opsd`. OPD with a separate external teacher model is not supported yet.

## Features

- Supported models
  - Llama3/Qwen2/Qwen2.5/Qwen3 language models
  - Qwen2-VL/Qwen2.5-VL/Qwen3-VL vision language models
  - DeepSeek-R1 distill models

- Supported algorithms
  - OPSD ![new](https://img.shields.io/badge/new-orange)
  - GRPO
  - DAPO
  - Reinforce++
  - ReMax
  - RLOO
  - GSPO
  - CISPO

- OPSD changes on top of EasyR1
  - Adds `algorithm.adv_estimator=opsd`
  - Adds teacher-prompt construction with `data.build_opsd_teacher`
  - Adds teacher log-prob recomputation on sampled response tokens
  - Replaces GRPO/group-normalized reward advantages with detached teacher/student log-ratio advantages
  - Reuses EasyR1's actor policy-gradient loss, optional reference KL, FSDP, vLLM rollout, and logger stack

- Supported datasets
  - Any text or vision-text dataset in the EasyR1 data format

- Supported tricks
  - Padding-free training
  - LoRA training
  - Resuming from the latest/best checkpoint
  - Wandb & SwanLab & Mlflow & Tensorboard tracking

## Requirements

### Software Requirements

- Python 3.9+
- transformers>=4.54.0
- flash-attn>=2.4.3
- vllm>=0.8.3

We provide a [Dockerfile](./Dockerfile) inherited from EasyR1 to build environments.

The EasyR1 Docker image remains a practical starting point:

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

If your environment does not support Docker, you can use **Apptainer**:

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

Use `USE_MODELSCOPE_HUB=1` to download models from the ModelScope hub.

### Hardware Requirements

\* *estimated from EasyR1; OPSD adds teacher log-prob recomputation and may need extra memory/time.*

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| OPSD/GRPO Full Fine-Tuning | AMP | 2*24GB | 4*40GB | 8*40GB | 16*80GB | 32*80GB |
| OPSD/GRPO Full Fine-Tuning | BF16 | 1*24GB | 1*40GB | 4*40GB | 8*80GB | 16*80GB |
| OPSD/GRPO LoRA Fine-Tuning | AMP | 1*12GB | 1*24GB | 2*32GB | 2*80GB | 4*80GB |

> [!NOTE]
> Use `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16` to enable bf16 training.

## Tutorial: Run Qwen2.5-VL OPSD on Geometry3K in 3 Steps

### Installation

```bash
git clone https://github.com/k-irona/EasyOPD.git
cd EasyOPD
pip install -e .
```

### OPSD Full Training

Set your local model path and launch the example script:

```bash
export MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
bash examples/qwen2_5_vl_3b_geo3k_opsd.sh
```

The key OPSD switch is:

```bash
algorithm.adv_estimator=opsd
```

### GRPO Baseline

To run a GRPO baseline, keep the same EasyR1 training configuration and use:

```bash
algorithm.adv_estimator=grpo
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/EasyOPD/exp_name/global_step_1/actor
```

> [!TIP]
> If you encounter issues with connecting to Hugging Face, consider using `export HF_ENDPOINT=https://hf-mirror.com`.

## OPSD Metrics

For OPSD, reward is still logged by the inherited EasyR1 pipeline, but reward is not the optimization target. The most relevant training curves are:

- `actor/pg_loss`: policy-gradient objective using OPSD advantages
- `actor/kl_loss`: reference KL loss when `algorithm.use_kl_loss=true`
- `actor/grad_norm`: update stability
- `opsd/advantage_mean`, `opsd/advantage_max`, `opsd/advantage_min`: teacher/student log-ratio scale

The actor loss follows the original EasyR1 actor update path:

```text
actor_loss = actor/pg_loss + algorithm.kl_coef * actor/kl_loss
```

## Custom Dataset

Please refer to the EasyR1 example datasets to prepare your own dataset.

- Text dataset: https://huggingface.co/datasets/hiyouga/math12k
- Image-text dataset: https://huggingface.co/datasets/hiyouga/geometry3k
- Multi-image-text dataset: https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa
- Text-image mixed dataset: https://huggingface.co/datasets/hiyouga/rl-mixed-dataset

For OPSD, each training sample must provide the normal prompt fields and an answer field. EasyOPD uses the answer to build a privileged teacher prompt for teacher log-prob scoring.

## How to Understand EasyOPD

EasyOPD inherits EasyR1's GRPO/PPO-style actor update and changes the advantage source for OPSD.

- EasyR1 GRPO: group-normalized reward advantage
- EasyOPD OPSD: detached token-level teacher/student log-ratio advantage

The implementation is intentionally close to "a one-line change on top of RL implementations": it swaps the advantage tensor while keeping the policy-gradient training path.

## How to Run 70B+ Model in Multi-node Environment

1. Start the Ray head node.

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

2. Start the Ray worker node and connect to the head node.

```bash
ray start --address=<head_node_ip>:6379
```

3. Check the Ray resource pool.

```bash
ray status
```

4. Run the training script on the Ray head node only.

```bash
bash examples/qwen2_5_vl_3b_geo3k_opsd.sh
```

See the [veRL multi-node documentation](https://verl.readthedocs.io/en/latest/start/multinode.html) for more details about multi-node training and Ray debugger.

## TODO

- Support external teacher models for OPD-style training.
- Support more VLM architectures.
- Support ulysses parallelism for VLMs.

## FAQs

> ValueError: Image features and image tokens do not match: tokens: 8192, features 9800

Increase `data.max_prompt_length` or reduce `data.max_pixels`.

> RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:62

Reduce `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.

> ValueError: No available memory for the cache blocks.

Reduce `worker.rollout.gpu_memory_utilization`, use fewer concurrent processes on the same GPU, or reduce rollout/model memory usage.

> RuntimeError: 0 active drivers ([]). There should only be one.

Uninstall `deepspeed` from the current python environment.

## Citation

EasyOPD is built on EasyR1 and veRL. Please cite and credit the upstream projects when using this repository.

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong, Richong Zhang},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```
