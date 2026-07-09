#!/usr/bin/env bash

set -euo pipefail

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.format_prompt=./examples/format_prompt/math.jinja \
    data.max_teacher_prompt_length=2048 \
    data.max_response_length=512 \
    data.rollout_batch_size=32 \
    data.max_pixels=524288 \
    worker.actor.model.model_path="${MODEL_PATH:-/path/to/Qwen2.5-VL-3B-Instruct}" \
    worker.actor.global_batch_size=16 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.fsdp.enable_rank0_init=false \
    worker.actor.fsdp.enable_cpu_offload=false \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.rollout.n=1 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.25 \
    worker.rollout.enforce_eager=true \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    algorithm.adv_estimator=opsd \
    trainer.project_name=EasyOPD \
    trainer.total_epochs=3 \
    trainer.val_before_train=false \
    trainer.val_freq=195 \
    trainer.save_freq=20 \
    trainer.save_limit=3 \
    trainer.val_generations_to_log=8 \
    trainer.logger='["file","wandb"]' \
    trainer.experiment_name=qwen2_5_vl_3b_geo3k_opsd_tp2_r512_b32_gbs16_n1_epoch3 \
    trainer.n_gpus_per_node=2
