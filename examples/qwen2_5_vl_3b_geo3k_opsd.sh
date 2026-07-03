#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.format_prompt=./examples/format_prompt/math.jinja \
    data.max_teacher_prompt_length=3072 \
    data.max_response_length=1024 \
    data.rollout_batch_size=32 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.opsd_divergence=teacher_topk_ce \
    worker.actor.opsd_top_k=64 \
    worker.actor.opsd_temperature=1.0 \
    worker.actor.opsd_kl_clip=0.0 \
    worker.rollout.n=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    algorithm.objective=opsd \
    trainer.experiment_name=qwen2_5_vl_3b_geo3k_opsd \
    trainer.n_gpus_per_node=2
