#!/usr/bin/env bash

set -euo pipefail

GPU_IDS="${GPU_IDS:-}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_USED_MB="${MAX_USED_MB:-1024}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MODEL_PATH="${MODEL_PATH:-/data/jinda/models/Qwen2.5-VL-3B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_vl_3b_geo3k_opsd_dryrun}"

if [[ -n "${GPU_IDS}" ]]; then
    echo "Waiting for fixed GPUs ${GPU_IDS} to have <= ${MAX_USED_MB} MiB used memory each."
else
    echo "Waiting for any ${NUM_GPUS} GPUs from ${GPU_CANDIDATES} to have <= ${MAX_USED_MB} MiB used memory each."
fi
echo "Polling every ${POLL_SECONDS}s."

while true; do
    if [[ -n "${GPU_IDS}" ]]; then
        mapfile -t gpu_rows < <(
            nvidia-smi \
                --id="${GPU_IDS}" \
                --query-gpu=index,memory.used \
                --format=csv,noheader,nounits
        )
    else
        mapfile -t gpu_rows < <(
            nvidia-smi \
                --id="${GPU_CANDIDATES}" \
                --query-gpu=index,memory.used \
                --format=csv,noheader,nounits
        )
    fi

    free_ids=()
    status_parts=()
    for row in "${gpu_rows[@]}"; do
        gpu_id="${row%%,*}"
        used="${row##*,}"
        gpu_id="${gpu_id//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        status_parts+=("${gpu_id}:${used}MiB")
        if (( used <= MAX_USED_MB )); then
            free_ids+=("${gpu_id}")
        fi
    done

    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${timestamp}] GPU used memory: ${status_parts[*]}"

    if [[ -n "${GPU_IDS}" ]]; then
        required_count="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
        if (( ${#free_ids[@]} == required_count )); then
            selected_gpu_ids="${GPU_IDS}"
            break
        fi
    elif (( ${#free_ids[@]} >= NUM_GPUS )); then
        selected_gpu_ids="$(IFS=,; echo "${free_ids[*]:0:NUM_GPUS}")"
        break
    fi

    sleep "${POLL_SECONDS}"
done

echo "Selected GPUs: ${selected_gpu_ids}"
echo "GPUs are free enough. Stopping stale Ray processes and starting OPSD dry run."
ray stop --force || true

CUDA_VISIBLE_DEVICES="${selected_gpu_ids}" python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.format_prompt=./examples/format_prompt/math.jinja \
    data.max_teacher_prompt_length=1536 \
    data.max_response_length=128 \
    data.rollout_batch_size=2 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=2 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.fsdp.enable_rank0_init=false \
    worker.actor.fsdp.enable_cpu_offload=true \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.actor.opsd_divergence=teacher_topk_ce \
    worker.actor.opsd_top_k=32 \
    worker.actor.opsd_temperature=1.0 \
    worker.actor.opsd_kl_clip=0.0 \
    worker.rollout.n=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.35 \
    worker.rollout.enforce_eager=true \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    algorithm.objective=opsd \
    trainer.max_steps=1 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.logger='["console"]' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=2
