# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_opsd_loss, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            logits = self._forward_micro_batch_logits(micro_batch, temperature=temperature)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _forward_micro_batch_logits(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        input_ids_key: str = "input_ids",
        attention_mask_key: str = "attention_mask",
        position_ids_key: str = "position_ids",
    ) -> torch.Tensor:
        input_ids = micro_batch[input_ids_key]
        attention_mask = micro_batch[attention_mask_key]
        position_ids = micro_batch[position_ids_key]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            use_cache=False,
        )
        logits: torch.Tensor = output.logits
        logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
        return logits / temperature

    def _build_teacher_micro_batch(self, micro_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        teacher_prompts = micro_batch["teacher_prompts"]
        teacher_attention_mask = micro_batch["teacher_attention_mask"]
        teacher_position_ids = micro_batch["teacher_position_ids"]
        responses = micro_batch["responses"]
        response_mask = micro_batch["response_mask"]
        batch_size, response_length = responses.shape

        delta_position_id = torch.arange(
            1, response_length + 1, device=teacher_position_ids.device, dtype=teacher_position_ids.dtype
        )
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if teacher_position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, teacher_position_ids.size(1), -1
            )

        teacher_response_position_ids = teacher_position_ids[..., -1:] + delta_position_id
        teacher_batch = dict(micro_batch)
        teacher_batch["teacher_input_ids"] = torch.cat([teacher_prompts, responses], dim=-1)
        teacher_batch["teacher_full_attention_mask"] = torch.cat([teacher_attention_mask, response_mask], dim=-1)
        teacher_batch["teacher_full_position_ids"] = torch.cat(
            [teacher_position_ids, teacher_response_position_ids], dim=-1
        )
        return teacher_batch

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        if self.config.loss_type == "opsd":
            select_keys.extend(["teacher_prompts", "teacher_attention_mask", "teacher_position_ids"])
            if self.config.opsd_format_pg != "none" and self.config.opsd_format_pg_loss_coef > 0:
                select_keys.append("format_advantages")
            if self.config.opsd_reward_pg != "none" and self.config.opsd_reward_pg_loss_coef > 0:
                select_keys.append("opsd_reward_advantages")
        else:
            select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)
                if total_response_tokens <= 0:
                    continue

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    if self.config.loss_type == "opsd":
                        max_input_len = max(
                            max_input_len,
                            mini_batch.batch["teacher_prompts"].size(-1) + mini_batch.batch["responses"].size(-1),
                        )

                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    if self.config.loss_type == "opsd":
                        teacher_inputs = self._build_teacher_micro_batch(model_inputs)
                        was_training = self.actor_module.training
                        self.actor_module.eval()
                        with torch.no_grad():
                            teacher_logits = self._forward_micro_batch_logits(
                                teacher_inputs,
                                temperature=1.0,
                                input_ids_key="teacher_input_ids",
                                attention_mask_key="teacher_full_attention_mask",
                                position_ids_key="teacher_full_position_ids",
                            ).detach()

                        if was_training:
                            self.actor_module.train()

                        student_logits = self._forward_micro_batch_logits(model_inputs, temperature=1.0)
                        loss, opsd_metrics = compute_opsd_loss(
                            student_logits=student_logits,
                            teacher_logits=teacher_logits,
                            response_mask=response_mask,
                            divergence=self.config.opsd_divergence,
                            top_k=self.config.opsd_top_k,
                            temperature=self.config.opsd_temperature,
                            kl_clip=self.config.opsd_kl_clip,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        batch_metrics = {f"opsd/{k}": v for k, v in opsd_metrics.items()}
                        log_probs = None
                        if self.config.opsd_format_pg_loss_coef > 0 and "format_advantages" in model_inputs:
                            log_probs = self.log_probs_from_logits(student_logits, model_inputs["responses"])
                            format_advantages = model_inputs["format_advantages"].to(log_probs.dtype).unsqueeze(-1)
                            format_pg_loss = average_loss(
                                -format_advantages * log_probs,
                                response_mask,
                                mode=self.config.loss_avg_mode,
                            )
                            loss = loss + self.config.opsd_format_pg_loss_coef * format_pg_loss
                            batch_metrics["opsd/format_pg_loss"] = format_pg_loss.detach().item()
                            batch_metrics["opsd/format_pg_loss_coef"] = self.config.opsd_format_pg_loss_coef
                            batch_metrics["opsd/format_advantage"] = (
                                (format_advantages * response_mask).sum() / (response_mask.sum() + 1e-8)
                            ).detach().item()
                        if (
                            self.config.opsd_reward_pg_loss_coef > 0
                            and "opsd_reward_advantages" in model_inputs
                        ):
                            if log_probs is None:
                                log_probs = self.log_probs_from_logits(student_logits, model_inputs["responses"])
                            reward_advantages = model_inputs["opsd_reward_advantages"].to(log_probs.dtype).unsqueeze(-1)
                            reward_pg_loss = average_loss(
                                -reward_advantages * log_probs,
                                response_mask,
                                mode=self.config.loss_avg_mode,
                            )
                            loss = loss + self.config.opsd_reward_pg_loss_coef * reward_pg_loss
                            batch_metrics["opsd/reward_pg_loss"] = reward_pg_loss.detach().item()
                            batch_metrics["opsd/reward_pg_loss_coef"] = self.config.opsd_reward_pg_loss_coef
                            batch_metrics["opsd/reward_advantage"] = (
                                (reward_advantages * response_mask).sum() / (response_mask.sum() + 1e-8)
                            ).detach().item()
                    else:
                        old_log_probs = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]

                        # all return: (bsz, response_length)
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

                        pg_loss, pg_metrics = compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                            tau_positive=self.config.tau_positive,
                            tau_negative=self.config.tau_negative,
                            loss_type=self.config.loss_type,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                            ref_log_probs = model_inputs["ref_log_probs"]
                            # compute kl loss
                            kld = compute_kl(
                                log_probs=log_probs,
                                ref_log_probs=ref_log_probs,
                                kl_penalty=self.config.kl_penalty,
                            )
                            kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                            loss = pg_loss + kl_loss * self.config.kl_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_coef
                        else:
                            loss = pg_loss

                        batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                        batch_metrics["actor/pg_loss"] = pg_loss.detach().item()

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
