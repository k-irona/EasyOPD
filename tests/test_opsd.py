from types import SimpleNamespace

import torch
from torch import nn

from verl.trainer.core_algos import compute_opsd_loss
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        vocab = torch.arange(self.vocab_size, dtype=torch.float32, device=input_ids.device)
        logits = input_ids.float().unsqueeze(-1) * 0.13 + vocab.view(1, 1, -1) * 0.07
        return SimpleNamespace(logits=logits)


def test_response_logits_gather_matches_forward_micro_batch_log_probs():
    config = ActorConfig(padding_free=False, use_torch_compile=False)
    actor = DataParallelPPOActor(config=config, actor_module=DummyCausalLM(vocab_size=17))
    micro_batch = {
        "input_ids": torch.tensor([[4, 5, 6, 7, 8, 0], [3, 2, 1, 9, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 0, 0]]),
        "responses": torch.tensor([[7, 8, 0], [9, 0, 0]]),
    }

    logits = actor._forward_micro_batch_logits(micro_batch, temperature=1.0)
    gathered = actor.log_probs_from_logits(logits, micro_batch["responses"])
    existing = actor._forward_micro_batch(micro_batch, temperature=1.0)

    assert torch.allclose(gathered, existing)


def test_opsd_teacher_topk_ce_logs_topk_mass_and_clips():
    teacher_logits = torch.tensor([[[4.0, 3.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0]]])
    student_logits = torch.zeros_like(teacher_logits)
    response_mask = torch.tensor([[1, 1]])

    loss, metrics = compute_opsd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        response_mask=response_mask,
        divergence="teacher_topk_ce",
        top_k=2,
        temperature=1.0,
        kl_clip=1.0,
        loss_avg_mode="token",
    )

    assert torch.isfinite(loss)
    assert metrics["clip_fraction"] >= 0
    assert 0 < metrics["topk_mass"] <= 1


def test_opsd_forward_kl_is_zero_for_identical_logits():
    logits = torch.randn(2, 3, 11)
    response_mask = torch.ones(2, 3)

    loss, _ = compute_opsd_loss(
        student_logits=logits,
        teacher_logits=logits,
        response_mask=response_mask,
        divergence="forward_kl",
        top_k=0,
        temperature=1.0,
        kl_clip=0.0,
        loss_avg_mode="token",
    )

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)
