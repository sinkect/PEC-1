from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer


@dataclass(frozen=True)
class GateL1WarmupConfig:
    """Configuration for dynamic L1 warm-up on gate scores."""

    max_lambda: float = 1e-3
    warmup_ratio: float = 0.1


@dataclass(frozen=True)
class KnowledgeDistillationConfig:
    """Configuration for Stage 1 answer-only KL distillation."""

    lambda_value: float = 0.3
    temperature: float = 2.0


def _resolve_max_steps(trainer: Trainer) -> int:
    max_steps = int(getattr(trainer.state, "max_steps", 0) or 0)
    if max_steps <= 0:
        max_steps = int(getattr(trainer.args, "max_steps", 0) or 0)
    return max(1, max_steps)


def _compute_current_lambda(trainer: Trainer, config: GateL1WarmupConfig) -> float:
    max_steps = _resolve_max_steps(trainer)
    warmup_steps = max(1, int(max_steps * config.warmup_ratio))
    current_step = int(getattr(trainer.state, "global_step", 0) or 0)

    if current_step >= warmup_steps:
        return config.max_lambda

    progress = float(current_step) / float(warmup_steps)
    return config.max_lambda * progress


def _split_student_teacher_inputs(inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    student_inputs = {key: value for key, value in inputs.items() if not key.startswith("teacher_")}
    teacher_inputs = {
        key.removeprefix("teacher_"): value
        for key, value in inputs.items()
        if key.startswith("teacher_")
    }
    return student_inputs, teacher_inputs


def _gather_answer_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> List[torch.Tensor]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100

    answer_logits: List[torch.Tensor] = []
    for sample_logits, sample_mask in zip(shift_logits, valid_mask):
        answer_logits.append(sample_logits[sample_mask].float())
    return answer_logits


def _align_labels_to_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    seq_diff = int(logits.shape[1] - labels.shape[1])
    if seq_diff == 0:
        return labels
    if seq_diff > 0:
        prefix = labels.new_full((labels.shape[0], seq_diff), -100)
        return torch.cat([prefix, labels], dim=1)
    raise ValueError(
        "Labels are longer than logits during answer-only KL computation: "
        f"logits_seq={logits.shape[1]}, labels_seq={labels.shape[1]}"
    )


def compute_answer_only_kl(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Computes KL only on answer-token positions for each sample."""

    temperature = max(float(temperature), 1e-5)
    student_labels = _align_labels_to_logits(student_logits, student_labels)
    teacher_labels = _align_labels_to_logits(teacher_logits, teacher_labels)
    student_answer_logits = _gather_answer_logits(student_logits, student_labels)
    teacher_answer_logits = _gather_answer_logits(teacher_logits, teacher_labels)

    losses: List[torch.Tensor] = []
    for student_sample_logits, teacher_sample_logits in zip(student_answer_logits, teacher_answer_logits):
        if student_sample_logits.numel() == 0 or teacher_sample_logits.numel() == 0:
            continue

        valid_steps = min(student_sample_logits.shape[0], teacher_sample_logits.shape[0])
        student_sample_logits = student_sample_logits[:valid_steps]
        teacher_sample_logits = teacher_sample_logits[:valid_steps]

        student_log_probs = F.log_softmax(student_sample_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_sample_logits / temperature, dim=-1)
        losses.append(
            F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
        )

    if not losses:
        return student_logits.new_zeros(())
    return torch.stack(losses).mean()


def compute_total_loss_with_gate_l1(
    standard_loss: torch.Tensor,
    gate_scores: torch.Tensor,
    lambda_current: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    l1_norm = gate_scores.abs().mean()
    total_loss = standard_loss + (lambda_current * l1_norm)
    return total_loss, l1_norm


def trainer_compute_loss_with_gate_l1(
    trainer: Trainer,
    model: torch.nn.Module,
    inputs: Dict[str, Any],
    config: Optional[GateL1WarmupConfig] = None,
    kd_config: Optional[KnowledgeDistillationConfig] = None,
    teacher_model: Optional[torch.nn.Module] = None,
    return_outputs: bool = False,
) -> Any:
    if config is None:
        config = GateL1WarmupConfig()
    if kd_config is None:
        kd_config = KnowledgeDistillationConfig()

    student_inputs, teacher_inputs = _split_student_teacher_inputs(inputs)
    needs_student_logits = teacher_model is not None and teacher_inputs and kd_config.lambda_value > 0.0
    outputs = model(**student_inputs, return_logits=needs_student_logits)

    standard_loss = outputs["loss"]
    if standard_loss is None:
        raise ValueError("Model outputs must contain 'loss'.")

    gate_scores = outputs["gate_scores"]
    if gate_scores is None:
        raise ValueError("Model outputs must contain 'gate_scores' for L1 regularization.")

    lambda_current = _compute_current_lambda(trainer, config)
    total_loss, gate_l1_loss = compute_total_loss_with_gate_l1(
        standard_loss=standard_loss,
        gate_scores=gate_scores,
        lambda_current=lambda_current,
    )
    outputs["gate_l1_loss"] = gate_l1_loss.detach()

    if teacher_model is not None and teacher_inputs and kd_config.lambda_value > 0.0:
        teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=teacher_inputs["input_ids"],
                attention_mask=teacher_inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            )

        distill_kl_loss = compute_answer_only_kl(
            student_logits=outputs["logits"],
            teacher_logits=teacher_outputs.logits,
            student_labels=student_inputs["labels"],
            teacher_labels=teacher_inputs["labels"],
            temperature=kd_config.temperature,
        )
        total_loss = total_loss + (kd_config.lambda_value * distill_kl_loss)
        outputs["distill_kl_loss"] = distill_kl_loss.detach()

    if return_outputs:
        return total_loss, outputs
    return total_loss


class GateL1Trainer(Trainer):
    """HuggingFace Trainer with gate L1 regularization and optional KD."""

    def __init__(
        self,
        *args: Any,
        gate_l1_max_lambda: float = 1e-3,
        gate_l1_warmup_ratio: float = 0.1,
        teacher_model: Optional[torch.nn.Module] = None,
        distill_kl_lambda: float = 0.0,
        distill_kl_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gate_l1_config = GateL1WarmupConfig(
            max_lambda=gate_l1_max_lambda,
            warmup_ratio=gate_l1_warmup_ratio,
        )
        self.kd_config = KnowledgeDistillationConfig(
            lambda_value=distill_kl_lambda,
            temperature=distill_kl_temperature,
        )
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        del num_items_in_batch
        return trainer_compute_loss_with_gate_l1(
            trainer=self,
            model=model,
            inputs=inputs,
            config=self.gate_l1_config,
            kd_config=self.kd_config,
            teacher_model=self.teacher_model,
            return_outputs=return_outputs,
        )
