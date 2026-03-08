from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import Trainer


@dataclass(frozen=True)
class GateL1WarmupConfig:
    """Configuration for dynamic L1 warm-up on gate scores."""

    max_lambda: float = 1e-3
    warmup_ratio: float = 0.1


def _resolve_max_steps(trainer: Trainer) -> int:
    max_steps = int(getattr(trainer.state, "max_steps", 0) or 0)
    if max_steps <= 0:
        max_steps = int(getattr(trainer.args, "max_steps", 0) or 0)
    return max(1, max_steps)


def _compute_current_lambda(trainer: Trainer, config: GateL1WarmupConfig) -> float:
    """Computes linear warm-up lambda based on global training step."""
    max_steps = _resolve_max_steps(trainer)
    warmup_steps = max(1, int(max_steps * config.warmup_ratio))
    current_step = int(getattr(trainer.state, "global_step", 0) or 0)

    if current_step >= warmup_steps:
        return config.max_lambda

    progress = float(current_step) / float(warmup_steps)
    return config.max_lambda * progress


def compute_total_loss_with_gate_l1(
    standard_loss: torch.Tensor,
    gate_scores: torch.Tensor,
    lambda_current: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes total loss = CE loss + lambda * L1(gate_scores).

    Args:
        standard_loss: Scalar CE loss tensor.  # []
        gate_scores: Gate activations from model outputs.  # [B, ...]
        lambda_current: Current L1 coefficient.

    Returns:
        total_loss: Combined optimization objective.  # []
        l1_norm: Mean absolute gate value used as penalty.  # []
    """
    l1_norm = gate_scores.abs().mean()  # [B, ...] -> []
    total_loss = standard_loss + (lambda_current * l1_norm)  # [] + [] -> []
    return total_loss, l1_norm


def trainer_compute_loss_with_gate_l1(
    trainer: Trainer,
    model: torch.nn.Module,
    inputs: Dict[str, Any],
    config: Optional[GateL1WarmupConfig] = None,
    return_outputs: bool = False,
) -> Any:
    """Shared compute_loss implementation for Trainer subclasses."""
    if config is None:
        config = GateL1WarmupConfig()

    outputs = model(**inputs)

    standard_loss = outputs["loss"]
    if standard_loss is None:
        raise ValueError("Model outputs must contain 'loss'.")

    gate_scores = outputs["gate_scores"]
    if gate_scores is None:
        raise ValueError("Model outputs must contain 'gate_scores' for L1 regularization.")

    lambda_current = _compute_current_lambda(trainer, config)
    total_loss, _ = compute_total_loss_with_gate_l1(
        standard_loss=standard_loss,
        gate_scores=gate_scores,
        lambda_current=lambda_current,
    )

    if return_outputs:
        return total_loss, outputs
    return total_loss


class GateL1Trainer(Trainer):
    """HuggingFace Trainer with dynamic L1 regularization on gate scores."""

    def __init__(
        self,
        *args: Any,
        gate_l1_max_lambda: float = 1e-3,
        gate_l1_warmup_ratio: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gate_l1_config = GateL1WarmupConfig(
            max_lambda=gate_l1_max_lambda,
            warmup_ratio=gate_l1_warmup_ratio,
        )

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
            return_outputs=return_outputs,
        )