from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from transformers import Trainer
except ModuleNotFoundError:  # Optional for lightweight tests/utilities.
    class Trainer:  # type: ignore[override]
        pass


@dataclass(frozen=True)
class GateL1WarmupConfig:
    """Configuration for dynamic L1 warm-up on gate scores."""

    max_lambda: float = 1e-3
    warmup_ratio: float = 0.1


@dataclass(frozen=True)
class ProjectorRawL2Config:
    """Configuration for projector_raw L2 regularization."""

    lambda_value: float = 0.0


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


def compute_total_loss_with_gate_l1(
    standard_loss: torch.Tensor,
    gate_scores: torch.Tensor,
    lambda_current: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Gate scores are stacked as [B, L, N_q, D]. Regularize only the final layer.
    if gate_scores.ndim >= 4:
        if gate_scores.shape[1] == 0:
            l1_norm = gate_scores.new_zeros(())
        else:
            gate_scores = gate_scores[:, -1]
            l1_norm = gate_scores.abs().mean()
    else:
        l1_norm = gate_scores.abs().mean()
    total_loss = standard_loss + (lambda_current * l1_norm)
    return total_loss, l1_norm


def compute_projector_raw_l2_loss(projector_raw: torch.Tensor) -> torch.Tensor:
    return projector_raw.float().square().mean()


def trainer_compute_loss_with_gate_l1(
    trainer: Trainer,
    model: torch.nn.Module,
    inputs: Dict[str, Any],
    config: Optional[GateL1WarmupConfig] = None,
    projector_raw_l2_config: Optional[ProjectorRawL2Config] = None,
    return_outputs: bool = False,
) -> Any:
    if config is None:
        config = GateL1WarmupConfig()
    if projector_raw_l2_config is None:
        projector_raw_l2_config = ProjectorRawL2Config()

    needs_projector_raw = projector_raw_l2_config.lambda_value > 0.0
    outputs = model(
        **inputs,
        return_projector_raw=needs_projector_raw,
    )

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

    if projector_raw_l2_config.lambda_value > 0.0:
        projector_raw = outputs.get("projector_raw")
        if projector_raw is None:
            raise ValueError("Model outputs must contain 'projector_raw' when projector_raw L2 is enabled.")
        projector_raw_l2_loss = compute_projector_raw_l2_loss(projector_raw)
        total_loss = total_loss + (projector_raw_l2_config.lambda_value * projector_raw_l2_loss)
        outputs["projector_raw_l2_loss"] = projector_raw_l2_loss.detach()

    if return_outputs:
        return total_loss, outputs
    return total_loss


class GateL1Trainer(Trainer):
    """HuggingFace Trainer with gate L1 regularization."""

    def __init__(
        self,
        *args: Any,
        gate_l1_max_lambda: float = 1e-3,
        gate_l1_warmup_ratio: float = 0.1,
        projector_raw_l2_lambda: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gate_l1_config = GateL1WarmupConfig(
            max_lambda=gate_l1_max_lambda,
            warmup_ratio=gate_l1_warmup_ratio,
        )
        self.projector_raw_l2_config = ProjectorRawL2Config(lambda_value=projector_raw_l2_lambda)

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
            projector_raw_l2_config=self.projector_raw_l2_config,
            return_outputs=return_outputs,
        )
