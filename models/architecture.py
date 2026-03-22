import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from .bridge import Extruder

def _find_multiple(a, b):
    return (-(a // -b)) * b



class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, expansion: float = 4.0):
        super().__init__()
        inter = _find_multiple(round(hidden_dim * expansion * 2 / 3), 256)
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * inter)
        self.down_proj = nn.Linear(inter, hidden_dim)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class PECEngine(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(
            self,
            profiler_path="answerdotai/ModernBERT-base",
            composer_path="Qwen/Qwen3-1.7B",
            num_query_tokens=64,
            freeze_profiler=False,
            freeze_composer=True,
            freeze_extruder=False,

    ):
        super().__init__()


        self.profiler = AutoModel.from_pretrained(profiler_path, dtype=torch.bfloat16,attn_implementation="flash_attention_2") if torch.cuda.is_available() else AutoModel.from_pretrained(profiler_path)
        self.composer = AutoModelForCausalLM.from_pretrained(composer_path, dtype=torch.bfloat16,
                                                             attn_implementation="flash_attention_2") if torch.cuda.is_available() else AutoModelForCausalLM.from_pretrained(composer_path)

        self.tokenizer = AutoTokenizer.from_pretrained(composer_path)
        # 2. Dimensions
        self.prof_dim = self.profiler.config.hidden_size
        self.comp_dim = self.composer.config.hidden_size

        # 3. Freeze
        if freeze_profiler:
            self.profiler.requires_grad_(False)
        if freeze_composer:
            self.composer.requires_grad_(False)

        # 4. Bridge Modules
        self.extruder = Extruder(
            hidden_size=self.prof_dim,
            num_query_tokens=num_query_tokens
        )  # [B, S_doc, D_prof] -> [B, N_q, D_prof]
        if freeze_extruder:
            self.extruder.requires_grad_(False)

        self.projector = nn.Sequential(
            nn.Linear(self.prof_dim, self.comp_dim),
            SwiGLU(self.comp_dim),
            nn.RMSNorm(self.comp_dim, eps=1e-6),
            nn.Linear(self.comp_dim, self.comp_dim)
        ).to(dtype=torch.bfloat16) # [D_prof] -> [D_comp]

        self.soft_prompt_scale = nn.Parameter(torch.tensor(0.03))

        self.post_extruder_norm = nn.RMSNorm(self.prof_dim, eps=1e-6)
        self.sep_token = nn.Parameter(torch.randn(1, 1, self.comp_dim))  # [1, 1, D_comp]

        self.pad_token_id = self.composer.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.composer.config.eos_token_id

        self._gradient_checkpointing_enabled = False
        self._composer_use_cache_default = getattr(self.composer.config, "use_cache", None)

        self._init_weights()

    def _init_weights(self):
        for p in self.extruder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.04)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.sep_token, mean=0.0, std=0.04)

        nn.init.ones_(self.post_extruder_norm.weight)

    @property
    def is_gradient_checkpointing(self) -> bool:
        if self._gradient_checkpointing_enabled:
            return True
        return any(
            bool(getattr(module, "is_gradient_checkpointing", False))
            for module in (self.profiler, self.composer, self.extruder)
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self._gradient_checkpointing_enabled = True

        for module in (self.profiler, self.composer, self.extruder):
            method = getattr(module, "gradient_checkpointing_enable", None)
            if method is None:
                continue
            if gradient_checkpointing_kwargs is None:
                method()
                continue
            try:
                method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                method()

        if hasattr(self.composer.config, "use_cache"):
            self.composer.config.use_cache = False

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing_enabled = False

        for module in (self.profiler, self.composer, self.extruder):
            method = getattr(module, "gradient_checkpointing_disable", None)
            if method is not None:
                method()

        if self._composer_use_cache_default is not None and hasattr(self.composer.config, "use_cache"):
            self.composer.config.use_cache = self._composer_use_cache_default

    def build_soft_prompt_artifacts(
            self,
            profiler_input_ids: torch.Tensor,
            profiler_attention_mask: torch.Tensor,
            return_gate_scores: bool = False,
            return_gate_logits: bool = False,
    ):
        prof_outputs = self.profiler(
            input_ids=profiler_input_ids,
            attention_mask=profiler_attention_mask,
        )
        prof_hidden = prof_outputs.last_hidden_state

        extruder_outputs = self.extruder(
            context=prof_hidden,
            attn_mask=profiler_attention_mask,
            return_gate_scores=return_gate_scores,
            return_gate_logits=return_gate_logits,
        )
        if return_gate_scores:
            if return_gate_logits:
                extruder_latents, gate_scores, gate_logits = extruder_outputs
            else:
                extruder_latents, gate_scores = extruder_outputs
                gate_logits = None
        else:
            extruder_latents = extruder_outputs
            gate_scores = None
            gate_logits = None

        projected_input = self.post_extruder_norm(extruder_latents) # [B, N_q, D_prof]
        projector_dtype = next(self.projector.parameters()).dtype
        projector_raw = self.projector(projected_input.to(dtype=projector_dtype))
        soft_prompts = self.soft_prompt_scale * projector_raw  # [B, N_q, D_comp]

        return {
            "soft_prompts": soft_prompts,
            "extruder_latents": extruder_latents,
            "projected_input": projected_input,
            "projector_raw": projector_raw,
            "profiler_hidden": prof_hidden,
            "gate_scores": gate_scores,
            "gate_logits": gate_logits,
        }

    def encode_soft_prompts(
            self,
            profiler_input_ids: torch.Tensor,
            profiler_attention_mask: torch.Tensor,
            return_gate_scores: bool = False,
            return_gate_logits: bool = False,
    ):
        artifacts = self.build_soft_prompt_artifacts(
            profiler_input_ids=profiler_input_ids,
            profiler_attention_mask=profiler_attention_mask,
            return_gate_scores=return_gate_scores,
            return_gate_logits=return_gate_logits,
        )
        return artifacts["soft_prompts"], artifacts["gate_scores"], artifacts["gate_logits"]

    def forward(
            self,
            profiler_input_ids,
            profiler_attention_mask,
            composer_input_ids,
            composer_attention_mask,
            labels=None,
            mh_target_input_ids_list=None,
            mh_target_attention_mask_list=None,
            return_logits: bool = False,
            return_projector_raw: bool = False,
    ):
        """
        New Forward Logic for Dynamic Masking Pipeline.

        Args:
            profiler_input_ids: Clean text for Encoder [B, Seq_Enc]
            composer_input_ids: Full text for Decoder [B, Seq_Dec]
                                (includes Instruction + Masked Context + Answer)
            labels: Masked labels for Decoder [B, Seq_Dec]
        """
        device = composer_attention_mask.device

        # --- [Phase 1] Profiler & Extruder (Compression) ---
        artifacts = self.build_soft_prompt_artifacts(
            profiler_input_ids=profiler_input_ids,
            profiler_attention_mask=profiler_attention_mask,
            return_gate_scores=True,
        )
        soft_prompts = artifacts["soft_prompts"]
        gate_scores = artifacts["gate_scores"]
        projector_raw = artifacts["projector_raw"] if return_projector_raw else None
        z_pool = artifacts["extruder_latents"].mean(dim=1)


        # --- [Phase 2] Composer Input Injection ---
        # 1. Get Embeddings of the actual text input (Qwen)
        # inputs_embeds: [B, Seq_Dec, D_comp]
        inputs_embeds = self.composer.get_input_embeddings()(composer_input_ids)
        soft_prompts = soft_prompts.to(dtype=inputs_embeds.dtype)

        # 2. Prepend Soft Prompts to the text embeddings
        # Structure: [Soft Prompts] + [Instruction + Masked Context + Answer]
        # This acts like a "Memory Prefix" or "System Context"
        final_inputs_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)

        # --- [Phase 3] Mask & Label Adjustment ---
        batch_size = soft_prompts.shape[0]
        prompt_len = soft_prompts.shape[1]

        # 1. Extend Attention Mask
        # Soft prompts should be fully visible (1)
        prompt_mask = torch.ones((batch_size, prompt_len), device=device, dtype=composer_attention_mask.dtype)
        final_attention_mask = torch.cat([prompt_mask, composer_attention_mask], dim=1)

        # 2. Extend Labels (if training)
        if labels is not None:
            # Soft prompts are not targets, so mask them with -100
            prompt_labels = torch.full((batch_size, prompt_len), -100, device=device, dtype=labels.dtype)
            final_labels = torch.cat([prompt_labels, labels], dim=1)
        else:
            final_labels = None

        # --- [Phase 4] Generation / Loss ---
        outputs = self.composer(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            use_cache=False,
            return_dict=True,
        )

        answer_loss = outputs.loss
        mh_align_loss = self._compute_morehop_align_loss(
            z_pool=z_pool,
            mh_target_input_ids_list=mh_target_input_ids_list,
            mh_target_attention_mask_list=mh_target_attention_mask_list,
        )

        total_loss = answer_loss
        if mh_align_loss is not None:
            total_loss = mh_align_loss * 0.1 if total_loss is None else total_loss + (0.1 * mh_align_loss)

        return {
            "loss": total_loss,
            "answer_loss": answer_loss,
            "mh_align_loss": mh_align_loss,
            "logits": outputs.logits if return_logits else None,
            "gate_scores": gate_scores,
            "projector_raw": projector_raw,
        }

    def _compute_morehop_align_loss(
            self,
            *,
            z_pool: torch.Tensor,
            mh_target_input_ids_list: list[torch.Tensor] | None,
            mh_target_attention_mask_list: list[torch.Tensor] | None,
    ):
        if not mh_target_input_ids_list or not mh_target_attention_mask_list:
            return None

        total_loss = z_pool.new_zeros(())
        total_pairs = 0

        for target_input_ids, target_attention_mask in zip(
            mh_target_input_ids_list,
            mh_target_attention_mask_list,
        ):
            active_rows = torch.any(target_attention_mask.bool(), dim=1)
            if not torch.any(active_rows):
                continue

            target_outputs = self.profiler(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
            )
            target_hidden = target_outputs.last_hidden_state

            mask = target_attention_mask.unsqueeze(-1).to(dtype=target_hidden.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            r_k = (target_hidden * mask).sum(dim=1) / denom

            similarities = F.cosine_similarity(
                z_pool[active_rows].float(),
                r_k[active_rows].float(),
                dim=-1,
            )
            total_loss = total_loss + (1.0 - similarities).sum()
            total_pairs += int(active_rows.sum().item())

        if total_pairs == 0:
            return None

        return total_loss / total_pairs
