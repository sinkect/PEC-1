import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from .bridge import Extruder, SwiGLU


class PECEngine(nn.Module):
    def __init__(
            self,
            profiler_path="profiler",
            composer_path="Qwen/Qwen3-1.7B",
            num_query_tokens=64,
            freeze_profiler=False,
            freeze_composer=True,
    ):
        super().__init__()
        current_dir = Path(__file__).parent
        # 1. Load Models
        self.profiler = AutoModel.from_pretrained(str(current_dir / profiler_path), local_files_only=True,
                                                  dtype=torch.bfloat16,
                                                  attn_implementation="flash_attention_2") if torch.cuda.is_available() else AutoModel.from_pretrained(str(current_dir / profiler_path), local_files_only=True)
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
            num_query_tokens=num_query_tokens,
            max_position_embeddings=self.profiler.config.max_position_embeddings,
        )  # [B, S_doc, D_prof] -> [B, N_q, D_prof]

        self.projector = nn.Sequential(
            nn.Linear(self.prof_dim, self.comp_dim),
            SwiGLU(self.comp_dim),
            nn.Linear(self.comp_dim, self.comp_dim)
        ) # [D_prof] -> [D_comp]

        self.post_extruder_norm = nn.RMSNorm(self.prof_dim, eps=1e-6)
        self.sep_token = nn.Parameter(torch.randn(1, 1, self.comp_dim))  # [1, 1, D_comp]

        self.pad_token_id = self.composer.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.composer.config.eos_token_id


        # Prepare special tokens
        u_end_ids = self.tokenizer("<|im_end|>\n", return_tensors="pt", add_special_tokens=False).input_ids
        a_start_ids = self.tokenizer("<|im_start|>assistant\n", return_tensors="pt",
                                     add_special_tokens=False).input_ids

        self.register_buffer("u_end_ids", u_end_ids, persistent=False)
        self.register_buffer("a_start_ids", a_start_ids, persistent=False)

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

    def forward(
            self,
            profiler_input_ids,
            profiler_attention_mask,
            composer_input_ids,
            composer_attention_mask,
            labels=None
    ):
        """
        New Forward Logic for Dynamic Masking Pipeline.

        Args:
            profiler_input_ids: Clean text for Encoder [B, Seq_Enc]
            composer_input_ids: Full text for Decoder [B, Seq_Dec]
                                (includes Instruction + Masked Context + Answer)
            labels: Masked labels for Decoder [B, Seq_Dec]
        """
        device = self.composer.device

        # --- [Phase 1] Profiler & Extruder (Compression) ---
        # 1. Encode with Profiler (Modern-BERT)
        prof_outputs = self.profiler(
            input_ids=profiler_input_ids,
            attention_mask=profiler_attention_mask,
        )
        prof_hidden = prof_outputs.last_hidden_state  # [B, Seq_Enc, D_prof]

        # 2. Extrude (Cross-Attention Compression)
        # Result: [B, Num_Query, D_prof]
        extruded = self.extruder(context=prof_hidden, attn_mask=profiler_attention_mask)
        extruded = self.post_extruder_norm(extruded)

        # 3. Project to Composer Dimension
        # Result: [B, Num_Query, D_comp] aka "Soft Prompts"
        soft_prompts = self.projector(extruded)

        # --- [Phase 2] Composer Input Injection ---
        # 1. Get Embeddings of the actual text input (Qwen)
        # inputs_embeds: [B, Seq_Dec, D_comp]
        inputs_embeds = self.composer.get_input_embeddings()(composer_input_ids)

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
            labels=final_labels
        )

        return outputs