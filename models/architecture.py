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
            num_query_tokens=512,
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

    def forward(self, profiler_input_ids, profiler_attention_mask,qwen_prompt_ids,
                labels=None):
        """
        profiler_input_ids: [Batch, Doc_Len + Prompt_Len] (Full Context)
        composer_input_ids: [Batch, Trigger_Len] (Static Trigger e.g., <|im_start|>assistant)
        """
        device = self.composer.device
        batch_size = profiler_input_ids.shape[0]
        embed_fn = self.composer.get_input_embeddings()

        # --- [Phase 1] Profiler & Extruder (Compression) ---
        prof_outputs = self.profiler(
            input_ids=profiler_input_ids.to(device),
            attention_mask=profiler_attention_mask.to(device),
        )
        prof_hidden = prof_outputs.last_hidden_state  # [B, Seq_P, D_prof]

        extruded = self.extruder(context=prof_hidden,attn_mask=profiler_attention_mask.to(device))  # [B, N_q, D_prof]
        extruded = self.post_extruder_norm(extruded)
        projected = self.projector(extruded)  # [B, N_q, D_comp]

        # Add Separator
        sep = self.sep_token.expand(batch_size, -1, -1).to(projected.dtype)
        latents_embeds = torch.cat([projected, sep], dim=1)  # [B, N_q + 1, D_comp]

        # ------------------------------------------------------------------
        # [Phase 2] Construct ChatML Template
        # Structure: <|im_start|>user\n [Latents] <|im_end|>\n <|im_start|>assistant\n
        # ------------------------------------------------------------------

        # Embed and expand to batch size
        prompt_embeds = embed_fn(qwen_prompt_ids.to(device))
        u_end_emb = embed_fn(self.u_end_ids).expand(batch_size, -1, -1)  # [B, 1, D_comp]
        a_start_emb = embed_fn(self.a_start_ids).expand(batch_size, -1, -1)  # [B, 1, D_comp]

        # 2. Answer Embeddings (Labels)
        labels = labels.to(device)
        answer_embeds = embed_fn(labels)  # [B, Ans_Len, D_comp]

        # --- [Phase 3] Concatenation ---
        # Qwen sees: [User Start] + [Latents] + [User End] + [Assistant Start] + [Answer]
        inputs_embeds = torch.cat([prompt_embeds, latents_embeds, u_end_emb, a_start_emb, answer_embeds], dim=1).to(
            device)
        prefix_len = prompt_embeds.shape[1] + latents_embeds.shape[1] + u_end_emb.shape[1] + a_start_emb.shape[1]

        # --- [Phase 4] Masks & Labels ---
        # 1. Attention Mask
        inputs_mask = torch.ones((batch_size, prefix_len), device=device,dtype=torch.long)
        ans_mask = (labels != self.pad_token_id).long()

        combined_mask = torch.cat([inputs_mask, ans_mask], dim=1)

        # 2. Labels for Loss Calculation
        # Ignore Prefix (-100)
        ignore_prefix = torch.full((batch_size, prefix_len), -100, device=device, dtype=labels.dtype)

        labels_padding = labels.clone()
        labels_padding[labels == self.pad_token_id] = -100

        combined_labels = torch.cat([ignore_prefix, labels_padding], dim=1)

        # --- [Phase 5] Generation / Loss ---
        outputs = self.composer(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels
        )

        return outputs
