import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer

# Import your model definition
from models.architecture import PECEngine


class PECPredictor:
    """Wrapper class for running inference with the PEC model.

    Handles loading the trained weights, processing inputs for both Profiler
    and Composer, and executing the generation loop with soft prompt injection.
    """

    def __init__(
            self,
            checkpoint_dir: str,
            profiler_path: str = "models/profiler",  # Path to local or HF model
            composer_path: str = "Qwen/Qwen3-1.7B",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initializes the predictor.

        Args:
            checkpoint_dir: Path to the directory containing 'pytorch_model.bin' (Extruder weights).
            profiler_path: Path or HF ID for the Profiler (Encoder).
            composer_path: Path or HF ID for the Composer (Decoder).
            device: Device to run inference on.
        """
        self.device = torch.device(device)
        self.base_dir = Path(checkpoint_dir)

        print(f"Loading PEC Engine on {self.device}...")

        # 1. Initialize Model Architecture
        self.model = PECEngine(
            profiler_path=profiler_path,
            composer_path=composer_path,
            freeze_profiler=True,  # No gradients needed for inference
            freeze_composer=True
        )

        # 2. Load Trained Weights (Extruder & Projector)
        weights_path = self.base_dir / "pytorch_model.bin"
        if weights_path.exists():
            print(f"Loading trained weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            # Load with strict=False because we only care about Extruder/Projector
            # keys matching. Profiler/Composer might be loaded from base config.
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print("Warning: No checkpoint found! Using random initialization (Debug mode).")

        self.model.to(self.device)
        self.model.eval()

        # 3. Load Tokenizers
        print("Loading tokenizers...")
        self.profiler_tokenizer = AutoTokenizer.from_pretrained(profiler_path)
        self.composer_tokenizer = AutoTokenizer.from_pretrained(composer_path, trust_remote_code=True)

    @torch.no_grad()
    def predict(
            self,
            query: str,
            context: str,
            max_new_tokens: int = 512,
            temperature: float = 0.7
    ) -> str:
        """Generates an answer for a given query and context.

        Args:
            query: The user question.
            context: The long document context.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated answer string.
        """

        # --- Step 1: Profiler Processing (Compression) ---
        # Format input for Profiler
        prof_text = f"Instruction: {query}\nContext: {context}"
        prof_inputs = self.profiler_tokenizer(
            prof_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192  # Adjust based on your Profiler's limit
        ).to(self.device)

        # Forward pass through Profiler + Extruder + Projector
        # Extract logic from PECEngine to get soft prompts manually
        prof_outputs = self.model.profiler(**prof_inputs)
        prof_hidden = prof_outputs.last_hidden_state

        extruded = self.model.extruder(context=prof_hidden, attn_mask=prof_inputs["attention_mask"])
        extruded = self.model.post_extruder_norm(extruded)
        soft_prompts = self.model.projector(extruded)  # Shape: [1, Num_Query, D_comp]

        # --- Step 2: Composer Input Preparation ---
        # Unlike training, we provide the CLEAN context (or slightly processed)
        # to leverage the "Deep Reading" effect (Soft Prompts + Original Text).

        # ChatML Format
        prompt = (
            f"<|im_start|>user\n"
            f"{query}\n\nContext:\n{context}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        comp_inputs = self.composer_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # Leave space for generation
        ).to(self.device)

        # Get embeddings for the text part
        text_embeds = self.model.composer.get_input_embeddings()(comp_inputs["input_ids"])

        # --- Step 3: Combine Embeddings (Injection) ---
        # Prepend Soft Prompts to the actual text embeddings
        # Input: [Soft Prompts] + [User Instruction & Context]
        final_inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)

        # Create Attention Mask
        # 1s for soft prompts + original mask
        prompt_len = soft_prompts.shape[1]
        batch_size = soft_prompts.shape[0]
        prompt_mask = torch.ones((batch_size, prompt_len), device=self.device, dtype=torch.long)
        final_attention_mask = torch.cat([prompt_mask, comp_inputs["attention_mask"]], dim=1)

        # --- Step 4: Generation ---
        output_ids = self.model.composer.generate(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=self.composer_tokenizer.eos_token_id,
            eos_token_id=self.composer_tokenizer.eos_token_id
        )

        # Decode output
        generated_text = self.composer_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text


def main():
    # Example Usage
    checkpoint_dir = "outputs/extruder"  # Where train_pec.py saved the model

    # Check if checkpoint exists
    if not Path(checkpoint_dir).exists():
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found.")
        print("Please run train_pec.py first.")
        return

    predictor = PECPredictor(checkpoint_dir=checkpoint_dir)

    # Test Data
    context = """
        The 'PEC Model' is a novel architecture consisting of a Profiler, Extruder, and Composer.
        The Profiler uses Modern-BERT to encode long contexts. 
        The Extruder uses Learnable Queries to compress information into soft prompts.
        The Composer (Qwen) uses these prompts to generate answers.
        """
    query = "What represents the Profiler in the PEC architecture?"

    print("\n" + "=" * 50)
    print(f"Query: {query}")
    print("-" * 50)

    answer = predictor.predict(query, context)

    print(f"Answer: {answer}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()