import torch

from models.architecture import PECEngine


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def generate(model, doc_text, prompt_text):
    device = get_device()

    profiler_inputs = model.profiler_tokenizer(
        doc_text, return_tensors="pt", truncation=True, max_length=256
    )
    composer_inputs = model.composer_tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=256
    )

    profiler_input_ids = profiler_inputs["input_ids"].to(device)
    profiler_attention_mask = profiler_inputs["attention_mask"].to(device)
    composer_input_ids = composer_inputs["input_ids"].to(device)
    composer_attention_mask = composer_inputs["attention_mask"].to(device)

    with torch.no_grad():
        prefix, composer_hidden = model(
            profiler_input_ids,
            profiler_attention_mask,
            composer_input_ids,
            composer_attention_mask,
        )

    return prefix, composer_hidden


def main():
    model = PECEngine().to(get_device(), dtype=torch.float16)
    doc_text = "Profile this document."
    prompt_text = "Generate a concise summary."
    prefix, composer_hidden = generate(model, doc_text, prompt_text)
    print(prefix.shape, composer_hidden.shape)


if __name__ == "__main__":
    main()
