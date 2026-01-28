import json
from pathlib import Path


def main():
    """
    Placeholder preprocessing script for profiler data.
    Expects data/train_corpus.jsonl and produces data/qa_pairs.jsonl.
    """
    base_dir = Path(__file__).parent
    src_path = base_dir / "data" / "train_corpus.jsonl"
    dst_path = base_dir / "data" / "qa_pairs.jsonl"

    if not src_path.exists():
        raise FileNotFoundError(f"Missing source file: {src_path}")

    qa_pairs = []
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record.get("text", "")
            if not text:
                continue
            qa_pairs.append({"question": text, "answer": text})

    with dst_path.open("w", encoding="utf-8") as f:
        for item in qa_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
