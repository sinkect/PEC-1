from __future__ import annotations

import json
import math
import re
import signal
import statistics
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - optional at runtime
    np = None

try:
    import sympy as sp
except Exception:  # pragma: no cover - optional at runtime
    sp = None

try:
    import datetime
    import dateutil.relativedelta as relativedelta
except Exception:  # pragma: no cover - optional at runtime
    datetime = None
    relativedelta = None


RE2_REPO_DATA_BASE_URL = "https://raw.githubusercontent.com/Tebmer/Rereading-LLM-Reasoning/main/data"
RE2_DEFAULT_DATASETS: Sequence[str] = (
    "gsm",
    "svamp",
    "asdiv",
    "aqua",
    "multiarith",
    "mawpssingleeq",
    "mawpsaddsub",
    "commonsenseqa",
    "strategyqa",
    "arc_easy",
    "arc_challenge",
    "date_understanding",
    "coin_flip",
)
RE2_DEFAULT_ACTS: Sequence[str] = ("vanilla", "cot")
RE2_SUPPORTED_ACTS: Sequence[str] = ("vanilla", "cot", "ps", "pal")


@dataclass(frozen=True)
class Re2DatasetConfig:
    name: str
    file_name: str
    prompt_family: str
    category: str
    answer_kind: str
    pretty_name: str
    supports_pal: bool = False


DATASET_CONFIGS: Mapping[str, Re2DatasetConfig] = {
    "gsm": Re2DatasetConfig("gsm", "gsm.jsonl", "gsm", "arithmetic", "number", "GSM8K", True),
    "svamp": Re2DatasetConfig("svamp", "svamp.jsonl", "gsm", "arithmetic", "number", "SVAMP", True),
    "asdiv": Re2DatasetConfig("asdiv", "asdiv.jsonl", "gsm", "arithmetic", "number", "ASDiv", True),
    "aqua": Re2DatasetConfig("aqua", "aqua.jsonl", "aqua", "arithmetic", "choice", "AQuA", False),
    "multiarith": Re2DatasetConfig("multiarith", "multiarith.jsonl", "gsm", "arithmetic", "number", "MultiArith", True),
    "mawpssingleeq": Re2DatasetConfig("mawpssingleeq", "mawpssingleeq.jsonl", "gsm", "arithmetic", "number", "SingleEQ", True),
    "mawpsaddsub": Re2DatasetConfig("mawpsaddsub", "mawpsaddsub.jsonl", "gsm", "arithmetic", "number", "AddSub", True),
    "commonsenseqa": Re2DatasetConfig("commonsenseqa", "CommonsenseQA.jsonl", "commonsenseqa", "commonsense", "choice", "CommonsenseQA"),
    "strategyqa": Re2DatasetConfig("strategyqa", "strategyqa.jsonl", "strategyqa", "commonsense", "yes_no", "StrategyQA"),
    "arc_easy": Re2DatasetConfig("arc_easy", "arc_easy.jsonl", "commonsenseqa", "commonsense", "choice", "ARC-e"),
    "arc_challenge": Re2DatasetConfig("arc_challenge", "arc_challenge.jsonl", "commonsenseqa", "commonsense", "choice", "ARC-c"),
    "date_understanding": Re2DatasetConfig("date_understanding", "date_understanding.jsonl", "date_understanding", "symbolic", "date", "Date Understanding", True),
    "coin_flip": Re2DatasetConfig("coin_flip", "coin_flip.jsonl", "coin_flip", "symbolic", "yes_no", "Coin Flip", True),
}

DATASET_ALIASES = {
    "commonsenseqa.jsonl": "commonsenseqa",
    "commonsenseqa": "commonsenseqa",
    "csqa": "commonsenseqa",
    "arc-e": "arc_easy",
    "arc_easy": "arc_easy",
    "arceasy": "arc_easy",
    "arc-c": "arc_challenge",
    "arc_challenge": "arc_challenge",
    "arcchallenge": "arc_challenge",
    "date": "date_understanding",
    "date_understanding": "date_understanding",
    "coin": "coin_flip",
    "coin_flip": "coin_flip",
    "singleeq": "mawpssingleeq",
    "addsub": "mawpsaddsub",
}

ACT_TO_METHODOLOGY = {
    "vanilla": "",
    "cot": "Let's think step by step.",
    "ps": (
        "Let's first understand the problem and devise a plan to solve the problem. "
        "Then, let's carry out the plan, solve the problem step by step, and give the ultimate answer. "
        "Please explicitly generate the mentioned process: [Problem Understanding], [Plan], "
        "[Solving/Calculations], [Answer]. in your response."
    ),
    "pal": "",
}

TEXT_PROMPT = """
Q: {question}

Your answer should be in the form \\boxed{{answer}}.
"""

TEXT_PROMPT_CHOICES = """
Q: {question}

Your answer should be in the form \\boxed{{choice}}, e.g. \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}. There is only one correct choice.
"""

TEXT_PROMPT_YES_NO = """
Q: {question}

Your answer should be either \\boxed{{yes}} or \\boxed{{no}}, in the form \\boxed{{answer}}.
"""

TEXT_PROMPT_GSM = """
Q: {question}

Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.
"""

TEXT_PROMPT_COIN = """
Q: {question}

Your answer should be \\boxed{{yes}} or \\boxed{{no}}, in the form \\boxed{{answer}}.
"""

TEXT_PROMPT_DATE = """
Q: {question}

Your answer should be a date, in the format of \\boxed{{MM/DD/YYYY}}, e.g. \\boxed{{05/01/2022}}.
"""

PAL_PROMPT_GSM = """
#!/bin/python3

import math
import numpy as np
import statistics
import sympy as sp

#################### Task Instruction ####################
# You will write python program to solve math problems.
# You will only write code blocks.
# Please generate your code block in `def solution()` function, and thus it can be executed by python interpreter. You don't need to call `solution()` function because it will be called by the system.
# The concrete format of `solution()` is as follows:
# def solution():
#   \"\"\"<question>\"\"\"
#   <your code>
#   result = <your result>
#   return result
##########################################################

# To better solve the probelm, 1) you can define variables first and then use them in your code block. 2) You can add comments to the code. 3) You can use the imported python package to solve the problem.

# Q: {question}

# Your defined "solution()" function with comments here.
""".strip()

PAL_PROMPT_COIN = """
# You will write python program to solve the coin flip problems.
# You will only write code blocks.
# Please generate your code block in `def solution()` function, and thus it can be executed by python interpreter. You don't need to call `solution()` function because it will be called by the system.
# The concrete format of `solution()` is as follows:
# def solution():
#   \"\"\"
#   :return: "yes" or "no"
#   \"\"\"
#   <your code>
#   result = <your result>
#   assert result in ["yes", "no"]
#   return result # "yes" or "no"
#
# To better solve the probelm, 1) you can define variables first and then use them in your code block. 2) You can add comments to the code. 3) If you use the library, you must import it by `import {{module}}` in the first line of your code block.

#!/bin/python3

# Q: {question}
# Your "solution()" function here. Remember only to return "yes" or "no".
""".strip()

PAL_PROMPT_DATE = """
#!/bin/python3

import datetime
import dateutil.relativedelta as relativedelta

#################### Task Instruction ####################
# You will write python program to solve the date understanding problems.
# You will only write code blocks.
# Please generate your code block in `def solution()` function, and thus it can be executed by python interpreter. You don't need to call `solution()` function because it will be called by the system.
# The concrete format of `solution()` is as follows:
# def solution():
#   <your code>
#   return result # "MM/DD/YYYY" by strftime('%m/%d/%Y').
##########################################################

# [TIPS]: To better solve the problem, 1) you can define variables first and then use them in your code block. 2) You can add comments to the code. 3) You can use relevant libraries, including `datetime.datetime`, `dateutil.relativedelta.relativedelta`, among others. 4) NOTICE! The 'today' in the [QUESTION] does not mean the date of the execution of the code, so you don't need and can NOT use the `datetime.datetime.now()` or `datetime.date.today()` to get the today date.

# [QUESTION]: "{question}"
# [RESPONSE FORMAT] You need to infer the date from context, and return the result in the format of `MM/DD/YYYY`.
# Your defined "solution()" function here. Remember you DON'T need and CAN NOT use the `datetime.datetime.now()` or `datetime.date.today()`.
""".strip()

PROMPT_FAMILY_TO_TEMPLATE = {
    "gsm": {"text": TEXT_PROMPT_GSM, "pal": PAL_PROMPT_GSM},
    "aqua": {"text": TEXT_PROMPT_CHOICES, "pal": None},
    "commonsenseqa": {"text": TEXT_PROMPT_CHOICES, "pal": None},
    "strategyqa": {"text": TEXT_PROMPT_YES_NO, "pal": None},
    "date_understanding": {"text": TEXT_PROMPT_DATE, "pal": PAL_PROMPT_DATE},
    "coin_flip": {"text": TEXT_PROMPT_COIN, "pal": PAL_PROMPT_COIN},
}


def normalize_dataset_name(name: str) -> str:
    normalized = Path(str(name)).name.lower().removesuffix(".jsonl")
    normalized = normalized.replace("-", "_")
    return DATASET_ALIASES.get(normalized, normalized)


def get_dataset_config(name: str) -> Re2DatasetConfig:
    dataset_name = normalize_dataset_name(name)
    if dataset_name not in DATASET_CONFIGS:
        raise KeyError(f"Unsupported RE2 benchmark dataset: {name}")
    return DATASET_CONFIGS[dataset_name]


def dataset_supports_act(dataset_name: str, act: str) -> bool:
    if act not in RE2_SUPPORTED_ACTS:
        raise ValueError(f"Unsupported act: {act}")
    if act != "pal":
        return True
    return get_dataset_config(dataset_name).supports_pal


def ensure_re2_benchmark_dataset(
    dataset_name: str,
    data_dir: Path,
    *,
    download_if_missing: bool = True,
) -> Path:
    config = get_dataset_config(dataset_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / config.file_name
    if path.exists():
        return path
    if not download_if_missing:
        raise FileNotFoundError(f"Benchmark dataset not found: {path}")

    url = f"{RE2_REPO_DATA_BASE_URL}/{config.file_name}"
    try:
        with urllib.request.urlopen(url) as response:
            payload = response.read()
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Failed to download benchmark dataset from {url}") from exc

    path.write_bytes(payload)
    return path


def load_re2_benchmark_examples(
    dataset_name: str,
    data_dir: Path,
    *,
    max_samples: int | None = None,
    download_if_missing: bool = True,
) -> List[Dict[str, Any]]:
    dataset_path = ensure_re2_benchmark_dataset(
        dataset_name,
        data_dir,
        download_if_missing=download_if_missing,
    )
    examples: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def _build_commonsenseqa_question(example: Mapping[str, Any]) -> str:
    choice = "Answer Choices:"
    for answer_choice in example["question"]["choices"]:
        choice += f" ({answer_choice['label']}) {answer_choice['text']}"
    return f"{example['question']['stem'].strip()} {choice}"


def _build_aqua_question(example: Mapping[str, Any]) -> str:
    choice = "(" + "(".join(example["options"])
    choice = choice.replace("(", " (").replace(")", ") ")
    return f"{example['question'].strip()} Answer Choices:{choice}"


def build_re2_question(example: Mapping[str, Any], dataset_name: str) -> str:
    config = get_dataset_config(dataset_name)
    if config.name in {"commonsenseqa", "arc_easy", "arc_challenge"}:
        return _build_commonsenseqa_question(example)
    if config.name == "aqua":
        return _build_aqua_question(example)
    return str(example.get("input", example.get("question", ""))).strip()


def build_re2_prompt(
    example: Mapping[str, Any],
    dataset_name: str,
    *,
    act: str,
    read_times: int,
) -> str:
    if read_times < 1:
        raise ValueError("read_times must be at least 1")
    if act not in RE2_SUPPORTED_ACTS:
        raise ValueError(f"Unsupported act: {act}")
    if act == "pal" and not dataset_supports_act(dataset_name, act):
        raise ValueError(f"{dataset_name} does not support act={act}")

    question = build_re2_question(example, dataset_name)
    reread_prefix = "# Read the question again: " if act == "pal" else "Read the question again: "
    repeated_question = f" \n\n{reread_prefix}".join([question] * read_times)

    config = get_dataset_config(dataset_name)
    if act == "pal":
        prompt_template = PROMPT_FAMILY_TO_TEMPLATE[config.prompt_family]["pal"]
        if prompt_template is None:
            raise ValueError(f"{dataset_name} does not have a PAL prompt template")
        return prompt_template.format(question=repeated_question)

    prompt_template = PROMPT_FAMILY_TO_TEMPLATE[config.prompt_family]["text"]
    prompt = prompt_template.format(question=repeated_question)
    methodology = ACT_TO_METHODOLOGY[act]
    if methodology:
        prompt += f"\nA: {methodology}"
    else:
        prompt += "\nA:"
    return prompt


def get_reference_answer_text(example: Mapping[str, Any], dataset_name: str) -> str:
    config = get_dataset_config(dataset_name)
    if config.answer_kind == "number":
        return str(example["target"])
    if config.answer_kind == "choice":
        if config.name == "aqua":
            return str(example["correct"])
        return str(example["answerKey"])
    if config.answer_kind == "yes_no":
        return str(example["target"])
    if config.answer_kind == "date":
        for answer, score in example["target_scores"].items():
            if float(score) == 1.0:
                return str(answer)
        return next(iter(example["target_scores"]))
    raise ValueError(f"Unsupported answer kind: {config.answer_kind}")


def get_example_id(example: Mapping[str, Any], fallback_index: int) -> str:
    for key in ("id", "_id", "question_id"):
        value = example.get(key)
        if value is not None:
            return str(value)
    return str(fallback_index)


def _extract_first_number(text: str) -> str | None:
    matches = re.findall(r"[\-+]?\d*[\.,/]?\d+", text)
    return matches[0] if matches else None


def _extract_last_number(text: str) -> str | None:
    matches = re.findall(r"[\-+]?\d*[\.,/]?\d+", text)
    return matches[-1] if matches else None


def _extract_boxed_candidates(text: str) -> List[str]:
    return [candidate.strip() for candidate in re.findall(r"boxed\{(.*?)\}", str(text), flags=re.DOTALL) if candidate.strip()]


def _extract_preferred_number(text: str) -> str | None:
    equality_matches = re.findall(r"=\s*([\-+]?\d*[\.,/]?\d+)", text)
    if equality_matches:
        return equality_matches[-1].replace(",", "")

    cue_matches = re.findall(
        r"(?:final answer|answer is|therefore|thus|hence|so|result(?: is)?|we get)\D*([\-+]?\d*[\.,/]?\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if cue_matches:
        return cue_matches[-1].replace(",", "")

    matches = re.findall(r"[\-+]?\d*[\.,/]?\d+", text)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0].replace(",", "")
    return matches[-1].replace(",", "")


def parse_answer_number(text: str, example: Mapping[str, Any] | None = None) -> str | None:
    candidate_answer = None
    boxed_matches = _extract_boxed_candidates(text)
    if boxed_matches:
        candidate_answer = boxed_matches[-1]

    if candidate_answer is None:
        parts = text.split("answer is ")
        if len(parts) > 1:
            candidate_answer = parts[-1].strip()

    if candidate_answer is None:
        last_paragraph = text.split("\n")[-1]
        candidate_answer = last_paragraph.split(". ")[-1]

    if candidate_answer is not None:
        preferred = _extract_preferred_number(candidate_answer)
        if preferred is not None:
            return preferred

    fallback = _extract_last_number(text)
    if fallback is None:
        return None
    try:
        return str(float(fallback))
    except Exception:
        return None


def parse_answer_number_strict(text: str, example: Mapping[str, Any] | None = None) -> str | None:
    for candidate in reversed(_extract_boxed_candidates(text)):
        preferred = _extract_preferred_number(candidate)
        if preferred is not None:
            return preferred
    return None


def parse_answer_yes_or_no(text: str, example: Mapping[str, Any] | None = None) -> str | None:
    boxed_matches = re.findall(r"\{([a-zA-Z]*)\}", text)
    for match_str in reversed(boxed_matches):
        lowered = match_str.lower()
        if "yes" in lowered:
            return "yes"
        if "no" in lowered:
            return "no"

    matches = re.findall(r"(yes|no)", text.lower())
    return matches[-1] if matches else None


def parse_answer_yes_or_no_strict(text: str, example: Mapping[str, Any] | None = None) -> str | None:
    for match_str in reversed(_extract_boxed_candidates(text)):
        lowered = match_str.lower()
        if "yes" in lowered:
            return "yes"
        if "no" in lowered:
            return "no"
    return None


def parse_answer_date(text: str, example: Mapping[str, Any] | None = None) -> str | None:
    def normalize_date(answer: str) -> str | None:
        parts = [part.strip() for part in str(answer).split("/") if part.strip()]
        if len(parts) != 3:
            return None
        month, day, year = parts
        if not (month.isdigit() and day.isdigit() and year.isdigit()):
            return None
        if len(month) == 1:
            month = "0" + month
        if len(day) == 1:
            day = "0" + day
        if len(year) == 2:
            year = "20" + year
        return "/".join((month, day, year))

    for match_str in reversed(_extract_boxed_candidates(text)):
        normalized = normalize_date(match_str)
        if normalized is not None:
            return normalized

    matches = re.findall(r"(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/([0-9]{4})", text)
    if matches:
        return normalize_date("/".join(matches[-1]))
    return None


def parse_answer_date_strict(text: str, example: Mapping[str, Any] | None = None) -> str | None:
    for match_str in reversed(_extract_boxed_candidates(text)):
        parts = [part.strip() for part in str(match_str).split("/") if part.strip()]
        if len(parts) != 3:
            continue
        month, day, year = parts
        if not (month.isdigit() and day.isdigit() and year.isdigit()):
            continue
        if len(month) == 1:
            month = "0" + month
        if len(day) == 1:
            day = "0" + day
        if len(year) == 2:
            year = "20" + year
        return "/".join((month, day, year))
    return None


def _score_string_similarity(left: str, right: str) -> float:
    if left == right:
        return 2.0
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(set(left_tokens) & set(right_tokens))
    return overlap / max(len(left_tokens), len(right_tokens))


def _best_matching_index(options: Sequence[str], prediction_text: str) -> int:
    scores = [_score_string_similarity(option, prediction_text) for option in options]
    if np is not None:
        return int(np.argmax(scores))
    return max(range(len(scores)), key=scores.__getitem__)


def parse_multiple_choice_answer(text: str, example: Mapping[str, Any]) -> str:
    boxed_matches = _extract_boxed_candidates(text)
    if boxed_matches:
        text = boxed_matches[-1]
    else:
        text = text.split("\n")[-1]
    prediction = text.strip()

    explicit_choice = re.findall(r"\(([A-E])\)", prediction)
    if explicit_choice:
        return explicit_choice[-1]

    bare_choice = re.findall(r"(?<![a-zA-Z])[ABCDE](?![a-zA-Z])", prediction)
    if bare_choice:
        return bare_choice[-1]

    options = [choice["text"] for choice in example["question"]["choices"]]
    labels = [choice["label"] for choice in example["question"]["choices"]]
    return labels[_best_matching_index(options, text)]


def parse_multiple_choice_answer_strict(text: str, example: Mapping[str, Any]) -> str | None:
    boxed_matches = _extract_boxed_candidates(text)
    if not boxed_matches:
        return None
    prediction = boxed_matches[-1].strip()

    explicit_choice = re.findall(r"\(([A-E])\)", prediction)
    if explicit_choice:
        return explicit_choice[-1]

    bare_choice = re.findall(r"(?<![a-zA-Z])[ABCDE](?![a-zA-Z])", prediction)
    if bare_choice:
        return bare_choice[-1]

    options = [choice["text"] for choice in example["question"]["choices"]]
    labels = [choice["label"] for choice in example["question"]["choices"]]
    return labels[_best_matching_index(options, prediction)]


def parse_multiple_choice_aqua(text: str, example: Mapping[str, Any]) -> str:
    boxed_matches = _extract_boxed_candidates(text)
    if boxed_matches:
        text = boxed_matches[-1]
    else:
        text = text.split("\n")[-1]
    prediction = text.strip()

    explicit_choice = re.findall(r"\(([A-E])\)", prediction)
    if explicit_choice:
        return explicit_choice[-1]

    bare_choice = re.findall(r"(?<![a-zA-Z])[ABCDE](?![a-zA-Z])", prediction)
    if bare_choice:
        return bare_choice[-1]

    labels = ["A", "B", "C", "D", "E"]
    return labels[_best_matching_index(example["options"], text)]


def parse_multiple_choice_aqua_strict(text: str, example: Mapping[str, Any]) -> str | None:
    boxed_matches = _extract_boxed_candidates(text)
    if not boxed_matches:
        return None
    prediction = boxed_matches[-1].strip()

    explicit_choice = re.findall(r"\(([A-E])\)", prediction)
    if explicit_choice:
        return explicit_choice[-1]

    bare_choice = re.findall(r"(?<![a-zA-Z])[ABCDE](?![a-zA-Z])", prediction)
    if bare_choice:
        return bare_choice[-1]

    labels = ["A", "B", "C", "D", "E"]
    return labels[_best_matching_index(example["options"], prediction)]


def _strip_math_string(value: str) -> str:
    text = str(value).strip()
    text = text.replace(" ", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "")
    text = text.replace("$", "")
    text = text.replace("\\,", "")
    return text


def _remove_boxed(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("\\boxed{") and value.endswith("}"):
        return value[len("\\boxed{"):-1]
    if value.startswith("boxed{") and value.endswith("}"):
        return value[len("boxed{"):-1]
    return value


def _last_boxed_only_string(text: str) -> str | None:
    match_positions = list(re.finditer(r"(\\boxed|boxed)\{", text))
    if not match_positions:
        return None
    start = match_positions[-1].start()
    brace_depth = 0
    started = False
    result_chars: List[str] = []
    for char in text[start:]:
        result_chars.append(char)
        if char == "{":
            brace_depth += 1
            started = True
        elif char == "}":
            brace_depth -= 1
            if started and brace_depth == 0:
                break
    return "".join(result_chars)


def parse_math_answer(text: str, example: Mapping[str, Any] | None = None) -> str | None:
    candidate = _remove_boxed(_last_boxed_only_string(text))
    if candidate is None:
        return None
    return _strip_math_string(candidate)


def parse_text_answer(text: str, example: Mapping[str, Any] | None = None) -> str | None:
    boxed_matches = re.findall(r"boxed\{(.*?)\}", text)
    return boxed_matches[-1].strip() if boxed_matches else None


def parse_text_answer_strict(text: str, example: Mapping[str, Any] | None = None) -> str | None:
    boxed_matches = _extract_boxed_candidates(text)
    return boxed_matches[-1] if boxed_matches else None


def normalize_numeric_answer(text: str, unit: str | None = None) -> str:
    normalized = str(text).strip()
    normalized = re.sub(r"^\$", "", normalized)
    normalized = re.sub(r"[,\./]$", "", normalized)
    if re.match(r"^[-+]?[\d,./]+$", normalized):
        normalized = normalized.replace(",", "")
        if re.match(r"[-+]?\d+$", normalized):
            number = int(normalized)
        elif "/" in normalized:
            numerator, denominator = normalized.split("/", 1)
            number = round(float(numerator) / float(denominator), 3)
        else:
            number = round(float(normalized), 3)
        normalized = re.sub(r"\.[0]+$", "", str(number))
        return normalized
    if unit:
        normalized = normalized.replace(unit, "").strip()
    return normalized


@contextmanager
def _time_limit(seconds: int):
    if seconds <= 0:
        yield
        return

    def handler(signum, frame):  # pragma: no cover - signal handler
        raise TimeoutError("PAL program execution timed out")

    previous_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:  # pragma: no cover - trivial cleanup
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _extract_code_block(text: str) -> str:
    fenced_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_blocks:
        return fenced_blocks[-1].strip()
    return text.strip()


def execute_pal_solution(text: str, dataset_name: str, *, timeout_seconds: int = 3) -> str | None:
    source = _extract_code_block(text)
    if not source:
        return None

    execution_globals: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "math": math,
        "statistics": statistics,
    }
    if np is not None:
        execution_globals["np"] = np
    if sp is not None:
        execution_globals["sp"] = sp
    if datetime is not None:
        execution_globals["datetime"] = datetime
    if relativedelta is not None:
        execution_globals["relativedelta"] = relativedelta

    try:
        with _time_limit(timeout_seconds):
            exec(source, execution_globals, execution_globals)
            solution = execution_globals.get("solution")
            if not callable(solution):
                return None
            result = solution()
    except Exception:
        return None

    if result is None:
        return None
    return str(result)


def parse_re2_answer(
    text: str,
    dataset_name: str,
    example: Mapping[str, Any],
    *,
    act: str = "vanilla",
    mode: str = "relaxed",
) -> Any:
    if mode not in {"strict", "relaxed"}:
        raise ValueError(f"Unsupported parse mode: {mode}")

    if act == "pal":
        executed_result = execute_pal_solution(text, dataset_name)
        if executed_result is None:
            return None
        text = executed_result
        mode = "relaxed"

    config = get_dataset_config(dataset_name)
    if mode == "strict":
        if config.answer_kind == "number":
            return parse_answer_number_strict(text, example)
        if config.answer_kind == "yes_no":
            return parse_answer_yes_or_no_strict(text, example)
        if config.answer_kind == "date":
            return parse_answer_date_strict(text, example)
        if config.name == "aqua":
            return parse_multiple_choice_aqua_strict(text, example)
        if config.answer_kind == "choice":
            return parse_multiple_choice_answer_strict(text, example)
        if config.name == "MATH":
            return parse_math_answer(text, example)
        return parse_text_answer_strict(text, example)

    if config.answer_kind == "number":
        return parse_answer_number(text, example)
    if config.answer_kind == "yes_no":
        return parse_answer_yes_or_no(text, example)
    if config.answer_kind == "date":
        return parse_answer_date(text, example)
    if config.name == "aqua":
        return parse_multiple_choice_aqua(text, example)
    if config.answer_kind == "choice":
        return parse_multiple_choice_answer(text, example)
    if config.name == "MATH":
        return parse_math_answer(text, example)
    return parse_text_answer(text, example)


def score_re2_prediction(dataset_name: str, prediction: Any, example: Mapping[str, Any]) -> int:
    config = get_dataset_config(dataset_name)
    if prediction is None:
        return 0
    if config.answer_kind == "number":
        try:
            return int(abs(float(prediction) - float(example["target"])) < 1e-3)
        except Exception:
            return 0
    if config.answer_kind == "yes_no":
        return int(str(prediction).lower() == str(example["target"]).lower())
    if config.answer_kind == "date":
        return int(float(example["target_scores"].get(str(prediction), 0)) > 0)
    if config.name == "aqua":
        return int(str(prediction).lower() == str(example["correct"]).lower())
    if config.answer_kind == "choice":
        return int(str(prediction).lower() == str(example["answerKey"]).lower())
    return 0


def summarize_re2_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    num_samples = len(records)
    if num_samples == 0:
        return {
            "num_samples": 0,
            "strict_correct_num": 0,
            "strict_accuracy": 0.0,
            "strict_null_num": 0,
            "strict_null_rate": 0.0,
            "relaxed_correct_num": 0,
            "relaxed_accuracy": 0.0,
            "relaxed_null_num": 0,
            "relaxed_null_rate": 0.0,
            "correct_num": 0,
            "accuracy": 0.0,
            "null_num": 0,
            "null_rate": 0.0,
            "no_boxed_num": 0,
            "no_boxed_rate": 0.0,
        }

    strict_correct_num = sum(int(record.get("strict_score", 0)) for record in records)
    strict_null_num = sum(1 for record in records if record.get("strict_parsed_prediction") is None)
    relaxed_correct_num = sum(int(record.get("relaxed_score", record.get("score", 0))) for record in records)
    relaxed_null_num = sum(
        1
        for record in records
        if record.get("relaxed_parsed_prediction", record.get("parsed_prediction")) is None
    )
    no_boxed_num = sum(int(record.get("no_boxed", 0)) for record in records)
    return {
        "num_samples": num_samples,
        "strict_correct_num": strict_correct_num,
        "strict_accuracy": strict_correct_num / num_samples,
        "strict_null_num": strict_null_num,
        "strict_null_rate": strict_null_num / num_samples,
        "relaxed_correct_num": relaxed_correct_num,
        "relaxed_accuracy": relaxed_correct_num / num_samples,
        "relaxed_null_num": relaxed_null_num,
        "relaxed_null_rate": relaxed_null_num / num_samples,
        "correct_num": relaxed_correct_num,
        "accuracy": relaxed_correct_num / num_samples,
        "null_num": relaxed_null_num,
        "null_rate": relaxed_null_num / num_samples,
        "no_boxed_num": no_boxed_num,
        "no_boxed_rate": no_boxed_num / num_samples,
    }


def build_arc_total_summary(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        dataset_name = str(row.get("dataset_name"))
        if dataset_name not in {"arc_easy", "arc_challenge"}:
            continue
        grouping_key = (
            row.get("kind"),
            row.get("model_name"),
            row.get("act"),
            row.get("read_times"),
        )
        bucket = grouped.setdefault(
            grouping_key,
            {
                "dataset_name": "arc_total",
                "dataset_label": "ARC-t",
                "category": "commonsense",
                "kind": row.get("kind"),
                "model_name": row.get("model_name"),
                "act": row.get("act"),
                "read_times": row.get("read_times"),
                "num_samples": 0,
                "strict_correct_num": 0,
                "strict_null_num": 0,
                "relaxed_correct_num": 0,
                "relaxed_null_num": 0,
                "correct_num": 0,
                "null_num": 0,
                "no_boxed_num": 0,
            },
        )
        bucket["num_samples"] += int(row.get("num_samples", 0))
        bucket["strict_correct_num"] += int(row.get("strict_correct_num", 0))
        bucket["strict_null_num"] += int(row.get("strict_null_num", 0))
        bucket["relaxed_correct_num"] += int(row.get("relaxed_correct_num", row.get("correct_num", 0)))
        bucket["relaxed_null_num"] += int(row.get("relaxed_null_num", row.get("null_num", 0)))
        bucket["correct_num"] += int(row.get("correct_num", 0))
        bucket["null_num"] += int(row.get("null_num", 0))
        bucket["no_boxed_num"] += int(row.get("no_boxed_num", 0))

    summaries: List[Dict[str, Any]] = []
    for row in grouped.values():
        num_samples = row["num_samples"]
        if num_samples == 0:
            continue
        row["strict_accuracy"] = row["strict_correct_num"] / num_samples
        row["strict_null_rate"] = row["strict_null_num"] / num_samples
        row["relaxed_accuracy"] = row["relaxed_correct_num"] / num_samples
        row["relaxed_null_rate"] = row["relaxed_null_num"] / num_samples
        row["accuracy"] = row["correct_num"] / num_samples
        row["null_rate"] = row["null_num"] / num_samples
        row["no_boxed_rate"] = row["no_boxed_num"] / num_samples
        summaries.append(row)
    return summaries
