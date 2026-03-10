import random
import sys
import unittest
from unittest import mock
from models.eval_utils import (
    aggregate_metrics,
    apply_mask_deterministically,
    exact_match_score,
    render_qwen_user_prompt,
    split_dataset_indices,
    strip_thinking_trace,
    thinking_mode_name,
    token_f1_score,
)


class FakeMasker:
    def __call__(self, text: str) -> str:
        return f"{text}|{random.random():.8f}"


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        payload = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        self.calls.append(payload)
        return "rendered"


class EvalUtilsTest(unittest.TestCase):
    def test_split_dataset_indices_is_disjoint_and_deterministic(self):
        first = split_dataset_indices(dataset_len=100, test_size=0.02, seed=42)
        second = split_dataset_indices(dataset_len=100, test_size=0.02, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first["eval"]), 2)
        self.assertEqual(len(first["train"]), 98)
        self.assertTrue(set(first["train"]).isdisjoint(set(first["eval"])))
        self.assertEqual(set(first["train"]) | set(first["eval"]), set(range(100)))

    def test_apply_mask_deterministically_restores_global_random_state(self):
        random.seed(123)
        expected_before = random.random()

        random.seed(123)
        masked_a = apply_mask_deterministically("prompt", FakeMasker(), seed=99)
        masked_b = apply_mask_deterministically("prompt", FakeMasker(), seed=99)
        after = random.random()

        self.assertEqual(masked_a, masked_b)
        self.assertEqual(after, expected_before)

    def test_metrics_helpers(self):
        self.assertEqual(exact_match_score("Answer", "answer"), 1.0)
        self.assertAlmostEqual(token_f1_score("a b c", "a c"), 0.8)

        summary = aggregate_metrics(
            [
                {"exact_match": 1.0, "token_f1": 0.5, "rougeL_f1": 0.4},
                {"exact_match": 0.0, "token_f1": 1.0, "rougeL_f1": 0.6},
            ]
        )
        self.assertEqual(summary["num_samples"], 2)
        self.assertAlmostEqual(summary["exact_match"], 0.5)
        self.assertAlmostEqual(summary["token_f1"], 0.75)
        self.assertAlmostEqual(summary["rougeL_f1"], 0.5)

    def test_render_qwen_user_prompt_forwards_enable_thinking(self):
        tokenizer = FakeTokenizer()
        rendered = render_qwen_user_prompt(tokenizer, "question", enable_thinking=True)

        self.assertEqual(rendered, "rendered")
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(tokenizer.calls[0]["messages"], [{"role": "user", "content": "question"}])
        self.assertFalse(tokenizer.calls[0]["tokenize"])
        self.assertTrue(tokenizer.calls[0]["add_generation_prompt"])
        self.assertTrue(tokenizer.calls[0]["enable_thinking"])
        self.assertEqual(thinking_mode_name(True), "thinking_on")
        self.assertEqual(thinking_mode_name(False), "thinking_off")

    def _import_get_sampling_params(self):
        """Import get_sampling_params while mocking heavy dependencies."""
        heavy_modules = ["torch", "safetensors", "safetensors.torch", "transformers"]
        sentinel = mock.MagicMock()
        patches = {m: sentinel for m in heavy_modules if m not in sys.modules}
        with mock.patch.dict(sys.modules, patches):
            # Remove cached evaluate_pec if already imported with stubs
            sys.modules.pop("evaluate_pec", None)
            from evaluate_pec import get_sampling_params
        return get_sampling_params

    def test_get_sampling_params_thinking_on(self):
        get_sampling_params = self._import_get_sampling_params()
        params = get_sampling_params(enable_thinking=True)
        self.assertAlmostEqual(params["temperature"], 0.6)
        self.assertAlmostEqual(params["top_p"], 0.95)
        self.assertEqual(params["top_k"], 20)
        self.assertAlmostEqual(params["min_p"], 0.0)

    def test_get_sampling_params_thinking_off(self):
        get_sampling_params = self._import_get_sampling_params()
        params = get_sampling_params(enable_thinking=False)
        self.assertAlmostEqual(params["temperature"], 0.7)
        self.assertAlmostEqual(params["top_p"], 0.8)
        self.assertEqual(params["top_k"], 20)
        self.assertAlmostEqual(params["min_p"], 0.0)

    def test_strip_thinking_trace(self):
        raw = "<think>step by step</think> final answer"
        self.assertEqual(strip_thinking_trace(raw), "final answer")
        self.assertEqual(strip_thinking_trace("plain answer"), "plain answer")


if __name__ == "__main__":
    unittest.main()

