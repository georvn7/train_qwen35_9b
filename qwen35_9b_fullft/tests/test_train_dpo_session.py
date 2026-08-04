#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_dpo_session.py"
SPEC = importlib.util.spec_from_file_location("train_dpo_session", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN_DPO_SESSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_DPO_SESSION)


class FakeTokenizer:
    eos_token_id = 999

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
        **_kwargs,
    ):
        assert tokenize is False
        parts = []
        for message in messages:
            parts.append(f"<{message['role']}>")
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                parts.append(f"<think>{reasoning}</think>")
            parts.append(str(message["content"]))
            parts.append(f"</{message['role']}>")
        text = "".join(parts)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, value, *, add_special_tokens=False):
        if not isinstance(value, str):
            raise TypeError(f"expected rendered string, got {type(value).__name__}")
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in value]}


class BuildTokenizedRowsTests(unittest.TestCase):
    def test_preserves_conversational_fields_while_tokenizing_rendered_text(self):
        row = {
            "prompt": [
                {"role": "system", "content": "debug"},
                {"role": "user", "content": "next"},
            ],
            "chosen": [{"role": "assistant", "content": "inspect parser"}],
            "rejected": [{"role": "assistant", "content": "guess"}],
        }

        tokenized, stats = TRAIN_DPO_SESSION.build_tokenized_rows(
            [row], FakeTokenizer(), max_prompt_length=4096, max_completion_length=1024
        )

        self.assertEqual(tokenized[0]["prompt"], row["prompt"])
        self.assertEqual(tokenized[0]["chosen"], row["chosen"])
        self.assertEqual(tokenized[0]["rejected"], row["rejected"])
        self.assertEqual(tokenized[0]["chosen_input_ids"][-1], 999)
        self.assertEqual(tokenized[0]["rejected_input_ids"][-1], 999)
        self.assertEqual(stats["rows"], 1)

    def test_keeps_plain_string_rows_supported(self):
        row = {"prompt": "debug", "chosen": "inspect", "rejected": "guess"}

        tokenized, _ = TRAIN_DPO_SESSION.build_tokenized_rows(
            [row], FakeTokenizer(), max_prompt_length=4096, max_completion_length=1024
        )

        self.assertEqual(tokenized[0]["prompt_input_ids"], [ord(c) for c in "debug"])
        self.assertEqual(tokenized[0]["chosen_input_ids"], [ord(c) for c in "inspect"] + [999])

    def test_maps_hen_thinking_to_qwen_reasoning_content(self):
        row = {
            "prompt": [{"role": "user", "content": "next"}],
            "chosen": [
                {
                    "role": "assistant",
                    "thinking": "inspect the observed value flow",
                    "content": "chosen action",
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "thinking": "repeat the stale hypothesis",
                    "content": "rejected action",
                }
            ],
        }

        rendered = TRAIN_DPO_SESSION.render_dpo_row_for_tokenization(
            row, FakeTokenizer()
        )

        self.assertIn("<think>inspect the observed value flow</think>", rendered["chosen"])
        self.assertIn("<think>repeat the stale hypothesis</think>", rendered["rejected"])
        self.assertNotIn("reasoning_content", row["chosen"][0])


if __name__ == "__main__":
    unittest.main()
