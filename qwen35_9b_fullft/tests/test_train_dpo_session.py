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
    eos_token = "<eos>"

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

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids)


class BuildTokenizedRowsTests(unittest.TestCase):
    def test_does_not_duplicate_eos_emitted_by_chat_template(self):
        tokenizer = FakeTokenizer()
        token_ids = [1, tokenizer.eos_token_id, 10]

        result = TRAIN_DPO_SESSION.append_eos_if_missing(
            token_ids,
            "answer<eos>\n",
            tokenizer,
        )

        self.assertIs(result, token_ids)

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

    def test_truncates_oversized_user_content_without_losing_role_headers(self):
        row = {
            "prompt": [
                {"role": "system", "content": "stable contract"},
                {"role": "user", "content": "old evidence " * 80 + "DECISIVE_TAIL"},
            ],
            "chosen": [{"role": "assistant", "content": "inspect parser"}],
            "rejected": [{"role": "assistant", "content": "guess"}],
        }

        tokenized, stats = TRAIN_DPO_SESSION.build_tokenized_rows(
            [row], FakeTokenizer(), max_prompt_length=100, max_completion_length=1024
        )
        final_prompt = FakeTokenizer().decode(tokenized[0]["prompt_input_ids"])

        self.assertTrue(final_prompt.startswith("<system>stable contract</system><user>"))
        self.assertIn("DECISIVE_TAIL", final_prompt)
        self.assertLessEqual(len(tokenized[0]["prompt_input_ids"]), 100)
        self.assertEqual(stats["prompt_truncation_modes"], {"trim_user_content": 1})

    def test_drops_complete_old_turns_before_trimming_message_content(self):
        row = {
            "prompt": [
                {"role": "system", "content": "stable contract"},
                {"role": "user", "content": "obsolete evidence " * 30},
                {"role": "assistant", "content": "obsolete action"},
                {"role": "user", "content": "current decisive evidence"},
            ],
            "chosen": [{"role": "assistant", "content": "inspect parser"}],
            "rejected": [{"role": "assistant", "content": "guess"}],
        }

        tokenized, stats = TRAIN_DPO_SESSION.build_tokenized_rows(
            [row], FakeTokenizer(), max_prompt_length=100, max_completion_length=1024
        )
        final_prompt = FakeTokenizer().decode(tokenized[0]["prompt_input_ids"])

        self.assertEqual(
            final_prompt,
            "<system>stable contract</system><user>current decisive evidence</user>"
            "<assistant>",
        )
        self.assertEqual(stats["prompt_truncation_modes"], {"drop_messages": 1})

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
