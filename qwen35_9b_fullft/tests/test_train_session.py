#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_session.py"
SPEC = importlib.util.spec_from_file_location("train_session", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        **_kwargs,
    ):
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            reasoning = message.get("reasoning_content")
            if reasoning:
                text += f"<think>{reasoning}</think>"
            text += str(message.get("content", ""))
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        values = [ord(character) for character in text]
        return values if tokenize else text


class ThinkingTokenOrderTests(unittest.TestCase):
    def test_hen_thinking_precedes_final_assistant_content(self):
        messages = [
            {"role": "user", "content": "grounded evidence"},
            {
                "role": "assistant",
                "content": "fix the producer",
                "thinking": "trace the earliest mismatch",
            },
        ]

        normalized = TRAIN.normalize_messages_for_chat_template(messages)
        token_ids = TRAIN.apply_chat_template_token_ids(
            normalized,
            FakeTokenizer(),
            reasoning_effort="",
            add_generation_prompt=False,
        )
        rendered = "".join(chr(token_id) for token_id in token_ids)

        self.assertLess(
            rendered.index("<think>trace the earliest mismatch</think>"),
            rendered.index("fix the producer"),
        )
        self.assertNotIn("reasoning_content", messages[-1])


if __name__ == "__main__":
    unittest.main()
