#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "train_awr_session.py"
SPEC = importlib.util.spec_from_file_location("train_awr_session", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AWR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AWR)


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        self.calls.append(dict(kwargs))
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
        return [ord(character) for character in text]


def row(sample: str, weight: float) -> dict:
    return {
        "objective": "repair_distance_awr",
        "group_id": "repair-1",
        "sample_id": sample,
        "sample_weight": weight,
        "prompt": [{"role": "user", "content": "grounded evidence"}],
        "completion": [
            {
                "role": "assistant",
                "thinking": "trace the blocker",
                "content": "fix the producer",
            }
        ],
    }


class RepairDistanceAwrTrainingTests(unittest.TestCase):
    def test_tokenization_masks_prompt_and_keeps_thinking(self):
        records, stats = AWR.flatten_awr_rows(
            [row("sample-1", 0.4)], FakeTokenizer(), 4096
        )
        self.assertEqual(1, len(records))
        self.assertEqual(0.4, records[0]["weight"])
        self.assertEqual(1, stats["thinking_sequences"])
        self.assertGreater(records[0]["target_start"], 0)
        self.assertLess(records[0]["target_start"], len(records[0]["input_ids"]))
        self.assertEqual(stats["final_tokens_total"], len(records[0]["input_ids"]))

    def test_completion_nll_is_mean_negative_log_probability(self):
        self.assertAlmostEqual(
            1.5,
            float(AWR.completion_nll(torch.tensor([-1.0, -2.0]))),
        )

    def test_invalid_weight_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "sample_weight"):
            AWR.flatten_awr_rows(
                [row("sample-1", 0.0)], FakeTokenizer(), 4096
            )

    def test_missing_thinking_and_duplicate_identity_fail_closed(self):
        missing = row("sample-1", 0.5)
        del missing["completion"][0]["thinking"]
        with self.assertRaisesRegex(ValueError, "thinking-enabled"):
            AWR.flatten_awr_rows([missing], FakeTokenizer(), 4096)

        duplicate = row("sample-1", 0.6)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            AWR.flatten_awr_rows(
                [row("sample-1", 0.5), duplicate], FakeTokenizer(), 4096
            )


if __name__ == "__main__":
    unittest.main()
