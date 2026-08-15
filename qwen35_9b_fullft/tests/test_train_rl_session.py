#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_rl_session.py"
SPEC = importlib.util.spec_from_file_location("train_rl_session", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RL)


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
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

    def decode(self, values, **_kwargs):
        return "".join(chr(value) for value in values)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=256, hidden_size=12):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, logits_to_keep=None, **_kwargs):
        hidden = torch.cumsum(self.embedding(input_ids), dim=1)
        logits = self.projection(hidden)
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


def policy_step(thinking="inspect evidence", answer="fix parser"):
    return {
        "prompt": [
            {"role": "system", "content": "debug"},
            {"role": "user", "content": "next"},
        ],
        "completion": [
            {"role": "assistant", "thinking": thinking, "content": answer}
        ],
    }


class TokenizationTests(unittest.TestCase):
    def test_completion_mask_contains_thinking_and_final_answer(self):
        tokenized, stats = RL.tokenize_policy_step(
            policy_step(), FakeTokenizer(), max_length=4096
        )
        completion = FakeTokenizer().decode(
            tokenized["input_ids"][tokenized["target_start"] :]
        )
        self.assertIn("<think>inspect evidence</think>", completion)
        self.assertIn("fix parser", completion)
        self.assertLess(completion.index("<think>inspect evidence</think>"), completion.index("fix parser"))
        self.assertTrue(stats["thinking_present"])
        self.assertGreater(stats["completion_tokens"], len("fix parser"))

    def test_left_truncation_preserves_entire_completion(self):
        step = policy_step(answer="DECISIVE_ANSWER")
        step["prompt"][1]["content"] = "old evidence " * 100
        full, _ = RL.tokenize_policy_step(step, FakeTokenizer(), max_length=4096)
        clipped, stats = RL.tokenize_policy_step(step, FakeTokenizer(), max_length=150)
        full_completion = full["input_ids"][full["target_start"] :]
        clipped_completion = clipped["input_ids"][clipped["target_start"] :]
        self.assertEqual(full_completion, clipped_completion)
        self.assertGreater(stats["prompt_tokens_removed"], 0)
        self.assertEqual(len(clipped["input_ids"]), 150)

    def test_flatten_requires_normalized_step_weight(self):
        row = {
            "group_id": "g",
            "rollout_id": "r",
            "reward": 1.0,
            "advantage": 1.0,
            "policy_step_weight": 1.0,
            "policy_steps": [policy_step(), policy_step(answer="second")],
        }
        with self.assertRaisesRegex(ValueError, "policy_step_weight"):
            RL.flatten_rollouts([row], FakeTokenizer(), 4096)
        row["policy_step_weight"] = 0.5
        records, stats = RL.flatten_rollouts([row], FakeTokenizer(), 4096)
        self.assertEqual(len(records), 2)
        self.assertEqual(stats["thinking_sequences"], 2)


class ObjectiveTests(unittest.TestCase):
    def test_old_policy_precomputation_handles_multiple_records(self):
        model = TinyCausalLM()
        records = [
            {"input_ids": [1, 2, 3, 4], "target_start": 2},
            {"input_ids": [4, 3, 2, 1], "target_start": 1},
        ]

        values = RL.compute_old_policy_logps(model, records, autocast_dtype=None)

        self.assertEqual(2, len(values))
        self.assertEqual((2,), tuple(values[0].shape))
        self.assertEqual((3,), tuple(values[1].shape))
        self.assertTrue(all(not value.requires_grad for value in values))

    def test_identity_policy_has_expected_loss_and_zero_kl(self):
        old = torch.tensor([-1.0, -2.0])
        loss, metrics = RL.clipped_policy_objective(
            old.clone(), old, 2.0, clip_epsilon=0.2, kl_beta=0.01
        )
        self.assertAlmostEqual(float(loss), -2.0, places=6)
        self.assertEqual(float(metrics["approx_kl"]), 0.0)
        self.assertEqual(float(metrics["clip_fraction"]), 0.0)
        self.assertEqual(float(metrics["ratio_mean"]), 1.0)

    def test_negative_advantage_uses_clipped_lower_ratio(self):
        old = torch.zeros(2)
        current = torch.full((2,), math.log(0.5))
        loss, _ = RL.clipped_policy_objective(
            current, old, -1.0, clip_epsilon=0.2, kl_beta=0.0
        )
        self.assertAlmostEqual(float(loss), 0.8, places=6)

    def test_completion_logps_returns_only_completion_tokens(self):
        model = TinyCausalLM()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        values = RL.completion_logps(model, ids, target_start=3)
        self.assertEqual(tuple(values.shape), (2,))
        values.sum().backward()
        self.assertIsNotNone(model.projection.weight.grad)


class FrozenSubsetTests(unittest.TestCase):
    def test_selection_is_deterministic_and_group_diverse(self):
        records = [
            {"record_id": f"{group}/{rollout}/0", "group_id": group}
            for rollout in range(3)
            for group in ("a", "b", "c")
        ]
        first = RL.select_frozen_subset(records, 5)
        second = RL.select_frozen_subset(records, 5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual({records[index]["group_id"] for index in first[:3]}, {"a", "b", "c"})

    def test_weighted_metrics_follow_rollout_step_normalization(self):
        metrics = [
            {"loss": 2.0, "approx_kl": 0.1},
            {"loss": -2.0, "approx_kl": 0.3},
        ]

        result = RL.aggregate_weighted_metrics(metrics, [0.25, 0.75])

        self.assertAlmostEqual(result["loss"], -1.0)
        self.assertAlmostEqual(result["approx_kl"], 0.25)
        self.assertAlmostEqual(result["weight_sum"], 1.0)

    def test_weighted_metrics_reject_non_positive_weights(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            RL.aggregate_weighted_metrics([{"loss": 1.0}], [0.0])


class TrainingLimitTests(unittest.TestCase):
    def test_production_uses_every_optimizer_step(self):
        self.assertEqual((17, False), RL.resolve_optimizer_step_limit(17, 0))

    def test_smoke_is_bounded_without_exceeding_dataset(self):
        self.assertEqual((1, True), RL.resolve_optimizer_step_limit(17, 1))
        self.assertEqual((3, True), RL.resolve_optimizer_step_limit(3, 10))

    def test_invalid_training_limits_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            RL.resolve_optimizer_step_limit(0, 1)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            RL.resolve_optimizer_step_limit(3, -1)


if __name__ == "__main__":
    unittest.main()
