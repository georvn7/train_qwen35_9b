#!/usr/bin/env python3

import importlib.util
import copy
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_dpo_session.py"
SPEC = importlib.util.spec_from_file_location("train_dpo_session", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN_DPO_SESSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_DPO_SESSION)
sys.path.insert(0, str(MODULE_PATH.parent))
from split_dpo import single_completion_forward  # noqa: E402
from trl import DPOTrainer  # noqa: E402


class FakeTokenizer:
    eos_token_id = 999
    eos_token = "<eos>"

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
        **kwargs,
    ):
        assert tokenize is False
        self.calls.append(dict(kwargs))
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
    def test_frozen_subset_is_deterministic(self):
        rows = [
            {"prompt": f"p{index}", "chosen": f"c{index}", "rejected": f"r{index}"}
            for index in range(20)
        ]
        first = TRAIN_DPO_SESSION.select_frozen_dpo_indices(rows, maximum=8)
        second = TRAIN_DPO_SESSION.select_frozen_dpo_indices(rows, maximum=8)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first)), 8)

    def test_frozen_metrics_use_reference_adjusted_margin(self):
        metrics = TRAIN_DPO_SESSION.dpo_comparison_metrics(
            chosen_logps=[-1.0, -2.0],
            rejected_logps=[-3.0, -4.0],
            ref_chosen_logps=[-1.0, -2.0],
            ref_rejected_logps=[-2.0, -3.0],
            beta=0.1,
        )
        self.assertAlmostEqual(metrics["rewards_margin"], 0.1, places=6)
        self.assertEqual(metrics["rewards_accuracy"], 1.0)
        self.assertLess(metrics["loss"], math.log(2.0))

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
        self.assertEqual(
            stats["prompt_final_tokens_total"],
            len(tokenized[0]["prompt_input_ids"]),
        )
        self.assertEqual(
            stats["chosen_final_tokens_total"],
            len(tokenized[0]["chosen_input_ids"]),
        )
        self.assertEqual(
            stats["rejected_final_tokens_total"],
            len(tokenized[0]["rejected_input_ids"]),
        )
        self.assertEqual(
            stats["policy_branch_tokens_per_epoch"],
            2 * len(tokenized[0]["prompt_input_ids"])
            + len(tokenized[0]["chosen_input_ids"])
            + len(tokenized[0]["rejected_input_ids"]),
        )

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

        tokenizer = FakeTokenizer()
        rendered = TRAIN_DPO_SESSION.render_dpo_row_for_tokenization(
            row, tokenizer
        )

        self.assertIn("<think>inspect the observed value flow</think>", rendered["chosen"])
        self.assertIn("<think>repeat the stale hypothesis</think>", rendered["rejected"])
        self.assertLess(
            rendered["chosen"].index("<think>inspect the observed value flow</think>"),
            rendered["chosen"].index("chosen action"),
        )
        self.assertLess(
            rendered["rejected"].index("<think>repeat the stale hypothesis</think>"),
            rendered["rejected"].index("rejected action"),
        )
        self.assertNotIn("reasoning_content", row["chosen"][0])
        self.assertTrue(tokenizer.calls)
        self.assertTrue(all(call["enable_thinking"] for call in tokenizer.calls))
        self.assertTrue(
            all(call["reasoning_effort"] == "medium" for call in tokenizer.calls)
        )
        self.assertTrue(
            all(call["preserve_thinking"] is False for call in tokenizer.calls)
        )


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden_size=12):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, logits_to_keep=None, **_kwargs):
        hidden = torch.cumsum(self.embedding(input_ids), dim=1)
        logits = self.projection(hidden)
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


class DummyDPOState:
    pad_token_id = 0
    aux_loss_enabled = False
    is_encoder_decoder = False
    use_logits_to_keep = True
    padding_free = False
    use_weighting = False
    max_length = 16
    truncation_mode = "keep_end"
    loss_type = ["sigmoid"]
    args = SimpleNamespace(rpo_alpha=None, ld_alpha=None, use_liger_loss=False)
    concatenated_inputs = staticmethod(DPOTrainer.concatenated_inputs)


def dpo_batch():
    return {
        "prompt_input_ids": torch.tensor([[0, 2, 3], [4, 5, 6]]),
        "prompt_attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        "chosen_input_ids": torch.tensor([[7, 8, 9], [10, 11, 0]]),
        "chosen_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "rejected_input_ids": torch.tensor([[12, 13], [14, 15]]),
        "rejected_attention_mask": torch.tensor([[1, 1], [1, 1]]),
    }


def gradient_psnr(left, right):
    left_flat = torch.cat([parameter.grad.detach().float().flatten() for parameter in left.parameters()])
    right_flat = torch.cat([parameter.grad.detach().float().flatten() for parameter in right.parameters()])
    mse = torch.mean((left_flat - right_flat) ** 2).item()
    if mse == 0.0:
        return math.inf
    peak = torch.maximum(left_flat.abs(), right_flat.abs()).max().item()
    return 20.0 * math.log10(peak / math.sqrt(mse))


class SplitBackwardTests(unittest.TestCase):
    def test_serial_forwards_match_trl_concatenated_forward(self):
        torch.manual_seed(7)
        trainer = DummyDPOState()
        model = TinyCausalLM()
        batch = dpo_batch()

        combined = DPOTrainer.concatenated_forward(trainer, model, batch)
        chosen = single_completion_forward(trainer, model, batch, "chosen")
        rejected = single_completion_forward(trainer, model, batch, "rejected")

        torch.testing.assert_close(chosen["logps"], combined["chosen_logps"])
        torch.testing.assert_close(rejected["logps"], combined["rejected_logps"])
        torch.testing.assert_close(chosen["mean_logits"], combined["mean_chosen_logits"])
        torch.testing.assert_close(rejected["mean_logits"], combined["mean_rejected_logits"])

    def test_split_backward_matches_loss_gradients_and_optimizer_update(self):
        torch.manual_seed(11)
        trainer = DummyDPOState()
        batch = dpo_batch()
        batched_model = TinyCausalLM()
        split_model = copy.deepcopy(batched_model)
        beta = 0.05
        ref_chosen = torch.tensor([-8.0, -7.0])
        ref_rejected = torch.tensor([-9.0, -8.5])

        combined = DPOTrainer.concatenated_forward(trainer, batched_model, batch)
        batched_logits = (
            combined["chosen_logps"]
            - combined["rejected_logps"]
            - ref_chosen
            + ref_rejected
        )
        batched_loss = -torch.nn.functional.logsigmoid(beta * batched_logits).mean()
        batched_loss.backward()

        with torch.no_grad():
            chosen_probe = single_completion_forward(trainer, split_model, batch, "chosen")["logps"]
            rejected_probe = single_completion_forward(trainer, split_model, batch, "rejected")["logps"]
        chosen_leaf = chosen_probe.detach().requires_grad_(True)
        rejected_leaf = rejected_probe.detach().requires_grad_(True)
        split_logits = chosen_leaf - rejected_leaf - ref_chosen + ref_rejected
        split_loss = -torch.nn.functional.logsigmoid(beta * split_logits).mean()
        chosen_coefficient, rejected_coefficient = torch.autograd.grad(
            split_loss, (chosen_leaf, rejected_leaf)
        )
        chosen_train = single_completion_forward(trainer, split_model, batch, "chosen")["logps"]
        (chosen_coefficient.detach() * chosen_train).sum().backward()
        rejected_train = single_completion_forward(trainer, split_model, batch, "rejected")["logps"]
        (rejected_coefficient.detach() * rejected_train).sum().backward()

        self.assertAlmostEqual(batched_loss.item(), split_loss.item(), places=6)
        for batched_parameter, split_parameter in zip(
            batched_model.parameters(), split_model.parameters()
        ):
            torch.testing.assert_close(
                batched_parameter.grad,
                split_parameter.grad,
                rtol=2e-5,
                atol=2e-6,
            )
        self.assertGreater(gradient_psnr(batched_model, split_model), 100.0)

        batched_optimizer = torch.optim.SGD(batched_model.parameters(), lr=1e-3)
        split_optimizer = torch.optim.SGD(split_model.parameters(), lr=1e-3)
        batched_optimizer.step()
        split_optimizer.step()
        for batched_parameter, split_parameter in zip(
            batched_model.parameters(), split_model.parameters()
        ):
            torch.testing.assert_close(
                batched_parameter,
                split_parameter,
                rtol=2e-5,
                atol=2e-6,
            )

    def test_split_contributions_use_one_gradient_accumulation_divisor(self):
        torch.manual_seed(19)
        trainer = DummyDPOState()
        batch = dpo_batch()
        batched_model = TinyCausalLM()
        split_model = copy.deepcopy(batched_model)
        beta = 0.05
        ref_chosen = torch.tensor([-8.0, -7.0])
        ref_rejected = torch.tensor([-9.0, -8.5])
        accumulation_divisor = 2

        combined = DPOTrainer.concatenated_forward(trainer, batched_model, batch)
        logits = combined["chosen_logps"] - combined["rejected_logps"] - ref_chosen + ref_rejected
        (-torch.nn.functional.logsigmoid(beta * logits).mean() / accumulation_divisor).backward()

        with torch.no_grad():
            chosen_probe = single_completion_forward(trainer, split_model, batch, "chosen")["logps"]
            rejected_probe = single_completion_forward(trainer, split_model, batch, "rejected")["logps"]
        chosen_leaf = chosen_probe.detach().requires_grad_(True)
        rejected_leaf = rejected_probe.detach().requires_grad_(True)
        loss = -torch.nn.functional.logsigmoid(
            beta * (chosen_leaf - rejected_leaf - ref_chosen + ref_rejected)
        ).mean()
        chosen_coefficient, rejected_coefficient = torch.autograd.grad(
            loss, (chosen_leaf, rejected_leaf)
        )
        chosen_train = single_completion_forward(trainer, split_model, batch, "chosen")["logps"]
        ((chosen_coefficient.detach() * chosen_train).sum() / accumulation_divisor).backward()
        rejected_train = single_completion_forward(trainer, split_model, batch, "rejected")["logps"]
        ((rejected_coefficient.detach() * rejected_train).sum() / accumulation_divisor).backward()

        self.assertGreater(gradient_psnr(batched_model, split_model), 100.0)


if __name__ == "__main__":
    unittest.main()
