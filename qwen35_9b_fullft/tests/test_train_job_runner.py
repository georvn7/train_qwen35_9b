#!/usr/bin/env python3
"""Fixture tests for the remote training job runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import train_job_runner as runner  # noqa: E402


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def sft_rows(count: int = 6) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "system", "content": "debug assistant"},
                {"role": "user", "content": f"case {idx}"},
                {"role": "assistant", "content": '{"action_type":"run_test","action_subject":"none"}'},
            ]
        }
        for idx in range(count)
    ]


def dpo_rows(count: int = 13) -> list[dict]:
    return [
        {
            "prompt": f"prompt {idx}",
            "chosen": '{"action_type":"function_info","action_subject":"parse_function_body"}',
            "rejected": '{"action_type":"log_info","action_subject":"none"}',
            "meta": {"idx": idx},
        }
        for idx in range(count)
    ]


def conversational_dpo_rows(count: int = 13) -> list[dict]:
    return [
        {
            "prompt": [
                {"role": "system", "content": "debug assistant"},
                {"role": "user", "content": f"case {idx}"},
            ],
            "chosen": [{"role": "assistant", "content": "chosen action"}],
            "rejected": [{"role": "assistant", "content": "rejected action"}],
            "meta": {"idx": idx},
        }
        for idx in range(count)
    ]


def thinking_sft_rows(count: int = 6, thinking: str = "inspect evidence") -> list[dict]:
    rows = sft_rows(count)
    for row in rows:
        row["messages"][-1]["thinking"] = thinking
    return rows


def thinking_dpo_rows(count: int = 13, thinking: str = "compare evidence") -> list[dict]:
    rows = conversational_dpo_rows(count)
    for row in rows:
        row["chosen"][-1]["thinking"] = thinking
        row["rejected"][-1]["thinking"] = thinking
    return rows


def rl_rows() -> list[dict]:
    rows = []
    for group_index in range(2):
        for rollout_index, advantage in enumerate((-1.0, 1.0)):
            rows.append(
                {
                    "group_id": f"group-{group_index}",
                    "rollout_id": f"rollout-{rollout_index}",
                    "reward": advantage,
                    "advantage": advantage,
                    "policy_step_weight": 0.5,
                    "policy_steps": [
                        {
                            "prompt": [
                                {"role": "system", "content": "debug assistant"},
                                {"role": "user", "content": "choose the next action"},
                            ],
                            "completion": [
                                {
                                    "role": "assistant",
                                    "thinking": "inspect grounded evidence",
                                    "content": "selected action",
                                }
                            ],
                        },
                        {
                            "prompt": [
                                {"role": "system", "content": "debug assistant"},
                                {"role": "user", "content": "continue"},
                            ],
                            "completion": [
                                {
                                    "role": "assistant",
                                    "thinking": "select the causal repair",
                                    "content": "terminal fix",
                                }
                            ],
                        },
                    ],
                }
            )
    return rows


def awr_rows() -> list[dict]:
    return [
        {
            "objective": "repair_distance_awr",
            "group_id": "repair-1",
            "sample_id": f"repair-1-{index}",
            "sample_weight": 0.25 + 0.1 * index,
            "prompt": [
                {"role": "system", "content": "debug assistant"},
                {"role": "user", "content": "grounded failed trajectory"},
            ],
            "completion": [
                {
                    "role": "assistant",
                    "thinking": "trace the terminal blocker",
                    "content": "selected action",
                }
            ],
        }
        for index in range(2)
    ]


def make_job(
    jobs_root: Path,
    job_id: str = "simplec-s0_2-micro-001",
    bad_checksum: bool = False,
    malformed_sft: bool = False,
    malformed_dpo: bool = False,
    conversational_dpo: bool = False,
    custom_dpo_rows: list[dict] | None = None,
    custom_sft_rows: list[dict] | None = None,
    sft_enabled: bool = True,
    dpo_enabled: bool = True,
    assistant_reasoning: bool = False,
    thinking_max_chars: int = 1800,
    dpo_execution_mode: str | None = None,
    rl_enabled: bool = False,
    custom_rl_rows: list[dict] | None = None,
    awr_enabled: bool = False,
    custom_awr_rows: list[dict] | None = None,
) -> Path:
    job_dir = jobs_root / "incoming" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    sft_payload = custom_sft_rows if custom_sft_rows is not None else sft_rows()
    sft_text = '{"messages": "bad"}\n' if malformed_sft else jsonl(sft_payload)
    if custom_dpo_rows is not None:
        dpo_payload = custom_dpo_rows
    elif malformed_dpo:
        dpo_payload = [{"prompt": "p", "chosen": "c"}]
    elif conversational_dpo:
        dpo_payload = conversational_dpo_rows()
    else:
        dpo_payload = dpo_rows()
    dpo_text = jsonl(dpo_payload)
    if sft_enabled:
        (job_dir / "train_sft.jsonl").write_text(sft_text, encoding="utf-8")
    if not rl_enabled and not awr_enabled:
        (job_dir / "train_dpo.jsonl").write_text(dpo_text, encoding="utf-8")
    sft_sha = sha256_text(sft_text)
    dpo_sha = sha256_text(dpo_text)
    if bad_checksum:
        sft_sha = "0" * 64
    manifest = {
        "format_version": 1,
        "job_id": job_id,
        "curriculum_round": 1,
        "created_at": "2026-07-14T00:00:00Z",
        "base_checkpoint": "fixture-base-checkpoint",
        "output_checkpoint": f"hayabusa-9b-{job_id}",
        "max_sequence_length": 32768,
        "training_profile": "micro_contract_validation",
        "assistant_reasoning": {
            "mode": "required" if assistant_reasoning else "disabled",
            "field": "thinking",
            "thinking_max_chars": thinking_max_chars,
            "semantic_judging": "not_used" if awr_enabled else "final_content_only",
        },
        "inputs": {},
        "stages": {
            "sft": {"enabled": sft_enabled and not rl_enabled and not awr_enabled, "overrides": {}},
            "dpo": {"enabled": dpo_enabled and not rl_enabled and not awr_enabled, "overrides": {}},
            "rl": {
                "enabled": rl_enabled,
                "execution_mode": "serialized",
                "overrides": {
                    "clip_epsilon": 0.2,
                    "kl_beta": 0.01,
                    "epochs": 1,
                } if rl_enabled else {},
            },
            "awr": {
                "enabled": awr_enabled,
                "execution_mode": "serialized",
                "overrides": {"epochs": 1} if awr_enabled else {},
            },
        },
        "deployment": {"enabled": True, "served_model_name": f"hayabusa-9b-{job_id}"},
    }
    if dpo_execution_mode is not None:
        manifest["dpo_execution_mode"] = dpo_execution_mode
    if awr_enabled:
        payload = custom_awr_rows if custom_awr_rows is not None else awr_rows()
        awr_text = jsonl(payload)
        (job_dir / "train_awr.jsonl").write_text(awr_text, encoding="utf-8")
        manifest["inputs"]["awr"] = {
            "path": "train_awr.jsonl",
            "sha256": sha256_text(awr_text),
            "rows": len(payload),
        }
    elif rl_enabled:
        payload = custom_rl_rows if custom_rl_rows is not None else rl_rows()
        rl_text = jsonl(payload)
        (job_dir / "train_rl.jsonl").write_text(rl_text, encoding="utf-8")
        manifest["inputs"]["rl"] = {
            "path": "train_rl.jsonl",
            "sha256": sha256_text(rl_text),
            "rows": len(payload),
        }
    else:
        manifest["inputs"]["dpo"] = {
            "path": "train_dpo.jsonl",
            "sha256": dpo_sha,
            "rows": len(dpo_payload),
        }
    if sft_enabled and not rl_enabled and not awr_enabled:
        manifest["inputs"]["sft"] = {
            "path": "train_sft.jsonl",
            "sha256": sft_sha,
            "rows": 1 if malformed_sft else 6,
        }
    (job_dir / "job.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (job_dir / "READY").write_text("", encoding="utf-8")
    return job_dir


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TrainJobRunnerTests(unittest.TestCase):
    def make_config(self, tmp: str, fail_stage: str = "", sleep_seconds: float = 0.0) -> runner.RunnerConfig:
        root = Path(tmp)
        return runner.RunnerConfig(
            jobs_root=root / "jobs",
            workspace_root=root / "workspace",
            mode="fixture",
            once=True,
            fixture=runner.FixtureConfig(fail_stage=fail_stage, sleep_seconds=sleep_seconds),
        )

    def test_real_commands_use_twenty_step_checkpoint_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_data = runner.ValidatedInput(
                path=root / "train.jsonl",
                rows=63,
                sha256="0" * 64,
            )
            job = runner.ValidatedJob(
                manifest={},
                job_id="checkpoint-policy",
                base_checkpoint="base",
                output_checkpoint="output",
                max_sequence_length=32768,
                training_profile="micro_contract_validation",
                sft_enabled=True,
                dpo_enabled=True,
                sft_input=input_data,
                dpo_input=input_data,
                deployment_enabled=False,
                served_model_name="output",
                assistant_reasoning="disabled",
                thinking_max_chars=1800,
            )
            config = runner.RunnerConfig(
                jobs_root=root / "jobs",
                workspace_root=root / "workspace",
                mode="real",
            )

            for command in (
                runner.build_sft_command(config, job, root / "sft"),
                runner.build_dpo_command(config, job, root / "dpo", "base"),
            ):
                save_index = command.index("--save-steps")
                self.assertEqual(command[save_index + 1], "20")
                self.assertNotIn("--extra-save-steps", command)

    def test_dpo_execution_modes_control_effective_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_data = runner.ValidatedInput(
                path=root / "train.jsonl", rows=1, sha256="0" * 64
            )
            config = runner.RunnerConfig(
                jobs_root=root / "jobs",
                workspace_root=root / "workspace",
                mode="real",
            )
            for requested, expected_effective, expected_length in (
                ("batched", "batched", 16384),
                ("split_backward", "split_backward", 32768),
                ("auto", "split_backward", 32768),
            ):
                with self.subTest(requested=requested):
                    job = runner.ValidatedJob(
                        manifest={},
                        job_id="execution-mode",
                        base_checkpoint="base",
                        output_checkpoint="output",
                        max_sequence_length=32768,
                        training_profile="micro_contract_validation",
                        sft_enabled=False,
                        dpo_enabled=True,
                        sft_input=None,
                        dpo_input=input_data,
                        deployment_enabled=False,
                        served_model_name="output",
                        assistant_reasoning="disabled",
                        thinking_max_chars=1800,
                        dpo_execution_mode=requested,
                    )
                    command = runner.build_dpo_command(
                        config, job, root / "dpo", "base"
                    )
                    mode_index = command.index("--dpo-execution-mode")
                    length_index = command.index("--max-length")
                    requested_index = command.index("--requested-dpo-execution-mode")
                    self.assertEqual(command[mode_index + 1], expected_effective)
                    self.assertEqual(int(command[length_index + 1]), expected_length)
                    self.assertEqual(command[requested_index + 1], requested)

    def test_invalid_dpo_execution_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(config.jobs_root, dpo_execution_mode="parallel_magic")
            rc = runner.run_once(config)
            self.assertEqual(rc, 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            result = read_json(failed / "result.json")
            self.assertEqual(result["failed_stage"], "validating")
            self.assertIn("dpo_execution_mode", result["error"])

    def test_valid_tiny_bundle_reaches_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(config.jobs_root)
            rc = runner.run_once(config)
            self.assertEqual(rc, 0)
            completed = config.jobs_root / "completed" / "simplec-s0_2-micro-001"
            self.assertTrue(completed.exists())
            self.assertEqual(read_json(completed / "status.json")["state"], "complete")
            result = read_json(completed / "result.json")
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["health_check"]["passed"])
            self.assertEqual(result["dpo_execution"]["requested_mode"], "batched")
            self.assertEqual(result["dpo_execution"]["effective_mode"], "batched")
            self.assertEqual(result["metrics"]["sft"], {"stage": "sft"})
            self.assertEqual(result["metrics"]["dpo"], {"stage": "dpo"})
            stages = read_json(completed / "stage_sessions.json")
            self.assertEqual(stages["sft"]["metrics"], {"stage": "sft"})
            self.assertEqual(stages["dpo"]["metrics"], {"stage": "dpo"})
            self.assertEqual(
                result["dpo_execution"]["effective_max_sequence_length"], 16384
            )

    def test_stage_metrics_must_exist_and_be_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            with self.assertRaisesRegex(runner.StageError, "training metrics are missing"):
                runner.load_stage_metrics(session, "dpo")
            (session / "metadata").mkdir()
            runner.atomic_write_json(session / "metadata" / "train_metrics.json", {})
            with self.assertRaisesRegex(runner.StageError, "training metrics are empty"):
                runner.load_stage_metrics(session, "dpo")

    def test_auto_mode_is_recorded_in_completed_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(config.jobs_root, dpo_execution_mode="auto")
            self.assertEqual(runner.run_once(config), 0)
            completed = config.jobs_root / "completed" / "simplec-s0_2-micro-001"
            execution = read_json(completed / "result.json")["dpo_execution"]
            self.assertEqual(execution["requested_mode"], "auto")
            self.assertEqual(execution["effective_mode"], "split_backward")
            self.assertEqual(execution["requested_max_sequence_length"], 32768)
            self.assertEqual(execution["effective_max_sequence_length"], 32768)

    def test_valid_conversational_dpo_bundle_reaches_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(config.jobs_root, conversational_dpo=True)
            rc = runner.run_once(config)
            self.assertEqual(rc, 0)
            completed = config.jobs_root / "completed" / "simplec-s0_2-micro-001"
            result = read_json(completed / "result.json")
            self.assertEqual(result["status"], "complete")
            self.assertIsNotNone(result["sft_checkpoint"])

    def test_valid_thinking_bundle_reaches_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(
                config.jobs_root,
                custom_sft_rows=thinking_sft_rows(),
                custom_dpo_rows=thinking_dpo_rows(),
                assistant_reasoning=True,
            )
            rc = runner.run_once(config)
            self.assertEqual(rc, 0)
            completed = config.jobs_root / "completed" / "simplec-s0_2-micro-001"
            result = read_json(completed / "result.json")
            self.assertEqual(result["health_check"]["assistant_reasoning"], "required")

    def test_valid_rl_only_bundle_reaches_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(
                config.jobs_root,
                sft_enabled=False,
                dpo_enabled=False,
                rl_enabled=True,
                assistant_reasoning=True,
            )
            self.assertEqual(runner.run_once(config), 0)
            completed = config.jobs_root / "completed" / "simplec-s0_2-micro-001"
            result = read_json(completed / "result.json")
            self.assertEqual(result["metrics"]["rl"], {"stage": "rl"})
            self.assertEqual(result["metrics"]["dpo"], {})
            self.assertEqual(result["rl_execution"]["execution_mode"], "serialized")
            stages = read_json(completed / "stage_sessions.json")
            self.assertIsNone(stages["dpo"])
            self.assertEqual(stages["rl"]["metrics"], {"stage": "rl"})

    def test_valid_awr_only_bundle_reaches_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(
                config.jobs_root,
                sft_enabled=False,
                dpo_enabled=False,
                awr_enabled=True,
                assistant_reasoning=True,
            )
            self.assertEqual(runner.run_once(config), 0)
            completed = config.jobs_root / "completed" / "simplec-s0_2-micro-001"
            result = read_json(completed / "result.json")
            self.assertEqual(result["metrics"]["awr"], {"stage": "awr"})
            self.assertEqual(result["metrics"]["rl"], {})
            self.assertEqual(result["awr_execution"]["execution_mode"], "serialized")
            stages = read_json(completed / "stage_sessions.json")
            self.assertIsNone(stages["rl"])
            self.assertEqual(stages["awr"]["metrics"], {"stage": "awr"})

    def test_awr_bundle_without_thinking_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            rows = awr_rows()
            del rows[0]["completion"][0]["thinking"]
            make_job(
                config.jobs_root,
                sft_enabled=False,
                dpo_enabled=False,
                awr_enabled=True,
                assistant_reasoning=True,
                custom_awr_rows=rows,
            )
            self.assertEqual(runner.run_once(config), 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            result = read_json(failed / "result.json")
            self.assertEqual("validating", result["failed_stage"])
            self.assertIn("thinking", result["error"])

    def test_rl_accepts_only_isolated_oversized_reasoning_negative(self) -> None:
        rows = rl_rows()[:2]
        negative = rows[0]
        step = negative["policy_steps"][0]
        step["completion"][0]["thinking"] = "x" * 1801
        step["reasoning_structure_valid"] = False
        step["reasoning_structure_reasons"] = ["thinking_too_long"]
        step["thinking_chars"] = 1801
        step["thinking_max_chars"] = 1800
        negative["policy_steps"] = [step]
        negative["policy_step_weight"] = 1.0
        negative["reward"] = -1.0
        negative["advantage"] = -1.0
        negative["reward_evidence"] = {
            "kind": "reasoning_structure_negative",
            "reason": "thinking_too_long",
            "response_kind": "SystemDebugAnalysis",
            "thinking_chars": 1801,
            "thinking_max_chars": 1800,
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(
                config.jobs_root,
                sft_enabled=False,
                dpo_enabled=False,
                rl_enabled=True,
                assistant_reasoning=True,
                custom_rl_rows=rows,
            )
            self.assertEqual(runner.run_once(config), 0)

        negative["reward"] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(
                config.jobs_root,
                sft_enabled=False,
                dpo_enabled=False,
                rl_enabled=True,
                assistant_reasoning=True,
                custom_rl_rows=rows,
            )
            self.assertEqual(runner.run_once(config), 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            self.assertIn(
                "must isolate one -1 response",
                read_json(failed / "result.json")["error"],
            )

    def test_rl_group_without_advantage_diversity_is_rejected(self) -> None:
        rows = rl_rows()
        for row in rows:
            row["advantage"] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(
                config.jobs_root,
                sft_enabled=False,
                dpo_enabled=False,
                rl_enabled=True,
                assistant_reasoning=True,
                custom_rl_rows=rows,
            )
            self.assertEqual(runner.run_once(config), 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            self.assertIn("advantage diversity", read_json(failed / "result.json")["error"])

    def test_rl_inconsistent_policy_step_weight_is_rejected(self) -> None:
        rows = rl_rows()
        rows[0]["policy_step_weight"] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(
                config.jobs_root,
                sft_enabled=False,
                dpo_enabled=False,
                rl_enabled=True,
                assistant_reasoning=True,
                custom_rl_rows=rows,
            )
            self.assertEqual(runner.run_once(config), 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            self.assertIn(
                "1 / policy_steps", read_json(failed / "result.json")["error"]
            )

    def test_required_thinking_rejects_missing_and_oversized_reasoning(self) -> None:
        cases = [
            ("missing", sft_rows(), thinking_dpo_rows(), 1800, "thinking is required"),
            (
                "oversized",
                thinking_sft_rows(thinking="12345"),
                thinking_dpo_rows(thinking="12345"),
                4,
                "thinking exceeds 4 characters",
            ),
        ]
        for name, sft_payload, dpo_payload, limit, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                config = self.make_config(tmp)
                make_job(
                    config.jobs_root,
                    custom_sft_rows=sft_payload,
                    custom_dpo_rows=dpo_payload,
                    assistant_reasoning=True,
                    thinking_max_chars=limit,
                )
                rc = runner.run_once(config)
                self.assertEqual(rc, 1)
                failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
                result = read_json(failed / "result.json")
                self.assertEqual(result["failed_stage"], "validating")
                self.assertIn(expected, result["error"])

    def test_training_environment_is_prepared_before_sft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(config.jobs_root)
            rc = runner.run_once(config)
            self.assertEqual(rc, 0)
            completed = config.jobs_root / "completed" / "simplec-s0_2-micro-001"
            environment = read_json(completed / "training_environment.json")
            self.assertEqual(environment["mode"], "fixture")
            self.assertFalse(environment["serving_stopped"])

    def test_training_preparation_failure_prevents_sft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp, fail_stage="prepare_training")
            make_job(config.jobs_root)
            rc = runner.run_once(config)
            self.assertEqual(rc, 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            result = read_json(failed / "result.json")
            self.assertEqual(result["failed_stage"], "prepare_training")
            self.assertFalse((failed / "logs" / "sft.log").exists())

    def test_dpo_only_job_starts_from_base_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(config.jobs_root, sft_enabled=False, conversational_dpo=True)
            rc = runner.run_once(config)
            self.assertEqual(rc, 0)
            completed = config.jobs_root / "completed" / "simplec-s0_2-micro-001"
            result = read_json(completed / "result.json")
            self.assertIsNone(result["sft_checkpoint"])
            stages = read_json(completed / "stage_sessions.json")
            self.assertIsNone(stages["sft"])
            self.assertIn("fixture-base-checkpoint", " ".join(stages["dpo"]["command"]))
            self.assertFalse((completed / "logs" / "sft.log").exists())

    def test_dpo_disabled_job_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(config.jobs_root, dpo_enabled=False)
            rc = runner.run_once(config)
            self.assertEqual(rc, 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            result = read_json(failed / "result.json")
            self.assertEqual(result["failed_stage"], "validating")
            self.assertIn("exactly one of DPO, RL, or AWR", result["error"])

    def test_dpo_message_array_validation_errors(self) -> None:
        cases = [
            ("empty_array", [{"prompt": [], "chosen": [{"role": "assistant", "content": "c"}], "rejected": [{"role": "assistant", "content": "r"}]}], "non-empty"),
            ("missing_role", [{"prompt": [{"content": "p"}], "chosen": [{"role": "assistant", "content": "c"}], "rejected": [{"role": "assistant", "content": "r"}]}], ".role"),
            ("missing_content", [{"prompt": [{"role": "user"}], "chosen": [{"role": "assistant", "content": "c"}], "rejected": [{"role": "assistant", "content": "r"}]}], ".content"),
            ("empty_content", [{"prompt": [{"role": "user", "content": ""}], "chosen": [{"role": "assistant", "content": "c"}], "rejected": [{"role": "assistant", "content": "r"}]}], ".content"),
        ]
        for name, rows, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                config = self.make_config(tmp)
                make_job(config.jobs_root, custom_dpo_rows=rows)
                rc = runner.run_once(config)
                self.assertEqual(rc, 1)
                failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
                result = read_json(failed / "result.json")
                self.assertEqual(result["failed_stage"], "validating")
                self.assertIn(expected, result["error"])

    def test_invalid_checksum_fails_during_validating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(config.jobs_root, bad_checksum=True)
            rc = runner.run_once(config)
            self.assertEqual(rc, 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            result = read_json(failed / "result.json")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failed_stage"], "validating")
            self.assertIn("sha256 mismatch", result["error"])

    def test_malformed_sft_and_dpo_fail_before_training(self) -> None:
        for field in ("sft", "dpo"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                config = self.make_config(tmp)
                make_job(
                    config.jobs_root,
                    malformed_sft=field == "sft",
                    malformed_dpo=field == "dpo",
                )
                rc = runner.run_once(config)
                self.assertEqual(rc, 1)
                failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
                result = read_json(failed / "result.json")
                self.assertEqual(result["failed_stage"], "validating")
                self.assertFalse((failed / "logs" / "sft.log").exists())

    def test_two_concurrent_runners_do_not_train_two_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            workspace_root = root / "workspace"
            make_job(jobs_root, job_id="job-a")
            make_job(jobs_root, job_id="job-b")
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "train_job_runner.py"),
                "--jobs-root",
                str(jobs_root),
                "--workspace-root",
                str(workspace_root),
                "--once",
                "--fixture-mode",
                "--fixture-sleep-seconds",
                "1.0",
            ]
            first = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(0.2)
            second = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            _, first_stderr = first.communicate(timeout=10)
            first_rc = first.returncode
            self.assertEqual(first_stderr, "")
            self.assertEqual(first_rc, 0)
            self.assertEqual(second.returncode, 75)
            completed_jobs = list((jobs_root / "completed").glob("*"))
            self.assertEqual(len(completed_jobs), 1)
            incoming_jobs = list((jobs_root / "incoming").glob("*"))
            self.assertEqual(len(incoming_jobs), 1)

    def test_failed_sft_prevents_dpo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp, fail_stage="sft")
            make_job(config.jobs_root)
            rc = runner.run_once(config)
            self.assertEqual(rc, 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            result = read_json(failed / "result.json")
            self.assertEqual(result["failed_stage"], "sft")
            self.assertFalse((failed / "logs" / "dpo.log").exists())

    def test_failed_dpo_prevents_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp, fail_stage="dpo")
            make_job(config.jobs_root)
            rc = runner.run_once(config)
            self.assertEqual(rc, 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            result = read_json(failed / "result.json")
            self.assertEqual(result["failed_stage"], "dpo")
            self.assertTrue(result["recoverable_dpo_only"])
            self.assertTrue(Path(result["sft_checkpoint"]).is_dir())
            self.assertFalse((failed / "logs" / "deploy.log").exists())

    def test_failed_health_check_prevents_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp, fail_stage="health_check")
            make_job(config.jobs_root)
            rc = runner.run_once(config)
            self.assertEqual(rc, 1)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            result = read_json(failed / "result.json")
            self.assertEqual(result["failed_stage"], "health_check")

    def test_status_and_result_are_valid_json_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp, fail_stage="dpo")
            make_job(config.jobs_root)
            runner.run_once(config)
            failed = config.jobs_root / "failed" / "simplec-s0_2-micro-001"
            self.assertIsInstance(read_json(failed / "status.json"), dict)
            self.assertIsInstance(read_json(failed / "result.json"), dict)

    def test_rerun_once_does_not_repeat_completed_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            make_job(config.jobs_root)
            self.assertEqual(runner.run_once(config), 0)
            completed = config.jobs_root / "completed" / "simplec-s0_2-micro-001"
            events_before = (completed / "trainer_events.jsonl").read_text(encoding="utf-8")
            self.assertEqual(runner.run_once(config), 0)
            events_after = (completed / "trainer_events.jsonl").read_text(encoding="utf-8")
            self.assertEqual(events_before, events_after)


if __name__ == "__main__":
    unittest.main()
