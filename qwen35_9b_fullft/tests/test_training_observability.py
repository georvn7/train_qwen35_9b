#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from training_observability import (  # noqa: E402
    model_parameter_observability,
    module_residency_observability,
    optimizer_observability,
    peak_process_rss_mib,
    token_throughput,
)


class TrainingObservabilityTests(unittest.TestCase):
    def test_token_throughput_is_explicit_estimate(self) -> None:
        self.assertEqual(
            {
                "tokens_per_epoch": 120,
                "completed_epochs": 1.5,
                "estimated_tokens_processed": 180.0,
                "estimated_tokens_per_second": 45.0,
            },
            token_throughput(120, 1.5, 4.0),
        )

    def test_peak_rss_is_positive(self) -> None:
        self.assertGreater(peak_process_rss_mib(), 0.0)

    def test_optimizer_and_parameter_dtypes_are_materialized(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        model = torch.nn.Linear(2, 1, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        optimizer_report = optimizer_observability(optimizer, "adamw")
        self.assertEqual("adamw", optimizer_report["configured_name"])
        self.assertEqual(2, optimizer_report["state_entries"])
        self.assertIn("float32", optimizer_report["state_tensors"]["tensor_counts"])

        model_report = model_parameter_observability(model)
        self.assertEqual(3, model_report["total_parameters"])
        self.assertEqual(3, model_report["trainable_parameters"])
        self.assertEqual(
            2,
            model_report["trainable_parameter_tensors"]["tensor_counts"]["float32"],
        )
        residency = module_residency_observability(model)
        self.assertTrue(residency["present"])
        self.assertEqual(["cpu"], residency["devices"])
        self.assertEqual(12, residency["parameter_bytes"])
        self.assertEqual(0, residency["cuda_parameter_bytes"])
        self.assertFalse(module_residency_observability(None)["present"])


if __name__ == "__main__":
    unittest.main()
