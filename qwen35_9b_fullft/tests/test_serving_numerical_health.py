"""Check the numerical-health environment inherited by the serving process."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start_vllm_fullft_bf16_openai.sh"


class ServingNumericalHealthTests(unittest.TestCase):
    def setting(self, value):
        prefix = SCRIPT.read_text().split("MODEL_PATH_DEFAULT=", 1)[0]
        self.assertIn("export VLLM_COMPUTE_NANS_IN_LOGITS=", prefix)
        env = os.environ.copy()
        env.pop("VLLM_COMPUTE_NANS_IN_LOGITS", None)
        if value is not None:
            env["VLLM_COMPUTE_NANS_IN_LOGITS"] = value
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "probe.sh"
            script.write_text(prefix + '\nprintenv VLLM_COMPUTE_NANS_IN_LOGITS\n')
            result = subprocess.run(
                ["bash", str(script)], env=env, check=True,
                capture_output=True, text=True,
            )
        return result.stdout.strip()

    def test_enabled_by_default(self):
        self.assertEqual(self.setting(None), "1")

    def test_empty_uses_safe_default(self):
        self.assertEqual(self.setting(""), "1")

    def test_explicit_override_is_preserved(self):
        self.assertEqual(self.setting("0"), "0")


if __name__ == "__main__":
    unittest.main()
