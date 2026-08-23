#!/usr/bin/env python3
"""Tests for cross-workspace resident vLLM shutdown."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stop_vllm_fullft_bf16_openai.sh"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(port: int) -> None:
    for _ in range(100):
        with socket.socket() as sock:
            sock.settimeout(0.05)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.02)
    raise AssertionError(f"fixture listener on port {port} did not start")


def write_listener(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import socket
import sys
import time

port = int(sys.argv[sys.argv.index("--port") + 1])
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen()
    while True:
        time.sleep(1)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


class StopVllmServiceTests(unittest.TestCase):
    def run_stop(self, port: int) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "PORT": str(port),
                "PY_BIN": sys.executable,
                "TERM_WAIT_ATTEMPTS": "5",
                "PORT_WAIT_ATTEMPTS": "2",
                "WAIT_SLEEP_SECONDS": "0.05",
            }
        )
        return subprocess.run(
            [str(SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_stops_vllm_listener_without_workspace_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "vllm"
            write_listener(executable)
            port = free_port()
            process = subprocess.Popen(
                [str(executable), "serve", "fixture", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                wait_for_port(port)
                result = self.run_stop(port)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                process.wait(timeout=5)
                with socket.socket() as sock:
                    self.assertNotEqual(sock.connect_ex(("127.0.0.1", port)), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_fails_closed_for_unidentified_listener(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "unrelated-listener"
            write_listener(executable)
            port = free_port()
            process = subprocess.Popen(
                [str(executable), "serve", "fixture", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                wait_for_port(port)
                result = self.run_stop(port)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("still accepting connections", result.stderr)
                self.assertIsNone(process.poll())
            finally:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
