import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliIntegrationTests(unittest.TestCase):
    def run_cli(self, *args: str, expect: int = 0) -> subprocess.CompletedProcess:
        environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
        completed = subprocess.run([sys.executable, "-m", "promptos.cli", *args], text=True,
                                   capture_output=True, env=environment, check=False)
        self.assertEqual(completed.returncode, expect, completed.stderr + completed.stdout)
        return completed

    def test_frozen_final_test_is_one_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "samples.jsonl"
            dataset.write_text("\n".join(json.dumps(item) for item in [
                {"id": "a", "text": "a", "expected": "A"},
                {"id": "b", "text": "b", "expected": "B"},
                {"id": "c", "text": "c", "expected": "C"},
            ]) + "\n")
            signals, split, runs = root / "signals.json", root / "split", root / "runs"
            self.run_cli("draft-signals", "--acceptance", "Return uppercase text only.", "--output", str(signals))
            self.run_cli("approve-signals", "--signals", str(signals))
            self.run_cli("split-dataset", "--dataset", str(dataset), "--inputs", "text", "--expected-field", "expected", "--output", str(split))
            command = ("final-evaluate", "--dataset", str(split / "final_test.jsonl"), "--split-manifest", str(split / "split_manifest.json"),
                       "--inputs", "text", "--signals", str(signals), "--prompt", "Convert input to uppercase.", "--runs", str(runs))
            self.run_cli(*command)
            self.run_cli(*command, expect=1)


if __name__ == "__main__":
    unittest.main()
