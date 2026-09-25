"""Unit tests for cases.infra.loghygiene."""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cases.infra.loghygiene import (SummaryTee, rotate_by_size,
                                   tee_process_output)


class SummaryTeeTest(unittest.TestCase):
    def test_classification(self) -> None:
        lines = ["[10:00] START job w1\n", "verbose payload line\n",
                 "[10:05] END job rc=0\n", "[10:06] HEARTBEAT ok\n"]
        classified = list(SummaryTee(lines).classified())
        self.assertEqual([is_summary for _, is_summary in classified],
                         [True, False, True, True])


class TeeProcessOutputTest(unittest.TestCase):
    def test_full_and_summary_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            job_log = Path(tmp) / "job.log"
            main_log = Path(tmp) / "main.log"
            proc = subprocess.Popen(
                ["bash", "-c", "echo 'START j'; echo 'noise'; "
                               "echo 'END j rc=0'"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True)
            tee_process_output(proc, job_log, main_log)
            self.assertEqual(proc.wait(), 0)
            job_text = job_log.read_text()
            main_text = main_log.read_text()
        self.assertIn("noise", job_text)
        self.assertNotIn("noise", main_text)
        self.assertIn("START j", main_text)
        self.assertIn("END j rc=0", main_text)


class RotateBySizeTest(unittest.TestCase):
    def test_no_rotation_under_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.log"
            path.write_text("x" * 10)
            rotated = rotate_by_size(path, max_bytes=100, keep=3)
            self.assertFalse(rotated)
            self.assertTrue(path.exists())

    def test_rotation_chain(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.log"
            path.write_text("x" * 100)
            rotated = rotate_by_size(path, max_bytes=10, keep=2)
            self.assertTrue(rotated)
            self.assertFalse(path.exists())
            self.assertTrue((Path(tmp) / "app.log.1").exists())


if __name__ == "__main__":
    unittest.main()
