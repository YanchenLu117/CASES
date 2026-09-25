"""Unit tests for cases.infra.supervisor (fast policies, no real waits)."""
from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cases.infra.supervisor import (SupervisionPolicy, run_under_supervisor,
                                   touch_heartbeat, wait_while_paused)

FAST = SupervisionPolicy(
    max_restarts=5, restart_window_s=60.0,
    backoff_init_s=0.02, backoff_max_s=0.05,
    beat_interval_s=0.02, pause_poll_s=0.02)


class SupervisorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.hb = self.root / "heartbeat.json"
        self.logs = self.root / "supervisor"
        self.alerts = self.root / "alerts.jsonl"

    def test_restarts_then_succeeds(self) -> None:
        marker = self.root / "done"
        cmd = ["bash", "-c",
               f"if [ -f {marker} ]; then echo 'END ok rc=0'; exit 0; "
               f"else touch {marker}; echo 'WARN first crash'; exit 3; fi"]
        rc = run_under_supervisor(cmd, heartbeat=self.hb, log_dir=self.logs,
                                  policy=FAST, alert_path=self.alerts)
        self.assertEqual(rc, 0)
        heartbeat = json.loads(self.hb.read_text())
        self.assertEqual(heartbeat["attempt"], 2)
        self.assertEqual(len(list(self.logs.glob("attempt_*.log"))), 2)
        self.assertFalse(self.alerts.exists())

    def test_crash_loop_gives_up_with_alert(self) -> None:
        policy = SupervisionPolicy(
            max_restarts=1, restart_window_s=60.0,
            backoff_init_s=0.01, backoff_max_s=0.02,
            beat_interval_s=0.02, pause_poll_s=0.02)
        rc = run_under_supervisor(["bash", "-c", "exit 7"],
                                  heartbeat=self.hb, log_dir=self.logs,
                                  policy=policy, alert_path=self.alerts)
        self.assertEqual(rc, 7)
        alert = json.loads(self.alerts.read_text().splitlines()[0])
        self.assertEqual(alert["kind"], "supervisor_crash_loop")
        self.assertEqual(alert["restarts"], 2)
        self.assertEqual(len(list(self.logs.glob("attempt_*.log"))), 2)

    def test_summary_log_receives_only_summary_lines(self) -> None:
        summary = self.root / "main.log"
        cmd = ["bash", "-c", "echo 'START job'; echo 'noise noise'; "
                              "echo 'END job rc=0'"]
        rc = run_under_supervisor(cmd, heartbeat=self.hb, log_dir=self.logs,
                                  policy=FAST, summary_log=summary)
        self.assertEqual(rc, 0)
        summary_text = summary.read_text()
        self.assertIn("START job", summary_text)
        self.assertIn("END job rc=0", summary_text)
        self.assertNotIn("noise noise", summary_text)
        full_text = "".join(p.read_text()
                            for p in self.logs.glob("attempt_*.log"))
        self.assertIn("noise noise", full_text)

    def test_wait_while_paused_blocks_then_releases(self) -> None:
        pause = self.root / "pause.flag"
        pause.write_text('{"owner": "canary"}')
        done = threading.Event()

        def _wait() -> None:
            wait_while_paused(pause, poll_s=0.02)
            done.set()

        thread = threading.Thread(target=_wait)
        thread.start()
        time.sleep(0.2)
        self.assertFalse(done.is_set())  # still blocked
        pause.unlink()
        done.wait(timeout=2.0)
        self.assertTrue(done.is_set())
        thread.join(timeout=2.0)

    def test_touch_heartbeat_shape(self) -> None:
        touch_heartbeat(self.hb, attempt=4, pid=123, note="running")
        data = json.loads(self.hb.read_text())
        self.assertEqual(data["attempt"], 4)
        self.assertEqual(data["pid"], 123)
        self.assertGreater(data["ts"], 0)


if __name__ == "__main__":
    unittest.main()
