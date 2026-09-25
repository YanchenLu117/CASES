"""Unit tests for cases.infra.watchdog."""
from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cases.infra.watchdog import Watchdog


class WatchdogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = self.root / "wd_state.json"
        self.alerts = self.root / "wd_alerts.jsonl"
        self.pause = self.root / "wd_pause.flag"
        self.heartbeat = self.root / "heartbeat.json"

    def _watchdog(self, **overrides) -> Watchdog:
        kwargs = dict(
            state_path=self.state, alerts_path=self.alerts,
            pause_path=self.pause, heartbeat=self.heartbeat,
        )
        kwargs.update(overrides)
        return Watchdog(**kwargs)

    def _write_heartbeat(self, age_s: float) -> None:
        self.heartbeat.write_text(json.dumps(
            {"schema": 1, "ts": time.time() - age_s, "attempt": 1,
             "pid": None, "note": "running"}))

    def test_stale_heartbeat_alerts_and_pauses(self) -> None:
        self._write_heartbeat(age_s=99999)
        active = self._watchdog(max_heartbeat_age_s=1800).step()
        self.assertIn("heartbeat", active)
        self.assertTrue(self.pause.exists())
        lines = self.alerts.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["kind"], "watchdog_heartbeat")

    def test_fresh_heartbeat_is_quiet(self) -> None:
        self._write_heartbeat(age_s=5)
        active = self._watchdog().step()
        self.assertEqual(active, {})
        self.assertFalse(self.pause.exists())

    def test_rc_counting_is_incremental(self) -> None:
        rc_log = self.root / "batch.log"
        rc_log.write_text("END a rc=0\nEND b rc=1\n")
        watchdog = self._watchdog(rc_log=rc_log, heartbeat=None)
        active = watchdog.step()
        self.assertIn("rc_failures", active)
        self.assertIn("1 new rc!=0", active["rc_failures"])
        with rc_log.open("a") as fh:
            fh.write("END c rc=2\nEND d rc=0\n")
        active = watchdog.step()
        self.assertIn("1 new rc!=0", active["rc_failures"])
        active = watchdog.step()  # no new lines -> failure stream stops
        self.assertEqual(active, {})
        kinds = [json.loads(line)["kind"]
                 for line in self.alerts.read_text().splitlines()]
        self.assertEqual(kinds, ["watchdog_rc_failures",
                                 "watchdog_rc_failures",
                                 "watchdog_rc_failures_recovered"])

    def test_recovery_removes_owned_flag_only(self) -> None:
        self._write_heartbeat(age_s=99999)
        watchdog = self._watchdog()
        watchdog.step()
        self.assertTrue(self.pause.exists())
        self._write_heartbeat(age_s=1)
        active = watchdog.step()
        self.assertEqual(active, {})
        self.assertFalse(self.pause.exists())
        kinds = [json.loads(line)["kind"]
                 for line in self.alerts.read_text().splitlines()]
        self.assertIn("watchdog_heartbeat_recovered", kinds)

    def test_foreign_flag_survives_recovery(self) -> None:
        self.pause.write_text('{"owner": "canary"}')
        self._write_heartbeat(age_s=1)
        self._watchdog().step()
        # owned by canary, not by the watchdog: left in place
        self.assertTrue(self.pause.exists())

    def test_disk_threshold(self) -> None:
        active = self._watchdog(disk_checks={str(self.root): 0}).step()
        self.assertIn("disk", active)  # any usage trips a 0% threshold
        active = self._watchdog(disk_checks={str(self.root): 100}).step()
        self.assertNotIn("disk", active)


if __name__ == "__main__":
    unittest.main()
