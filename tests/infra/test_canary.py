"""Unit tests for cases.infra.canary (no network: probe_fn injected)."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cases.infra.canary import Canary, Probe


def make_canary(tmp: Path, probe_fn, **overrides) -> Canary:
    kwargs = dict(
        endpoint="http://endpoint.test/v1", api_key="k", model="m",
        state_path=tmp / "canary_state.json",
        pause_path=tmp / "canary_pause.flag",
        probe_fn=probe_fn,
    )
    kwargs.update(overrides)
    return Canary(**kwargs)


class CanaryCircuitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_consecutive_failures_open_circuit(self) -> None:
        canary = make_canary(self.root, lambda: Probe(False, 0.1, "http 502"))
        canary.step()
        self.assertEqual(canary.circuit, "closed")  # 1 failure: not yet
        canary.step()
        self.assertEqual(canary.circuit, "open")    # 2 consecutive -> open
        self.assertTrue(canary.pause_path.exists())
        owner = canary._flag_owner()
        self.assertEqual(owner, "canary")

    def test_window_trip_without_consecutive(self) -> None:
        results = iter([Probe(False, 0.1), Probe(True, 0.1),
                        Probe(False, 0.1), Probe(False, 0.1)])
        canary = make_canary(self.root, lambda: next(results),
                             consecutive_fail=99, window=4, trip_failures=3)
        for _ in range(4):
            canary.step()
        self.assertEqual(canary.circuit, "open")

    def test_recovery_closes_and_removes_flag(self) -> None:
        state = {"ok": False}
        canary = make_canary(
            self.root, lambda: Probe(state["ok"], 0.1),
            reset_lookback=3)
        canary.step()
        canary.step()
        self.assertEqual(canary.circuit, "open")
        state["ok"] = True
        canary.step()
        canary.step()
        self.assertEqual(canary.circuit, "open")  # 2 ok: not yet
        canary.step()
        self.assertEqual(canary.circuit, "closed")
        self.assertFalse(canary.pause_path.exists())

    def test_state_roundtrip(self) -> None:
        canary = make_canary(self.root, lambda: Probe(True, 0.2))
        for _ in range(3):
            canary.step()
        revived = make_canary(self.root, lambda: Probe(True, 0.2))
        revived.load()
        self.assertEqual(len(revived.probes), 3)
        self.assertEqual(revived.circuit, canary.circuit)

    def test_manual_flag_is_never_removed(self) -> None:
        canary = make_canary(
            self.root, lambda: Probe(True, 0.1), reset_lookback=1)
        canary.pause_path.write_text('{"owner": "human", "reason": "manual"}')
        canary.step()
        # closed circuit, but the flag belongs to a human: untouched
        self.assertTrue(canary.pause_path.exists())


if __name__ == "__main__":
    unittest.main()
