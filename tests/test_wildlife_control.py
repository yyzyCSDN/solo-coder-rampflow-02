import unittest
from datetime import datetime, timezone

from rampflow.wildlife_control import (
    Area,
    IncidentState,
    RECOVERY_PLAN,
    Position,
    SightingReport,
    SurfaceElement,
    WildlifeControlService,
    WildlifeError,
)

UTC = timezone.utc
T = lambda h, m=0, s=0: datetime(2026, 5, 4, h, m, s, tzinfo=UTC)


def build_service(buffer_m=100.0):
    svc = WildlifeControlService(safety_buffer_m=buffer_m)
    # Runway 09/27 strip runs east-west around y=0.
    svc.register_element(
        SurfaceElement("RWY-09", "runway", Area(-2000, -30, 2000, 30))
    )
    # Parallel taxiway A sits 200 m north of the runway centreline, beyond the
    # default 100 m containment buffer.
    svc.register_element(SurfaceElement("TWY-A", "taxiway", Area(-2000, 190, 2000, 210)))
    # A remote stand well clear of the runway edge.
    svc.register_element(SurfaceElement("STD-901", "stand", Area(1500, 600, 1600, 700)))
    # Movements: arrival on final along the strip, taxiing aircraft on TWY-A,
    # and an apron bus that never goes near the runway edge.
    svc.register_movement("ARR-301", "flight", Area(0, -30, 400, 30))
    svc.register_movement("TAXI-17", "flight", Area(0, 190, 300, 210))
    svc.register_movement("BUS-5", "vehicle", Area(1500, 600, 1620, 700))
    return svc


class ContainmentTests(unittest.TestCase):
    def test_containment_marks_only_truly_affected_area(self):
        svc = build_service()
        decision = svc.intake(
            SightingReport(
                "R-1", "patrol", T(8, 0), T(8, 1), Position(100, 20), flock_radius_m=20
            )
        )
        self.assertEqual(decision.code, "contained")
        # Flock at runway edge + 100 m buffer clips the runway but not TWY-A
        # (centreline 150 m away) and never the remote stand.
        self.assertEqual(decision.affected_elements, frozenset({"RWY-09"}))
        # Only the arrival is paused; the TWY-A taxi and apron bus keep moving.
        self.assertEqual(decision.paused_movements, ("ARR-301",))
        view = svc.incidents[decision.incident_id]
        self.assertEqual(view.state, IncidentState.CONTAINED)
        self.assertTrue(view.containment.contains(Position(100, 20)))
        self.assertFalse(view.containment.contains(Position(100, 150)))

    def test_duplicate_report_does_not_re_contain(self):
        svc = build_service()
        first = svc.intake(
            SightingReport("R-1", "tower", T(8, 0), T(8, 1), Position(100, 0))
        )
        # Same report re-sent (e.g. tower and apron car relay the same phone call).
        second = svc.intake(
            SightingReport("R-1", "tower", T(8, 0), T(8, 5), Position(100, 0))
        )
        self.assertEqual(second.code, "duplicate")
        self.assertEqual(first.incident_id, second.incident_id)
        self.assertEqual(second.event_ids, ())
        self.assertFalse(second.contained)
        self.assertEqual(len(svc.incidents), 1)
        # Only the original report/container events exist.
        self.assertEqual(
            [e.kind for e in svc.log], ["incident.reported", "area.contained"]
        )

    def test_same_location_new_report_attaches_as_observation(self):
        svc = build_service()
        svc.intake(
            SightingReport("R-1", "tower", T(8, 0), T(8, 1), Position(100, 0))
        )
        follow = svc.intake(
            SightingReport("R-2", "apron", T(8, 3), T(8, 4), Position(105, 5))
        )
        self.assertTrue(follow.code.startswith("observation-attached"))
        self.assertEqual(len(svc.incidents), 1)
        view = svc.incidents[follow.incident_id]
        self.assertEqual(view.observation_reports, ("R-2",))
        self.assertEqual(view.paused, ("ARR-301",))


class RecoveryOrderTests(unittest.TestCase):
    def _contain(self, svc):
        return svc.intake(
            SightingReport("R-1", "tower", T(8, 0), T(8, 1), Position(100, 0))
        )

    def test_checklist_runs_in_dependency_order(self):
        svc = build_service()
        decision = self._contain(svc)
        iid = decision.incident_id
        self.assertEqual(
            [a.action_id for a in svc.ready_actions(iid)], ["dispatch_patrol"]
        )
        with self.assertRaises(WildlifeError):
            svc.complete_action(iid, "area_clear", T(8, 10), "tower")
        steps = [a.action_id for a in RECOVERY_PLAN]
        for idx, action_id in enumerate(steps):
            ready = [a.action_id for a in svc.ready_actions(iid)]
            self.assertEqual(ready, [action_id])
            svc.complete_action(iid, action_id, T(8, 5 + idx), "patrol-7")
        view = svc.incidents[iid]
        self.assertEqual(view.state, IncidentState.RESOLVED)
        self.assertEqual(view.resumed, ("ARR-301",))
        self.assertEqual(svc.ready_actions(iid), ())

    def test_resume_happens_only_after_clearance(self):
        svc = build_service()
        iid = self._contain(svc).incident_id
        for action_id in ("dispatch_patrol", "patrol_on_scene", "sweep_complete"):
            svc.complete_action(iid, action_id, T(8, 6), "patrol-7")
        view = svc.incidents[iid]
        self.assertEqual(view.resumed, ())
        svc.complete_action(iid, "area_clear", T(8, 7), "tower")
        svc.complete_action(iid, "resume_movements", T(8, 8), "apron")
        view = svc.incidents[iid]
        self.assertEqual(view.resumed, ("ARR-301",))

    def test_completed_step_cannot_be_completed_again(self):
        svc = build_service()
        iid = self._contain(svc).incident_id
        svc.complete_action(iid, "dispatch_patrol", T(8, 5), "patrol-7")
        with self.assertRaises(WildlifeError):
            svc.complete_action(iid, "dispatch_patrol", T(8, 6), "patrol-7")


class LateRecordTests(unittest.TestCase):
    def test_late_sighting_does_not_rewind_confirmed_disposition(self):
        svc = build_service()
        svc.intake(
            SightingReport("R-1", "tower", T(8, 0), T(8, 1), Position(100, 0))
        )
        # Patrol confirms the sweep at 08:10.
        for action_id, who, at in [
            ("dispatch_patrol", "apron", T(8, 2)),
            ("patrol_on_scene", "patrol-7", T(8, 4)),
            ("sweep_complete", "patrol-7", T(8, 10)),
        ]:
            svc.complete_action("WCI-0001", action_id, at, who)

        before = svc.incidents["WCI-0001"]
        version_before = before.version
        paused_before = before.paused
        completed_before = before.completed

        # A handwritten note from the patrol car arrives late: a sighting at
        # 07:55, inside the already contained area, delivered only at 08:20.
        late = svc.intake(
            SightingReport(
                "R-LATE", "patrol", T(7, 55), T(8, 20), Position(90, 5)
            )
        )
        self.assertTrue(late.late)
        self.assertEqual(late.code, "observation-attached-late")
        after = svc.incidents["WCI-0001"]
        # Confirmed disposition is untouched; only the audit trail grows.
        self.assertEqual(after.completed, completed_before)
        self.assertEqual(after.paused, paused_before)
        self.assertEqual(after.state, IncidentState.RECOVERING)
        self.assertEqual(after.version, version_before + 1)
        self.assertIn("R-LATE", after.observation_reports)
        # Next step is still exactly where the checklist left off.
        self.assertEqual(
            [a.action_id for a in svc.ready_actions("WCI-0001")], ["area_clear"]
        )

    def test_event_sequence_is_monotonic_and_replay_is_identical(self):
        svc = build_service()
        svc.intake(
            SightingReport("R-1", "tower", T(8, 0), T(8, 1), Position(100, 0))
        )
        svc.intake(
            SightingReport("R-2", "patrol", T(7, 50), T(8, 30), Position(100, 0))
        )  # late observation
        svc.complete_action("WCI-0001", "dispatch_patrol", T(8, 2), "apron")

        seqs = [e.seq for e in svc.log]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))

        rebuilt = svc.replay()
        self.assertEqual(len(rebuilt.log), len(svc.log))
        for a, b in zip(svc.log, rebuilt.log):
            self.assertEqual(a.event_id, b.event_id)
            self.assertEqual(a.kind, b.kind)
            self.assertEqual(a.payload, b.payload)
        view = rebuilt.incidents["WCI-0001"]
        self.assertEqual(view.state, IncidentState.RECOVERING)
        self.assertEqual(view.completed, frozenset({"dispatch_patrol"}))
        self.assertEqual(view.observation_reports, ("R-2",))
        # Idempotency survives replay.
        again = rebuilt.intake(
            SightingReport("R-1", "tower", T(8, 0), T(8, 1), Position(100, 0))
        )
        self.assertEqual(again.code, "duplicate")

    def test_late_delivery_after_resolution_does_not_reopen_incident(self):
        svc = build_service()
        iid = svc.intake(
            SightingReport("R-1", "tower", T(8, 0), T(8, 1), Position(100, 0))
        ).incident_id
        for n, action_id in enumerate(a.action_id for a in RECOVERY_PLAN):
            svc.complete_action(iid, action_id, T(8, 2 + n), "patrol-7")
        self.assertEqual(svc.incidents[iid].state, IncidentState.RESOLVED)

        # Handwritten note sighted at 07:55, delivered at 09:30.
        late = svc.intake(
            SightingReport("R-LATE", "patrol", T(7, 55), T(9, 30), Position(90, 5))
        )
        self.assertEqual(late.code, "observation-attached-late")
        self.assertTrue(late.late)
        self.assertFalse(late.contained)
        self.assertEqual(len(svc.incidents), 1)
        self.assertEqual(svc.incidents[iid].state, IncidentState.RESOLVED)
        self.assertEqual(svc.incidents[iid].paused, ("ARR-301",))

        # But a genuinely new sighting at 09:40 (after clearance) opens fresh.
        fresh = svc.intake(
            SightingReport("R-9", "tower", T(9, 40), T(9, 41), Position(100, 0))
        )
        self.assertEqual(fresh.code, "contained")
        self.assertEqual(len(svc.incidents), 2)

    def test_action_timestamp_before_last_disposition_rejected(self):
        svc = build_service()
        iid = svc.intake(
            SightingReport("R-1", "tower", T(8, 0), T(8, 1), Position(100, 0))
        ).incident_id
        svc.complete_action(iid, "dispatch_patrol", T(8, 9), "apron")
        with self.assertRaises(WildlifeError):
            svc.complete_action(iid, "patrol_on_scene", T(8, 5), "patrol-7")

    def test_report_received_before_sighted_rejected(self):
        with self.assertRaises(ValueError):
            SightingReport("R-X", "tower", T(8, 10), T(8, 0), Position(0, 0))


if __name__ == "__main__":
    unittest.main()
