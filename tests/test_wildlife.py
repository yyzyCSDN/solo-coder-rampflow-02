import unittest
from datetime import datetime, timezone, timedelta

from rampflow.wildlife import (
    CaseState,
    OperationArea,
    WildlifeControl,
    WildlifeReport,
)
from rampflow.workflows.control import RampWorkflowControl

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def make_control(**kw):
    areas = (
        OperationArea("RWY", "runway", 0.0, 0.0, 500.0),
        OperationArea("TWY-A", "taxiway", 800.0, 0.0, 100.0),
        OperationArea("APRON-1", "apron", 0.0, 2000.0, 300.0),
    )
    return WildlifeControl(areas, **kw)


class WildlifeReportTests(unittest.TestCase):
    def test_report_circles_only_truly_affected_areas(self):
        c = make_control(hazard_radius_m=200.0)
        case = c.report(WildlifeReport("R1", "tower", D(10), 100.0, 0.0), D(10, 1))
        self.assertEqual(case.areas, ("RWY",))
        self.assertEqual(case.state, CaseState.CONTAINED)
        self.assertEqual([h.area_id for h in case.holds], ["RWY"])
        self.assertEqual([r.area_id for r in case.reinspections], ["RWY"])
        self.assertEqual(case.reinspections[0].due_at, D(10, 21))

    def test_report_without_affected_area_is_logged_only(self):
        c = make_control(hazard_radius_m=100.0)
        case = c.report(WildlifeReport("R1", "patrol", D(10), 5000.0, 5000.0), D(10, 1))
        self.assertEqual(case.state, CaseState.LOGGED)
        self.assertEqual(case.holds, ())
        self.assertTrue(c.movement_allowed("RWY").accepted)

    def test_movement_paused_only_in_held_areas(self):
        c = make_control()
        c.report(WildlifeReport("R1", "tower", D(10), 0.0, 0.0), D(10, 1))
        self.assertFalse(c.movement_allowed("RWY").accepted)
        self.assertTrue(c.movement_allowed("APRON-1").accepted)
        self.assertEqual(c.active_holds(), (("RWY", "WC-0001"),))

    def test_same_report_resubmitted_does_not_recontain(self):
        c = make_control()
        first = c.report(WildlifeReport("R1", "tower", D(10), 0.0, 0.0), D(10, 1))
        again = c.report(WildlifeReport("R1", "tower", D(10), 0.0, 0.0), D(10, 5))
        self.assertEqual(again.case_id, first.case_id)
        self.assertEqual(again.version, 1)
        self.assertEqual(len(again.holds), 1)
        self.assertEqual(len(c.cases()), 1)

    def test_relayed_sighting_folds_into_existing_case(self):
        c = make_control()
        first = c.report(WildlifeReport("R1", "tower", D(10), 0.0, 0.0), D(10, 1))
        relay = c.report(WildlifeReport("R2", "apron", D(10, 4), 20.0, 0.0), D(10, 11))
        self.assertEqual(relay.case_id, first.case_id)
        self.assertEqual(relay.report_ids, ("R1", "R2"))
        self.assertEqual(len(relay.holds), 1)
        self.assertEqual(relay.sequence, first.sequence)
        self.assertEqual(len(c.cases()), 1)

    def test_late_duplicate_after_closure_does_not_reopen(self):
        c = make_control()
        case = c.report(WildlifeReport("R1", "tower", D(10), 0.0, 0.0), D(10, 1))
        c.start_reinspection(case.case_id, "RWY", D(10, 22))
        c.record_reinspection(case.case_id, "RWY", True, D(10, 30))
        c.lift_holds(case.case_id, D(10, 31))
        closed = c.close(case.case_id, D(10, 32))
        late = c.report(WildlifeReport("R9", "patrol", D(10, 2), 10.0, 0.0), D(11))
        self.assertEqual(late.case_id, closed.case_id)
        self.assertEqual(late.state, CaseState.CLOSED)
        self.assertTrue(c.movement_allowed("RWY").accepted)
        self.assertEqual(len(c.cases()), 1)

    def test_late_record_keeps_confirmed_sequence(self):
        c = make_control()
        first = c.report(WildlifeReport("R1", "tower", D(10), 0.0, 0.0), D(10, 1))
        late = c.report(WildlifeReport("R2", "patrol", D(9, 30), 0.0, 2000.0), D(10, 2))
        self.assertEqual((first.sequence, late.sequence), (1, 2))
        self.assertEqual([x.case_id for x in c.cases()], [first.case_id, late.case_id])
        self.assertEqual(late.areas, ("APRON-1",))


class WildlifeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.c = make_control()
        self.case = self.c.report(WildlifeReport("R1", "tower", D(10), 0.0, 0.0), D(10, 1))

    def test_recovery_steps_cannot_be_skipped(self):
        c = self.c
        with self.assertRaises(RuntimeError):
            c.record_reinspection(self.case.case_id, "RWY", True, D(10, 25))
        with self.assertRaises(RuntimeError):
            c.lift_holds(self.case.case_id, D(10, 25))
        c.start_reinspection(self.case.case_id, "RWY", D(10, 22))
        c.record_reinspection(self.case.case_id, "RWY", True, D(10, 30))
        lifted = c.lift_holds(self.case.case_id, D(10, 31))
        self.assertEqual(lifted.state, CaseState.CLEARED)
        self.assertTrue(c.movement_allowed("RWY").accepted)
        closed = c.close(self.case.case_id, D(10, 32))
        self.assertEqual(closed.state, CaseState.CLOSED)

    def test_not_clear_reinspection_keeps_hold_and_reschedules(self):
        c = self.c
        c.start_reinspection(self.case.case_id, "RWY", D(10, 22))
        case = c.record_reinspection(self.case.case_id, "RWY", False, D(10, 30))
        self.assertFalse(c.movement_allowed("RWY").accepted)
        rounds = [r for r in case.reinspections if r.area_id == "RWY"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[-1].due_at, D(10, 50))
        with self.assertRaises(RuntimeError):
            c.lift_holds(self.case.case_id, D(10, 31))
        c.start_reinspection(self.case.case_id, "RWY", D(10, 51))
        c.record_reinspection(self.case.case_id, "RWY", True, D(10, 58))
        c.lift_holds(self.case.case_id, D(10, 59))
        self.assertTrue(c.movement_allowed("RWY").accepted)

    def test_late_conflicting_outcome_rejected_identical_replay_ignored(self):
        c = self.c
        c.start_reinspection(self.case.case_id, "RWY", D(10, 22))
        confirmed = c.record_reinspection(self.case.case_id, "RWY", True, D(10, 30))
        replay = c.record_reinspection(self.case.case_id, "RWY", True, D(10, 35))
        self.assertEqual(replay.version, confirmed.version)
        with self.assertRaises(RuntimeError):
            c.record_reinspection(self.case.case_id, "RWY", False, D(10, 35))

    def test_pending_reinspections_work_list(self):
        due = self.c.pending_reinspections(D(10, 30))
        self.assertEqual([r.area_id for r in due], ["RWY"])
        self.assertEqual(self.c.pending_reinspections(D(10, 5)), ())

    def test_overlapping_cases_hold_area_until_all_clear(self):
        c = self.c
        other = c.report(WildlifeReport("R2", "patrol", D(11), 100.0, 0.0), D(11, 1))
        for case, at in ((self.case, D(10, 30)), (other, D(11, 30))):
            c.start_reinspection(case.case_id, "RWY", at)
            c.record_reinspection(case.case_id, "RWY", True, at + timedelta(minutes=5))
        c.lift_holds(self.case.case_id, D(10, 40))
        self.assertFalse(c.movement_allowed("RWY").accepted)
        c.lift_holds(other.case_id, D(11, 40))
        self.assertTrue(c.movement_allowed("RWY").accepted)


class WildlifeWorkflowTests(unittest.TestCase):
    def test_containment_and_clearance_commit_through_workflow(self):
        c = make_control()
        wf = RampWorkflowControl()
        receipt = wf.commit_wildlife_containment(
            "op-contain-1", "RWY", D(10, 1),
            lambda: c.report(WildlifeReport("R1", "tower", D(10), 0.0, 0.0), D(10, 1)),
        )
        self.assertEqual(receipt.event.topic, "wildlife.containment.committed")
        replay = wf.commit_wildlife_containment(
            "op-contain-1", "RWY", D(10, 2),
            lambda: c.report(WildlifeReport("R1", "tower", D(10), 0.0, 0.0), D(10, 2)),
        )
        self.assertIs(replay, receipt)
        self.assertEqual(len(c.cases()), 1)
        case = receipt.result
        c.start_reinspection(case.case_id, "RWY", D(10, 22))
        c.record_reinspection(case.case_id, "RWY", True, D(10, 30))
        cleared = wf.commit_wildlife_clearance(
            "op-clear-1", "RWY", D(10, 31),
            lambda: c.lift_holds(case.case_id, D(10, 31)),
        )
        self.assertEqual(cleared.event.topic, "wildlife.clearance.committed")
        self.assertTrue(c.movement_allowed("RWY").accepted)
        self.assertTrue(wf.stream.verify_chain())


if __name__ == "__main__":
    unittest.main()
