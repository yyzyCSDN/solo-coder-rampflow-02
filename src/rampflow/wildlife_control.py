from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from math import hypot
from typing import Iterable, Mapping

from .timeutil import utc

# Reporting channels: tower, patrol vehicle (bird-control car) and apron control.
SOURCES = ("tower", "patrol", "apron")


class WildlifeError(ValueError):
    """Domain rule violation (unknown subject, illegal transition, stale record)."""


@dataclass(frozen=True)
class Position:
    x_m: float
    y_m: float

    def distance_to(self, other: "Position") -> float:
        return hypot(self.x_m - other.x_m, self.y_m - other.y_m)


@dataclass(frozen=True)
class Area:
    """Axis-aligned surface footprint in metre coordinates."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self):
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise ValueError("degenerate area")

    @classmethod
    def disk(cls, center: Position, radius_m: float) -> "Area":
        if radius_m < 0:
            raise ValueError("radius must be non-negative")
        return cls(
            center.x_m - radius_m,
            center.y_m - radius_m,
            center.x_m + radius_m,
            center.y_m + radius_m,
        )

    def expanded(self, metres: float) -> "Area":
        if metres < 0:
            raise ValueError("buffer must be non-negative")
        return Area(
            self.min_x - metres,
            self.min_y - metres,
            self.max_x + metres,
            self.max_y + metres,
        )

    def union(self, other: "Area") -> "Area":
        return Area(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )

    def intersects(self, other: "Area") -> bool:
        return not (
            self.max_x < other.min_x
            or other.max_x < self.min_x
            or self.max_y < other.min_y
            or other.max_y < self.min_y
        )

    def contains(self, point: Position) -> bool:
        return (
            self.min_x <= point.x_m <= self.max_x
            and self.min_y <= point.y_m <= self.max_y
        )


@dataclass(frozen=True)
class SurfaceElement:
    """A mappable piece of the movement area: runway, taxiway segment or stand."""

    element_id: str
    kind: str
    footprint: Area


@dataclass(frozen=True)
class SightingReport:
    report_id: str
    source: str
    sighted_at: datetime
    received_at: datetime
    position: Position
    flock_radius_m: float = 25.0
    species: str = "bird"
    count: int = 1

    def __post_init__(self):
        if not self.report_id:
            raise ValueError("report_id required")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source}")
        if self.flock_radius_m < 0 or self.count < 1:
            raise ValueError("invalid flock description")
        seen = utc(self.sighted_at)
        received = utc(self.received_at)
        if received < seen:
            raise ValueError("report cannot be received before it is sighted")
        object.__setattr__(self, "sighted_at", seen)
        object.__setattr__(self, "received_at", received)


@dataclass(frozen=True)
class RecoveryAction:
    action_id: str
    label: str
    dependencies: frozenset[str] = frozenset()
    ordered: bool = True


RECOVERY_PLAN: tuple[RecoveryAction, ...] = (
    RecoveryAction("dispatch_patrol", "dispatch bird-control patrol"),
    RecoveryAction("patrol_on_scene", "patrol on scene", frozenset({"dispatch_patrol"})),
    RecoveryAction("sweep_complete", "hazard sweep complete", frozenset({"patrol_on_scene"})),
    RecoveryAction("area_clear", "controller confirms area clear", frozenset({"sweep_complete"})),
    RecoveryAction("resume_movements", "resume paused movements", frozenset({"area_clear"})),
    RecoveryAction("resolve", "close wildlife incident", frozenset({"resume_movements"})),
)


class IncidentState:
    REPORTED = "reported"
    CONTAINED = "contained"
    RECOVERING = "recovering"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


# --- Immutable domain events: the append-only authority on disposition order ---


@dataclass(frozen=True)
class WildlifeEvent:
    seq: int
    event_id: str
    incident_id: str
    at: datetime
    kind: str
    payload: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "at", utc(self.at))


@dataclass(frozen=True)
class IncidentView:
    incident_id: str
    state: str
    sighting: Area
    containment: Area
    affected_elements: frozenset[str]
    paused: tuple[str, ...]
    resumed: tuple[str, ...]
    completed: frozenset[str]
    log: tuple[WildlifeEvent, ...]
    observation_reports: tuple[str, ...]
    latest_sighted_at: datetime
    contained_at: datetime | None = None

    @property
    def version(self) -> int:
        return len(self.log)


@dataclass(frozen=True)
class IntakeDecision:
    accepted: bool
    code: str
    reason: str
    incident_id: str
    affected_elements: frozenset[str]
    paused_movements: tuple[str, ...]
    contained: bool
    late: bool
    report_id: str
    event_ids: tuple[str, ...]
    details: Mapping[str, object] = field(default_factory=dict)


_MOVEMENT_KINDS = {"flight", "vehicle", "tow"}


class WildlifeControlService:
    """Coordinates bird-flock containment around the runway.

    Intake is idempotent by report_id.  Every disposition is an immutable event
    appended under a monotonic sequence, so a record delivered late or re-sent
    after a crash cannot reopen, reorder or duplicate a confirmed containment.
    """

    def __init__(self, safety_buffer_m: float = 100.0):
        if safety_buffer_m < 0:
            raise ValueError("safety_buffer_m must be non-negative")
        self.safety_buffer_m = safety_buffer_m
        self.elements: dict[str, SurfaceElement] = {}
        self._movements: dict[str, Area] = {}
        self.incidents: dict[str, IncidentView] = {}
        self.log: list[WildlifeEvent] = []
        self._reports: dict[str, str] = {}
        self._seq = 0

    # -- topology registration -------------------------------------------------

    def register_element(self, element: SurfaceElement) -> None:
        if element.element_id in self.elements:
            raise WildlifeError(f"duplicate element {element.element_id}")
        self.elements[element.element_id] = element

    def register_movement(self, movement_id: str, kind: str, footprint: Area) -> None:
        if not movement_id:
            raise WildlifeError("movement_id required")
        if kind not in _MOVEMENT_KINDS:
            raise WildlifeError(f"unknown movement kind {kind}")
        if movement_id in self._movements:
            raise WildlifeError(f"duplicate movement {movement_id}")
        self._movements[movement_id] = footprint

    # -- report intake ----------------------------------------------------------

    def intake(self, report: SightingReport) -> IntakeDecision:
        """Accept a report exactly once.

        Duplicate report_ids return the original decision.  A sighting inside an
        already-contained footprint is attached as an observation; a record that
        arrives late (its sighted time precedes the containment decision) is
        recorded but never rewinds the confirmed disposition.
        """
        if report.report_id in self._reports:
            return self._duplicate_decision(report)

        host = self._host_incident(report)
        if host is not None:
            resolved_late = host.state == IncidentState.RESOLVED
            late = resolved_late or bool(
                host.state not in {IncidentState.REPORTED, IncidentState.RESOLVED}
                and report.sighted_at <= (host.contained_at or report.received_at)
            )
            self._append(
                host.incident_id,
                report.received_at,
                "observation.recorded",
                {
                    "report_id": report.report_id,
                    "source": report.source,
                    "sighted_at": report.sighted_at.isoformat(),
                    "late": late,
                    "in_containment": True,
                    "post_resolution": resolved_late,
                },
            )
            self._reports[report.report_id] = host.incident_id
            view = self.incidents[host.incident_id]
            code = (
                "observation-attached-late"
                if late
                else "observation-attached"
            )
            reason = (
                "sighting predates a confirmed closure; audit-only, order unchanged"
                if resolved_late
                else "sighting inside active containment; logged without new closure"
            )
            return IntakeDecision(
                True,
                code,
                reason,
                view.incident_id,
                view.affected_elements,
                view.paused,
                False,
                late,
                report.report_id,
                tuple(e.event_id for e in view.log[-1:]),
            )

        incident_id = f"WCI-{len(self.incidents) + 1:04d}"
        sighting = Area.disk(report.position, report.flock_radius_m)
        containment = sighting.expanded(self.safety_buffer_m)
        affected = frozenset(
            eid
            for eid, element in self.elements.items()
            if element.footprint.intersects(containment)
        )
        paused = tuple(
            mid
            for mid, footprint in sorted(self._movements.items())
            if footprint.intersects(containment)
        )

        event_ids: list[str] = []
        event_ids.append(
            self._append(
                incident_id,
                report.received_at,
                "incident.reported",
                {
                    "report_id": report.report_id,
                    "source": report.source,
                    "sighted_at": report.sighted_at.isoformat(),
                    "species": report.species,
                    "count": report.count,
                    "sighting_area": _area_payload(sighting),
                },
            ).event_id
        )
        event_ids.append(
            self._append(
                incident_id,
                report.received_at,
                "area.contained",
                {
                    "containment_area": _area_payload(containment),
                    "affected_elements": sorted(affected),
                    "paused_movements": list(paused),
                    "safety_buffer_m": self.safety_buffer_m,
                },
            ).event_id
        )
        self._reports[report.report_id] = incident_id
        view = self.incidents[incident_id]
        return IntakeDecision(
            True,
            "contained",
            "movement area contained; patrol re-inspection scheduled",
            incident_id,
            affected,
            paused,
            True,
            False,
            report.report_id,
            tuple(event_ids),
        )

    def _duplicate_decision(self, report: SightingReport) -> IntakeDecision:
        incident_id = self._reports[report.report_id]
        view = self.incidents[incident_id]
        return IntakeDecision(
            True,
            "duplicate",
            "report already applied; containment not repeated",
            incident_id,
            view.affected_elements,
            view.paused,
            False,
            False,
            report.report_id,
            (),
        )

    def _host_incident(self, report: SightingReport) -> IncidentView | None:
        """Locate the incident a sighting belongs to.

        Active footprints always host new sightings.  A resolved incident hosts
        a record only when the sighting predates its resolution (a genuinely
        late delivery); a sighting after resolution is a new hazard.
        """
        best: IncidentView | None = None
        for view in self.incidents.values():
            if not view.containment.contains(report.position):
                continue
            if view.state != IncidentState.RESOLVED:
                if best is None or best.state == IncidentState.RESOLVED:
                    best = view
                continue
            resolved_at = view.log[-1].at
            if report.sighted_at <= resolved_at:
                if best is None or best.state == IncidentState.RESOLVED:
                    best = view
        return best

    # -- recovery checklist -----------------------------------------------------

    def ready_actions(self, incident_id: str) -> tuple[RecoveryAction, ...]:
        view = self._require(incident_id)
        if view.state not in {IncidentState.CONTAINED, IncidentState.RECOVERING}:
            return ()
        return tuple(
            action
            for action in RECOVERY_PLAN
            if action.action_id not in view.completed
            and action.dependencies <= view.completed
        )

    def complete_action(
        self, incident_id: str, action_id: str, at: datetime, actor: str
    ) -> IncidentView:
        at = utc(at)
        ready = {a.action_id: a for a in self.ready_actions(incident_id)}
        if action_id not in ready:
            raise WildlifeError(
                f"action {action_id} not ready; follow the confirmed recovery order"
            )
        view = self._require(incident_id)
        # Observations (which may themselves arrive late) do not anchor the
        # disposition clock; only confirmed state-changing steps do.
        dispositions = [e for e in view.log if e.kind != "observation.recorded"]
        if dispositions and at < dispositions[-1].at:
            raise WildlifeError("late record cannot precede the last disposition")
        payload: dict[str, object] = {"actor": actor}
        if action_id == "resume_movements":
            pending = [m for m in view.paused if m not in view.resumed]
            payload["resumed_movements"] = pending
        self._append(incident_id, at, f"action.{action_id}.completed", payload)
        view = self.incidents[incident_id]
        if view.state == IncidentState.CONTAINED:
            self._append(incident_id, at, "incident.recovering", {"actor": actor})
            view = self.incidents[incident_id]
        return view

    def rollback_order(self, incident_id: str) -> tuple[str, ...]:
        """Order in which confirmed reversible steps would be unwound."""
        view = self._require(incident_id)
        return tuple(
            action.action_id
            for action in reversed(RECOVERY_PLAN)
            if action.action_id in view.completed and action.ordered
        )

    # -- event log and replay ---------------------------------------------------

    def _append(self, incident_id: str, at: datetime, kind: str, payload: Mapping[str, object]) -> WildlifeEvent:
        self._seq += 1
        event = WildlifeEvent(
            self._seq,
            f"WCE-{self._seq:06d}",
            incident_id,
            utc(at),
            kind,
            dict(payload),
        )
        self.log.append(event)
        self.incidents[incident_id] = self._apply(
            self.incidents.get(incident_id), event
        )
        return event

    def replay(self) -> "WildlifeControlService":
        """Rebuild identical projections from the immutable log (determinism)."""
        clone = WildlifeControlService(self.safety_buffer_m)
        clone.elements = dict(self.elements)
        clone._movements = dict(self._movements)
        for event in self.log:
            clone._seq = event.seq - 1
            clone._append(event.incident_id, event.at, event.kind, event.payload)
            if event.kind in {"incident.reported", "observation.recorded"}:
                clone._reports[str(event.payload["report_id"])] = event.incident_id
        return clone

    def _apply(self, view: IncidentView | None, event: WildlifeEvent) -> IncidentView:
        kind = event.kind
        if kind == "incident.reported":
            if view is not None:
                raise WildlifeError("incident already opened")
            sighting = _payload_area(event.payload["sighting_area"])
            return IncidentView(
                event.incident_id,
                IncidentState.REPORTED,
                sighting,
                sighting,
                frozenset(),
                (),
                (),
                frozenset(),
                (event,),
                (),
                utc(datetime.fromisoformat(str(event.payload["sighted_at"]))),
            )

        assert view is not None, event.incident_id
        log = view.log + (event,)

        if kind == "area.contained":
            if view.state != IncidentState.REPORTED:
                raise WildlifeError("containment can only follow a report")
            return replace(
                view,
                state=IncidentState.CONTAINED,
                containment=_payload_area(event.payload["containment_area"]),
                affected_elements=frozenset(event.payload["affected_elements"]),
                paused=tuple(event.payload["paused_movements"]),
                contained_at=event.at,
                log=log,
            )

        if kind == "observation.recorded":
            sighted_at = utc(datetime.fromisoformat(str(event.payload["sighted_at"])))
            # Late observations are retained for the audit trail but never move
            # the state, footprint or confirmed disposition order.
            return replace(
                view,
                observation_reports=view.observation_reports
                + (str(event.payload["report_id"]),),
                latest_sighted_at=max(view.latest_sighted_at, sighted_at),
                log=log,
            )

        if kind == "incident.recovering":
            if view.state != IncidentState.CONTAINED:
                raise WildlifeError("recovery can only start from containment")
            return replace(view, state=IncidentState.RECOVERING, log=log)

        if kind.startswith("action.") and kind.endswith(".completed"):
            action_id = kind[len("action.") : -len(".completed")]
            allowed = {a.action_id for a in RECOVERY_PLAN}
            if action_id not in allowed:
                raise WildlifeError(f"unknown recovery action {action_id}")
            deps = next(a.dependencies for a in RECOVERY_PLAN if a.action_id == action_id)
            if action_id in view.completed or not deps <= view.completed:
                raise WildlifeError(f"recovery step {action_id} out of order")
            resumed = view.resumed
            if action_id == "resume_movements":
                resumed = tuple(view.paused)
            completed = view.completed | {action_id}
            state = view.state
            if action_id == "resolve":
                state = IncidentState.RESOLVED
            return replace(
                view,
                state=state,
                completed=completed,
                resumed=resumed,
                log=log,
            )

        raise WildlifeError(f"unknown event kind {kind}")

    def _require(self, incident_id: str) -> IncidentView:
        try:
            return self.incidents[incident_id]
        except KeyError:
            raise WildlifeError(f"unknown incident {incident_id}") from None


def _area_payload(area: Area) -> Mapping[str, float]:
    return {
        "min_x": area.min_x,
        "min_y": area.min_y,
        "max_x": area.max_x,
        "max_y": area.max_y,
    }


def _payload_area(payload: object) -> Area:
    data = payload  # type: ignore[assignment]
    return Area(
        float(data["min_x"]),  # type: ignore[index]
        float(data["min_y"]),  # type: ignore[index]
        float(data["max_x"]),  # type: ignore[index]
        float(data["max_y"]),  # type: ignore[index]
    )
