from __future__ import annotations
from dataclasses import dataclass,replace
from datetime import datetime,timedelta
from enum import Enum
from math import hypot
from typing import Iterable
from .timeutil import utc
from .models import Decision

class CaseState(str,Enum):LOGGED='logged';CONTAINED='contained';REINSPECTING='reinspecting';CLEARED='cleared';CLOSED='closed'

@dataclass(frozen=True)
class OperationArea:
    area_id:str
    kind:str
    center_x:float
    center_y:float
    radius_m:float
    def __post_init__(self):
        if not self.area_id or self.radius_m<=0:raise ValueError('invalid operation area')

@dataclass(frozen=True)
class WildlifeReport:
    report_id:str
    channel:str
    observed_at:datetime
    x:float
    y:float
    flock_size:int=1
    species:str=''
    severity:int=1
    def __post_init__(self):
        if not self.report_id:raise ValueError('report id required')
        if self.flock_size<1:raise ValueError('flock size must be positive')
        object.__setattr__(self,'observed_at',utc(self.observed_at))

@dataclass(frozen=True)
class MovementHold:
    area_id:str
    held_at:datetime
    lifted_at:datetime|None=None
    def __post_init__(self):
        object.__setattr__(self,'held_at',utc(self.held_at))
        if self.lifted_at is not None:object.__setattr__(self,'lifted_at',utc(self.lifted_at))
    @property
    def active(self)->bool:return self.lifted_at is None

@dataclass(frozen=True)
class Reinspection:
    area_id:str
    round:int
    due_at:datetime
    started_at:datetime|None=None
    completed_at:datetime|None=None
    clear:bool|None=None
    def __post_init__(self):
        object.__setattr__(self,'due_at',utc(self.due_at))
        if self.started_at is not None:object.__setattr__(self,'started_at',utc(self.started_at))
        if self.completed_at is not None:object.__setattr__(self,'completed_at',utc(self.completed_at))

@dataclass(frozen=True)
class WildlifeCase:
    case_id:str
    sequence:int
    state:CaseState
    observed_at:datetime
    contained_at:datetime
    x:float
    y:float
    severity:int
    areas:tuple[str,...]
    holds:tuple[MovementHold,...]
    reinspections:tuple[Reinspection,...]
    report_ids:tuple[str,...]
    version:int=1
    def __post_init__(self):
        object.__setattr__(self,'observed_at',utc(self.observed_at))
        object.__setattr__(self,'contained_at',utc(self.contained_at))

class WildlifeControl:
    """Bird hazard containment with idempotent reporting and guarded recovery.

    A report circles only the operational areas its hazard circle truly touches,
    pauses movements there and schedules reinspection in one step.  The same
    sighting relayed again over phone or radio folds into the existing case
    instead of re-containing, and the confirmed disposition sequence is assigned
    at acceptance time so late records never reorder it.
    """
    def __init__(self,areas:Iterable[OperationArea]=(),hazard_radius_m:float=300.0,reinspection_delay_min:int=20,dedup_radius_m:float=150.0,dedup_window_min:int=30):
        areas=tuple(areas)
        if len({a.area_id for a in areas})!=len(areas):raise ValueError('duplicate operation area')
        if hazard_radius_m<=0:raise ValueError('hazard radius must be positive')
        if reinspection_delay_min<=0:raise ValueError('reinspection delay must be positive')
        self.areas={a.area_id:a for a in areas}
        self.hazard_radius_m=hazard_radius_m
        self.reinspection_delay=timedelta(minutes=reinspection_delay_min)
        self.dedup_radius_m=dedup_radius_m
        self.dedup_window=timedelta(minutes=dedup_window_min)
        self._cases:dict[str,WildlifeCase]={}
        self._by_report:dict[str,str]={}
        self._sequence=0

    def affected_areas(self,x:float,y:float)->tuple[str,...]:
        reach=self.hazard_radius_m
        return tuple(sorted(a.area_id for a in self.areas.values() if hypot(a.center_x-x,a.center_y-y)<=a.radius_m+reach))

    def _duplicate_of(self,report:WildlifeReport)->WildlifeCase|None:
        case_id=self._by_report.get(report.report_id)
        if case_id is not None:return self._cases[case_id]
        window=self.dedup_window.total_seconds()
        for case in self._cases.values():
            if abs((case.observed_at-report.observed_at).total_seconds())<=window and hypot(case.x-report.x,case.y-report.y)<=self.dedup_radius_m:
                return case
        return None

    def report(self,report:WildlifeReport,received_at:datetime)->WildlifeCase:
        received_at=utc(received_at)
        duplicate=self._duplicate_of(report)
        if duplicate is not None:
            if report.report_id in duplicate.report_ids:return duplicate
            duplicate=replace(duplicate,report_ids=duplicate.report_ids+(report.report_id,),version=duplicate.version+1)
            self._cases[duplicate.case_id]=duplicate;self._by_report[report.report_id]=duplicate.case_id
            return duplicate
        areas=self.affected_areas(report.x,report.y)
        self._sequence+=1
        state=CaseState.CONTAINED if areas else CaseState.LOGGED
        case=WildlifeCase(f'WC-{self._sequence:04d}',self._sequence,state,report.observed_at,received_at,report.x,report.y,report.severity,areas,tuple(MovementHold(a,received_at) for a in areas),tuple(Reinspection(a,1,received_at+self.reinspection_delay) for a in areas),(report.report_id,))
        self._cases[case.case_id]=case;self._by_report[report.report_id]=case.case_id
        return case

    def case(self,case_id:str)->WildlifeCase:return self._cases[case_id]

    def cases(self)->tuple[WildlifeCase,...]:
        return tuple(sorted(self._cases.values(),key=lambda c:c.sequence))

    def _latest(self,case:WildlifeCase,area_id:str)->Reinspection:
        rounds=[r for r in case.reinspections if r.area_id==area_id]
        if not rounds:raise KeyError(area_id)
        return rounds[-1]

    @staticmethod
    def _swap(case:WildlifeCase,old:Reinspection,new:Reinspection)->tuple[Reinspection,...]:
        return tuple(new if (r.area_id==old.area_id and r.round==old.round) else r for r in case.reinspections)

    def start_reinspection(self,case_id:str,area_id:str,at:datetime)->WildlifeCase:
        at=utc(at);case=self._cases[case_id]
        if case.state not in (CaseState.CONTAINED,CaseState.REINSPECTING):raise RuntimeError('case not awaiting reinspection')
        current=self._latest(case,area_id)
        if current.completed_at is not None:raise RuntimeError('reinspection round already confirmed')
        if current.started_at is not None:return case
        updated=replace(current,started_at=at)
        case=replace(case,state=CaseState.REINSPECTING,reinspections=self._swap(case,current,updated),version=case.version+1)
        self._cases[case_id]=case;return case

    def record_reinspection(self,case_id:str,area_id:str,clear:bool,at:datetime)->WildlifeCase:
        at=utc(at);case=self._cases[case_id]
        if case.state!=CaseState.REINSPECTING:raise RuntimeError('reinspection not underway')
        current=self._latest(case,area_id)
        if current.started_at is None:raise RuntimeError('reinspection not dispatched')
        if current.completed_at is not None:
            if current.clear==clear:return case
            raise RuntimeError('reinspection outcome already confirmed')
        done=replace(current,completed_at=at,clear=clear)
        reins=list(self._swap(case,current,done))
        if not clear:reins.append(Reinspection(area_id,current.round+1,at+self.reinspection_delay))
        case=replace(case,reinspections=tuple(reins),version=case.version+1)
        self._cases[case_id]=case;return case

    def lift_holds(self,case_id:str,at:datetime)->WildlifeCase:
        at=utc(at);case=self._cases[case_id]
        if case.state==CaseState.CLOSED:raise RuntimeError('case closed')
        if case.state==CaseState.CLEARED:return case
        if case.state==CaseState.LOGGED:raise RuntimeError('case holds nothing')
        pending=sorted(a for a in case.areas if self._latest(case,a).clear is not True)
        if pending:raise RuntimeError(f'reinspection pending: {pending}')
        holds=tuple(replace(h,lifted_at=at) for h in case.holds)
        case=replace(case,state=CaseState.CLEARED,holds=holds,version=case.version+1)
        self._cases[case_id]=case;return case

    def close(self,case_id:str,at:datetime)->WildlifeCase:
        utc(at);case=self._cases[case_id]
        if case.state==CaseState.CLOSED:return case
        if case.state!=CaseState.CLEARED:raise RuntimeError('case not cleared')
        case=replace(case,state=CaseState.CLOSED,version=case.version+1)
        self._cases[case_id]=case;return case

    def movement_allowed(self,area_id:str)->Decision:
        for case in self.cases():
            if case.state in (CaseState.CONTAINED,CaseState.REINSPECTING):
                if any(h.area_id==area_id and h.active for h in case.holds):
                    return Decision(False,'wildlife-hold','movement paused by wildlife containment',{'case_id':case.case_id,'sequence':case.sequence})
        return Decision(True,'clear','no active wildlife hold')

    def active_holds(self)->tuple[tuple[str,str],...]:
        out=[]
        for case in self.cases():
            if case.state in (CaseState.CONTAINED,CaseState.REINSPECTING):
                out.extend((h.area_id,case.case_id) for h in case.holds if h.active)
        return tuple(sorted(out))

    def pending_reinspections(self,at:datetime)->tuple[Reinspection,...]:
        at=utc(at);out=[]
        for case in self._cases.values():
            if case.state not in (CaseState.CONTAINED,CaseState.REINSPECTING):continue
            for area in case.areas:
                current=self._latest(case,area)
                if current.completed_at is None and current.due_at<=at:out.append(current)
        return tuple(sorted(out,key=lambda r:(r.due_at,r.area_id,r.round)))
