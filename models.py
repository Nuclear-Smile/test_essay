from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
import copy
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from patient_data import AdherenceState, Patient


@dataclass
class Doctor:
    doctor_id: str
    current_node: str
    queue_name: str
    available: bool = True
    history: List[str] = field(default_factory=list)
    transfers: List[Tuple[str, str]] = field(default_factory=list)
    skill_level: float = 1.0
    service_speed: float = 1.0
    adherence_boost: float = 0.0


class HealthcareSystem:
    """
    负责构建完整的医疗系统，包括患者生成、医生调度以及节点间距离矩阵等逻辑。
    该实现直接来源于 `原件 copy.ipynb` 中的核心代码，并保持行为一致。
    """

    def __init__(
        self,
        hospital_name: str = "综合医院",
        referral_name: str = "转诊中心",
        community_count: int = 27,
        hospital_outpatient_rooms: int = 3,
        hospital_emergency_rooms: int = 5,
        community_rooms: int = 3,
        patient_dataset_path: str | Path = "patient_dataset.csv",
        location_path: str | Path = "location.xlsx",
    ) -> None:
        self.hospital_name = hospital_name
        self.referral_name = referral_name
        self.patient_dataset_path = Path(patient_dataset_path)
        self.location_path = Path(location_path)

        self.community_names = np.array(
            [f"社康机构_{i + 1}" for i in range(community_count)],
            dtype="<U16",
        )
        self.all_nodes: List[str] = [self.hospital_name, self.referral_name] + list(
            self.community_names
        )

        self.all_patients: List[Patient] = []
        self.patient_generation_stats: Dict[str, int] = {}
        self.busy_rates: Dict[str, Dict[str, Dict[str, float]]] = {}

        self.p_hospital_outpatient_rooms = hospital_outpatient_rooms
        self.q_hospital_emergency_rooms = hospital_emergency_rooms
        self.n_community_rooms = community_rooms

        self.contiunity_matrix = np.random.rand(len(self.all_nodes), len(self.all_nodes))

        self.node_index_map = {name: idx for idx, name in enumerate(self.all_nodes)}
        self.index_node_map = {idx: name for idx, name in enumerate(self.all_nodes)}
        self.node_queues = self._initialize_queues()
        self.doctors = self._initialize_doctors()
        self.busy_rates = self._initialize_busy_rates()
        self._busy_rates_stale = True
        self.doctor_transfer_log: List[Dict[str, str]] = []
        self.doctor_transfer_unit_cost = 800.0
        self.transfer_matrix: Optional[np.ndarray] = None
        self.pending_revisits: Dict[int, List[Patient]] = defaultdict(list)
        self.revisit_delay_scale = 30.0  # mean delay for revisit in time steps
        self.adherence_effect_magnitude = 0.25  # 控制依从度对疗效影响的幅度
        self.adherence_improve_floor = 0.35  # 依从度极差时的转好概率下限
        self.adherence_improve_ceiling = 0.9  # 依从度最佳时的转好概率上限
        self.total_simulation_steps: int = 0

        self.distance_matrix = self._initialize_distance_matrix()

    # ------------------------------------------------------------------ #
    # 初始化逻辑
    # ------------------------------------------------------------------ #
    def _initialize_queues(self) -> Dict[str, Dict[str, Dict[str, object]]]:
        node_queues: Dict[str, Dict[str, Dict[str, object]]] = {
            self.hospital_name: {
                "门诊": {
                    "rooms": self.p_hospital_outpatient_rooms,
                    "queue": deque(),
                    "ongoing": {},
                },
                "急诊": {
                    "rooms": self.q_hospital_emergency_rooms,
                    "queue": deque(),
                    "ongoing": {},
                },
            },
            self.referral_name: {
                "转诊队列": {"rooms": 0, "queue": deque(), "ongoing": {}}
            },
        }

        for cname in self.community_names:
            node_queues[cname] = {
                "门诊": {
                    "rooms": self.n_community_rooms,
                    "queue": deque(),
                    "ongoing": {},
                }
            }
        return node_queues

    def _initialize_doctors(self) -> Dict[str, Dict[str, Dict[str, object]]]:
        self.doctor_registry: Dict[str, Doctor] = {}
        doctors_by_node: Dict[str, Dict[str, Dict[str, object]]] = {}
        doctor_counter = 1

        for node_name, queues in self.node_queues.items():
            doctors_by_node[node_name] = {}
            for q_name, q_data in queues.items():
                rooms = q_data["rooms"]
                if rooms <= 0:
                    continue

                doctor_capacity = max(1, rooms - 1)
                doctor_capacity = min(doctor_capacity, rooms)
                roster: List[Doctor] = []
                for _ in range(doctor_capacity):
                    doctor_id = f"D{doctor_counter:04d}"
                    doctor_counter += 1
                    doctor = Doctor(
                        doctor_id=doctor_id,
                        current_node=node_name,
                        queue_name=q_name,
                    )
                    doctor.skill_level = random.uniform(0.85, 1.25)
                    doctor.service_speed = random.uniform(0.8, 1.3)
                    doctor.adherence_boost = random.uniform(0.02, 0.15)
                    doctor.history.append(node_name)
                    roster.append(doctor)
                    self.doctor_registry[doctor_id] = doctor

                doctors_by_node[node_name][q_name] = {
                    "roster": roster,
                    "available": deque(roster),
                }
        return doctors_by_node

    def _initialize_busy_rates(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        rates: Dict[str, Dict[str, Dict[str, float]]] = {}
        for node_name, queues in self.node_queues.items():
            rates[node_name] = {}
            for q_name in queues.keys():
                rates[node_name][q_name] = {
                    "busy_doctors": 0,
                    "total_doctors": 0,
                    "observations": 0,
                    "queued_patients": 0,
                    "queued_samples": 0,
                }
        return rates

    # ------------------------------------------------------------------ #
    # 医生调度
    # ------------------------------------------------------------------ #
    def _available_doctor_count(self, node_name: str, queue_name: str) -> int:
        queue_info = self.doctors.get(node_name, {}).get(queue_name)
        if not queue_info:
            return 0
        return len(queue_info["available"])

    def _acquire_doctor(self, node_name: str, queue_name: str) -> Optional[Doctor]:
        queue_info = self.doctors.get(node_name, {}).get(queue_name)
        if not queue_info or len(queue_info["available"]) == 0:
            return None
        doctor = queue_info["available"].popleft()
        doctor.available = False
        return doctor

    def _release_doctor(self, node_name: str, queue_name: str, doctor: Optional[Doctor]) -> None:
        if doctor is None:
            return
        node_info = self.doctors.setdefault(node_name, {})
        queue_info = node_info.setdefault(queue_name, {"roster": [], "available": deque()})
        if doctor not in queue_info["roster"]:
            queue_info["roster"].append(doctor)
        if doctor not in queue_info["available"]:
            queue_info["available"].append(doctor)
        doctor.available = True

    def _remove_doctor_from_node(self, doctor: Doctor) -> None:
        node_info = self.doctors.get(doctor.current_node, {})
        queue_info = node_info.get(doctor.queue_name)
        if not queue_info:
            return
        if doctor in queue_info["available"]:
            queue_info["available"].remove(doctor)
        if doctor in queue_info["roster"]:
            queue_info["roster"].remove(doctor)

    def _get_adherence_state(self, patient: Patient) -> AdherenceState:
        adherence = getattr(patient, "adherence", None)
        if adherence is None:
            initial_prob = float(getattr(patient, "followup", random.random()))
            initial_prob = float(max(0.0, min(1.0, initial_prob)))
            adherence = AdherenceState(
                probability=initial_prob,
                reward=float(getattr(patient, "adherence_reward", 0.0)),
                base_probability=float(getattr(patient, "base_followup", initial_prob)),
                base_reward=float(getattr(patient, "base_adherence_reward", 0.0)),
            )
            patient.adherence = adherence
        return adherence

    def transfer_doctor(self, doctor_id: str, to_node: str, queue_name: str) -> None:
        doctor = self.doctor_registry.get(doctor_id)
        if doctor is None:
            raise ValueError(f"未找到编号为 {doctor_id} 的医生。")

        if to_node not in self.node_queues or queue_name not in self.node_queues[to_node]:
            raise ValueError(f"目标机构 {to_node}-{queue_name} 未配置诊室。")

        rooms = self.node_queues[to_node][queue_name]["rooms"]
        if rooms <= 0:
            raise ValueError(f"目标机构 {to_node}-{queue_name} 无法配置医生。")

        target_info = self.doctors.setdefault(to_node, {})
        queue_info = target_info.setdefault(queue_name, {"roster": [], "available": deque()})
        if len(queue_info["roster"]) >= rooms:
            raise ValueError(f"{to_node}-{queue_name} 的医生数量已达到诊室数量上限。")

        if not doctor.available:
            raise RuntimeError(f"医生 {doctor_id} 正在出诊，无法转移。")

        from_node = doctor.current_node
        self._remove_doctor_from_node(doctor)

        doctor.current_node = to_node
        doctor.queue_name = queue_name
        doctor.available = True
        doctor.transfers.append((from_node, to_node))
        doctor.history.append(to_node)

        queue_info["roster"].append(doctor)
        queue_info["available"].append(doctor)

        self.doctor_transfer_log.append(
            {"doctor_id": doctor_id, "from": from_node, "to": to_node}
        )
        self._busy_rates_stale = True

    def _record_patient_doctor(self, patient: Patient, doctor: Optional[Doctor]) -> None:
        if doctor is None or patient is None:
            return
        if not hasattr(patient, "doctor_history"):
            patient.doctor_history = []
        if not hasattr(patient, "continuity_reward"):
            patient.continuity_reward = 0.0
        if not hasattr(patient, "doctor_continuity_streak"):
            patient.doctor_continuity_streak = 0

        adherence_state = self._get_adherence_state(patient)

        previous_doctor = patient.doctor_history[-1] if patient.doctor_history else None
        patient.doctor_history.append(doctor.doctor_id)
        if previous_doctor == doctor.doctor_id:
            patient.doctor_continuity_streak += 1
        else:
            patient.doctor_continuity_streak = 1
        if patient.doctor_continuity_streak > 1:
            patient.continuity_reward += 1.0
            adherence_state.reward += 1.0
        patient.assigned_doctor = doctor.doctor_id
        patient.last_doctor_id = doctor.doctor_id

    def apply_treatment_effect(self, patient: Patient, doctor: Optional[Doctor]) -> None:
        if patient is None or doctor is None:
            return
        if not hasattr(patient, "severity_score_after") or patient.severity_score_after is None:
            patient.severity_score_after = float(patient.severity_score)
        base_improvement = doctor.skill_level * max(0.1, patient.severity_score / 10.0)
        patient.severity_score_after = float(
            max(0.0, patient.severity_score_after - base_improvement)
        )
        adherence_state = self._get_adherence_state(patient)
        adherence_state.set_probability(adherence_state.probability + doctor.adherence_boost)
        self._apply_adherence_outcome(patient, adherence_state, stochastic=True)
        patient.experience_score += doctor.skill_level * 0.5
        if patient.primary_doctor_id is None:
            patient.primary_doctor_id = doctor.doctor_id

    def _apply_adherence_outcome(
        self,
        patient: Patient,
        adherence_state: AdherenceState,
        *,
        stochastic: bool = True,
    ) -> None:
        """根据依从度在有限范围内调整治疗效果。"""
        if patient.severity_score_after is None:
            patient.severity_score_after = float(patient.severity_score)
        severity_after = float(patient.severity_score_after)
        adjustment = min(
            1.5,
            max(0.2, severity_after * self.adherence_effect_magnitude),
        )
        adherence_level = float(max(0.0, min(1.0, adherence_state.probability)))
        improve_prob = (
            self.adherence_improve_floor
            + (self.adherence_improve_ceiling - self.adherence_improve_floor) * adherence_level
        )
        improve_prob = float(max(0.0, min(1.0, improve_prob)))
        if stochastic:
            if random.random() < improve_prob:
                severity_after = max(0.0, severity_after - adjustment)
            else:
                severity_after = min(10.0, severity_after + adjustment)
        else:
            expected_delta = adjustment * (1.0 - 2.0 * improve_prob)
            severity_after = min(10.0, max(0.0, severity_after + expected_delta))
        patient.severity_score_after = severity_after

    def snapshot_state(self) -> Dict[str, object]:
        return {
            "all_patients": copy.deepcopy(self.all_patients),
            "doctors": copy.deepcopy(self.doctors),
            "doctor_registry": copy.deepcopy(self.doctor_registry),
            "doctor_transfer_log": copy.deepcopy(self.doctor_transfer_log),
            "busy_rates": copy.deepcopy(self.busy_rates),
            "busy_rates_stale": self._busy_rates_stale,
        }

    def restore_state(self, state: Dict[str, object]) -> None:
        self.all_patients = state["all_patients"]
        self.doctors = state["doctors"]
        self.doctor_registry = state["doctor_registry"]
        self.doctor_transfer_log = state["doctor_transfer_log"]
        self.busy_rates = state["busy_rates"]
        self._busy_rates_stale = state.get("busy_rates_stale", False)

    def _determine_revisit_referral(self, patient: Patient) -> Tuple[str, str]:
        severity = max(1, int(round(getattr(patient, "severity_score_after", patient.severity_score))))
        if severity >= 6:
            return self.hospital_name, "急诊"
        if patient.referral_node in {self.hospital_name, self.referral_name}:
            return self.hospital_name, "门诊"
        return patient.referral_node, "门诊"

    def _create_revisit_patient(self, patient: Patient, arrival_time: int) -> Patient:
        severity = max(1, int(round(getattr(patient, "severity_score_after", patient.severity_score))))
        adherence_state = self._get_adherence_state(patient)
        referral_node, target_queue = self._determine_revisit_referral(patient)
        new_visit_index = patient.visit_index + 1
        new_id = f"{patient.base_id}_rev{new_visit_index}"
        service_time = self._get_service_time(severity)
        new_patient = Patient(
            pid=new_id,
            severity_score=severity,
            arrival_time=arrival_time,
            current_node=None,
            referral_node=referral_node,
            target_queue=target_queue,
            service_time=service_time,
            remaining_service=service_time,
            followup=adherence_state.probability,
        )
        new_patient.base_id = patient.base_id
        new_patient.visit_index = new_visit_index
        new_patient.cost_map = dict(patient.cost_map)
        new_patient.primary_doctor_id = patient.primary_doctor_id
        new_patient.experience_score = patient.experience_score
        new_patient.doctor_history = list(getattr(patient, "doctor_history", []))
        new_patient.doctor_continuity_streak = getattr(patient, "doctor_continuity_streak", 0)
        new_patient.continuity_reward = 0.0
        new_patient.base_continuity_reward = getattr(patient, "base_continuity_reward", 0.0)

        new_adherence_state = self._get_adherence_state(new_patient)
        new_adherence_state.reward = 0.0
        new_adherence_state.base_reward = float(adherence_state.base_reward)
        new_adherence_state.base_probability = float(adherence_state.base_probability)
        return new_patient

    def _schedule_revisit(self, patient: Patient, current_time: int) -> None:
        severity_after = getattr(patient, "severity_score_after", patient.severity_score)
        if severity_after <= 0:
            return
        self._get_adherence_state(patient)
        delay = max(1, int(np.ceil(np.random.exponential(self.revisit_delay_scale))))
        arrival_time = current_time + delay
        cutoff = getattr(self, "total_simulation_steps", 0)
        if cutoff and arrival_time >= cutoff:
            return
        new_patient = self._create_revisit_patient(patient, arrival_time)
        self.pending_revisits[arrival_time].append(new_patient)
        self.all_patients.append(new_patient)
        patient.visit_index += 1

    def cache_patient_baselines(self) -> None:
        for patient in self.all_patients:
            adherence_state = self._get_adherence_state(patient)
            patient.base_continuity_reward = getattr(patient, "continuity_reward", 0.0)
            adherence_state.base_reward = adherence_state.reward
            patient.base_severity_after = getattr(
                patient,
                "severity_score_after",
                float(patient.severity_score),
            )
            adherence_state.base_probability = adherence_state.probability
            patient.base_experience_score = getattr(patient, "experience_score", 0.0)
            if patient.start_service_time is not None:
                patient.base_waiting_time = max(0.0, patient.start_service_time - patient.arrival_time)
            else:
                patient.base_waiting_time = None

            base_doctor_id: Optional[str] = None
            if patient.doctor_history:
                base_doctor_id = patient.doctor_history[-1]
            elif patient.primary_doctor_id:
                base_doctor_id = patient.primary_doctor_id
            patient.base_doctor_id = base_doctor_id

            if base_doctor_id and base_doctor_id in self.doctor_registry:
                base_doctor = self.doctor_registry[base_doctor_id]
                patient.base_doctor_skill = base_doctor.skill_level
                patient.base_doctor_speed = base_doctor.service_speed
            else:
                patient.base_doctor_skill = None
                patient.base_doctor_speed = None

    def _select_doctor_for_patient(self, patient: Patient, node_name: Optional[str]) -> Optional[Doctor]:
        if not node_name:
            return None
        queue_map = self.doctors.get(node_name, {})
        preferred_queue = None
        if patient.target_queue in queue_map:
            preferred_queue = patient.target_queue
        elif queue_map:
            preferred_queue = next(iter(queue_map))
        if preferred_queue is None:
            return None
        doctor_info = queue_map.get(preferred_queue)
        if not doctor_info:
            return None
        roster: List[Doctor] = doctor_info.get("roster", [])
        if not roster:
            return None
        return max(roster, key=lambda doc: doc.skill_level)

    def update_patient_metrics_after_moves(self) -> None:
        for patient in self.all_patients:
            adherence_state = self._get_adherence_state(patient)
            base_cont = getattr(patient, "base_continuity_reward", 0.0)
            base_reward = getattr(adherence_state, "base_reward", adherence_state.reward)
            base_follow = getattr(adherence_state, "base_probability", adherence_state.probability)
            base_exp = getattr(patient, "base_experience_score", getattr(patient, "experience_score", 0.0))
            base_wait = getattr(patient, "base_waiting_time", None)
            base_severity_after = getattr(
                patient,
                "base_severity_after",
                getattr(patient, "severity_score_after", float(patient.severity_score)),
            )
            base_doc_speed = getattr(patient, "base_doctor_speed", None)

            # continuity & adherence depend on primary doctor alignment
            patient.continuity_reward = 0.0
            adherence_state.reward = 0.0
            primary_id = getattr(patient, "primary_doctor_id", None)

            final_node: Optional[str] = patient.current_node
            if not final_node:
                if patient.travel_history:
                    final_node = patient.travel_history[-1][1]
                else:
                    final_node = patient.referral_node

            if primary_id and primary_id in self.doctor_registry:
                primary_doctor = self.doctor_registry[primary_id]
                if primary_doctor.current_node == final_node:
                    patient.continuity_reward = base_cont
                    adherence_state.reward = base_reward

            # treatment effect & waiting time adjustments based on representative doctor at new node
            repr_doctor = self._select_doctor_for_patient(patient, final_node)
            if repr_doctor is None and primary_id and primary_id in self.doctor_registry:
                repr_doctor = self.doctor_registry[primary_id]

            adherence_state.set_probability(base_follow)
            patient.experience_score = float(base_exp)
            patient.severity_score_after = float(base_severity_after)

            new_speed = getattr(repr_doctor, "service_speed", None)
            if repr_doctor is not None:
                raw_severity = float(patient.severity_score)
                improvement = repr_doctor.skill_level * max(0.1, raw_severity / 10.0)
                patient.severity_score_after = float(max(0.0, raw_severity - improvement))
                adherence_state.set_probability(base_follow + repr_doctor.adherence_boost)
                patient.experience_score = float(base_exp + repr_doctor.skill_level * 0.5)
                self._apply_adherence_outcome(
                    patient,
                    adherence_state,
                    stochastic=False,
                )
            else:
                new_speed = base_doc_speed
                patient.severity_score_after = float(base_severity_after)
                adherence_state.set_probability(base_follow)
                patient.experience_score = float(base_exp)

            if base_wait is not None:
                waiting_time = float(base_wait)
                if base_doc_speed and new_speed:
                    waiting_time = waiting_time * (base_doc_speed / new_speed)
                waiting_time = max(0.0, waiting_time)
                patient.start_service_time = patient.arrival_time + waiting_time

    # ------------------------------------------------------------------ #
    # 运营成本估算
    # ------------------------------------------------------------------ #
    def _infer_patient_service_point(self, patient: Patient) -> Tuple[Optional[str], Optional[str]]:
        final_node: Optional[str] = patient.current_node
        if not final_node:
            if patient.travel_history:
                final_node = patient.travel_history[-1][1]
            else:
                final_node = patient.referral_node
        if final_node is None:
            return None, None

        queues = self.node_queues.get(final_node)
        if not queues:
            return final_node, None

        queue_name = getattr(patient, "target_queue", None)
        if queue_name not in queues:
            severity_after = getattr(patient, "severity_score_after", None)
            severity_score = (
                severity_after if severity_after is not None else getattr(patient, "severity_score", 5)
            )
            if final_node == self.hospital_name:
                if severity_score >= 6 and "急诊" in queues:
                    queue_name = "急诊"
                elif "门诊" in queues:
                    queue_name = "门诊"
                else:
                    queue_name = next(iter(queues.keys()), None)
            else:
                queue_name = "门诊" if "门诊" in queues else next(iter(queues.keys()), None)
        return final_node, queue_name

    def refresh_busy_rates_from_state(self) -> None:
        busy_rates = self._initialize_busy_rates()
        patient_load: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for patient in self.all_patients:
            node_name, queue_name = self._infer_patient_service_point(patient)
            if node_name is None or queue_name is None:
                continue
            patient_load[node_name][queue_name] += 1

        for node_name, queues in self.node_queues.items():
            node_load = patient_load.get(node_name, {})
            doctor_queues = self.doctors.get(node_name, {})
            for queue_name in queues.keys():
                entry = busy_rates[node_name][queue_name]
                doc_info = doctor_queues.get(queue_name)
                total_doctors = len(doc_info.get("roster", [])) if doc_info else 0
                load = node_load.get(queue_name, 0)
                entry["total_doctors"] = float(total_doctors)
                if total_doctors > 0:
                    entry["observations"] = 1.0
                    busy_doctors = float(min(load, total_doctors)) if load > 0 else 0.0
                else:
                    entry["observations"] = 0.0
                    busy_doctors = 0.0
                entry["busy_doctors"] = busy_doctors
                entry["queued_patients"] = float(max(0, load - total_doctors))
                entry["queued_samples"] = 1.0 if entry["queued_patients"] > 0 else 0.0

        self.busy_rates = busy_rates
        self._busy_rates_stale = False

    def ensure_busy_rates_ready(self) -> None:
        if self._busy_rates_stale:
            self.refresh_busy_rates_from_state()

    # ------------------------------------------------------------------ #
    # 地理与矩阵
    # ------------------------------------------------------------------ #
    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return 6371.0 * c

    def _initialize_distance_matrix(self) -> np.ndarray:
        if not self.location_path.exists():
            raise FileNotFoundError(f"地点信息文件不存在: {self.location_path}")

        df = pd.read_excel(self.location_path, sheet_name=0)
        n = len(self.all_nodes)
        dist_matrix = np.zeros((n, n), dtype=float)
        new_continuity_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                if i == j:
                    dist_matrix[i, j] = 0.0
                    new_continuity_matrix[i, j] = 1.0
                elif i != j and (i == 0 or j == 0):
                    dist_matrix[i, j] = self.haversine(
                        df.iloc[i, 6], df.iloc[i, 5], df.iloc[j, 6], df.iloc[j, 5]
                    )
                    new_continuity_matrix[i, j] = self.contiunity_matrix[i, j]
                elif i != j:
                    dist_matrix[i, j] = self.haversine(
                        df.iloc[i, 6], df.iloc[i, 5], df.iloc[0, 6], df.iloc[0, 5]
                    ) + self.haversine(df.iloc[0, 6], df.iloc[0, 5], df.iloc[j, 6], df.iloc[j, 5])
                    new_continuity_matrix[i, j] = (
                        self.contiunity_matrix[i, 0] * self.contiunity_matrix[0, j]
                    )
        self.contiunity_matrix = new_continuity_matrix
        return dist_matrix

    # ------------------------------------------------------------------ #
    # 患者数据加载
    # ------------------------------------------------------------------ #
    def _load_patient_dataset(self) -> pd.DataFrame:
        if not self.patient_dataset_path.exists():
            raise FileNotFoundError(f"患者数据集文件不存在: {self.patient_dataset_path}")
        suffix = self.patient_dataset_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self.patient_dataset_path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(self.patient_dataset_path)
        raise ValueError(f"不支持的患者数据集文件格式: {suffix}")

    def _refresh_nodes_from_dataset(self, dataset: pd.DataFrame) -> None:
        community_nodes = sorted(
            {
                node
                for node in dataset["referral_node"].unique()
                if node not in {self.hospital_name, self.referral_name}
            }
        )
        if community_nodes:
            self.community_names = np.array(community_nodes, dtype="<U32")
        else:
            self.community_names = np.array([], dtype="<U1")
        self.all_nodes = [self.hospital_name, self.referral_name] + list(self.community_names)
        self.contiunity_matrix = np.random.rand(len(self.all_nodes), len(self.all_nodes))
        self.node_index_map = {name: idx for idx, name in enumerate(self.all_nodes)}
        self.index_node_map = {idx: name for idx, name in enumerate(self.all_nodes)}
        self.distance_matrix = self._initialize_distance_matrix()
        self.node_queues = self._initialize_queues()
        self.doctors = self._initialize_doctors()
        self.busy_rates = self._initialize_busy_rates()
        self._busy_rates_stale = True
        self.doctor_transfer_log = []
        self.pending_revisits.clear()

    # ------------------------------------------------------------------ #
    # 患者到达与模拟
    # ------------------------------------------------------------------ #
    def create_arrival_schedule(
        self,
        total_light: int,
        total_severe: int,
        total_steps: int,
        light_hospital_prob: float = 0.5,
        worse_prob: float = 0.1,
        better_prob: float = 0.3,
    ) -> Dict[int, List[Patient]]:
        dataset = self._load_patient_dataset()

        required_columns = {
            "patient_id",
            "arrival_time",
            "referral_node",
            "target_queue",
            "severity_score",
        }
        missing_columns = required_columns - set(dataset.columns)
        if missing_columns:
            raise ValueError(f"患者数据集中缺少必要字段: {missing_columns}")

        dataset = dataset.sort_values("arrival_time").reset_index(drop=True)
        self._refresh_nodes_from_dataset(dataset)

        def _is_missing(value):
            if value is None:
                return True
            try:
                return pd.isna(value)
            except TypeError:
                return False

        def _parse_json_field(value, default):
            if isinstance(value, (list, dict)):
                return value
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return default
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return default
            if _is_missing(value):
                return default
            return default

        schedule: Dict[int, List[Patient]] = defaultdict(list)
        patients: List[Patient] = []

        for row in dataset.itertuples(index=False):
            arrival_time = int(getattr(row, "arrival_time"))
            severity_score = int(getattr(row, "severity_score"))

            service_time_value = getattr(row, "service_time", np.nan)
            service_time = (
                service_time_value
                if not _is_missing(service_time_value)
                else self._get_service_time(severity_score)
            )
            try:
                service_time = float(service_time)
            except (TypeError, ValueError):
                service_time = float(self._get_service_time(severity_score))

            remaining_service_value = getattr(row, "remaining_service", service_time)
            remaining_service = (
                remaining_service_value
                if not _is_missing(remaining_service_value)
                else service_time
            )
            try:
                remaining_service = float(remaining_service)
            except (TypeError, ValueError):
                remaining_service = service_time

            followup_raw = getattr(row, "followup", None)
            if _is_missing(followup_raw):
                followup_value = random.random()
            else:
                try:
                    followup_value = float(followup_raw)
                except (TypeError, ValueError):
                    followup_value = random.random()

            patient = Patient(
                pid=str(getattr(row, "patient_id")),
                severity_score=severity_score,
                arrival_time=arrival_time,
                current_node=None,
                referral_node=getattr(row, "referral_node"),
                target_queue=getattr(row, "target_queue"),
                service_time=service_time,
                remaining_service=remaining_service,
                followup=followup_value,
            )
            patient.assigned_doctor = None
            patient.doctor_history = []
            patient.continuity_reward = 0.0
            adherence_state = self._get_adherence_state(patient)
            adherence_reward_value = getattr(row, "adherence_reward", None)
            if _is_missing(adherence_reward_value):
                adherence_state.reward = 0.0
            else:
                try:
                    adherence_state.reward = float(adherence_reward_value)
                except (TypeError, ValueError):
                    adherence_state.reward = 0.0
            base_followup_value = getattr(row, "base_followup", None)
            if _is_missing(base_followup_value):
                adherence_state.base_probability = adherence_state.probability
            else:
                try:
                    adherence_state.base_probability = float(base_followup_value)
                except (TypeError, ValueError):
                    adherence_state.base_probability = adherence_state.probability
            base_adherence_reward_value = getattr(row, "base_adherence_reward", None)
            if _is_missing(base_adherence_reward_value):
                adherence_state.base_reward = adherence_state.reward
            else:
                try:
                    adherence_state.base_reward = float(base_adherence_reward_value)
                except (TypeError, ValueError):
                    adherence_state.base_reward = adherence_state.reward
            patient.doctor_continuity_streak = 0
            patient.last_doctor_id = None
            patient.base_id = patient.id.split("_rev")[0]
            patient.visit_index = 0

            severity_after = getattr(row, "severity_score_after", None)
            if not _is_missing(severity_after):
                patient.severity_score_after = int(severity_after)

            current_node_value = getattr(row, "current_node", None)
            if not _is_missing(current_node_value):
                patient.current_node = str(current_node_value)

            start_service_value = getattr(row, "start_service_time", None)
            if not _is_missing(start_service_value):
                try:
                    patient.start_service_time = float(start_service_value)
                except (TypeError, ValueError):
                    patient.start_service_time = None

            travel_history_value = getattr(row, "travel_history", None)
            travel_history_data = _parse_json_field(travel_history_value, [])
            travel_history: List[tuple] = []
            if isinstance(travel_history_data, list):
                for item in travel_history_data:
                    if isinstance(item, (list, tuple)) and len(item) >= 4:
                        try:
                            from_node, to_node, distance, continuity = item[:4]
                            travel_history.append(
                                (
                                    str(from_node),
                                    str(to_node),
                                    float(distance),
                                    float(continuity),
                                )
                            )
                        except (TypeError, ValueError):
                            continue
            patient.travel_history = travel_history

            cost_map_value = getattr(row, "cost_map", None)
            cost_map_data = _parse_json_field(cost_map_value, {})
            if isinstance(cost_map_data, dict):
                parsed_cost_map = {}
                for key, val in cost_map_data.items():
                    if _is_missing(val):
                        continue
                    try:
                        parsed_cost_map[str(key)] = float(val)
                    except (TypeError, ValueError):
                        continue
                patient.cost_map = parsed_cost_map

            patients.append(patient)
            schedule[arrival_time].append(patient)

        actual_light_count = int((dataset["severity_score"] <= 5).sum())
        actual_severe_count = int((dataset["severity_score"] > 5).sum())
        stats = {
            "actual_light_count": actual_light_count,
            "actual_severe_count": actual_severe_count,
        }

        self.all_patients = patients
        self.patient_generation_stats = stats
        print(f"实际轻症患者总人次: {actual_light_count}")
        print(f"实际重症患者总人次: {actual_severe_count}")
        return schedule

    def simulate_one_step(self, t: int, schedule: Dict[int, List[Patient]]) -> None:
        for node_name, queues in self.node_queues.items():
            for q_name, q_data in queues.items():
                ongoing = q_data["ongoing"]
                queue_size = len(q_data["queue"])
                doctor_info = self.doctors.get(node_name, {}).get(q_name)
                total_doctors = len(doctor_info["roster"]) if doctor_info else 0
                available_doctors = len(doctor_info["available"]) if doctor_info else 0
                busy_entry = self.busy_rates.setdefault(node_name, {}).setdefault(
                    q_name,
                    {
                        "busy_doctors": 0,
                        "total_doctors": 0,
                        "observations": 0,
                        "queued_patients": 0,
                        "queued_samples": 0,
                    },
                )
                busy_entry["observations"] += 1
                busy_entry["busy_doctors"] += len(ongoing)
                busy_entry["total_doctors"] += total_doctors
                busy_entry["queued_patients"] += max(0, queue_size - available_doctors)
                busy_entry["queued_samples"] += 1

        arriving_patients = schedule.get(t, [])
        for patient in arriving_patients:
            self.node_queues[patient.referral_node][patient.target_queue]["queue"].append(
                patient
            )
            patient.current_node = patient.referral_node

        for node_name, queues in self.node_queues.items():
            for q_name, q_data in queues.items():
                ongoing = q_data["ongoing"]
                queue_ = q_data["queue"]

                finished_assignments: List[Tuple[str, Patient, Optional[Doctor]]] = []
                for slot_id, info in list(ongoing.items()):
                    if len(info) == 3:
                        patient, remain, doctor = info
                    else:
                        patient, remain = info
                        doctor = None
                    new_remain = remain - 1.0
                    if new_remain <= 0:
                        finished_assignments.append((slot_id, patient, doctor))
                        patient.remaining_service = 0.0
                    else:
                        patient.remaining_service = new_remain
                        ongoing[slot_id] = (patient, new_remain, doctor)

                for slot_id, patient, doctor in finished_assignments:
                    del ongoing[slot_id]
                    self.apply_treatment_effect(patient, doctor)
                    self._schedule_revisit(patient, t)
                    if hasattr(patient, "assigned_doctor"):
                        patient.assigned_doctor = None
                    if doctor is not None:
                        self._release_doctor(node_name, q_name, doctor)

                while queue_:
                    doctor = self._acquire_doctor(node_name, q_name)
                    if doctor is None:
                        break
                    next_patient = queue_.popleft()
                    if next_patient.start_service_time is None:
                        next_patient.start_service_time = t
                    next_patient.current_node = node_name
                    service_time = (
                        next_patient.remaining_service
                        if next_patient.remaining_service > 0
                        else next_patient.service_time
                    )
                    adjusted_service_time = max(0.5, service_time / max(0.1, doctor.service_speed))
                    next_patient.remaining_service = adjusted_service_time
                    slot_id = doctor.doctor_id
                    self._record_patient_doctor(next_patient, doctor)
                    ongoing[slot_id] = (next_patient, adjusted_service_time, doctor)
        self._busy_rates_stale = False

    def travel_patient(self, patient: Patient, from_node: str, to_node: str) -> None:
        from_idx = self.node_index_map[from_node]
        to_idx = self.node_index_map[to_node]
        dist = self.distance_matrix[from_idx, to_idx]
        continuity = self.contiunity_matrix[from_idx, to_idx]
        patient.travel_history.append((from_node, to_node, dist, continuity))
        patient.current_node = to_node

    def run_simulation(
        self,
        total_steps: int = 160,
        total_light: int = 200,
        total_severe: int = 50,
        light_hospital_prob: float = 0.5,
    ) -> None:
        self.node_queues = self._initialize_queues()
        self.doctors = self._initialize_doctors()
        self.busy_rates = self._initialize_busy_rates()
        self._busy_rates_stale = True
        self.doctor_transfer_log.clear()
        self.all_patients.clear()
        self.pending_revisits.clear()

        self.total_simulation_steps = total_steps

        schedule = self.create_arrival_schedule(
            total_light=total_light,
            total_severe=total_severe,
            total_steps=total_steps,
            light_hospital_prob=light_hospital_prob,
        )

        self.pending_revisits.clear()
        for current_step in range(total_steps):
            self.simulate_one_step(current_step, schedule)
            if self.pending_revisits:
                for arrival_time, patients in list(self.pending_revisits.items()):
                    if arrival_time < total_steps:
                        schedule[arrival_time].extend(patients)
                self.pending_revisits.clear()

        self.cache_patient_baselines()

        print("=== 模拟结束后，各节点队列 & 正在服务情况 ===")
        for node, queues in self.node_queues.items():
            print(f"\n节点: {node}")
            for q_name, q_data in queues.items():
                q_size = len(q_data["queue"])
                ongoing_size = len(q_data["ongoing"])
                print(f"  队列: {q_name}, 等待数: {q_size}, 正在服务数: {ongoing_size}")
        print(f"总患者人数: {len(self.all_patients)}")

    # ------------------------------------------------------------------ #
    # 染色体辅助函数
    # ------------------------------------------------------------------ #
    def transfer_patients_by_matrix(self, transition_matrix: np.ndarray) -> None:
        for patient in self.all_patients:
            patient.travel_history.clear()

        transition_matrix = self.validate_and_fix_matrix(transition_matrix)
        self.transfer_matrix = transition_matrix

        index = 0
        for row in transition_matrix:
            patient_id = str(row[0])
            from_node = row[1]
            to_node = row[2]

            if index >= len(self.all_patients):
                print(f"[警告] 找不到编号为 {patient_id} 的患者，跳过该条转移。")
                continue
            patient = self.all_patients[index]
            index += 1

            self.travel_patient(patient, from_node, to_node)
        self._busy_rates_stale = True

    def generate_transition_matrix(self) -> np.ndarray:
        transitions: List[List[object]] = []
        for patient in self.all_patients:
            from_node = patient.current_node if patient.current_node else patient.referral_node
            to_node = np.random.choice(self.all_nodes)
            transitions.append([patient.id, from_node, to_node])

        transition_matrix = np.array(transitions, dtype=object)
        transition_matrix = self.validate_and_fix_matrix(transition_matrix)
        return transition_matrix

    def validate_and_fix_matrix(self, transition_matrix: np.ndarray) -> np.ndarray:
        matrix_dict: Dict[str, Tuple[str, str]] = {}
        for row in transition_matrix:
            pid = str(row[0])
            matrix_dict[pid] = (row[1], row[2])

        fixed_matrix = transition_matrix.copy()
        for idx, row in enumerate(fixed_matrix):
            pid = str(row[0])
            from_node = row[1]
            if "_rev" not in pid:
                continue
            base_id = pid.split("_rev")[0]
            prev_pid = f"{base_id}_rev{int(pid.split('_rev')[1]) - 1}" if "_rev" in pid else base_id
            if prev_pid not in matrix_dict:
                continue
            prev_to_node = matrix_dict[prev_pid][1]
            if from_node != prev_to_node:
                fixed_matrix[idx, 1] = prev_to_node
        return fixed_matrix

    def _get_service_time(self, score: int) -> float:
        """根据 1~10 的病情分值来确定服务时长。"""
        return 6.0 if score <= 5 else 12.0


