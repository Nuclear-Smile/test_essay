from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import json
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_LIGHT_PATIENTS = 540
DEFAULT_SEVERE_PATIENTS = 100


@dataclass
class AdherenceState:
    """统一管理患者依从度状态。"""

    probability: float = 0.0
    reward: float = 0.0
    base_probability: float = 0.0
    base_reward: float = 0.0

    def clamp_probability(self) -> None:
        self.probability = float(max(0.0, min(1.0, self.probability)))

    def set_probability(self, value: float) -> None:
        self.probability = float(max(0.0, min(1.0, value)))


class Patient:
    """表示一个患者的基础信息及轨迹。"""

    def __init__(
        self,
        pid: str,
        severity_score: int,
        arrival_time: int,
        current_node: Optional[str],
        referral_node: str,
        target_queue: str,
        service_time: float,
        remaining_service: float,
        followup: Optional[float] = None,
    ) -> None:
        self.id = pid
        self.base_id = pid
        self.visit_index = 0
        self.severity_score = severity_score
        self.severity_score_after = severity_score
        self.arrival_time = arrival_time
        self.current_node = current_node
        self.referral_node = referral_node
        self.target_queue = target_queue
        self.service_time = service_time
        self.remaining_service = remaining_service
        self.cost_map: Dict[str, float] = {}
        initial_followup = float(np.random.rand() if followup is None else followup)
        initial_followup = float(max(0.0, min(1.0, initial_followup)))
        self.adherence = AdherenceState(
            probability=initial_followup,
            base_probability=initial_followup,
        )
        self.travel_history: List[Tuple[str, str, float, float]] = []
        self.start_service_time: Optional[int] = None
        self.experience_score: float = 0.0
        self.primary_doctor_id: Optional[str] = None
        self.doctor_history: List[str] = []
        self.doctor_continuity_streak: int = 0
        self.continuity_reward: float = 0.0
        self.base_continuity_reward: float = 0.0
        self.assigned_doctor: Optional[str] = None

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            "Patient("
            f"id={self.id}, "
            f"severity_score={self.severity_score}, "
            f"arrival_time={self.arrival_time}, "
            f"current_node={self.current_node}, "
            f"referral_node={self.referral_node}, "
            f"target_queue={self.target_queue}, "
            f"service_time={self.service_time}, "
            f"remaining_service={self.remaining_service}, "
            f"travel_history={self.travel_history}"
            ")"
        )


def default_service_time(score: int) -> float:
    """根据病情分值返回默认服务时长。"""

    return 6.0 if score <= 5 else 12.0


def _choose_referral_node(
    hospital_name: str,
    community_names: Sequence[str],
    light_hospital_prob: float,
) -> Tuple[str, str]:
    if np.random.rand() < light_hospital_prob:
        return hospital_name, "门诊"
    return np.random.choice(community_names), "门诊"


def _create_patient(
    pid: str,
    severity_score: int,
    arrival_time: int,
    referral_node: str,
    target_queue: str,
    service_time_fn: Callable[[int], float],
    followup: Optional[float] = None,
) -> Patient:
    service_time = service_time_fn(severity_score)
    return Patient(
        pid=pid,
        severity_score=severity_score,
        arrival_time=arrival_time,
        current_node=None,
        referral_node=referral_node,
        target_queue=target_queue,
        service_time=service_time,
        remaining_service=service_time,
        followup=followup,
    )


def _write_patient_dataset(
    dataset: pd.DataFrame,
    output_path: str,
    output_format: Optional[str] = None,
) -> Path:
    """将患者数据集写入磁盘并返回生成的路径。"""

    path = Path(output_path)
    fmt = (output_format or path.suffix.lstrip(".")).lower()

    if not fmt:
        fmt = "csv"
    if not path.suffix:
        path = path.with_suffix(f".{fmt}")

    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        dataset.to_csv(path, index=False)
    elif fmt in {"xlsx", "xls"}:
        dataset.to_excel(path, index=False)
    else:
        raise ValueError(f"不支持的输出格式: {fmt}")

    return path


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _parse_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_field(value: object, default: object) -> object:
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


def _parse_optional_str(value: object) -> Optional[str]:
    if _is_missing(value):
        return None
    return str(value)


def _coerce_travel_history(value: object) -> List[Tuple[str, str, float, float]]:
    data = _parse_json_field(value, [])
    history: List[Tuple[str, str, float, float]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                from_node, to_node, distance, continuity = item[:4]
                dist_val = _parse_float(distance, 0.0) or 0.0
                cont_val = _parse_float(continuity, 0.0) or 0.0
                history.append((str(from_node), str(to_node), dist_val, cont_val))
    return history


def _coerce_cost_map(value: object) -> Dict[str, float]:
    data = _parse_json_field(value, {})
    result: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, val in data.items():
            parsed = _parse_float(val)
            if parsed is not None:
                result[str(key)] = parsed
    return result


def create_patient_schedule(
    total_light: int,
    total_severe: int,
    total_steps: int,
    hospital_name: str,
    community_names: Sequence[str],
    *,
    light_hospital_prob: float = 0.5,
    service_time_fn: Optional[Callable[[int], float]] = None,
    random_seed: Optional[int] = None,
) -> Tuple[Dict[int, List[Patient]], List[Patient], Dict[str, int]]:
    """生成患者到达队列（不包含任何复诊记录）。"""

    if service_time_fn is None:
        service_time_fn = default_service_time

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    total_light = int(total_light)
    total_severe = int(total_severe)

    schedule: Dict[int, List[Patient]] = defaultdict(list)
    all_patients: List[Patient] = []

    light_arrivals = np.sort(np.random.randint(0, total_steps, size=total_light)) if total_light > 0 else np.array([])
    severe_arrivals = np.sort(np.random.randint(0, total_steps, size=total_severe)) if total_severe > 0 else np.array([])

    patient_counter = 1

    for arrival_time in light_arrivals:
        pid = f"P{patient_counter:04d}"
        patient_counter += 1
        severity_score = np.random.randint(1, 6)
        referral_node, target_queue = _choose_referral_node(
            hospital_name, community_names, light_hospital_prob
        )
        patient = _create_patient(
            pid=pid,
            severity_score=severity_score,
            arrival_time=int(arrival_time),
            referral_node=referral_node,
            target_queue=target_queue,
            service_time_fn=service_time_fn,
        )
        patient.severity_score_after = max(1, severity_score - np.random.randint(1, 3))
        schedule[patient.arrival_time].append(patient)
        all_patients.append(patient)

    for arrival_time in severe_arrivals:
        pid = f"P{patient_counter:04d}"
        patient_counter += 1
        severity_score = np.random.randint(6, 11)
        patient = _create_patient(
            pid=pid,
            severity_score=severity_score,
            arrival_time=int(arrival_time),
            referral_node=hospital_name,
            target_queue="急诊",
            service_time_fn=service_time_fn,
        )
        patient.severity_score_after = max(1, severity_score - np.random.randint(1, 4))
        schedule[patient.arrival_time].append(patient)
        all_patients.append(patient)

    stats = {
        "actual_light_count": total_light,
        "actual_severe_count": total_severe,
    }
    return schedule, all_patients, stats


def _build_node_metadata(
    location_df: pd.DataFrame,
    all_nodes: Sequence[str],
) -> Dict[str, Dict[str, Optional[str]]]:
    metadata: Dict[str, Dict[str, Optional[str]]] = {}
    usable_rows = min(len(location_df), len(all_nodes))

    for idx in range(usable_rows):
        row = location_df.iloc[idx]
        metadata[all_nodes[idx]] = {
            "address": row.iloc[1] if location_df.shape[1] > 1 else None,
            "longitude": row.iloc[5] if location_df.shape[1] > 5 else None,
            "latitude": row.iloc[6] if location_df.shape[1] > 6 else None,
        }
    return metadata


def patients_to_dataframe(
    patients: Iterable[Patient],
    node_metadata: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for patient in patients:
        revisit_count = patient.visit_index
        record: Dict[str, object] = {
            "patient_id": patient.id,
            "arrival_time": patient.arrival_time,
            "referral_node": patient.referral_node,
            "target_queue": patient.target_queue,
            "severity_score": patient.severity_score,
            "severity_score_after": patient.severity_score_after,
            "followup": patient.adherence.probability,
            "revisit_count": revisit_count,
            "service_time": patient.service_time,
            "remaining_service": patient.remaining_service,
            "current_node": patient.current_node,
            "start_service_time": patient.start_service_time,
            "travel_history": json.dumps(patient.travel_history, ensure_ascii=False),
            "cost_map": json.dumps(patient.cost_map, ensure_ascii=False),
        }

        if node_metadata and patient.referral_node in node_metadata:
            meta = node_metadata[patient.referral_node]
            record["referral_address"] = meta.get("address")
            record["referral_longitude"] = meta.get("longitude")
            record["referral_latitude"] = meta.get("latitude")

        records.append(record)

    return pd.DataFrame(records)


def generate_patient_dataset(
    location_path: str,
    total_light: int,
    total_severe: int,
    total_steps: int,
    *,
    hospital_name: str = "综合医院",
    referral_name: str = "转诊中心",
    community_names: Optional[Sequence[str]] = None,
    light_hospital_prob: float = 0.5,
    random_seed: Optional[int] = None,
    output_path: Optional[str] = None,
    output_format: Optional[str] = None,
) -> Tuple[Dict[int, List[Patient]], List[Patient], pd.DataFrame]:
    """根据地理位置表生成患者数据集。"""

    location_df = pd.read_excel(location_path)

    if community_names is None:
        community_count = max(len(location_df) - 2, 0)
        community_names = [f"社康机构_{idx + 1}" for idx in range(community_count)]

    all_nodes = [hospital_name, referral_name] + list(community_names)
    if len(location_df) < len(all_nodes):
        raise ValueError("location.xlsx 行数不足以匹配全部节点。")

    schedule, patients, stats = create_patient_schedule(
        total_light=total_light,
        total_severe=total_severe,
        total_steps=total_steps,
        hospital_name=hospital_name,
        community_names=community_names,
        light_hospital_prob=light_hospital_prob,
        service_time_fn=default_service_time,
        random_seed=random_seed,
    )

    metadata = _build_node_metadata(location_df, all_nodes)
    dataset = patients_to_dataframe(patients, metadata)

    dataset.attrs["stats"] = stats
    dataset.attrs["all_nodes"] = all_nodes

    if output_path:
        saved_path = _write_patient_dataset(dataset, output_path, output_format)
        dataset.attrs["file_path"] = str(saved_path)

    return schedule, patients, dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成患者数据集文件")
    parser.add_argument("--location", default="location.xlsx", help="地点信息表路径")
    parser.add_argument("--light", type=int, default=DEFAULT_LIGHT_PATIENTS, help="轻症患者数量")
    parser.add_argument("--severe", type=int, default=DEFAULT_SEVERE_PATIENTS, help="重症患者数量")
    parser.add_argument("--steps", type=int, default=160, help="模拟总步数")
    parser.add_argument("--output", default="healthcare_app/data/patient_dataset.csv", help="输出文件路径")
    parser.add_argument("--format", default=None, help="输出文件格式(csv/xlsx)，默认根据后缀判断")
    parser.add_argument("--seed", type=int, default=21, help="随机数种子")

    args = parser.parse_args()

    _, _, dataset = generate_patient_dataset(
        location_path=args.location,
        total_light=args.light,
        total_severe=args.severe,
        total_steps=args.steps,
        random_seed=args.seed,
        output_path=args.output,
        output_format=args.format,
    )

    file_path = dataset.attrs.get("file_path", args.output)
    print(f"患者数据集已生成，共 {len(dataset)} 条记录，保存至: {file_path}")
