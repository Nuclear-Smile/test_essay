from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from models import HealthcareSystem


class ObjectiveEvaluator:
    """
    负责对 `HealthcareSystem` 运行后的数据进行统计并输出多目标优化所需指标。
    该实现直接复刻自 `原件 copy.ipynb`，并将过程中的日志保存在实例属性中。
    """

    def __init__(self, healthcare_system: HealthcareSystem) -> None:
        self.system = healthcare_system
        self.mid_records: List[List[float]] = []
        self.profit_records: List[Dict[str, float]] = []
        self.generate_random_treatment_costs()

    # ------------------------------------------------------------------ #
    # 随机成本初始化
    # ------------------------------------------------------------------ #
    def generate_random_treatment_costs(self, min_cost: int = 100, max_cost: int = 500) -> None:
        for patient in self.system.all_patients:
            patient.cost_map = {}
            for node in self.system.all_nodes:
                patient.cost_map[node] = float(random.randint(min_cost, max_cost))

    # ------------------------------------------------------------------ #
    # 基础统计
    # ------------------------------------------------------------------ #
    def compute_total_waiting_time(self) -> float:
        total_waiting = 0.0
        for patient in self.system.all_patients:
            if patient.start_service_time is None:
                continue
            wait = patient.start_service_time - patient.arrival_time
            if wait > 0:
                total_waiting += wait
        return total_waiting

    def count_visits(self) -> Dict[str, int]:
        visits: Dict[str, int] = defaultdict(int)
        for patient in self.system.all_patients:
            visits[patient.referral_node] += 1
            for travel in patient.travel_history:
                visits[travel[1]] += 1
        return visits

    def count_referrals(self) -> Dict[str, int]:
        referrals: Dict[str, int] = defaultdict(int)
        for patient in self.system.all_patients:
            for travel in patient.travel_history:
                from_node, to_node = travel[0], travel[1]
                if from_node != to_node:
                    referrals[from_node] += 1
        return referrals

    def count_patients_visited_hospital(self) -> int:
        hospital_name = self.system.hospital_name
        count = 0
        for patient in self.system.all_patients:
            visited_nodes = {patient.referral_node}
            for travel in patient.travel_history:
                visited_nodes.add(travel[1])
            if hospital_name in visited_nodes:
                count += 1
        return count

    def count_patients_visited_community_centers(self) -> int:
        community_names = set(self.system.community_names)
        count = 0
        for patient in self.system.all_patients:
            visited_nodes = {patient.referral_node}
            for travel in patient.travel_history:
                visited_nodes.add(travel[1])
            if visited_nodes & community_names:
                count += 1
        return count

    def count_patients_transferred(self) -> int:
        count = 0
        for patient in self.system.all_patients:
            has_transfer = False
            for travel in patient.travel_history:
                if travel[0] != travel[1]:
                    has_transfer = True
                    break
            if has_transfer:
                count += 1
        return count

    def total_transfer_events(self) -> int:
        total = 0
        for patient in self.system.all_patients:
            total += sum(1 for from_node, to_node, _, _ in patient.travel_history if from_node != to_node)
        return total

    def compute_doctor_rewards(self) -> Tuple[float, float]:
        continuity = 0.0
        adherence = 0.0
        for patient in self.system.all_patients:
            continuity += getattr(patient, "continuity_reward", 0.0)
            adherence_state = self.system._get_adherence_state(patient)
            adherence += adherence_state.reward
        return continuity, adherence

    # ------------------------------------------------------------------ #
    # 成本与收益
    # ------------------------------------------------------------------ #
    def TC_doctor_transfer(self) -> float:
        return self.system.doctor_transfer_unit_cost * len(self.system.doctor_transfer_log)

    def hosp_treatment_R(self) -> float:
        return self.count_patients_visited_hospital() * 1000.0

    def hosp_service_R(self) -> float:
        visits = self.count_patients_visited_hospital()
        return visits * (150.0 + 200.0) + visits * 0.1 * 500.0

    def hosp_referral_R(self) -> float:
        return self.count_patients_transferred() * (30.0 + 5.0)

    def commu_treatment_R(self) -> float:
        return self.count_patients_visited_community_centers() * 800.0

    def commu_referral_R(self) -> float:
        return self.count_patients_transferred() * (15.0 + 2.0)

    def total_revenue(self) -> float:
        return (
            self.hosp_treatment_R()
            + self.hosp_service_R()
            + self.hosp_referral_R()
            + self.commu_treatment_R()
            + self.commu_referral_R()
        )

    def TC_hosp_operation(self) -> float:
        self.system.ensure_busy_rates_ready()
        total_operation_cost = 0.0
        resource_config = {
            "outpatient_rooms": {"rho_busy_cost": 500, "rho_idle_cost": 100},
            "emergency_rooms": {"rho_busy_cost": 750, "rho_idle_cost": 260},
            "community_rooms": {"rho_busy_cost": 200, "rho_idle_cost": 80},
        }
        for node_name, queue_rates in self.system.busy_rates.items():
            if node_name == self.system.referral_name:
                continue
            for q_name, metrics in queue_rates.items():
                observations = metrics.get("observations", 0)
                total_doctors = metrics.get("total_doctors", 0)
                if observations == 0 or total_doctors == 0:
                    continue
                if node_name == self.system.hospital_name:
                    res_type = "emergency_rooms" if q_name == "急诊" else "outpatient_rooms"
                else:
                    res_type = "community_rooms"
                costs = resource_config[res_type]
                busy_doctors = metrics.get("busy_doctors", 0)
                rho = busy_doctors / total_doctors if total_doctors else 0
                avg_doctors = total_doctors / observations
                busy_cost = costs["rho_busy_cost"]
                idle_cost = costs["rho_idle_cost"]
                total_operation_cost += avg_doctors * (rho * busy_cost + (1 - rho) * idle_cost)
        return total_operation_cost

    def TC_hosp_referral(self, alpha_distance_cost: float, beta_admin_cost: float) -> float:
        total_referral_cost = 0.0
        for patient in self.system.all_patients:
            for transfer in patient.travel_history:
                from_node, _, _, _ = transfer
                if from_node == self.system.hospital_name:
                    distance_cost = 0.0
                    total_cost = distance_cost + beta_admin_cost
                    total_referral_cost += total_cost
        return total_referral_cost

    def TC_waiting(self) -> float:
        return self.compute_total_waiting_time() * 2.0

    def TC_centre(self, gamma_parallel: float, gamma_bidirectional: float, delta_admin: float) -> float:
        total_centre_cost = self.TC_doctor_transfer()
        for patient in self.system.all_patients:
            for transfer in patient.travel_history:
                from_node, to_node, _, _ = transfer
                if (
                    from_node == self.system.referral_name
                    or to_node == self.system.referral_name
                ):
                    if from_node.startswith("社康机构") and to_node.startswith("社康机构"):
                        coordination_cost = gamma_parallel
                    elif (
                        from_node == self.system.hospital_name
                        or to_node == self.system.hospital_name
                    ):
                        coordination_cost = gamma_bidirectional
                    else:
                        continue
                    total_centre_cost += coordination_cost + delta_admin
        return total_centre_cost

    def TC_ins(self) -> float:
        total_insurance_cost = 0.0
        for patient in self.system.all_patients:
            lambda_i = 0.9 if patient.severity_score_after <= 5 else 1.35
            if patient.current_node in self.system.community_names:
                rho_ji = 0.7
                base_rate = 800.0
            else:
                rho_ji = 0.6
                base_rate = 1000.0
            rw_i = (patient.severity_score_after**2 / 100) * 3.0
            treatment_cost = rw_i * base_rate
            insurance_cost = lambda_i * rho_ji * treatment_cost
            total_insurance_cost += insurance_cost
        return total_insurance_cost

    def transfer_cost(self) -> float:
        total_transfer_cost = 0.0
        for patient in self.system.all_patients:
            total_transfer_cost += sum(
                patient.cost_map.get(to_node, 0.0) for _, to_node, _, _ in patient.travel_history
            )
        return total_transfer_cost

    def treatment_cost(self) -> float:
        total_treatment_cost = 0.0
        for patient in self.system.all_patients:
            if not patient.travel_history:
                continue
            first_to_node = patient.travel_history[0][1]
            cost_base = patient.cost_map.get(first_to_node, 0.0)
            total_treatment_cost += cost_base * (patient.severity_score**2 / 25.0)
        return total_treatment_cost

    def admin_cost(self) -> float:
        total_admin_cost = 0.0
        for patient in self.system.all_patients:
            admin_cost = sum(
                1
                for from_node, to_node, _, _ in patient.travel_history
                if patient.referral_node != to_node
            )
            total_admin_cost += admin_cost
        return total_admin_cost

    def evaluate_experience_improvement(self) -> float:
        total_referral_cost = 0.0
        total_continuity = 0.0
        total_experience = 0.0
        for patient in self.system.all_patients:
            total_experience += getattr(patient, "experience_score", 0.0)
            for _, _, distance, continuity in patient.travel_history:
                total_referral_cost -= distance
                total_continuity += continuity
        return total_referral_cost + total_continuity + total_experience

    def evaluate_patient_effect(self) -> float:
        total_effect = 0.0
        for patient in self.system.all_patients:
            s_before = patient.severity_score
            s_after = patient.severity_score_after
            adherence_state = self.system._get_adherence_state(patient)
            followup = adherence_state.probability
            if patient.travel_history:
                _, to_node, _, _ = patient.travel_history[0]
                is_hospital = to_node == self.system.hospital_name
            else:
                is_hospital = patient.referral_node == self.system.hospital_name
            q_term = 1 if is_hospital else 1
            total_effect += (s_before - s_after) + followup + q_term
        return total_effect

    def TU_patient(
        self,
        omega_effect: float,
        omega_waiting: float,
        omega_transfer_cost: float,
        omega_treatment_cost: float,
        omega_experience: float,
        omega_admin_cost: float,
    ) -> float:
        effect = self.evaluate_patient_effect()
        waiting_time = self.compute_total_waiting_time()
        transfer_cost = self.transfer_cost()
        treatment_cost = self.treatment_cost()
        admin_cost = self.admin_cost()
        utility = (
            omega_effect * effect
            - omega_waiting * waiting_time
            - omega_transfer_cost * transfer_cost
            - omega_treatment_cost * treatment_cost
            - omega_admin_cost * admin_cost
        )

        experience_improvement = self.evaluate_experience_improvement()
        total_utility = utility + omega_experience * experience_improvement

        self.mid_records.append(
            [
                effect,
                waiting_time,
                transfer_cost,
                treatment_cost,
                experience_improvement,
                admin_cost,
            ]
        )
        return total_utility

    def TC_hosp_operation_per_node(self) -> Dict[str, float]:
        self.system.ensure_busy_rates_ready()
        node_operation_cost = {node_name: 0.0 for node_name in self.system.all_nodes}
        resource_config = {
            "outpatient_rooms": {"rho_busy_cost": 500, "rho_idle_cost": 100},
            "emergency_rooms": {"rho_busy_cost": 750, "rho_idle_cost": 260},
            "community_rooms": {"rho_busy_cost": 200, "rho_idle_cost": 80},
        }

        for node_name, queue_rates in self.system.busy_rates.items():
            if node_name == self.system.referral_name:
                continue
            for q_name, metrics in queue_rates.items():
                observations = metrics.get("observations", 0)
                total_doctors = metrics.get("total_doctors", 0)
                if observations == 0 or total_doctors == 0:
                    continue
                if node_name == self.system.hospital_name:
                    res_type = "emergency_rooms" if q_name == "急诊" else "outpatient_rooms"
                else:
                    res_type = "community_rooms"
                costs = resource_config[res_type]
                busy_doctors = metrics.get("busy_doctors", 0)
                rho = busy_doctors / total_doctors if total_doctors else 0
                avg_doctors = total_doctors / observations
                queue_cost = avg_doctors * (rho * costs["rho_busy_cost"] + (1 - rho) * costs["rho_idle_cost"])
                node_operation_cost[node_name] += queue_cost
        return node_operation_cost

    def calculate_profit_per_node(self) -> Dict[str, float]:
        visits = self.count_visits()
        referrals = self.count_referrals()
        node_operation_cost = self.TC_hosp_operation_per_node()

        profits: Dict[str, float] = {}

        hospital_name = self.system.hospital_name
        hosp_visits = visits.get(hospital_name, 0)
        hosp_referrals = referrals.get(hospital_name, 0)
        hosp_treatment = hosp_visits * 1000.0
        hosp_service = hosp_visits * (150.0 + 200.0) + hosp_visits * 0.1 * 500.0
        hosp_referral_income = hosp_referrals * (30.0 + 5.0)
        hosp_operation_cost = node_operation_cost.get(hospital_name, 0.0)
        hosp_referral_cost = hosp_referrals * 50.0
        profits[hospital_name] = (
            hosp_treatment + hosp_service + hosp_referral_income - hosp_operation_cost - hosp_referral_cost
        )

        for comm in self.system.community_names:
            comm_visits = visits.get(comm, 0)
            comm_referrals = referrals.get(comm, 0)
            comm_treatment = comm_visits * 800.0
            comm_referral_income = comm_referrals * (15.0 + 2.0)
            comm_operation_cost = node_operation_cost.get(comm, 0.0)
            comm_referral_cost = comm_referrals * 20.0
            profits[comm] = (
                comm_treatment + comm_referral_income - comm_operation_cost - comm_referral_cost
            )

        rc_name = self.system.referral_name
        rc_coordination_cost = self.TC_centre(50, 200, 100.0)
        profits[rc_name] = -rc_coordination_cost

        self.profit_records.append(profits)
        return profits


