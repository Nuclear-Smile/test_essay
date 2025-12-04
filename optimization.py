from __future__ import annotations

import copy
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
from deap import base, creator, tools

from models import HealthcareSystem
from objective_evaluator import ObjectiveEvaluator
from tqdm import trange  # 新增
import tqdm
class DecisionIndividual:
    """Composite decision encoding patients and doctors."""

    __slots__ = (
        "patient_moves",
        "doctor_moves",
        "fitness",
        "original_objectives",
        "crowding_dist",
        "transfer_events",
    )

    def __init__(self, patient_moves: np.ndarray, doctor_moves: np.ndarray):
        self.patient_moves = patient_moves
        self.doctor_moves = doctor_moves
        self.fitness = creator.FitnessMin()
        self.original_objectives = None
        self.crowding_dist = 0.0
        self.transfer_events = 0

    def __deepcopy__(self, memo):
        patient_moves = copy.deepcopy(self.patient_moves, memo)
        doctor_moves = copy.deepcopy(self.doctor_moves, memo)
        clone = DecisionIndividual(patient_moves, doctor_moves)
        if self.fitness.valid:
            clone.fitness.values = tuple(self.fitness.values)
        if self.original_objectives is not None:
            clone.original_objectives = tuple(self.original_objectives)
        clone.crowding_dist = self.crowding_dist
        clone.transfer_events = self.transfer_events
        return clone


def normalize(values: Sequence[float], min_vals: Sequence[float], max_vals: Sequence[float]) -> List[float]:
    return [
        (v - min_v) / (max_v - min_v) if max_v != min_v else 0.0
        for v, min_v, max_v in zip(values, min_vals, max_vals)
    ]


def global_parameter_tuning(cxpb: float, mutpb: float, cx_end: float, mut_end: float, generation: int, max_gen: int) -> Tuple[float, float]:
    ratio = float(generation) / float(max_gen) if max_gen else 0.0
    new_cxpb = cxpb - (cxpb - cx_end) * ratio
    new_mutpb = mutpb - (mutpb - mut_end) * ratio
    return max(new_cxpb, cx_end), max(new_mutpb, mut_end)


def local_parameter_tuning(
    p1,
    p2,
    cxpb: float,
    mutpb: float,
    old_revenue: float,
    old_tc: float,
    old_util: float,
) -> Tuple[float, float, float]:
    p1_stat = 0
    p2_stat = 0
    turnup = 1.1
    turndown = 0.9
    stat = [old_revenue, old_tc, old_util]
    for i in range(3):
        if p1.original_objectives[i] < stat[i]:
            p1_stat += 1
        if p2.original_objectives[i] < stat[i]:
            p2_stat += 1

    if p1_stat + p2_stat >= 4:
        cxpb *= turnup
    elif p1_stat + p2_stat <= 2:
        cxpb *= turndown

    mutpb_p1 = mutpb * (turnup if p1_stat >= 2 else turndown if p1_stat <= 1 else 1.0)
    mutpb_p2 = mutpb * (turnup if p2_stat >= 2 else turndown if p2_stat <= 1 else 1.0)
    return cxpb, mutpb_p1, mutpb_p2


def _dominates(ind1, ind2) -> bool:
    better_or_equal = True
    strictly_better = False
    for f1, f2 in zip(ind1.fitness.values, ind2.fitness.values):
        if f1 > f2:
            better_or_equal = False
            break
        if f1 < f2:
            strictly_better = True
    return better_or_equal and strictly_better


def non_dominated_sort(population: List[DecisionIndividual]) -> List[List[DecisionIndividual]]:
    S = [[] for _ in range(len(population))]
    n = [0 for _ in range(len(population))]
    rank = [0 for _ in range(len(population))]

    for p in range(len(population)):
        S[p] = []
        n[p] = 0
        for q in range(len(population)):
            if _dominates(population[p], population[q]):
                S[p].append(q)
            elif _dominates(population[q], population[p]):
                n[p] += 1
    front = [[]]
    for i in range(len(population)):
        if n[i] == 0:
            rank[i] = 0
            front[0].append(i)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        front.append(Q)

    fronts: List[List[DecisionIndividual]] = []
    for f in front:
        if f:
            fronts.append([population[idx] for idx in f])
    return fronts


def crowding_distance(front: List[DecisionIndividual]) -> None:
    if not front:
        return
    length = len(front)
    for ind in front:
        ind.crowding_dist = 0.0

    num_objs = len(front[0].fitness.values)
    for m in range(num_objs):
        front.sort(key=lambda ind: ind.fitness.values[m])
        front[0].crowding_dist = float("inf")
        front[-1].crowding_dist = float("inf")
        f_min = front[0].fitness.values[m]
        f_max = front[-1].fitness.values[m]
        if abs(f_max - f_min) < 1e-14:
            continue
        for i in range(1, length - 1):
            front[i].crowding_dist += (
                (front[i + 1].fitness.values[m] - front[i - 1].fitness.values[m]) / (f_max - f_min)
            )


def select_population_by_rank_and_crowding(population: List[DecisionIndividual], pop_size: int) -> List[DecisionIndividual]:
    fronts = non_dominated_sort(population)
    new_population: List[DecisionIndividual] = []
    for front in fronts:
        crowding_distance(front)
        front.sort(key=lambda ind: ind.crowding_dist, reverse=True)
        if len(new_population) + len(front) <= pop_size:
            new_population.extend(front)
        else:
            needed = pop_size - len(new_population)
            new_population.extend(front[:needed])
            break
    return new_population


class HealthcareOptimizationProblem:
    def __init__(self, healthcare_system: HealthcareSystem) -> None:
        self.healthcare_system = healthcare_system
        self.evaluator = ObjectiveEvaluator(healthcare_system)
        self.mid_records = self.evaluator.mid_records
        self.profit_history: List[List[Tuple[str, float]]] = []

    def objectives(self, individual: DecisionIndividual) -> List[float]:
        system = self.healthcare_system
        snapshot = system.snapshot_state()
        try:
            system.doctor_transfer_log = []
            doctor_moves = getattr(individual, "doctor_moves", None)
            if doctor_moves is not None and len(doctor_moves) > 0:
                for row in doctor_moves:
                    doctor_id = str(row[0])
                    target_node = str(row[2])
                    queue_name = str(row[3]) if len(row) > 3 else None
                    if not queue_name or doctor_id not in system.doctor_registry:
                        continue
                    doctor = system.doctor_registry[doctor_id]
                    if doctor.current_node == target_node and doctor.queue_name == queue_name:
                        continue
                    try:
                        system.transfer_doctor(doctor_id, target_node, queue_name)
                    except Exception:
                        continue

            patient_moves = getattr(individual, "patient_moves", None)
            if patient_moves is None:
                patient_moves = np.empty((0, 3), dtype=object)
            system.transfer_matrix = patient_moves
            system.transfer_patients_by_matrix(patient_moves)

            system.update_patient_metrics_after_moves()

            revenue = self.evaluator.total_revenue()
            cost1 = self.evaluator.TC_hosp_operation()
            cost2 = self.evaluator.TC_hosp_referral(2, 150)
            cost3 = self.evaluator.TC_centre(50, 200, 100.0)
            cost4 = self.evaluator.TC_doctor_transfer()
            total_cost = cost1 + cost2 + cost3 + cost4
            revenue -= total_cost

            tc_inc = self.evaluator.TC_ins()
            total_utility = self.evaluator.TU_patient(
                omega_effect=0.4,
                omega_waiting=1.0,
                omega_transfer_cost=0.5,
                omega_treatment_cost=0.5,
                omega_experience=1.0,
                omega_admin_cost=0.5,
            )

            profits = self.evaluator.calculate_profit_per_node()
            filtered = [
                (node, profit)
                for node, profit in profits.items()
                if node != self.healthcare_system.referral_name
            ]
            self.profit_history.append(filtered)

            if hasattr(individual, "transfer_events"):
                individual.transfer_events = self.evaluator.total_transfer_events()

            return [-revenue, tc_inc, -total_utility]
        finally:
            system.restore_state(snapshot)


class DVG_NSGAIIOptimization:
    def __init__(
        self,
        healthcare_problem: HealthcareOptimizationProblem,
        grouping_strategy: str = "by_initial_location",
        subgroup_count: int = 2,
    ) -> None:
        self.problem = healthcare_problem
        self.healthcare_system = healthcare_problem.healthcare_system
        self.num_patients = len(self.healthcare_system.all_patients)
        if self.num_patients == 0:
            raise ValueError("Healthcare system has no patients initialized. 请先运行仿真生成患者。")

        self.grouping_strategy = grouping_strategy
        self.subgroup_count = max(subgroup_count, 1)
        self.comm_ls = list(self.healthcare_system.all_nodes)
        if not self.comm_ls:
            raise ValueError("系统节点列表为空，无法构建变量。")

        self.node_queue_options: Dict[str, List[str]] = {
            node: [queue for queue, data in queues.items() if data["rooms"] > 0]
            for node, queues in self.healthcare_system.node_queues.items()
        }
        self.doctor_ids = list(self.healthcare_system.doctor_registry.keys())
        self.doctor_index_map = {doc_id: idx for idx, doc_id in enumerate(self.doctor_ids)}

        self.patient_groups = self._create_patient_groups()
        self.doctor_groups = self._create_doctor_groups()

        self.min_values = [0.0, 0.0, 0.0]
        self.max_values = [1.0, 1.0, 1.0]

        self.toolbox = base.Toolbox()
        self._register_toolbox()

    def _register_toolbox(self) -> None:
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))

        self.toolbox.register("clone", copy.deepcopy)
        self.toolbox.register("individual", self._create_decision_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.grouped_crossover)
        self.toolbox.register("mutate", self.grouped_mutation, indpb=0.1)
        self.toolbox.register("select", select_population_by_rank_and_crowding)
        self.toolbox.register("evaluate", self.evaluated_objectives)
        self.toolbox.register("local_search", self.local_search)

    def _create_patient_groups(self) -> List[List[int]]:
        if self.grouping_strategy == "by_initial_location":
            hospital_group: List[int] = []
            community_group: List[int] = []
            hospital_name = self.healthcare_system.hospital_name

            for idx, patient in enumerate(self.healthcare_system.all_patients):
                if patient.referral_node == hospital_name:
                    hospital_group.append(idx)
                else:
                    community_group.append(idx)

            def split_group(indices: List[int]) -> List[List[int]]:
                if not indices:
                    return []
                chunks = max(1, min(self.subgroup_count, len(indices)))
                groups = [[] for _ in range(chunks)]
                for i, patient_idx in enumerate(indices):
                    groups[i % chunks].append(patient_idx)
                return groups

            groups = split_group(hospital_group) + split_group(community_group)
            if not groups:
                groups = [list(range(self.num_patients))]
            print(
                f"Created groups by initial location. Hospital: {len(hospital_group)} patients, Community: {len(community_group)} patients."
            )
            print(f"Total groups created: {len(groups)}")
            return groups

        if self.grouping_strategy == "modulo":
            groups = [[] for _ in range(self.subgroup_count)]
            for i in range(self.num_patients):
                groups[i % self.subgroup_count].append(i)
            print(f"Created {self.subgroup_count} variable groups using modulo method.")
            return groups

        raise ValueError(f"Unknown grouping strategy: {self.grouping_strategy}")

    def _create_doctor_groups(self) -> List[List[int]]:
        if not self.doctor_ids:
            return []
        # All doctors share one group so every crossover/mutation sees the full vector.
        return [list(range(len(self.doctor_ids)))]

    def _create_decision_individual(self) -> DecisionIndividual:
        if not self.healthcare_system.all_patients:
            patient_moves = np.empty((0, 3), dtype=object)
        else:
            patient_moves = self.healthcare_system.generate_transition_matrix()

        doctor_rows: List[List[object]] = []
        valid_nodes = [node for node, queues in self.node_queue_options.items() if queues]
        if not valid_nodes:
            valid_nodes = [self.healthcare_system.hospital_name]

        for doctor_id in self.doctor_ids:
            doctor = self.healthcare_system.doctor_registry[doctor_id]
            target_node = random.choice(valid_nodes)
            queue_options = self.node_queue_options.get(target_node, [doctor.queue_name])
            queue_name = random.choice(queue_options) if queue_options else doctor.queue_name
            doctor_rows.append([doctor_id, doctor.current_node, target_node, queue_name])

        doctor_moves = np.array(doctor_rows, dtype=object)
        return DecisionIndividual(patient_moves, doctor_moves)

    def grouped_crossover(self, ind1: DecisionIndividual, ind2: DecisionIndividual):
        if self.patient_groups:
            group = random.choice(self.patient_groups)
            pm1 = ind1.patient_moves
            pm2 = ind2.patient_moves
            for idx in group:
                if idx < len(pm1) and idx < len(pm2) and random.random() < 0.5:
                    pm1[idx, 2], pm2[idx, 2] = pm2[idx, 2], pm1[idx, 2]

        if self.doctor_groups:
            group = random.choice(self.doctor_groups)
            dm1 = ind1.doctor_moves
            dm2 = ind2.doctor_moves
            for idx in group:
                if idx < len(dm1) and idx < len(dm2):
                    if random.random() < 0.5:
                        dm1[idx, 2], dm2[idx, 2] = dm2[idx, 2], dm1[idx, 2]
                    if random.random() < 0.5:
                        dm1[idx, 3], dm2[idx, 3] = dm2[idx, 3], dm1[idx, 3]
        return ind1, ind2

    def grouped_mutation(self, individual: DecisionIndividual, indpb):
        if not self.comm_ls:
            print("[Warning] comm_ls (node list) is empty in grouped_mutation. Cannot mutate.")
            return (individual,)

        if self.patient_groups:
            group = random.choice(self.patient_groups)
            pm = individual.patient_moves
            for idx in group:
                if idx < len(pm) and random.random() < indpb:
                    old_node = pm[idx, 2]
                    candidates = [node for node in self.comm_ls if node != old_node]
                    if candidates:
                        pm[idx, 2] = random.choice(candidates)

        if self.doctor_groups:
            group = random.choice(self.doctor_groups)
            dm = individual.doctor_moves
            valid_nodes = [node for node, queues in self.node_queue_options.items() if queues]
            if not valid_nodes:
                valid_nodes = [self.healthcare_system.hospital_name]
            for idx in group:
                if idx < len(dm) and random.random() < indpb:
                    target_node = random.choice(valid_nodes)
                    queue_choices = self.node_queue_options.get(target_node, [dm[idx, 3]])
                    queue_name = random.choice(queue_choices) if queue_choices else dm[idx, 3]
                    dm[idx, 2] = target_node
                    dm[idx, 3] = queue_name
        return (individual,)

    def local_search(self, individual: DecisionIndividual) -> DecisionIndividual:
        new_ind = copy.deepcopy(individual)
        if random.random() < 0.5 and len(new_ind.patient_moves) > 0:
            idx = random.randrange(len(new_ind.patient_moves))
            old_node = new_ind.patient_moves[idx, 2]
            candidates = [node for node in self.comm_ls if node != old_node]
            if candidates:
                new_ind.patient_moves[idx, 2] = random.choice(candidates)
        elif len(new_ind.doctor_moves) > 0:
            idx = random.randrange(len(new_ind.doctor_moves))
            valid_nodes = [node for node, queues in self.node_queue_options.items() if queues]
            if not valid_nodes:
                valid_nodes = [self.healthcare_system.hospital_name]
            target_node = random.choice(valid_nodes)
            queue_choices = self.node_queue_options.get(target_node, [new_ind.doctor_moves[idx, 3]])
            queue_name = random.choice(queue_choices) if queue_choices else new_ind.doctor_moves[idx, 3]
            new_ind.doctor_moves[idx, 2] = target_node
            new_ind.doctor_moves[idx, 3] = queue_name
        return new_ind

    def evaluated_objectives(self, individual: DecisionIndividual):
        raw_objectives = self.problem.objectives(individual)
        normalized = normalize(raw_objectives, self.min_values, self.max_values)
        individual.original_objectives = raw_objectives
        individual.fitness.values = normalized
        return normalized

    def optimize(
        self,
        n_generations: int = 100,
        population_size: int = 50,
        cxpb: float = 0.9,
        mutpb: float = 0.1,
        cx_end: float = 0.1,
        mut_end: float = 0.01,
    ):
        if self.num_patients == 0:
            print("[Error] Cannot optimize: No patients found in the healthcare system.")
            return [], None

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        stats_orig = tools.Statistics(
            key=lambda ind: ind.original_objectives
            if hasattr(ind, "original_objectives") and ind.original_objectives is not None
            else [np.nan] * 3
        )
        stats_orig.register("avg_orig", np.mean, axis=0)
        stats_orig.register("min_orig", np.min, axis=0)
        stats_orig.register("max_orig", np.max, axis=0)

        multi_stats = tools.MultiStatistics(fitness=stats, original=stats_orig)
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "fitness", "original"
        logbook.chapters["fitness"].header = "min", "max", "avg", "std"
        logbook.chapters["original"].header = "min_orig", "max_orig", "avg_orig"

        print(f"Initializing population of size {population_size}...")
        try:
            population = self.toolbox.population(n=population_size)
        except Exception as exc:  # pragma: no cover
            print(f"[Error] Failed to initialize population: {exc}")
            return [], None

        print("Evaluating initial population...")
        invalid = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            if isinstance(fit, (list, tuple)) and len(fit) == len(ind.fitness.weights):
                ind.fitness.values = fit
            else:
                ind.fitness.values = tuple(1e12 for _ in range(len(ind.fitness.weights)))

        valid_objectives = [
            ind.original_objectives
            for ind in population
            if hasattr(ind, "original_objectives") and ind.original_objectives is not None
        ]
        if valid_objectives:
            objs_arr = np.array(valid_objectives)
            self.min_values = np.nanmin(objs_arr, axis=0).tolist()
            self.max_values = np.nanmax(objs_arr, axis=0).tolist()
            for idx in range(len(self.max_values)):
                self.max_values[idx] += abs(self.max_values[idx] * 0.01) + 1e-6
            print(f"Updated Min Bounds: {self.min_values}")
            print(f"Updated Max Bounds: {self.max_values}")

        record = multi_stats.compile(population) if multi_stats else stats.compile(population)
        logbook.record(gen=0, evals=len(invalid), **record)
        print(logbook.stream)

        def composite_score(individual: DecisionIndividual):
            revenue, tc_inc, total_utility = individual.original_objectives
            return -revenue * tc_inc * total_utility

        print("\n--- Starting Evolution ---")
        old_revenue = old_tc = old_util = 0.0
        if valid_objectives:
            means = np.mean(np.array(valid_objectives), axis=0)
            old_revenue, old_tc, old_util = means.tolist()
        print("51121")
        for gen in trange(n_generations, desc="Evolution", leave=True):
            offspring: List[DecisionIndividual] = []
            local_search_offspring: List[DecisionIndividual] = []
            cxpb, mutpb = global_parameter_tuning(cxpb, mutpb, cx_end, mut_end, gen, n_generations)

            population[:] = self.toolbox.select(population, population_size)
            population_sorted = sorted(population, key=composite_score, reverse=False)

            select_good_count = max(1, int(0.1 * population_size))
            for ind in population_sorted[:select_good_count]:
                search_ind = self.toolbox.local_search(ind)
                self.toolbox.evaluate(search_ind)
                local_search_offspring.append(search_ind)

            while len(offspring) < population_size * 2:
                p1, p2 = random.sample(population, 2)
                c1 = self.toolbox.clone(p1)
                c2 = self.toolbox.clone(p2)

                if gen > 0:
                    cxpb_tmp, mutpb_p1, mutpb_p2 = local_parameter_tuning(
                        p1, p2, cxpb, mutpb, old_revenue, old_tc, old_util
                    )
                else:
                    cxpb_tmp, mutpb_p1, mutpb_p2 = cxpb, mutpb, mutpb

                if random.random() < cxpb_tmp:
                    c1, c2 = self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values
                    self.toolbox.evaluate(c1)
                    self.toolbox.evaluate(c2)
                    offspring.extend([c1, c2])

                if gen > 0:
                    _, mutpb_p1, mutpb_p2 = local_parameter_tuning(
                        c1, c2, cxpb, mutpb, old_revenue, old_tc, old_util
                    )

                if random.random() < mutpb_p1:
                    mutated, = self.toolbox.mutate(c1)
                    del mutated.fitness.values
                    self.toolbox.evaluate(mutated)
                    offspring.append(mutated)
                if random.random() < mutpb_p2:
                    mutated, = self.toolbox.mutate(c2)
                    del mutated.fitness.values
                    self.toolbox.evaluate(mutated)
                    offspring.append(mutated)

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                if isinstance(fit, (list, tuple)) and len(fit) == len(ind.fitness.weights):
                    ind.fitness.values = fit
                    ind.fitness.valid = True
                else:
                    ind.fitness.values = tuple(1e12 for _ in range(len(ind.fitness.weights)))

            population[:] = self.toolbox.select(
                population + offspring + local_search_offspring, population_size
            )

            record = multi_stats.compile(population) if multi_stats else stats.compile(population)
            revenues = [ind.original_objectives[0] for ind in population]
            tc_incs = [ind.original_objectives[1] for ind in population]
            utils = [ind.original_objectives[2] for ind in population]
            old_revenue = float(np.mean(revenues))
            old_tc = float(np.mean(tc_incs))
            old_util = float(np.mean(utils))
            logbook.record(gen=gen, evals=len(invalid), **record)

        print("\n--- Evolution Finished ---")
        return population, logbook


