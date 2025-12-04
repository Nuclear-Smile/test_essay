from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from deap import tools

from models import HealthcareSystem
from optimization import DVG_NSGAIIOptimization, HealthcareOptimizationProblem
from patient_data import DEFAULT_LIGHT_PATIENTS, DEFAULT_SEVERE_PATIENTS, generate_patient_dataset
from visualization import (
    plot_optimization_convergence,
    plot_pareto_front,
    plot_pareto_front_3d_animated,
)


def save_logbook(logbook, output_path: Path) -> None:
    try:
        with output_path.open("wb") as fh:
            pickle.dump(logbook, fh)
        print(f"Logbook saved to: {output_path}")
    except Exception as exc:
        print(f"Error saving logbook: {exc}")


def save_pareto_front(population, node_count: int, output_dir: Path) -> None:
    try:
        pareto_front_inds = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    except Exception as exc:
        print(f"Error sorting Pareto front: {exc}")
        return

    pareto_data = []
    for idx, ind in enumerate(pareto_front_inds):
        if hasattr(ind, "original_objectives") and ind.original_objectives is not None:
            profit = -ind.original_objectives[0]
            tc_ins = ind.original_objectives[1]
            utility = -ind.original_objectives[2]
            transfer_events = getattr(ind, "transfer_events", None)
            pareto_data.append(
                {
                    "Solution": idx,
                    "NetProfit": profit,
                    "TC_Insurance": tc_ins,
                    "PatientUtility": utility,
                    "TotalTransfers": transfer_events,
                }
            )
    if not pareto_data:
        print("No Pareto data to save.")
        return

    df = pd.DataFrame(pareto_data)
    output_path = output_dir / f"pareto_front_nodes_{node_count}.xlsx"
    try:
        df.to_excel(output_path, index=False)
        print(f"Pareto front objectives saved to: {output_path}")
    except Exception as exc:
        print(f"Error saving Pareto front data: {exc}")

    csv_path = output_dir / f"pareto_front_nodes_{node_count}.csv"
    try:
        df.to_csv(csv_path, index=False)
        print(f"Pareto front CSV saved to: {csv_path}")
    except Exception as exc:
        print(f"Error saving Pareto front CSV: {exc}")


def save_mid_records(mid_records: List[List[float]], output_dir: Path, node_count: int) -> None:
    if not mid_records:
        print("No intermediate records to save.")
        return
    columns = [
        "治疗效果",
        "等待时间",
        "转诊距离成本",
        "治疗费用",
        "患者体验分",
        "行政负担",
    ]
    df = pd.DataFrame(mid_records, columns=columns)
    output_path = output_dir / f"mids_nodes_{node_count}.xlsx"
    try:
        df.to_excel(output_path, index=False)
        print(f"Intermediate utility components saved to: {output_path}")
    except Exception as exc:
        print(f"Error saving intermediate utility data: {exc}")


def save_node_profits(profit_history: List[List], output_dir: Path, node_count: int, population_size: int) -> None:
    if not profit_history:
        print("No profit history available.")
        return
    try:
        last_evals = profit_history[-population_size:]
    except IndexError:
        last_evals = profit_history

    avg_profits = {}
    for eval_list in last_evals:
        for node, profit in eval_list:
            avg_profits.setdefault(node, []).append(profit)
    final_avg_profits = {node: float(np.mean(values)) for node, values in avg_profits.items()}
    df = pd.DataFrame(list(final_avg_profits.items()), columns=["Node", "AverageProfit_LastGen"])
    output_path = output_dir / f"node_profits_nodes_{node_count}.xlsx"
    try:
        df.to_excel(output_path, index=False)
        print(f"Average node profits saved to: {output_path}")
    except Exception as exc:
        print(f"Error saving node profits: {exc}")


def save_objective_matrix(population, output_dir: Path, filename: str = "objs.npy") -> None:
    objectives = []
    for ind in population:
        if hasattr(ind, "original_objectives") and ind.original_objectives is not None:
            net_profit = -ind.original_objectives[0]
            tc_ins = ind.original_objectives[1]
            total_utility = -ind.original_objectives[2]
            objectives.append([net_profit, tc_ins, total_utility])
    if not objectives:
        print("No objective data to save.")
        return
    obj_array = np.array(objectives).T
    output_path = output_dir / filename
    np.save(output_path, obj_array)
    print(f"Objective matrix saved to: {output_path}")


def run_experiment(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).resolve()
    data_dir = base_dir / "data"
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    patient_dataset = args.patient_dataset
    if not patient_dataset:
        patient_dataset = data_dir / "patient_dataset.csv"
        patient_dataset.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Generating patient dataset: light={args.light_patients}, severe={args.severe_patients}, steps={args.sim_steps}"
        )
        _, _, dataset = generate_patient_dataset(
            location_path=args.location,
            total_light=args.light_patients,
            total_severe=args.severe_patients,
            total_steps=args.sim_steps,
            random_seed=args.seed,
            output_path=str(patient_dataset),
        )
        print(f"Patient dataset generated with {len(dataset)} records at {patient_dataset}")

    patient_dataset = Path(patient_dataset)
    location_path = Path(args.location) if args.location else (data_dir / "location.xlsx")

    for mutation_start in args.mutation_start:
        for node_count in args.node_counts:
            print(f"\n{'=' * 20} Running for {node_count} total nodes {'=' * 20}")
            totallist = []

            random_seed = args.seed
            if random_seed is not None:
                np.random.seed(random_seed)

            community_count = node_count - 1
            if community_count < 0:
                raise ValueError(f"Node count {node_count} is too small. Must be at least 2.")

            print(f"Initializing HealthcareSystem with {community_count} community centers...")
            system = HealthcareSystem(
                hospital_name="综合医院",
                referral_name="转诊中心",
                community_count=community_count,
                hospital_outpatient_rooms=args.hospital_outpatient_rooms,
                hospital_emergency_rooms=args.hospital_emergency_rooms,
                community_rooms=args.community_rooms,
                patient_dataset_path=patient_dataset,
                location_path=location_path,
            )

            print(
                f"Running initial simulation to generate patients ({args.light_patients} light, {args.severe_patients} severe)..."
            )
            system.run_simulation(
                total_steps=args.sim_steps,
                total_light=args.light_patients,
                total_severe=args.severe_patients,
                light_hospital_prob=args.light_hospital_prob,
            )

            problem = HealthcareOptimizationProblem(system)
            optimizer = DVG_NSGAIIOptimization(
                healthcare_problem=problem,
                grouping_strategy=args.grouping_strategy,
                subgroup_count=args.subgroup_count,
            )

            print(
                f"Starting DVG-NSGA-II optimization: {args.generations} generations, Pop Size={args.population_size}..."
            )
            population, logbook = optimizer.optimize(
                n_generations=args.generations,
                population_size=args.population_size,
                cxpb=args.cxpb,
                mutpb=mutation_start,
                cx_end=args.cx_end,
                mut_end=args.mut_end,
            )

            if not population:
                print("[ERROR] Optimization did not return a final population.")
                continue

            print(f"Final population size: {len(population)}")

            experiment_dir = outputs_dir / f"nodes_{node_count}_mut_{mutation_start}"
            experiment_dir.mkdir(parents=True, exist_ok=True)

            plot_optimization_convergence(
                logbook,
                experiment_dir,
                base_filename="convergence",
                csv_filename=f"convergence_nodes_{node_count}.csv",
            )
            plot_pareto_front(
                population,
                experiment_dir,
                filename="pareto_front.png",
                csv_filename=f"pareto_front_nodes_{node_count}.csv",
            )
            plot_pareto_front_3d_animated(
                population,
                experiment_dir,
                filename=f"pareto_3d_nodes_{node_count}.gif",
            )

            save_logbook(logbook, experiment_dir / f"logbook_nodes_{node_count}.pkl")
            save_pareto_front(population, node_count, experiment_dir)
            save_mid_records(problem.evaluator.mid_records, experiment_dir, node_count)
            save_node_profits(problem.profit_history, experiment_dir, node_count, args.population_size)
            save_objective_matrix(population, experiment_dir, filename="objs.npy")

            print(f"\n--- Run for {node_count} nodes finished ---\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run healthcare system optimization pipeline")
    parser.add_argument("--base-dir", default="正式完整模型", help="工程根目录（默认当前项目 healthcare_app）")
    parser.add_argument("--patient-dataset", help="患者数据集路径（默认自动生成在 base_dir/data/patient_dataset.csv）")
    parser.add_argument("--location", default="正式完整模型/data/location.xlsx", help="地理位置表路径")
    parser.add_argument("--mutation-start", type=float, nargs="+", default=[0.10], help="起始变异概率列表")
    parser.add_argument("--node-counts", type=int, nargs="+", default=[28], help="节点数量配置")
    parser.add_argument("--generations", type=int, default=200, help="迭代代数")
    parser.add_argument("--population-size", type=int, default=200, help="种群规模")
    parser.add_argument("--cxpb", type=float, default=0.9, help="交叉概率起始值")
    parser.add_argument("--mut-end", type=float, default=0.01, help="变异概率下界")
    parser.add_argument("--cx-end", type=float, default=0.1, help="交叉概率下界")
    parser.add_argument("--grouping-strategy", default="by_initial_location", help="变量分组策略")
    parser.add_argument("--subgroup-count", type=int, default=5, help="每类子分组数量")
    parser.add_argument("--sim-steps", type=int, default=160, help="仿真时间步/可用于生成数据集")
    parser.add_argument("--light-patients", type=int, default=DEFAULT_LIGHT_PATIENTS, help="轻症患者数量")
    parser.add_argument("--severe-patients", type=int, default=DEFAULT_SEVERE_PATIENTS, help="重症患者数量")
    parser.add_argument("--light-hospital-prob", type=float, default=0.5, help="轻症患者首诊进入医院的概率")
    parser.add_argument("--hospital-outpatient-rooms", type=int, default=10, help="医院门诊诊室数量")
    parser.add_argument("--hospital-emergency-rooms", type=int, default=10, help="医院急诊诊室数量")
    parser.add_argument("--community-rooms", type=int, default=3, help="社区诊室数量")
    parser.add_argument("--seed", type=int, default=21, help="随机数种子（用于可重复）")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()


