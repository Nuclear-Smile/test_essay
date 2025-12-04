from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation


def select_best_compromise_solution_3d(objectives_list: Sequence[Sequence[float]]) -> int:
    obj_array = np.array(objectives_list)
    if obj_array.shape[1] != 3:
        raise ValueError("Expected 3 objectives.")

    ideal_point = np.min(obj_array, axis=0)
    min_vals = np.min(obj_array, axis=0)
    max_vals = np.max(obj_array, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges < 1e-9] = 1.0

    norm_obj = (obj_array - min_vals) / ranges
    norm_ideal = (ideal_point - min_vals) / ranges
    distances = np.linalg.norm(norm_obj - norm_ideal, axis=1)
    best_index = int(np.argmin(distances))
    return best_index


def plot_pareto_front(
    population: Iterable,
    output_dir: Path,
    filename: str = "pareto_front.png",
    csv_filename: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    from deap import tools  # lazy import to avoid optional dependency at module import

    population = list(population)
    if not population:
        print("Cannot plot Pareto front: Final population is empty.")
        return

    pareto_front_inds = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    objectives: List[Sequence[float]] = []
    valid_inds_for_plot = []
    for ind in pareto_front_inds:
        if hasattr(ind, "original_objectives") and ind.original_objectives is not None and len(ind.original_objectives) == 3:
            objectives.append(ind.original_objectives)
            valid_inds_for_plot.append(ind)
        else:
            print(f"[Warning] Individual {ind} missing valid original_objectives. Skipping for plot.")

    if not objectives:
        print("No valid solutions found on the Pareto front with original objectives stored.")
        return

    obj_array = np.array(objectives)
    outlier_mask = _compute_outlier_mask(obj_array)
    if outlier_mask.sum() == 0:
        print("All Pareto candidates flagged as outliers; skipping removal to retain data.")
    elif outlier_mask.sum() < len(obj_array):
        removed = len(obj_array) - int(outlier_mask.sum())
        print(f"Filtering out {removed} Pareto outlier(s) before plotting.")
        obj_array = obj_array[outlier_mask]
        valid_inds_for_plot = [ind for idx, ind in enumerate(valid_inds_for_plot) if outlier_mask[idx]]

    net_profit = -obj_array[:, 0]
    tc_ins = obj_array[:, 1]
    total_utility = -obj_array[:, 2]

    compromise_index = select_best_compromise_solution_3d(obj_array)
    compromise_x = net_profit[compromise_index]
    compromise_y1 = tc_ins[compromise_index]
    compromise_y2 = total_utility[compromise_index]

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(net_profit, tc_ins, label="Pareto Front", alpha=0.7)
    plt.scatter(compromise_x, compromise_y1, color="red", marker="*", s=150, label="Compromise")
    plt.xlabel("Net Profit (Higher is Better)")
    plt.ylabel("Total Insurance Cost (Lower is Better)")
    plt.title("Profit vs Insurance Cost")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.scatter(net_profit, total_utility, label="Pareto Front", alpha=0.7)
    plt.scatter(compromise_x, compromise_y2, color="red", marker="*", s=150, label="Compromise")
    plt.xlabel("Net Profit (Higher is Better)")
    plt.ylabel("Total Patient Utility (Higher is Better)")
    plt.title("Profit vs Patient Utility")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.scatter(tc_ins, total_utility, label="Pareto Front", alpha=0.7)
    plt.scatter(compromise_y1, compromise_y2, color="red", marker="*", s=150, label="Compromise")
    plt.xlabel("Total Insurance Cost (Lower is Better)")
    plt.ylabel("Total Patient Utility (Higher is Better)")
    plt.title("Insurance Cost vs Patient Utility")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Pareto Front Projections (Original Objectives)")
    output_path = output_dir / filename
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Pareto front saved to: {output_path}")

    if csv_filename:
        csv_path = output_dir / csv_filename
        df = pd.DataFrame(
            {
                "NetProfit": net_profit,
                "TC_Insurance": tc_ins,
                "PatientUtility": total_utility,
            }
        )
        df.to_csv(csv_path, index=False)
        print(f"Pareto front data saved to: {csv_path}")


def plot_pareto_front_3d_animated(population: Iterable, output_dir: Path, filename: str = "pareto_front_3d.gif") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    from deap import tools

    population = list(population)
    if not population:
        print("Cannot plot 3D Pareto front: Final population is empty.")
        return

    pareto_front_inds = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    objectives: List[Sequence[float]] = []
    valid_inds_for_plot = []
    for ind in pareto_front_inds:
        if hasattr(ind, "original_objectives") and ind.original_objectives is not None and len(ind.original_objectives) == 3:
            objectives.append(ind.original_objectives)
            valid_inds_for_plot.append(ind)

    if not objectives:
        print("No valid solutions on Pareto front for 3D plot.")
        return

    obj_array = np.array(objectives)
    outlier_mask = _compute_outlier_mask(obj_array)
    if outlier_mask.sum() == 0:
        print("All Pareto candidates flagged as outliers (3D); skipping removal to retain data.")
    elif outlier_mask.sum() < len(obj_array):
        removed = len(obj_array) - int(outlier_mask.sum())
        print(f"Filtering out {removed} Pareto outlier(s) before 3D plotting.")
        obj_array = obj_array[outlier_mask]
        valid_inds_for_plot = [ind for idx, ind in enumerate(valid_inds_for_plot) if outlier_mask[idx]]

    net_profit = -obj_array[:, 0]
    tc_ins = obj_array[:, 1]
    total_utility = -obj_array[:, 2]
    compromise_index = select_best_compromise_solution_3d(obj_array)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        net_profit,
        tc_ins,
        total_utility,
        label="Pareto Front",
        alpha=0.7,
        s=40,
        c=total_utility,
        cmap="viridis",
    )

    ax.scatter(
        net_profit[compromise_index],
        tc_ins[compromise_index],
        total_utility[compromise_index],
        color="red",
        marker="*",
        s=250,
        label="Compromise Solution",
        depthshade=False,
    )

    ax.set_xlabel("Net Profit (Higher)")
    ax.set_ylabel("Insurance Cost (Lower)")
    ax.set_zlabel("Patient Utility (Higher)")
    ax.set_title("3D Pareto Front (Original Objectives)")
    ax.legend()
    fig.colorbar(sc, label="Total Patient Utility")

    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return (fig,)

    try:
        ani = FuncAnimation(fig, update, frames=range(0, 360, 4), interval=100, blit=False)
        output_path = output_dir / filename
        ani.save(output_path, writer="pillow", fps=15)
        print(f"3D Pareto front animation saved to: {output_path}")
    except Exception as exc:
        print(f"Error saving animation (ensure 'pillow' library is installed): {exc}")
        output_path = output_dir / filename.replace(".gif", ".png")
        plt.savefig(output_path, dpi=300)
        print(f"Fallback static 3D plot saved to: {output_path}")
    finally:
        plt.close(fig)


def _sanitize_name(name: str) -> str:
    sanitized = name.replace("-", "neg_").replace(" ", "_")
    return "".join(ch for ch in sanitized if ch.isalnum() or ch == "_").lower()


def plot_optimization_convergence(
    logbook,
    output_dir: Path,
    base_filename: str = "convergence",
    csv_filename: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if logbook is None or len(logbook) == 0:
        print("Logbook is empty, cannot plot convergence.")
        return

    generations = logbook.select("gen")

    if "original" not in logbook.chapters:
        print("Logbook missing 'original' chapter, skipping convergence plots.")
        return

    try:
        avg_origs = logbook.chapters["original"].select("avg_orig")
    except KeyError:
        print("Could not find 'avg_orig' in logbook stats for original objectives.")
        return

    if not avg_origs or any(np.isnan(avg_origs[0])):
        print("Original objective statistics contain NaNs, skipping plot.")
        return

    obj_names = ["-NetProfit", "TC_ins", "-TotalUtility"]
    series_dict: Dict[str, List[float]] = {}

    for idx, name in enumerate(obj_names):
        series = [val[idx] for val in avg_origs]
        series_dict[name] = series
        plt.figure(figsize=(8, 5))
        plt.plot(generations, series, label=name, color="tab:blue")
        plt.xlabel("Generation")
        plt.ylabel(name)
        plt.title(f"Convergence of {name}")
        plt.grid(True)
        plt.tight_layout()
        sanitized = _sanitize_name(name)
        output_path = output_dir / f"{base_filename}_{sanitized}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Convergence plot saved to: {output_path}")

    if csv_filename:
        df = pd.DataFrame({"generation": generations})
        for name, series in series_dict.items():
            df[name] = series
        csv_path = output_dir / csv_filename
        df.to_csv(csv_path, index=False)
        print(f"Convergence data saved to: {csv_path}")


def _compute_outlier_mask(obj_array: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    if obj_array.size == 0 or obj_array.shape[0] < 4:
        return np.ones(obj_array.shape[0], dtype=bool)

    mask = np.ones(obj_array.shape[0], dtype=bool)
    for col in range(obj_array.shape[1]):
        column = obj_array[:, col]
        q1, q3 = np.percentile(column, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask &= (column >= lower) & (column <= upper)
    return mask


