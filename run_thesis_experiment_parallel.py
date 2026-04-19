"""
Parallelized experiment runner for thesis experiments.

Identical logic to run_thesis_experiment.py, but runs all (dim, run) jobs
in parallel using multiprocessing.Pool.

Key detail: each worker calls torch.set_num_threads(1) to prevent PyTorch's
internal OpenMP/MKL from spawning its own threads --> otherwise n_workers *
n_cpu_threads compete for the same cores and performance degrades.

Usage:
    python run_thesis_experiment_parallel.py \\
        -c experiments/thesis_experiments/configs/gibo_baseline.yaml \\
        -d experiments/thesis_experiments/data \\
        --n_workers 24

    # Smoke test:
    python run_thesis_experiment_parallel.py \\
        -c experiments/thesis_experiments/configs/gibo_baseline.yaml \\
        -d experiments/thesis_experiments/data \\
        --smoke --n_workers 4
"""

# ============================================================
# THESIS EXTENSION — BEGIN
# Description: Parallel wrapper around run_thesis_experiment logic
# ============================================================

import os
import pickle
import argparse
from multiprocessing import Pool, cpu_count

import torch
import yaml

from src.loop import call_counter
from src.optimizers import BayesianGradientAscent
from src.model import DerivativeExactGPSEModel
from src.acquisition_function import optimize_acqf_custom_bo
from src.synthetic_functions import generate_objective_from_gp_post

# Reuse build_optimizer and run_single from the sequential runner unchanged.
from run_thesis_experiment import build_optimizer, run_single


def _worker_init():
    """Called once per worker process at Pool startup.

    Limits PyTorch's internal thread count to 1 so that n_workers processes
    each use 1 core, instead of n_workers * n_cpu_threads all competing.
    """
    torch.set_num_threads(1)


def _run_job(args):
    """Top-level function (required for pickle-ability in multiprocessing).

    Unpacks job tuple, skips if output already exists, runs run_single,
    saves .pkl.
    """
    cfg, dim, run_idx, seed, train_x, train_y, lengthscale, f_max, out_path = args

    if os.path.exists(out_path):
        print(f"  dim={dim} run {run_idx:03d} — exists, skipping.", flush=True)
        return

    print(f"  dim={dim} run {run_idx:03d} (seed={seed}) starting...", flush=True)
    try:
        result = run_single(
            cfg=cfg,
            dim=dim,
            seed=seed,
            train_x=train_x,
            train_y=train_y,
            lengthscale=lengthscale,
            f_max=f_max,
        )
        with open(out_path, "wb") as fh:
            pickle.dump(result, fh)
        regret_final = result["regret_per_eval"][-1] if result["regret_per_eval"] else float("nan")
        n_iters = len(result["f_values"])
        print(
            f"  dim={dim} run {run_idx:03d} done. "
            f"iters={n_iters}, regret={regret_final:.4f}",
            flush=True,
        )
    except Exception as e:
        print(f"  dim={dim} run {run_idx:03d} ERROR: {e}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Parallelized thesis experiment runner."
    )
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to experiment config YAML.")
    parser.add_argument("-d", "--data_dir", type=str, required=True,
                        help="Path to pre-generated data directory.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: first dim only, 2 runs, 30 calls.")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPUs).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    #Load pre-generated data once in the main process.
    #Workers receive tensors via pickle --> acceptable since tensors are small
    #(1000 points x dim) compared to the per-run compute cost.
    data_dir = args.data_dir
    train_x_dict      = torch.load(os.path.join(data_dir, "train_x.pt"))
    train_y_dict      = torch.load(os.path.join(data_dir, "train_y.pt"))
    lengthscales_dict = torch.load(os.path.join(data_dir, "lengthscales.pt"))
    f_max_dict        = torch.load(os.path.join(data_dir, "f_max.pt"))

    if args.smoke:
        cfg["dimensions"] = [cfg["dimensions"][0]]
        cfg["n_runs"] = 2
        cfg["max_objective_calls"] = 30
        print("[SMOKE TEST] dim={}, n_runs=2, max_calls=30".format(cfg["dimensions"]))

    dimensions  = cfg["dimensions"]
    n_runs      = cfg["n_runs"]
    seed_start  = cfg.get("seed_start", 0)
    results_dir = cfg["results_dir"]
    name        = cfg["name"]

    # Build flat job list — one entry per (dim, run).
    jobs = []
    for dim in dimensions:
        dim_dir = os.path.join(results_dir, name, f"dim_{dim}")
        os.makedirs(dim_dir, exist_ok=True)
        for run_idx in range(n_runs):
            seed = seed_start + run_idx
            out_path = os.path.join(dim_dir, f"run_{run_idx:03d}.pkl")
            jobs.append((
                cfg,
                dim,
                run_idx,
                seed,
                train_x_dict[dim][run_idx],
                train_y_dict[dim][run_idx],
                lengthscales_dict[dim],
                float(f_max_dict[dim][run_idx]),
                out_path,
            ))

    # Skip already-finished jobs before launching workers.
    pending = [j for j in jobs if not os.path.exists(j[-1])]

    n_workers = min(args.n_workers or cpu_count(), len(pending)) if pending else 1

    print(f"\nExperiment : {name}")
    print(f"Mode       : {cfg['inner_loop_mode']}")
    print(f"Dimensions : {dimensions}")
    print(f"Total jobs : {len(jobs)}  ({len(jobs) - len(pending)} already done, {len(pending)} pending)")
    print(f"Workers    : {n_workers}\n")

    if not pending:
        print("All jobs already completed.")
        return

    with Pool(processes=n_workers, initializer=_worker_init) as pool:
        pool.map(_run_job, pending)

    print(f"\nDone. Results: {os.path.join(results_dir, name)}")


if __name__ == "__main__":
    main()

# ============================================================
# THESIS EXTENSION — END
# ============================================================
