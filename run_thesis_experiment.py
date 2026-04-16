"""
Unified experiment runner for thesis experiments.

Runs one variant (baseline, prob_wolfe, or det_ei) across all configured
dimensions and seeds. Saves per-run .pkl files with all thesis metrics.

Usage:
    python run_thesis_experiment.py \\
        -c experiments/thesis_experiments/configs/gibo_baseline.yaml \\
        -d experiments/thesis_experiments/data

    # Smoke test (fast):
    python run_thesis_experiment.py \\
        -c experiments/thesis_experiments/configs/gibo_baseline.yaml \\
        -d experiments/thesis_experiments/data \\
        --smoke

Output (one file per run):
    <results_dir>/<name>/dim_<D>/run_<NNN>.pkl

Each .pkl contains a dict with keys:
    regret_per_eval, f_values, inner_loop_samples, step_sizes,
    p_wolfe_values (Variant A), wolfe_satisfied (Variant B),
    armijo_ok, curvature_ok (Variant B),
    gradient_norms, posterior_variance_trace, config, seed, dimension
"""

# ============================================================
# THESIS EXTENSION — BEGIN
# Description: Unified experiment runner for all three thesis variants
# ============================================================

import os
import pickle
import argparse

import torch
import yaml

from src.loop import call_counter
from src.optimizers import BayesianGradientAscent
from src.model import DerivativeExactGPSEModel
from src.acquisition_function import optimize_acqf_custom_bo
from src.synthetic_functions import (
    generate_objective_from_gp_post,
    get_lengthscales,
    factor_hennig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optimizer(cfg: dict, dim: int, params_init: torch.Tensor,
                    objective, lengthscale: torch.Tensor) -> BayesianGradientAscent:
    """Instantiate BayesianGradientAscent from thesis config dict."""

    # ============================================================
    # THESIS EXTENSION — BEGIN
    # Description: Resolve YAML placeholder strings to concrete values,
    #              matching the original GIBO config.evaluate() behaviour.
    # ============================================================
    n_max_raw = cfg["N_max"]
    n_max = 5 * dim if n_max_raw == "variable" else int(n_max_raw)

    max_samples_raw = cfg["max_samples_per_iteration"]
    if max_samples_raw == "dim_search_space":
        max_samples = dim
    else:
        max_samples = int(max_samples_raw)
    # ============================================================
    # THESIS EXTENSION — END
    # ============================================================

    hypers = {
        "covar_module.base_kernel.lengthscale": lengthscale,
        "covar_module.outputscale": torch.tensor(cfg["gp_hypers"]["outputscale"]),
        "likelihood.noise": torch.tensor(cfg["noise_variance"]),
    }

    model_config = dict(
        N_max=n_max,
        ard_num_dims=dim,
        prior_mean=0.0,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint=None,
        noise_hyperprior=None,
    )

    hyperparameter_config = dict(
        hypers=hypers,
        no_noise_optimization=cfg["no_noise_optimization"],
        optimize_hyperparameters=cfg["optimize_hyperparameters"],
    )

    optimizer_torch_config = {"lr": cfg["lr"]}
    lr_schedular = cfg.get("lr_schedular", None)

    return BayesianGradientAscent(
        params_init=params_init,
        objective=objective,
        max_samples_per_iteration=max_samples,
        OptimizerTorch=torch.optim.SGD,
        optimizer_torch_config=optimizer_torch_config,
        lr_schedular=lr_schedular,
        Model=DerivativeExactGPSEModel,
        model_config=model_config,
        hyperparameter_config=hyperparameter_config,
        optimize_acqf=optimize_acqf_custom_bo,
        optimize_acqf_config=dict(
            q=1,
            num_restarts=cfg["acqf_num_restarts"],
            raw_samples=cfg["acqf_raw_samples"],
        ),
        bounds=None,
        delta=cfg["delta"],
        epsilon_diff_acq_value=cfg.get("epsilon_diff_acq_value", None),
        generate_initial_data=None,
        normalize_gradient=cfg.get("normalize_gradient", False),
        standard_deviation_scaling=cfg.get("standard_deviation_scaling", False),
        verbose=cfg.get("verbose", False),
        inner_loop_mode=cfg["inner_loop_mode"],
        c1=cfg["c1"],
        c2=cfg["c2"],
        c_W=cfg["c_W"],
        # ============================================================
        # THESIS EXPERIMENT EXTENSION — BEGIN
        # Description: Pass alpha_max and min_samples_per_iteration from
        #   YAML config, with safe defaults (None and 1) so that existing
        #   configs without these keys continue to work unchanged.
        # ============================================================
        alpha_max=cfg.get("alpha_max", None),
        min_samples_per_iteration=cfg.get("min_samples_per_iteration", 1),
        # ============================================================
        # THESIS EXPERIMENT EXTENSION — END
        # ============================================================
    )


def run_single(cfg: dict, dim: int, seed: int,
               train_x: torch.Tensor, train_y: torch.Tensor,
               lengthscale: torch.Tensor, f_max: float) -> dict:
    """Run one optimization trajectory. Returns a result dict."""

    torch.manual_seed(seed)

    objective_raw = generate_objective_from_gp_post(
        train_x=train_x,
        train_y=train_y,
        noise_variance=cfg["noise_variance"],
        gp_hypers={
            "covar_module.base_kernel.lengthscale": lengthscale,
            "covar_module.outputscale": torch.tensor(cfg["gp_hypers"]["outputscale"]),
        },
    )
    objective = call_counter(objective_raw)

    params_init = 0.5 * torch.ones(dim, dtype=torch.float32)
    optimizer = build_optimizer(cfg, dim, params_init, objective, lengthscale)

    # --- Metric accumulators ---
    f_values = []
    inner_loop_samples = []
    step_sizes = []
    p_wolfe_values = []        # Variant A only
    wolfe_satisfied_trace = [] # both adaptive variants
    armijo_ok_trace = []       # Variant B only
    curvature_ok_trace = []    # Variant B only
    gradient_norms = []
    posterior_variance_trace = []
    calls_at_iteration = []

    max_calls = cfg["max_objective_calls"]

    while objective._calls < max_calls:
        optimizer.step()

        info = optimizer.last_step_info
        f_values.append(optimizer.params_history_list[-1].clone())
        inner_loop_samples.append(info.get("n_inner_samples", None))
        step_sizes.append(info.get("alpha", None))
        gradient_norms.append(info.get("grad_norm", None))
        posterior_variance_trace.append(info.get("sigma2", None))
        calls_at_iteration.append(objective._calls)

        if cfg["inner_loop_mode"] == "prob_wolfe":
            p_wolfe_values.append(info.get("p_wolfe", None))
            wolfe_satisfied_trace.append(info.get("wolfe_satisfied", None))
        elif cfg["inner_loop_mode"] == "det_ei":
            wolfe_satisfied_trace.append(info.get("wolfe_satisfied", None))
            armijo_ok_trace.append(info.get("armijo_ok", None))
            curvature_ok_trace.append(info.get("curvature_ok", None))

    # Simple regret at each iteration: f_max - best_so_far
    best_so_far = float("-inf")
    regret_per_eval = []
    for params_t in f_values:
        val = objective_raw(params_t.view(1, -1),
                            observation_noise=False, requires_grad=False).item()
        best_so_far = max(best_so_far, val)
        regret_per_eval.append(float(f_max) - best_so_far)

    return {
        "regret_per_eval": regret_per_eval,
        "f_values": [p.numpy() for p in f_values],
        "inner_loop_samples": inner_loop_samples,
        "step_sizes": step_sizes,
        "p_wolfe_values": p_wolfe_values,
        "wolfe_satisfied": wolfe_satisfied_trace,
        "armijo_ok": armijo_ok_trace,
        "curvature_ok": curvature_ok_trace,
        "gradient_norms": gradient_norms,
        "posterior_variance_trace": posterior_variance_trace,
        "calls_at_iteration": calls_at_iteration,
        "f_max": float(f_max),
        "config": cfg,
        "seed": seed,
        "dimension": dim,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run thesis experiment for one variant."
    )
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to experiment config YAML.")
    parser.add_argument("-d", "--data_dir", type=str, required=True,
                        help="Path to pre-generated data directory.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: dim=[4], n_runs=1, max_calls=30.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load pre-generated data
    data_dir = args.data_dir
    train_x_dict   = torch.load(os.path.join(data_dir, "train_x.pt"))
    train_y_dict   = torch.load(os.path.join(data_dir, "train_y.pt"))
    lengthscales_dict = torch.load(os.path.join(data_dir, "lengthscales.pt"))
    f_max_dict     = torch.load(os.path.join(data_dir, "f_max.pt"))

    # Smoke-test overrides
    if args.smoke:
        cfg["dimensions"] = [cfg["dimensions"][0]]
        cfg["n_runs"] = 1
        cfg["max_objective_calls"] = 30
        print("[SMOKE TEST] dim={}, n_runs=1, max_calls=30".format(cfg["dimensions"]))

    dimensions  = cfg["dimensions"]
    n_runs      = cfg["n_runs"]
    seed_start  = cfg.get("seed_start", 0)
    results_dir = cfg["results_dir"]
    name        = cfg["name"]

    print(f"\nExperiment: {name}")
    print(f"Mode:       {cfg['inner_loop_mode']}")
    print(f"Dimensions: {dimensions}")
    print(f"Runs/dim:   {n_runs}")
    print(f"Budget:     {cfg['max_objective_calls']} objective calls\n")

    for dim in dimensions:
        dim_dir = os.path.join(results_dir, name, f"dim_{dim}")
        os.makedirs(dim_dir, exist_ok=True)

        print(f"dim={dim}:")
        for run_idx in range(n_runs):
            seed = seed_start + run_idx
            out_path = os.path.join(dim_dir, f"run_{run_idx:03d}.pkl")

            if os.path.exists(out_path):
                print(f"  run {run_idx:03d} — already exists, skipping.")
                continue

            print(f"  run {run_idx:03d} (seed={seed}) ...", end=" ", flush=True)
            try:
                result = run_single(
                    cfg=cfg,
                    dim=dim,
                    seed=seed,
                    train_x=train_x_dict[dim][run_idx],
                    train_y=train_y_dict[dim][run_idx],
                    lengthscale=lengthscales_dict[dim],
                    f_max=f_max_dict[dim][run_idx],
                )
                with open(out_path, "wb") as fh:
                    pickle.dump(result, fh)
                regret_final = result["regret_per_eval"][-1] if result["regret_per_eval"] else float("nan")
                n_iters = len(result["f_values"])
                print(f"done. iters={n_iters}, final_regret={regret_final:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                raise

        print()

    print(f"Results saved to: {os.path.join(results_dir, name)}")


if __name__ == "__main__":
    main()

# ============================================================
# THESIS EXTENSION — END
# ============================================================
