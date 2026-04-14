"""
Generate synthetic GP functions for thesis experiments.

Wrapper around the original generate_data_synthetic_functions logic,
adapted for the thesis data_generation.yaml config format.

Usage:
    python generate_thesis_data.py -c experiments/thesis_experiments/configs/data_generation.yaml

Output (in cfg['out_dir']):
    train_x.pt        -- list of train_x tensors per dimension and objective
    train_y.pt        -- list of train_y tensors per dimension and objective
    lengthscales.pt   -- lengthscale tensor per dimension
    f_max.pt          -- maximum function value per dimension and objective
    argmax.pt         -- argmax location per dimension and objective
"""

# ============================================================
# THESIS EXTENSION — BEGIN
# Description: Data generation script for within-model experiments
# ============================================================

import os
import argparse

import torch
import yaml

from src.synthetic_functions import (
    generate_training_samples,
    get_maxima_objectives,
    get_lengthscales,
    factor_hennig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic GP functions for thesis experiments."
    )
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to data_generation.yaml config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dimensions = cfg["dimensions"]
    num_objectives = cfg["num_objectives"]
    num_samples = cfg["num_samples"]
    noise_variance = cfg["noise_variance"]
    outputscale = cfg["gp_hypers"]["outputscale"]
    factor_lengthscale = cfg["factor_lengthscale"]
    gamma = cfg["gamma"]
    n_max = cfg["n_max"]
    out_dir = cfg["out_dir"]

    os.makedirs(out_dir, exist_ok=True)

    train_x_dict = {}
    train_y_dict = {}
    lengthscales_dict = {}

    print("Generating synthetic GP functions for thesis experiments.")
    print(f"Dimensions: {dimensions}, objectives per dim: {num_objectives}")

    for dim in dimensions:
        print(f"\n  dim={dim} ...", end=" ", flush=True)

        # Sample lengthscale from uniform distribution around the Hennig scale.
        l = get_lengthscales(dim, factor_hennig)
        dist = torch.distributions.Uniform(
            factor_lengthscale * l * (1 - gamma),
            factor_lengthscale * l * (1 + gamma),
        )
        lengthscale = dist.sample((1, dim))

        train_x, train_y = generate_training_samples(
            num_objectives=num_objectives,
            dim=dim,
            num_samples=num_samples,
            gp_hypers={
                "covar_module.base_kernel.lengthscale": lengthscale,
                "covar_module.outputscale": torch.tensor(outputscale),
            },
        )
        train_x_dict[dim] = train_x
        train_y_dict[dim] = train_y
        lengthscales_dict[dim] = lengthscale
        print("done.")

    print("\nComputing maxima (this may take a while) ...")
    f_max_dict, argmax_dict = get_maxima_objectives(
        lengthscales=lengthscales_dict,
        noise_variance=noise_variance,
        train_x=train_x_dict,
        train_y=train_y_dict,
        n_max=n_max,
    )

    torch.save(train_x_dict,    os.path.join(out_dir, "train_x.pt"))
    torch.save(train_y_dict,    os.path.join(out_dir, "train_y.pt"))
    torch.save(lengthscales_dict, os.path.join(out_dir, "lengthscales.pt"))
    torch.save(f_max_dict,      os.path.join(out_dir, "f_max.pt"))
    torch.save(argmax_dict,     os.path.join(out_dir, "argmax.pt"))

    print(f"\nData saved to: {out_dir}")
    for dim in dimensions:
        print(f"  dim={dim}: f_max range [{min(f_max_dict[dim]):.3f}, {max(f_max_dict[dim]):.3f}]")


if __name__ == "__main__":
    main()

# ============================================================
# THESIS EXTENSION — END
# ============================================================
