#!/usr/bin/env python3
"""Calibrate the value network for safety-critical deployment.

This script computes the calibration_value_adjustment parameter used by the
safety filter. The adjustment ensures that the predicted values are
conservative (i.e., the robot maintains safety even when the model
over-estimates the safety level).

The calibration uses conformal prediction to compute probabilistic
error bounds with two probability parameters:

    epsilon: The desired failure rate (e.g., 0.05 means at most 5% of
             predictions may be non-conservative)

    beta:    The confidence level for the bound itself. With probability
             at least (1 - beta), the computed error_bound achieves the
             desired epsilon. Default is 1e-12 (extremely high confidence).

The guarantee is: with probability >= (1 - beta), we have

    P(predicted_value - true_value > error_bound) <= epsilon

In other words, the model over-predicts by more than 'error_bound' at most
epsilon fraction of the time. The calibration_adjustment is the negation of
the error_bound:

    calibration_adjustment = -error_bound

Example usage:
    python scripts/calibrate_value_network.py
    python scripts/calibrate_value_network.py --show-plot

The output calibration_value_adjustment should be used in:
    - hardware/configs/default.yaml (filter.calibration_adjustment)
    - scripts/run_sims.py (calibration_value_adjustment variable)
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data.data import ValueDataset
from utils.value_network.models import LiDARValueNN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate value network for safety-critical deployment"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/training/checkpoints/epoch_05000.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--options",
        type=str,
        default="results/training/options.pickle",
        help="Path to training options file",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/environments/validation",
        help="Path to validation environments directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000000,
        help="Number of samples for calibration (more = tighter bounds)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-12,
        help="Confidence level: with prob >= (1-beta), the bound achieves epsilon.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=100,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--rel-radius",
        type=float,
        default=0.1,
        help="Relative state sampling radius (calibrate near ego frame origin)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save calibration plot (optional)",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show calibration plot interactively",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, options_path: str) -> tuple:
    """Load the trained value network model."""
    with open(options_path, "rb") as f:
        options = pickle.load(f)

    model = LiDARValueNN(
        options["input_means"].cuda(),
        options["input_stds"].cuda(),
        options["output_mean"].cuda(),
        options["output_std"].cuda(),
        input_dim=5 + options["num_rays"],
    ).cuda()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, options


def compute_prediction_errors(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_samples: int,
) -> np.ndarray:
    """Compute prediction errors on validation data.

    Returns:
        Sorted array of prediction errors (predicted - true).
        Positive errors mean over-estimation (dangerous).
        Negative errors mean under-estimation (conservative).
    """
    errors = []
    with tqdm(total=num_samples, desc="Computing prediction errors") as pbar:
        while len(errors) < num_samples:
            inputs, values, _ = next(iter(dataloader))
            inputs = torch.flatten(inputs, end_dim=1).float().cuda()
            values = torch.flatten(values, end_dim=1).float().cuda()

            with torch.no_grad():
                pred_values = model.forward(inputs)

            batch_errors = (pred_values - values).cpu().numpy()
            errors.extend(batch_errors)
            pbar.update(len(batch_errors))

    errors = np.array(errors[:num_samples])
    return np.sort(errors)


def compute_error_bound(
    errors: np.ndarray,
    epsilon: float,
    beta: float,
) -> tuple:
    """Compute over-prediction error bound using conformal prediction.

    Computes a probabilistic bound on prediction errors: with probability
    >= (1 - beta), at most epsilon fraction of predictions over-predict
    by more than the returned error_bound.

    Args:
        errors: Sorted array of prediction errors (predicted - true).
        epsilon: Desired failure probability (fraction of non-conservative predictions).
        beta: Confidence level. With probability >= (1 - beta), the computed
              bound achieves the desired epsilon.

    Returns:
        Tuple of (error_bound, epsilons) where error_bound is the conformal bound
        (None if not achievable with given samples). The calibration_adjustment
        for use in config files is the negation: -error_bound.
    """
    n = len(errors)

    # For each position k in sorted errors, compute what epsilon we can guarantee
    # with confidence (1 - beta). This uses the Beta distribution property of
    # order statistics: P(true_(1-eps)_quantile <= X_{(k)}) = Beta.cdf(1-eps; k, n-k+1)
    # Inverting: eps(k) = 1 - Beta.ppf(beta; k, n-k+1)
    #
    # Only compute for indices in [0.9n, 0.999n] which covers eps_list [0.01, 0.1].
    # If eps_list is changed, this range must be updated to match.
    vis_start, vis_end = 0.9, 0.999
    vis_indices = np.arange(int(vis_start * n), int(vis_end * n))
    epsilons = np.full(n, fill_value=np.nan)
    epsilons[vis_indices] = 1 - stats.beta.ppf(beta, vis_indices, n - vis_indices + 1)

    # Find the error corresponding to the desired epsilon
    valid_indices = np.argwhere(epsilons <= epsilon)
    if len(valid_indices) == 0:
        error_bound = None
    else:
        error_bound = errors[valid_indices[0, 0]]

    return error_bound, epsilons


def main():
    args = parse_args()

    print("=" * 60)
    print("Value Network Calibration")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation dir: {args.val_dir}")
    print(f"Num samples: {args.num_samples:,}")
    print(f"Beta: {args.beta}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, options = load_model(args.checkpoint, args.options)

    # Create validation dataset
    print("Creating validation dataset...")
    val_dataset = ValueDataset(
        args.val_dir,
        num_repeats=1,
        num_rays=options["num_rays"],
        num_egos=100,
        ego_radius=options["ego_radius"],
        num_rels=100,
        rel_radius=args.rel_radius,
        use_cupy=options["use_cupy"],
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Compute prediction errors
    print("\nComputing prediction errors...")
    errors = compute_prediction_errors(model, val_dataloader, args.num_samples)

    # Compute and print calibration adjustments for different epsilons
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    eps_list = np.linspace(0.01, 0.1, num=10)
    for eps in eps_list:
        bound, _ = compute_error_bound(errors, eps, args.beta)
        if bound is None:
            print(f"epsilon: {eps:.3f}  error_bound: N/A (insufficient samples)")
        else:
            print(f"epsilon: {eps:.3f}  error_bound: {bound:+.4f}  -->  calibration_adjustment: {-bound:+.4f}")
    print("=" * 60)
    print("\nSet calibration_adjustment (the negated error_bound) in:")
    print("  - hardware/configs/default.yaml (filter.calibration_adjustment)")
    print("  - scripts/run_sims.py (calibration_value_adjustment variable)")

    # Plot calibration curve
    if args.save_plot or args.show_plot:
        n = len(errors)
        vis_start, vis_end = 0.9, 0.999
        vis_indices = np.arange(int(vis_start * n), int(vis_end * n))

        # Compute epsilons for visualization range
        _, all_epsilons = compute_error_bound(errors, 0.05, args.beta)

        plt.figure(figsize=(10, 6))
        plt.plot(errors[vis_indices], all_epsilons[vis_indices], "b-", linewidth=2)
        plt.xlabel("Error Bound (predicted - true) [m]", fontsize=12)
        plt.ylabel("Epsilon (failure probability)", fontsize=12)
        plt.title(
            f"Calibration Curve (n={args.num_samples:,}, beta={args.beta})",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved to: {args.save_plot}")

        if args.show_plot:
            plt.show()


if __name__ == "__main__":
    main()
