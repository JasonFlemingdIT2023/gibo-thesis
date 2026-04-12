"""
Posterior evaluation utilities for line search termination criteria.

All computations are analytical evaluations of the GP posterior-->
no additional objective function calls are made here.

Mathematical conventions (maximization):
    theta   : current parameters, shape [1, D]
    p       : normalized search direction, shape [1, D]
    alpha   : scalar step size
    phi(a)  = mu_post(theta + a*p)        - 1D objective along p
    phi'(a) = p^T * mean_d(theta + a*p)    - directional derivative

SE kernel (ARD):
    k(x1, x2) = sigma_f^2 * exp(-sum_d (x1_d - x2_d)^2 / (2 * l_d^2))

Posterior kernel:
    k_post(x1, x2) = k(x1, x2) - k(x1, X) @ K_inv @ k(X, x2)
    where K_inv = (K(X,X) + sigma_n^2 * I)^{-1}
"""

from typing import Dict, Tuple

import torch


# ---------------------------------------------------------------------------
# Private helpers: analytic derivatives of the SE prior kernel
# ---------------------------------------------------------------------------

def _get_K_x1_x2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """k(x1, x2) — prior kernel value, shape [1, 1].

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Kernel value, shape [1, 1].
    """
    return model.covar_module(x1, x2).evaluate()


def _get_K_x_X(model, x: torch.Tensor) -> torch.Tensor:
    """k(x, X_train) — prior kernel between x and all training points, shape [1, N].

    Args:
        model: DerivativeExactGPSEModel.
        x: Shape [1, D].

    Returns:
        Kernel row vector, shape [1, N].
    """
    X = model.train_inputs[0]
    return model.covar_module(x, X).evaluate()


def _get_lengthscale_sq(model) -> torch.Tensor:
    """Squared ARD lengthscales, shape [D].

    For non-ARD (single lengthscale) this still works via broadcasting.
    """
    return model.covar_module.base_kernel.lengthscale.detach().squeeze() ** 2


def _get_dk_dx2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Analytic derivative of k(x1, x2) w.r.t. x2, shape [D].

    For the SE kernel:
        d k(x1, x2) / d x2_d = k(x1, x2) * (x1_d - x2_d) / l_d^2

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Gradient of k w.r.t. x2, shape [D].
    """
    k = _get_K_x1_x2(model, x1, x2).squeeze()          # scalar
    l2 = _get_lengthscale_sq(model)                      # [D]
    diff = (x1 - x2).squeeze()                           # [D]
    return k * diff / l2                                  # [D]


def _get_dk_dx1(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Analytic derivative of k(x1, x2) w.r.t. x1, shape [D].

    For the SE kernel:
        d k(x1, x2) / d x1_d = -k(x1, x2) * (x1_d - x2_d) / l_d^2
                               = -d k(x1, x2) / d x2_d

    This sign follows from d/dx1_d of exp(-(x1_d-x2_d)^2 / 2l^2).
    """
    return -_get_dk_dx2(model, x1, x2)                   # [D]


def _get_d2k_dx1dx2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Analytic mixed second derivative of k(x1, x2) w.r.t. x1 and x2, shape [D, D].

    For the SE kernel:
        d^2 k / (d x1_d  d x2_e) = k(x1,x2) * (delta_{de}/l_d^2
                                    - (x1_d-x2_d)*(x1_e-x2_e) / (l_d^2 * l_e^2))

    In matrix form:
        d^2 k / (d x1  d x2^T) = k(x1,x2) * (diag(1/l^2) - outer(diff/l^2, diff/l^2))

    At x1 = x2 this reduces to k(x,x) * diag(1/l^2) = sigma_f^2 * diag(1/l^2),
    which matches model._get_Kxx_dx2().

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Mixed Hessian of k, shape [D, D].
    """
    k = _get_K_x1_x2(model, x1, x2).squeeze()           # scalar
    l2 = _get_lengthscale_sq(model)                      # [D]
    diff = (x1 - x2).squeeze()                           # [D]
    diff_over_l2 = diff / l2                             # [D]
    return k * (torch.diag(1.0 / l2) - torch.outer(diff_over_l2, diff_over_l2))  # [D, D]


# ---------------------------------------------------------------------------
# Posterior covariance and its analytic derivatives
# ---------------------------------------------------------------------------

def posterior_cov(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Posterior kernel k_post(x1, x2) — scalar.

    k_post(x1, x2) = k(x1, x2) - k(x1, X) @ K_inv @ k(X, x2)

    At x1 = x2 this equals sigma2_post(x1), the posterior variance
    of the latent function (no observation noise).

    Args:
        model: DerivativeExactGPSEModel (prediction_strategy must be initialized).
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Scalar posterior covariance.
    """
    K_inv = model.get_KXX_inv()                          # [N, N]
    k_x1_X = _get_K_x_X(model, x1)                      # [1, N]
    k_X_x2 = _get_K_x_X(model, x2).T                    # [N, 1]
    k_prior = _get_K_x1_x2(model, x1, x2)               # [1, 1]
    return (k_prior - k_x1_X @ K_inv @ k_X_x2).squeeze()  # scalar


def posterior_dcov_dx2(
    model, x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """Derivative of k_post(x1, x2) w.r.t. x2, shape [D].

    d k_post(x1, x2) / d x2
        = d k(x1, x2) / d x2  -  k(x1, X) @ K_inv @ d k(X, x2) / d x2

    The matrix d k(X, x2) / d x2 has shape [N, D] where entry [j, d] is
        d k(X_j, x2) / d x2_d = k(X_j, x2) * (X_j_d - x2_d) / l_d^2

    This is exactly what model._get_KxX_dx(x2) gives after reshaping:
        model._get_KxX_dx(x2) has shape [1, D, N]
        Entry [0, d, j] = -(x2_d - X_j_d)/l_d^2 * k(x2, X_j)
                        = (X_j_d - x2_d)/l_d^2 * k(X_j, x2)   [same by symmetry]
    so d k(X, x2) / d x2  [N, D] = model._get_KxX_dx(x2).squeeze(0).T

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Gradient of posterior covariance w.r.t. x2, shape [D].
    """
    K_inv = model.get_KXX_inv()                          # [N, N]
    k_x1_X = _get_K_x_X(model, x1)                      # [1, N]
    dk_prior = _get_dk_dx2(model, x1, x2)                # [D]

    # d k(X, x2) / d x2: shape [N, D]
    dk_X_x2 = model._get_KxX_dx(x2).squeeze(0).T        # [N, D]

    # alpha = k(x1, X) @ K_inv: shape [1, N]
    alpha_vec = k_x1_X @ K_inv                           # [1, N]

    return dk_prior - (alpha_vec @ dk_X_x2).squeeze(0)   # [D]


def posterior_dcov_dx1(
    model, x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """Derivative of k_post(x1, x2) w.r.t. x1, shape [D].

    d k_post(x1, x2) / d x1
        = d k(x1, x2) / d x1  -  d k(x1, X) / d x1 @ K_inv @ k(X, x2)

    The matrix d k(x1, X) / d x1 has shape [D, N] where entry [d, j] is
        d k(x1, X_j) / d x1_d

    This is exactly model._get_KxX_dx(x1).squeeze(0), shape [D, N].

    Note: By symmetry of k_post,
        posterior_dcov_dx1(model, x0, xa) == posterior_dcov_dx2(model, xa, x0)
    Both formulations are used in different S-term definitions.

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Gradient of posterior covariance w.r.t. x1, shape [D].
    """
    K_inv = model.get_KXX_inv()                          # [N, N]
    k_X_x2 = _get_K_x_X(model, x2).T                    # [N, 1]
    dk_prior = _get_dk_dx1(model, x1, x2)                # [D]

    # d k(x1, X) / d x1: shape [D, N]
    dk_x1_X = model._get_KxX_dx(x1).squeeze(0)          # [D, N]

    return dk_prior - (dk_x1_X @ K_inv @ k_X_x2).squeeze()  # [D]


def posterior_d2cov_dx1dx2(
    model, x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """Mixed second derivative of k_post(x1, x2) w.r.t. x1 and x2, shape [D, D].

    d^2 k_post(x1, x2) / (d x1  d x2^T)
        = d^2 k(x1, x2) / (d x1  d x2^T)
          - d k(x1, X) / d x1 @ K_inv @ d k(X, x2) / d x2

    Shapes:
        d k(x1, X) / d x1 : [D, N]  (from model._get_KxX_dx(x1))
        K_inv              : [N, N]
        d k(X, x2) / d x2 : [N, D]  (from model._get_KxX_dx(x2).T)
        result             : [D, D]

    At x1 = x2 = x this gives the posterior gradient covariance,
    which matches model.posterior_derivative(x)[1].

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Mixed Hessian of posterior covariance, shape [D, D].
    """
    K_inv = model.get_KXX_inv()                          # [N, N]
    d2k_prior = _get_d2k_dx1dx2(model, x1, x2)          # [D, D]

    # d k(x1, X) / d x1: shape [D, N]
    dk_x1_X = model._get_KxX_dx(x1).squeeze(0)          # [D, N]

    # d k(X, x2) / d x2: shape [N, D]
    dk_X_x2 = model._get_KxX_dx(x2).squeeze(0).T        # [N, D]

    return d2k_prior - dk_x1_X @ K_inv @ dk_X_x2        # [D, D]


# ---------------------------------------------------------------------------
# S-terms: all 10 covariance entries for Variant A (probabilistic Wolfe)
# ---------------------------------------------------------------------------

def compute_s_terms(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute all cross-covariance terms for the probabilistic Wolfe conditions.

    The joint GP distribution of z = [f0, f0', fa, fa'] is characterized by
    these 10 scalar covariance entries:

        S11 = Var(f(theta))                    = sigma2_post(theta)
        S22 = Var(p^T grad f(theta))           = p^T variance_d(theta) p
        S33 = Var(f(theta + alpha*p))          = sigma2_post(theta + alpha*p)
        S44 = Var(p^T grad f(theta + alpha*p)) = p^T variance_d(theta+alpha*p) p
        S13 = Cov(f(theta), f(theta+alpha*p))  = k_post(theta, theta+alpha*p)
        S12 = Cov(f(theta), p^T grad f(theta))
            = p^T dk_post(theta, theta)/dx2
        S14 = Cov(f(theta), p^T grad f(theta+alpha*p))
            = p^T dk_post(theta, theta+alpha*p)/dx2
        S23 = Cov(p^T grad f(theta), f(theta+alpha*p))
            = p^T dk_post(theta+alpha*p, theta)/dx2  [= dk_post(theta,xa)/dx1 by symmetry]
        S24 = Cov(p^T grad f(theta), p^T grad f(theta+alpha*p))
            = p^T d2k_post(theta, theta+alpha*p)/(dx1 dx2^T) p
        S34 = Cov(f(theta+alpha*p), p^T grad f(theta+alpha*p))
            = p^T dk_post(theta+alpha*p, theta+alpha*p)/dx2

    Args:
        model: DerivativeExactGPSEModel (prediction_strategy must be initialized
               via model.posterior(theta) before calling this function).
        theta: Current parameters, shape [1, D].
        alpha: Scalar step size candidate.
        p: Normalized search direction, shape [1, D].

    Returns:
        Dictionary mapping 'S11', 'S12', ..., 'S34' to scalar tensors.
    """
    x0 = theta                                   # [1, D]
    xa = theta + alpha * p                       # [1, D]
    p_vec = p.squeeze()                          # [D]

    with torch.no_grad():
        # --Diagonal variances 

        #S11, S33: posterior variance (latent, no noise) at theta and theta+alpha*p
        S11 = model.posterior(x0).mvn.variance.squeeze()   # scalar
        S33 = model.posterior(xa).mvn.variance.squeeze()   #scalar

        # S22: posterior gradient variance projected onto p at theta
        _, var_d_0 = model.posterior_derivative(x0)        # [1, D, D] or [D, D]
        var_d_0 = var_d_0.squeeze()                        # [D, D]
        S22 = p_vec @ var_d_0 @ p_vec                     # scalar

        # S44: posterior gradient variance projected onto p at theta+alpha*p
        _, var_d_a = model.posterior_derivative(xa)        # [D, D]
        var_d_a = var_d_a.squeeze()                        # [D, D]
        S44 = p_vec @ var_d_a @ p_vec                     # scalar

        # --Cross-covariances between function values

        # S13: Cov(f(theta), f(theta+alpha*p))
        S13 = posterior_cov(model, x0, xa)                 # scalar

        # --Cross-covariances between function value and gradient 

        # S12: Cov(f(theta), f'(theta) along p)
        # = p^T d k_post(theta, theta) / d theta'
        S12 = p_vec @ posterior_dcov_dx2(model, x0, x0)   # scalar

        # S14: Cov(f(theta), f'(theta+alpha*p) along p)
        # = p^T d k_post(theta, theta+alpha*p) / d (theta+alpha*p)
        S14 = p_vec @ posterior_dcov_dx2(model, x0, xa)   # scalar

        # S23: Cov(f'(theta) along p, f(theta+alpha*p))
        # = p^T d k_post(theta+alpha*p, theta) / d theta   
        # = p^T d k_post(theta, theta+alpha*p) / d theta   [equiv. by symmetry]
        S23 = p_vec @ posterior_dcov_dx2(model, xa, x0)   # scalar

        # S34: Cov(f(theta+alpha*p), f'(theta+alpha*p) along p)
        # = p^T d k_post(theta+alpha*p, theta+alpha*p) / d (theta+alpha*p)'
        S34 = p_vec @ posterior_dcov_dx2(model, xa, xa)   # scalar

        # - Cross-covariance between gradients 

        # S24: Cov(f'(theta) along p, f'(theta+alpha*p) along p)
        # = p^T d^2 k_post(theta, theta+alpha*p)/(d theta d (theta+alpha*p)^T) p
        d2cov = posterior_d2cov_dx1dx2(model, x0, xa)     # [D, D]
        S24 = p_vec @ d2cov @ p_vec                       # scalar

    return {
        'S11': S11, 'S22': S22, 'S33': S33, 'S44': S44,
        'S13': S13,
        'S12': S12, 'S14': S14, 'S23': S23, 'S24': S24,
        'S34': S34,
    }


# ---------------------------------------------------------------------------
# Main line search utilities (shared by both variants)
# ---------------------------------------------------------------------------

def get_search_direction(
    model, theta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized search direction from posterior gradient mean.

    p = mean_d(theta) / ||mean_d(theta)||

    The search direction is the normalized posterior gradient. Since GIBO
    maximizes, ascending along this direction is the natural choice.

    Args:
        model: DerivativeExactGPSEModel.
        theta: Current parameters, shape [1, D].

    Returns:
        p: Normalized search direction, shape [1, D].
        mean_d: Raw (unnormalized) posterior gradient mean, shape [1, D].
    """
    with torch.no_grad():
        mean_d, _ = model.posterior_derivative(theta)   # [1, D]
        norm = mean_d.norm()
        if norm < 1e-10:
            # Degenerate: gradient is essentially zero -->no meaningful direction
            p = torch.zeros_like(mean_d)
        else:
            p = mean_d / norm                           # [1, D]
    return p, mean_d


def eval_phi(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the 1D surrogate phi(alpha) = mu_post(theta + alpha*p).

    Also computes the directional derivative phi'(alpha) and posterior
    variance sigma2(alpha). All values come from the GP posterior analytically.

    Args:
        model: DerivativeExactGPSEModel.
        theta: Current parameters, shape [1, D].
        alpha: Step size candidate (scalar float).
        p: Normalized search direction, shape [1, D].

    Returns:
        phi:       mu_post(theta + alpha*p), scalar tensor.
        phi_prime: p^T mean_d(theta + alpha*p), scalar tensor.
        sigma2:    sigma2_post(theta + alpha*p), scalar tensor.
    """
    with torch.no_grad():
        x_new = theta + alpha * p                              # [1, D]

        posterior = model.posterior(x_new)
        phi = posterior.mvn.mean.squeeze()                     #scalar
        sigma2 = posterior.mvn.variance.squeeze()              # scalar

        mean_d, _ = model.posterior_derivative(x_new)          # [1, D]
        phi_prime = (p * mean_d).sum()                         # scalar

    return phi, phi_prime, sigma2


def eval_phi_0(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate phi and phi' at alpha = 0 (i.e., at theta itself).

    Equivalent to eval_phi(model, theta, 0.0, p) but named separately
    for clarity — callers often need to cache this baseline value.

    Returns:
        phi_0:       mu_post(theta), scalar tensor.
        phi_prime_0: p^T mean_d(theta), scalar tensor.
        sigma2_0:    sigma2_post(theta), scalar tensor.
    """
    return eval_phi(model, theta, 0.0, p)
