"""
Variant B: Deterministic Wolfe Conditions + EI step size for inner loop termination.

Design:
    alpha_candidate = argmax_{alpha in (0, delta]} EI(alpha)
    Stop inner loop when strong Wolfe conditions hold on mu_post(alpha_candidate).

The EI acquisition selects the step size that balances expected improvement
with posterior uncertainty. The Wolfe check is then evaluated deterministically
on the posterior mean --> no probabilistic threshold.


EI for maximization (reference eta = mu_post(theta)):
    z(alpha)  = (mu_post(theta + alpha*p) - eta) / sigma_post(theta + alpha*p)
    EI(alpha) = (mu_post(theta + alpha*p) - eta) * Phi(z) + sigma_post * phi_pdf(z)

Wolfe conditions (deterministic, evaluated on mu_post):
    Armijo:         phi(alpha) >= phi(0) + c1 * alpha * phi'(0)
    Strong Curvature: |phi'(alpha)| <= c2 * |phi'(0)|

Reference:
    Jones et al. (1998), Efficient Global Optimization of Expensive Black-Box
    Functions. Journal of Global Optimization.
    Nocedal & Wright (2006), Numerical Optimization, Chapter 3.
    
Design Reason:
    The standard BoTorch botorch.acquisition.analytic.ExpectedImprovement operates over the full D-dimensional input space
    and is designed to select the next observation point. 
    In contrast, the line search requires optimizing EI over a single scalar step size alpha 
    along a fixed search direction p, i.e. EI(alpha) = EI(θ_t + alpha·p). 
    Adapting BoTorch EI to this 1D parametrization would require a non standard coordinate transformation 
    and would obscure the analytical structure of the line search. 
    A direct scalar implementation via scipy.optimize.minimize_scalar is therefore both simpler and more transparent.
"""

# ============================================================
# THESIS EXTENSION — BEGIN
# Description: Full implementation of Variant B
# ============================================================

from typing import Tuple

import torch
from scipy.optimize import minimize_scalar
from scipy.stats import norm as scipy_norm

from src.line_search.utils import (
    eval_phi, eval_phi_0, get_search_direction, compute_gradient_snr,
)
# ============================================================
# THESIS EXTENSION — BEGIN
# Description: Import compute_p_wolfe for ei_pwolfe combined check.
#   No circular import: prob_wolfe imports from utils only, not det_ei.
# ============================================================
from src.line_search.prob_wolfe import compute_p_wolfe
# ============================================================
# THESIS EXTENSION — END
# ============================================================


# ---------------------------------------------------------------------------
# Step 1: Compute EI at a given alpha
# ---------------------------------------------------------------------------

def compute_ei(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
    eta: torch.Tensor,
) -> float:
    """Compute Expected Improvement EI(alpha) for maximization.

    EI measures how much better than eta = mu_post(theta) we expect at
    theta + alpha*p, accounting for posterior uncertainty sigma_post.

    EI(alpha) = (mu - eta) * Phi(z) + sigma * phi_pdf(z)
    where z = (mu - eta) / sigma, mu = mu_post(theta + alpha*p),
    sigma = sqrt(sigma2_post(theta + alpha*p)).

    When sigma -> 0 (GP very confident), EI -> max(mu - eta, 0).
    When mu >> eta, the Phi(z) term dominates (pure exploitation).
    When sigma is large, the phi_pdf(z) term adds exploration.

    Args:
        model: DerivativeExactGPSEModel.
        theta: Current parameters, shape [1, D].
        alpha: Step size candidate (scalar float).
        p: Normalized search direction, shape [1, D].
        eta: Reference value (scalar tensor), typically mu_post(theta).

    Returns:
        EI(alpha) as float, >= 0.
    """
    phi, _, sigma2 = eval_phi(model, theta, alpha, p)
    sigma = sigma2.clamp(min=0.0).sqrt()

    mu_val = phi.item()
    sigma_val = sigma.item()
    eta_val = eta.item()

    if sigma_val < 1e-10:
        #GP fully confident: EI = max(mu - eta, 0)
        return float(max(mu_val - eta_val, 0.0))

    z = (mu_val - eta_val) / sigma_val
    ei = (mu_val - eta_val) * scipy_norm.cdf(z) + sigma_val * scipy_norm.pdf(z)
    return float(max(ei, 0.0))


# ---------------------------------------------------------------------------
# Step 2: Find alpha* = argmax EI(alpha) over (0, delta]
# ---------------------------------------------------------------------------

def find_alpha_star_ei(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    eta: torch.Tensor,
    delta: float = 0.2,
) -> float:
    """Find alpha* = argmax_{alpha in (0, delta]} EI(alpha).

    Uses scipy.optimize.minimize_scalar (Brent method, bounded) on -EI(alpha).
    EI is typically smooth and unimodal for the SE kernel, making this
    1D optimization reliable.

    All evaluations are analytical GP posterior calls --> no real function
    evaluations.

    Args:
        model: DerivativeExactGPSEModel with current training data.
        theta: Current parameters, shape [1, D].
        p: Normalized search direction, shape [1, D].
        eta: Reference value for EI (typically mu_post(theta)).
        delta: Upper bound for alpha search (local neighbourhood radius).

    Returns:
        Scalar float alpha*.
    """
    def neg_ei(alpha: float) -> float:
        return -compute_ei(model, theta, alpha, p, eta)

    result = minimize_scalar(
        neg_ei,
        bounds=(1e-6, delta),
        method='bounded',
        options={'xatol': 1e-5, 'maxiter': 500},
    )
    return float(result.x)


# ---------------------------------------------------------------------------
# Step 3: Deterministic strong Wolfe check on mu_post
# ---------------------------------------------------------------------------

def check_det_wolfe(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_prime_0: torch.Tensor,
    c1: float = 0.05,
    c2: float = 0.5,
) -> Tuple[bool, bool]:
    """Check strong Wolfe conditions deterministically on mu_post.

    Both conditions are evaluated on the posterior mean -->no probability,
    no thresholding. This is the classical line search termination criterion
    applied to the GP surrogate.

    Armijo (sufficient increase):
        phi(alpha) >= phi(0) + c1 * alpha * phi'(0)

    Strong Curvature:
        |phi'(alpha)| <= c2 * |phi'(0)|

    Note: phi'(0) = p^T mean_d(theta) > 0 by construction (p is the
    normalized gradient direction). The absolute value in the curvature
    condition covers both ascent and descent of the surrogate.

    Args:
        model: DerivativeExactGPSEModel.
        theta: Current parameters, shape [1, D].
        alpha: Step size candidate (scalar float).
        p: Normalized search direction, shape [1, D].
        phi_0: Cached mu_post(theta).
        phi_prime_0: Cached p^T mean_d(theta).
        c1: Armijo constant (default 0.05).
        c2: Curvature constant (default 0.5).

    Returns:
        armijo_ok: True if Armijo condition holds.
        curvature_ok: True if strong curvature condition holds.
    """
    phi_a, phi_prime_a, _ = eval_phi(model, theta, alpha, p)

    armijo_ok = bool(
        phi_a >= phi_0 + c1 * alpha * phi_prime_0
    )
    curvature_ok = bool(
        phi_prime_a.abs() <= c2 * phi_prime_0.abs()
    )
    return armijo_ok, curvature_ok


# ---------------------------------------------------------------------------
# Step 4: Combined check for use in the inner loop
# ---------------------------------------------------------------------------

def check_det_wolfe_combined(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_prime_0: torch.Tensor,
    eta: torch.Tensor,
    delta: float = 0.2,
    c1: float = 0.05,
    c2: float = 0.5,
) -> Tuple[bool, float, bool, bool]:
    """Perform one deterministic Wolfe check for the current GP state.

    Computes alpha_candidate via EI maximization, then evaluates strong
    Wolfe conditions at that candidate. Called once per inner loop iteration.

    Args:
        model: DerivativeExactGPSEModel (updated with latest GI sample).
        theta: Current parameters, shape [1, D].
        p: Normalized search direction (fixed for the inner loop).
        phi_0: Cached mu_post(theta).
        phi_prime_0: Cached p^T mean_d(theta).
        eta: Reference value for EI (typically phi_0).
        delta: Search radius for alpha.
        c1: Armijo constant.
        c2: Curvature constant.

    Returns:
        wolfe_satisfied: True if both Armijo and curvature hold.
        alpha_candidate: The EI-optimal step size.
        armijo_ok: Individual Armijo result.
        curvature_ok: Individual curvature result.
    """
    alpha_candidate = find_alpha_star_ei(model, theta, p, eta, delta)
    armijo_ok, curvature_ok = check_det_wolfe(
        model, theta, alpha_candidate, p,
        phi_0=phi_0, phi_prime_0=phi_prime_0,
        c1=c1, c2=c2,
    )
    wolfe_satisfied = armijo_ok and curvature_ok
    return wolfe_satisfied, alpha_candidate, armijo_ok, curvature_ok


# ---------------------------------------------------------------------------
# Step 5: EI step size + Probabilistic Wolfe termination (ei_pwolfe)
# ---------------------------------------------------------------------------

def check_ei_pwolfe(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_prime_0: torch.Tensor,
    eta: torch.Tensor,
    delta: float = 0.2,
    c1: float = 0.05,
    c2: float = 0.5,
    c_W: float = 0.3,
    sigma_floor: float = 0.1,
) -> Tuple[bool, float, float, bool, bool]:
    """EI step size + probabilistic Wolfe termination with sigma floor.

    Combines the EI step selector from det_ei with the uncertainty aware
    stopping criterion from prob_wolfe. The sigma_floor prevents p_Wolfe
    trivially collapsing to 0 near training data (simulates Mahsereci &
    Hennig's Wiener process which has non-zero variance everywhere).

    Step selection:   alpha* = argmax_{alpha in (0, delta]} EI(alpha)
    Termination:      stop when p_Wolfe(alpha*) > c_W

    armijo_ok and curvature_ok are logged but NOT used for termination.

    Returns:
        wolfe_satisfied: True if p_Wolfe(alpha*) > c_W.
        alpha_candidate: EI-optimal step size.
        p_wolfe_value: Computed p_Wolfe probability.
        armijo_ok: Deterministic Armijo on mu_post (logging only).
        curvature_ok: Deterministic curvature on mu_post (logging only).
    """
    # ============================================================
    # THESIS EXTENSION — BEGIN
    # Description: Combined EI step + probabilistic Wolfe termination
    # ============================================================
    alpha_candidate = find_alpha_star_ei(model, theta, p, eta, delta=delta)

    p_wolfe_value = compute_p_wolfe(
        model, theta, alpha_candidate, p,
        phi_0=phi_0, phi_prime_0=phi_prime_0,
        c1=c1, c2=c2, sigma_floor=sigma_floor,
    )

    armijo_ok, curvature_ok = check_det_wolfe(
        model, theta, alpha_candidate, p,
        phi_0=phi_0, phi_prime_0=phi_prime_0,
        c1=c1, c2=c2,
    )

    wolfe_satisfied = p_wolfe_value > c_W
    return wolfe_satisfied, alpha_candidate, p_wolfe_value, armijo_ok, curvature_ok
    # ============================================================
    # THESIS EXTENSION — END
    # ============================================================


# ---------------------------------------------------------------------------
# Step 6: EI step size + Gradient SNR termination (ei_snr)
# ---------------------------------------------------------------------------

def check_ei_snr(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_prime_0: torch.Tensor,
    eta: torch.Tensor,
    delta: float = 0.2,
    c1: float = 0.05,
    c2: float = 0.5,
    tau_snr: float = 1.0,
) -> Tuple[bool, float, float, bool, bool]:
    """EI step size + gradient SNR termination.

    Uses EI to select alpha*, then terminates based on gradient confidence:

        SNR = (p^T mean_d(theta))^2 / (p^T variance_d(theta) p)

    Termination: stop when SNR >= tau_snr
        SNR >= tau : signal dominates noise --> direction reliable → stop
        SNR <  tau : noise-dominated --> more GI samples needed

    armijo_ok and curvature_ok are logged but NOT used for termination.
    tau_snr=1.0: signal equals one noise std --> moderate confidence threshold.

    Returns:
        snr_satisfied: True if gradient_snr(theta) >= tau_snr.
        alpha_candidate: EI-optimal step size.
        snr_value: Computed gradient SNR (float, possibly inf).
        armijo_ok: Deterministic Armijo on mu_post (logging only).
        curvature_ok: Deterministic curvature on mu_post (logging only).
    """
    # ============================================================
    # THESIS EXTENSION — BEGIN
    # Description: Combined EI step + gradient SNR termination
    # ============================================================
    alpha_candidate = find_alpha_star_ei(model, theta, p, eta, delta=delta)

    snr_value = compute_gradient_snr(model, theta, p, phi_prime_0=phi_prime_0)

    armijo_ok, curvature_ok = check_det_wolfe(
        model, theta, alpha_candidate, p,
        phi_0=phi_0, phi_prime_0=phi_prime_0,
        c1=c1, c2=c2,
    )

    snr_satisfied = snr_value >= tau_snr
    return snr_satisfied, alpha_candidate, snr_value, armijo_ok, curvature_ok
    # ============================================================
    # THESIS EXTENSION — END
    # ============================================================


