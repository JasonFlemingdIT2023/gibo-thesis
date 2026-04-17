"""
Variant A: Probabilistic Wolfe Conditions for inner loop termination.

Design:
    alpha_candidate = cubic_hermite_max(phi(0), phi'(0), phi(probe), phi'(probe))
    Stop inner loop when p_Wolfe(alpha_candidate) > c_W.

Step size method: cubic Hermite interpolation using two GP posterior
evaluations — at alpha=0 (theta) and at alpha=probe (GI trust region
boundary, typically 0.2). The cubic is fitted with scipy.interpolate
.CubicHermiteSpline and its maximum is found with minimize_scalar on
[0, alpha_max], where alpha_max = 2*l (SE lengthscale).

The cubic extrapolates naturally beyond the probe interval: if phi is
still increasing at the probe, the maximum will lie beyond probe,
yielding larger steps than trust-region-bounded argmax.

Theoretical bound: at distance 2*l the SE prior correlation is
exp(-2) ≈ 13.5% — the GP is essentially extrapolating (Rasmussen &
Williams, 2006, Ch. 4), making 2*l a principled upper bound.

Guard: if phi'(0) <= 0, falls back to lr_fallback (gradient direction
not ascending, occurs near convergence where the GP is unreliable).

Wolfe conditions for maximization:
    W-I  (Armijo):   a_t = phi(alpha) - phi(0) - c1*alpha*phi'(0) >= 0
    W-II (Curvature): b_t = c2*phi'(0) - phi'(alpha) >= 0  (weak form)

References:
    Mahsereci & Hennig (2017), Probabilistic Line Searches for Stochastic
    Optimization. JMLR 18. Signs and kernel adapted for maximization and
    SE posterior (not Wiener process).
    Nocedal & Wright (2006), Numerical Optimization, Sections 3.4-3.5.
"""

# ============================================================
# THESIS EXTENSION — BEGIN
# Description: Full implementation of Variant A (Phase 2)
# ============================================================

from typing import Optional, Tuple

import torch
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm as scipy_norm

from src.line_search.utils import (
    eval_phi,
    eval_phi_0,
    compute_s_terms,
    get_search_direction,
)


# ---------------------------------------------------------------------------
# Step 1: Find alpha_candidate via cubic Hermite interpolation
# ---------------------------------------------------------------------------

def find_alpha_star(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    delta: Optional[float] = None,
    probe: float = 0.2,
    phi_0: Optional[torch.Tensor] = None,
    phi_prime_0: Optional[torch.Tensor] = None,
    lr_fallback: float = 0.25,
) -> float:
    """Find alpha* via cubic Hermite interpolation with probe at trust region boundary.

    Evaluates phi and phi' at alpha=0 and alpha=probe from the GP posterior
    (no real function evaluations). Fits a CubicHermiteSpline through those
    four values and finds its maximum on [0, alpha_max] via minimize_scalar.

    alpha_max = 2 * lengthscale  if delta is None (dynamic, model-derived)
              = delta             otherwise.

    The cubic can extrapolate beyond the probe interval, producing larger
    steps when phi is still increasing at the probe — fixing the systematic
    underestimation from trust-region-bounded argmax mu_post.

    Guard: returns lr_fallback if phi'(0) <= 0 (gradient direction not
    ascending; occurs near convergence where the posterior gradient is noisy).

    Args:
        model: DerivativeExactGPSEModel with current training data.
        theta: Current parameters, shape [1, D].
        p: Normalized search direction, shape [1, D].
        delta: alpha_max (upper clamp). The optimizer always passes 2*l here
               explicitly. If None (e.g. in unit tests), computed from model.
        probe: Second evaluation point for cubic fitting; should equal the GI
               trust region radius (optimizer's self.delta = 0.2).
        phi_0: Pre-computed mu_post(theta) — avoids redundant GP call.
        phi_prime_0: Pre-computed p^T mean_d(theta) — same.
        lr_fallback: Step returned when phi'(0) <= 0 (defensive; the optimizer
                     guards against this before calling, but kept for standalone
                     use in tests and check_prob_wolfe).

    Returns:
        Scalar float alpha* in (0, alpha_max].
    """
    # ============================================================
    # THESIS EXTENSION — BEGIN
    # Description: Replace argmax mu_post (minimize_scalar on posterior mean,
    #   bounded to [0, delta]) with cubic Hermite interpolation that can
    #   extrapolate beyond the GI trust region.
    # --- ORIGINAL GIBO CODE (replaced by thesis extension) ---
    # def neg_phi(alpha: float) -> float:
    #     phi, _, _ = eval_phi(model, theta, alpha, p)
    #     return -phi.item()
    # result = minimize_scalar(
    #     neg_phi,
    #     bounds=(1e-6, delta),
    #     method='bounded',
    #     options={'xatol': 1e-5, 'maxiter': 500},
    # )
    # return float(result.x)
    # --- END ORIGINAL GIBO CODE ---

    # 1. Baseline values at alpha=0 (use cache if provided)
    if phi_0 is None or phi_prime_0 is None:
        phi_0_t, phi_prime_0_t, _ = eval_phi_0(model, theta, p)
    else:
        phi_0_t, phi_prime_0_t = phi_0, phi_prime_0

    phi_prime_0_val = float(phi_prime_0_t)

    # 2. Guard: direction not ascending → fallback
    if phi_prime_0_val <= 0.0:
        return lr_fallback

    # 3. alpha_max: either passed explicitly or 2 * model lengthscale
    if delta is None:
        ls = model.covar_module.base_kernel.lengthscale.mean().item()
        alpha_max = 2.0 * ls
    else:
        alpha_max = float(delta)

    # 4. Probe point must not exceed alpha_max
    a = min(float(probe), alpha_max)
    if a < 1e-8:
        return min(lr_fallback, alpha_max)

    # 5. Evaluate phi and phi' at probe point (GP posterior, no rollout)
    phi_0_val = float(phi_0_t)
    phi_a_t, phi_prime_a_t, _ = eval_phi(model, theta, a, p)

    # 6. Fit cubic Hermite spline through (0, phi_0, phi'_0) and (a, phi_a, phi'_a)
    #    scipy.interpolate.CubicHermiteSpline takes knots, values, and derivatives.
    spline = CubicHermiteSpline(
        x=[0.0, a],
        y=[phi_0_val, float(phi_a_t)],
        dydx=[phi_prime_0_val, float(phi_prime_a_t)],
    )

    # 7. Maximise the spline on (0, alpha_max] via bounded Brent search on -spline
    result = minimize_scalar(
        lambda alpha: -float(spline(alpha)),
        bounds=(1e-8, alpha_max),
        method='bounded',
        options={'xatol': 1e-5, 'maxiter': 500},
    )
    # ============================================================
    # THESIS EXTENSION — END
    # ============================================================

    return float(result.x)


# ---------------------------------------------------------------------------
# Step 2: Compute p_Wolfe(alpha_candidate)
# ---------------------------------------------------------------------------

def compute_p_wolfe(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
    phi_0: torch.Tensor = None,
    phi_prime_0: torch.Tensor = None,
    c1: float = 0.05,
    c2: float = 0.5,
) -> float:
    """Compute p_Wolfe(alpha) = P(W-I holds AND W-II holds | GP posterior).

    The four quantities [f(theta), f'(theta)*p, f(theta+alpha*p), f'(theta+alpha*p)*p]
    are jointly Gaussian under the GP posterior with covariance matrix
    parameterized by the 10 S-terms from compute_s_terms().

    The Wolfe conditions define linear constraints on this joint Gaussian:
        a_t = [-1, -c1*alpha, 1, 0] @ z >= 0   (Armijo)
        b_t = [0, c2, 0, -1]        @ z >= 0   (Curvature, weak form)

    This gives a bivariate normal probability:
        p_Wolfe = P(a_t >= 0, 0 <= b_t <= b_bar)
                = Phi(upb) - Phi(h_b) - Phi2(h_a, upb; rho) + Phi2(h_a, h_b; rho)

    where Phi is the univariate and Phi2 the bivariate standard normal CDF.

    Args:
        model: DerivativeExactGPSEModel.
        theta: Current parameters, shape [1, D].
        alpha: Step size candidate (scalar float).
        p: Normalized search direction, shape [1, D].
        phi_0: Pre-computed mu_post(theta) — avoids redundant call if cached.
        phi_prime_0: Pre-computed p^T mean_d(theta) — same.
        c1: Armijo constant (default 0.05).
        c2: Curvature constant (default 0.5).

    Returns:
        p_Wolfe in [0, 1] as float.
    """
    # --Baseline values at alpha=0 (cached if provided) 
    if phi_0 is None or phi_prime_0 is None:
        phi_0, phi_prime_0, _ = eval_phi_0(model, theta, p)

    # --Values at alpha 
    phi_a, phi_prime_a, _ = eval_phi(model, theta, alpha, p)

    # --All 10 cross-covariance terms 
    s = compute_s_terms(model, theta, alpha, p)

    # --Means of a_t (Armijo) and b_t (curvature) 
    #m_a = phi(alpha) - phi(0) - c1*alpha*phi'(0)
    #m_b = c2*phi'(0) - phi'(alpha)
    m_a = phi_a - phi_0 - c1 * alpha * phi_prime_0
    m_b = c2 * phi_prime_0 - phi_prime_a

    # --Variances and cross-covariance 
    #C_aa = Var(a_t), C_bb = Var(b_t), C_ab = Cov(a_t, b_t)
    C_aa = (
        s['S11'] + s['S33'] - 2 * s['S13']
        + (c1 * alpha) ** 2 * s['S22']
        + 2 * c1 * alpha * s['S12']
        - 2 * c1 * alpha * s['S23']
    )
    C_bb = s['S44'] + c2 ** 2 * s['S22'] - 2 * c2 * s['S24']

    C_ab = (
        -c2 * s['S12']
        + s['S14']
        - c1 * alpha * c2 * s['S22']
        + c1 * alpha * s['S24']
        + c2 * s['S23']
        - s['S34']
    )

    # --Numerical stability
    C_aa = C_aa.clamp(min=1e-10)
    C_bb = C_bb.clamp(min=1e-10)
    sqrt_Caa = C_aa.sqrt()
    sqrt_Cbb = C_bb.sqrt()

    # --Correlation 
    rho = (C_ab / (sqrt_Caa * sqrt_Cbb)).clamp(-1 + 1e-6, 1 - 1e-6)
    rho_val = rho.item()

    # Standardized integration limits 
    # h_a: lower limit for U = (a_t - m_a) / sqrt(C_aa)
    # h_b: lower limit for V = (b_t - m_b) / sqrt(C_bb)
    h_a = (-m_a / sqrt_Caa).item()
    h_b = (-m_b / sqrt_Cbb).item()

    # Upper bound for curvature condition (strong, ~95% confidence):
    # b_bar = 2*c2*(phi'(0) + 2*sqrt(S22))
    #This captures the strong curvature condition |phi'(alpha)| <= c2*|phi'(0)|
    # accounting for uncertainty in the gradient at theta.
    b_bar = 2 * c2 * (phi_prime_0 + 2 * s['S22'].sqrt())
    upb = ((b_bar - m_b) / sqrt_Cbb).item()

    # Guard against empty integration interval
    if upb <= h_b:
        return 0.0
    '''
     Bivariate normal probability 
     p_Wolfe = P(U >= h_a, h_b <= V <= upb)
             = P(U >= h_a, V >= h_b) - P(U >= h_a, V >= upb)
    
     Using Phi2(a, b; rho) = P(U <= a, V <= b):
       P(U >= h_a, V >= h_b) = 1 - Phi(h_a) - Phi(h_b) + Phi2(h_a, h_b; rho)
       P(U >= h_a, V >= upb) = 1 - Phi(h_a) - Phi(upb) + Phi2(h_a, upb; rho)
    
     Difference:
       p_Wolfe = Phi(upb) - Phi(h_b) - Phi2(h_a, upb; rho) + Phi2(h_a, h_b; rho)
    '''
    cov_matrix = [[1.0, rho_val], [rho_val, 1.0]]
    mvn = multivariate_normal(mean=[0.0, 0.0], cov=cov_matrix)

    p_wolfe = (
        scipy_norm.cdf(upb)
        - scipy_norm.cdf(h_b)
        - mvn.cdf([h_a, upb])
        + mvn.cdf([h_a, h_b])
    )

    return float(max(0.0, p_wolfe))


# ---------------------------------------------------------------------------
# Step 3: Combined check for use in the inner loop
# ---------------------------------------------------------------------------

def check_prob_wolfe(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_prime_0: torch.Tensor,
    delta: Optional[float] = None,
    probe: float = 0.2,
    c1: float = 0.05,
    c2: float = 0.5,
    c_W: float = 0.3,
) -> Tuple[bool, float, float]:
    """Perform one probabilistic Wolfe check for the current GP state.

    Computes alpha_candidate via cubic Hermite interpolation, then evaluates
    p_Wolfe at that candidate. Called once per inner loop iteration.

    Args:
        model: DerivativeExactGPSEModel (updated with latest GI sample).
        theta: Current parameters, shape [1, D].
        p: Normalized search direction (fixed for the inner loop), shape [1, D].
        phi_0: Cached mu_post(theta) — computed once before inner loop.
        phi_prime_0: Cached p^T mean_d(theta) — same.
        delta: alpha_max for cubic search. If None, uses 2 * lengthscale.
        probe: Second evaluation point for cubic fitting (GI trust region radius).
        c1: Armijo constant.
        c2: Curvature constant.
        c_W: Wolfe probability threshold for termination.

    Returns:
        wolfe_satisfied: True if p_Wolfe(alpha_candidate) > c_W.
        alpha_candidate: The step size found by cubic Hermite.
        p_wolfe_value: The computed p_Wolfe probability.
    """
    alpha_candidate = find_alpha_star(
        model, theta, p,
        delta=delta, probe=probe,
        phi_0=phi_0, phi_prime_0=phi_prime_0,
    )
    p_wolfe_value = compute_p_wolfe(
        model, theta, alpha_candidate, p,
        phi_0=phi_0, phi_prime_0=phi_prime_0,
        c1=c1, c2=c2,
    )
    wolfe_satisfied = p_wolfe_value > c_W
    return wolfe_satisfied, alpha_candidate, p_wolfe_value

# ============================================================
# THESIS EXTENSION — END
# ============================================================
