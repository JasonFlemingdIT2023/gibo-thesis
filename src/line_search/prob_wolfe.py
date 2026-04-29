"""
Variant A: Probabilistic Wolfe Conditions for inner loop termination.

Design:
    alpha_candidate = argmax_{alpha in (0, delta]} mu_post(theta + alpha*p)
    Stop inner loop when p_Wolfe(alpha_candidate) > c_W.

Wolfe conditions for maximization as random variables:
    W-I  (Armijo):   a_t = phi(alpha) - phi(0) - c1*alpha*phi'(0) >= 0
    W-II (Curvature): b_t = c2*phi'(0) - phi'(alpha) >= 0  (weak form)-->strong form for upper bar.

Reference:
    Mahsereci & Hennig (2017), Probabilistic Line Searches for Stochastic
    Optimization. JMLR 18. 
    Adapted for maximization and
    SE posterior (not Wiener process).
"""
from typing import Tuple

import torch
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm as scipy_norm

from src.line_search.utils import (
    eval_phi,
    eval_phi_0,
    compute_s_terms,
    get_search_direction,
)

def find_alpha_star(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    delta: float = 0.2,
) -> float:
    """Find alpha* = argmax_{alpha in (0, delta]} mu_post(theta + alpha*p).

    Uses scipy.optimize.minimize_scalar (Brent method, bounded) on -phi(alpha).
    All evaluations are analytical GP posterior calls -->no real function
    evaluations.
    Args:
        model: DerivativeExactGPSEModel with current training data.
        theta: Current parameters, shape [1, D].
        p: Normalized search direction, shape [1, D].
        delta: Upper bound for alpha search (local neighbourhood radius).

    Returns:
        Scalar float alpha*.
    """
    def neg_phi(alpha: float) -> float:
        phi, _, _ = eval_phi(model, theta, alpha, p)
        return -phi.item()

    result = minimize_scalar(
        neg_phi,
        bounds=(1e-6, delta),
        method='bounded',
        options={'xatol': 1e-5, 'maxiter': 500},
    )
    return float(result.x)

def compute_p_wolfe(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
    phi_0: torch.Tensor = None,
    phi_prime_0: torch.Tensor = None,
    c1: float = 0.05,
    c2: float = 0.5,
    sigma_floor: float = 0.0,
) -> float:
    """Compute p_Wolfe(alpha) = P(W-I holds and W-II holds | GP posterior).

    The four terms [f(theta), f'(theta)*p, f(theta+alpha*p), f'(theta+alpha*p)*p]
    are jointly Gaussian ditributed under the GP posterior with covariance matrix
    by the 10 S-terms from compute_s_terms().

    The Wolfe conditions define linear constraints on this joint Gaussian:
        a_t = [-1, -c1*alpha, 1, 0] @ z >= 0   (Armijo)
        b_t = [0, c2, 0, -1]        @ z >= 0   (Curvature, weak form)
    -->linear Transformation results in GP
    Both variables are corellated. 

    This gives a bivariate normal probability:
        p_Wolfe = P(a_t >= 0, 0 <= b_t <= b_bar)
                = Phi(upb) - Phi(h_b) - Phi2(h_a, upb; rho) + Phi2(h_a, h_b; rho)

    Phi is the univariate and Phi2 the bivariate standard normal CDF.

    Args:
        model: DerivativeExactGPSEModel.
        theta: Current parameters, shape [1, D].
        alpha: Step size candidate (scalar float).
        p: Normalized search direction, shape [1, D].
        phi_0: Pre- omputed mu_post(theta) --> avoids redundant call if cached.
        phi_prime_0: Pre computed p^T mean_d(theta)
        c1: Armijo constant (default 0.05).
        c2: Curvature constant (default 0.5).
        sigma_floor: Minimum posterior std fraction passed to compute_s_terms.
            Applied to diagonal S-terms (S11, S22, S33, S44) to prevent
            p_Wolfe from collapsing near training data where variance --> 0.
            Default 0.0 preserves original behaviour. Use 0.1 for ei_pwolfe.

    Returns:
        p_Wolfe in [0, 1] as float.
    """
    # Baseline values at alpha=0 
    if phi_0 is None or phi_prime_0 is None:
        phi_0, phi_prime_0, _ = eval_phi_0(model, theta, p)

    phi_a, phi_prime_a, _ = eval_phi(model, theta, alpha, p)

    # cross-covariance terms
    s = compute_s_terms(model, theta, alpha, p, sigma_floor=sigma_floor)

    # Means of a_t (Armijo) and b_t (curvature) 
    # m_a = phi(alpha) - phi(0) - c1*alpha*phi'(0)
    # m_b = c2*phi'(0) - phi'(alpha)
    m_a = phi_a - phi_0 - c1 * alpha * phi_prime_0
    m_b = c2 * phi_prime_0 - phi_prime_a

    # C_aa = Var(a_t), C_bb = Var(b_t), C_ab = Cov(a_t, b_t)
    C_aa = (
        s['S11'] + s['S33'] - 2 * s['S13'] + (c1 * alpha) ** 2 * s['S22']
        + 2 * c1 * alpha * s['S12']- 2 * c1 * alpha * s['S23']
    )
    C_bb = s['S44'] + c2 ** 2 * s['S22'] - 2 * c2 * s['S24']

    C_ab = (
        -c2 * s['S12'] + s['S14']- c1 * alpha * c2 * s['S22']
        + c1 * alpha * s['S24'] + c2 * s['S23'] - s['S34']
    )

    # Numerical stability
    C_aa = C_aa.clamp(min=1e-10)
    C_bb = C_bb.clamp(min=1e-10)
    sqrt_Caa = C_aa.sqrt()
    sqrt_Cbb = C_bb.sqrt()

    # Correlation 
    rho = (C_ab / (sqrt_Caa * sqrt_Cbb)).clamp(-1 + 1e-6, 1 - 1e-6)
    rho_val = rho.item()

    # Standardized limits 
    # h_a: lower limit for U = (a_t - m_a) / sqrt(C_aa)
    # h_b: lower limit for V = (b_t - m_b) / sqrt(C_bb)
    h_a = (-m_a / sqrt_Caa).item()
    h_b = (-m_b / sqrt_Cbb).item()

    # Upper bound for curvature condition (strong, 95% confidence from paper):
    # b_bar = 2*c2*(phi'(0) + 2*sqrt(S22))
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


def check_prob_wolfe(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_prime_0: torch.Tensor,
    delta: float = 0.2,
    c1: float = 0.05,
    c2: float = 0.5,
    c_W: float = 0.3,
) -> Tuple[bool, float, float]:
    """Perform one probabilistic Wolfe check for the current GP state.

    Args:
        model: DerivativeExactGPSEModel.
        theta: Current parameters, shape [1, D].
        p: Normalized search direction, shape [1, D].
        phi_0: Cached mu_post(theta) -->computed once before inner loop.
        phi_prime_0: Cached p^T mean_d(theta) -->computed once before inner loop.
        delta: Search radius for alpha -->aphamax.
        c1: Armijo constant.
        c2: Curvature constant.
        c_W: Wolfe probability threshold for termination.

    Returns:
        wolfe_satisfied: True if p_Wolfe(alpha_candidate) > c_W.
        alpha_candidate: The step size found by argmax mu_post.
        p_wolfe_value: The computed p_Wolfe probability.
    """
    alpha_candidate = find_alpha_star(model, theta, p, delta)
    p_wolfe_value = compute_p_wolfe(
                    model, theta, alpha_candidate, p,
                    phi_0=phi_0, phi_prime_0=phi_prime_0,
                    c1=c1, c2=c2,
                 )
    wolfe_satisfied = p_wolfe_value > c_W
    return wolfe_satisfied, alpha_candidate, p_wolfe_value

