"""
Posterior evaluation utilities for line search termination criteria.

Mathematical conventions (maximization):
    theta: current parameters, shape [1, D]
    p: normalized search direction, shape [1, D]
    alpha: scalar step size
    phi(a)  = mu_post(theta + a*p) - 1D objective along p
    phi'(a) = p^T * mean_d(theta + a*p) - directional derivative

SE kernel (ARD):
    k(x1, x2) = sigma_f^2 * exp(-sum_d (x1_d - x2_d)^2 / (2 * l_d^2))

Posterior kernel:
    k_post(x1, x2) = k(x1, x2) - k(x1, X) @ K_inv @ k(X, x2)
    where K_inv = (K(X,X) + sigma_n^2 * I)^-1
"""

from typing import Dict, Tuple
import torch

# Private helpers: analytic derivatives of the SE prior kernel

def _get_K_x1_x2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """k(x1, x2) --> prior kernel value, shape [1, 1].

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Kernel value, shape [1, 1].
    """
    return model.covar_module(x1, x2).evaluate()


def _get_K_x_X(model, x: torch.Tensor) -> torch.Tensor:
    """k(x, X_train) --> kernel between x and all training points, shape [1, N].

    Args:
        model: DerivativeExactGPSEModel.
        x: Shape [1, D].

    Returns:
        Kernel row vector, shape [1, N].
    """
    X = model.train_inputs[0]
    return model.covar_module(x, X).evaluate()


def _get_lengthscale_sq(model) -> torch.Tensor:
    #Squared ARD lengthscales, shape [D].
    
    return model.covar_module.base_kernel.lengthscale.detach().squeeze() ** 2


def _get_dk_dx2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Analytic derivative of k(x1, x2) for x2, shape [D]

    For the SE kernel:
        d k(x1, x2) /d x2_d = k(x1, x2) * (x1_d - x2_d) / l_d^2

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Gradient of for x2, shape [D].
    """
    k = _get_K_x1_x2(model, x1, x2).squeeze()         
    l2 = _get_lengthscale_sq(model)                      
    diff = (x1 - x2).squeeze()                           
    return k * diff / l2                                  


def _get_dk_dx1(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Analytic derivative of k(x1, x2) for x1, shape [D].

    For the SE kernel:
        d k(x1, x2) / d x1_d = -k(x1, x2) * (x1_d - x2_d) / l_d^2
                               = -d k(x1, x2) / d x2_d

    This sign follows from d/dx1_d of exp(-(x1_d-x2_d)^2 / 2l^2).
    """
    return -_get_dk_dx2(model, x1, x2)                  


def _get_d2k_dx1dx2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Analytic mixed second derivative of k(x1, x2) for x1 and x2, shape [D, D].

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
    k = _get_K_x1_x2(model, x1, x2).squeeze()           
    l2 = _get_lengthscale_sq(model)                      
    diff = (x1 - x2).squeeze()                           
    diff_over_l2 = diff / l2                            
    return k * (torch.diag(1.0 / l2) - torch.outer(diff_over_l2, diff_over_l2))


# Posterior covariance and its analytic derivatives

def posterior_cov(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Posterior kernel k_post(x1, x2)
    
    k_post(x1, x2) = k(x1, x2) - k(x1, X) @ K_inv @ k(X, x2)

    At x1 = x2 this equals sigma2_post(x1), the posterior variance
    of the function (no observation noise).

    Args:
        model: DerivativeExactGPSEModel (prediction_strategy must be initialized).
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Scalar posterior covariance.
    """
    K_inv = model.get_KXX_inv()                          
    k_x1_X = _get_K_x_X(model, x1)                      
    k_X_x2 = _get_K_x_X(model, x2).T                    
    k_prior = _get_K_x1_x2(model, x1, x2)               
    return (k_prior - k_x1_X @ K_inv @ k_X_x2).squeeze()


def posterior_dcov_dx2(
    model, x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """Derivative of k_post(x1, x2) for x2, shape [D].

    d k_post(x1, x2) / d x2
        = d k(x1, x2) / d x2  - k(x1, X) @ K_inv @ d k(X, x2) / d x2

    The matrix d k(X, x2) / d x2 has shape [N, D] where entry [j, d] is
        d k(X_j, x2) /d x2_d = k(X_j, x2) * (X_j_d - x2_d) / l_d^2

    This is exactly what model._get_KxX_dx(x2) gives after reshaping:
        model._get_KxX_dx(x2) has shape [1, D, N]
        Entry [0, d, j] = -(x2_d - X_j_d)/l_d^2 * k(x2, X_j)
                        = (X_j_d - x2_d)/l_d^2 * k(X_j, x2) 
    so d k(X, x2) / d x2  [N, D] = model._get_KxX_dx(x2).squeeze(0).T

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Gradient of posterior covariance for x2, shape [D].
    """
    K_inv = model.get_KXX_inv()                         
    k_x1_X = _get_K_x_X(model, x1)                    
    dk_prior = _get_dk_dx2(model, x1, x2)                

    # d k(X, x2) / d x2: shape [N, D]
    dk_X_x2 = model._get_KxX_dx(x2).squeeze(0).T        

    # alpha = k(x1, X) @ K_inv: shape [1, N]
    alpha_vec = k_x1_X @ K_inv                           

    return dk_prior - (alpha_vec @ dk_X_x2).squeeze(0)  


def posterior_dcov_dx1(
    model, x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """Derivative of k_post(x1, x2) forx1, shape [D].

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
    K_inv = model.get_KXX_inv()                       
    k_X_x2 = _get_K_x_X(model, x2).T                   
    dk_prior = _get_dk_dx1(model, x1, x2)                

    # d k(x1, X) / d x1: shape [D, N]
    dk_x1_X = model._get_KxX_dx(x1).squeeze(0)          

    return dk_prior - (dk_x1_X @ K_inv @ k_X_x2).squeeze() 


def posterior_d2cov_dx1dx2(
    model, x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """Mixed second derivative of k_post(x1, x2) for x1 and x2, shape [D, D].

    d^2 k_post(x1, x2) / (d x1  d x2^T)
        = d^2 k(x1, x2) / (d x1  d x2^T)- d k(x1, X) / d x1 @ K_inv @ d k(X, x2) / d x2

    Terms:
        d k(x1, X) / d x1 -->from model._get_KxX_dx(x1)
        K_inv --> from model.get_KXX_inv()             
        d k(X, x2) / d x2 -->from model._get_KxX_dx(x2).T
    

    At x1 = x2 = x this gives the posterior gradient covariance,
    which matches model.posterior_derivative(x)[1].

    Args:
        model: DerivativeExactGPSEModel.
        x1: Shape [1, D].
        x2: Shape [1, D].

    Returns:
        Mixed Hessian of posterior covariance, shape [D, D].
    """
    K_inv = model.get_KXX_inv()                          
    d2k_prior = _get_d2k_dx1dx2(model, x1, x2)          
    
    # d k(x1, X) / d x1: shape [D, N]
    dk_x1_X = model._get_KxX_dx(x1).squeeze(0)          

    # d k(X, x2) / d x2: shape [N, D]
    dk_X_x2 = model._get_KxX_dx(x2).squeeze(0).T        

    return d2k_prior - dk_x1_X @ K_inv @ dk_X_x2        


# S-terms: all 10 covariance entries for Variant A

def compute_s_terms(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
    sigma_floor: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute all cross-covariance terms for the probabilistic Wolfe conditions.

    The joint GP distribution of z = [f0, f0', fa, fa'] is characterized by
    these 10 scalar covariance entries:

        S11 = Var(f(theta)) = sigma2_post(theta)
        S22 = Var(p^T grad f(theta)) = p^T variance_d(theta) p
        S33 = Var(f(theta + alpha*p)) = sigma2_post(theta + alpha*p)
        S44 = Var(p^T grad f(theta + alpha*p)) = p^T variance_d(theta+alpha*p) p
        S13 = Cov(f(theta), f(theta+alpha*p)) = k_post(theta, theta+alpha*p)
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
        model: DerivativeExactGPSEModel
        theta: Current parameters, shape [1, D]
        alpha: Scalar step size candidate.
        p: Normalized search direction, shape [1, D].
        sigma_floor: Minimum posterior std as a fraction of sqrt(outputscale).
            Applied to diagonal S-terms (S11, S22, S33, S44) to prevent
            p_Wolfe collapsing to 0 or 1 near training data where variance --> 0.
            Design choice: simulates Mahsereci & Hennig's Wiener process
            minimum variance, which never reaches zero along the path.

    Returns:
        Dictionary mapping 'S11', 'S12', ..., 'S34' to scalar tensors.
    """
    x0 = theta                                  
    xa = theta + alpha * p                       
    p_vec = p.squeeze()                          

    with torch.no_grad():
        #,Diagonal variances
        S11 = model.posterior(x0).mvn.variance.squeeze()   
        S33 = model.posterior(xa).mvn.variance.squeeze()  

        _, var_d_0 = model.posterior_derivative(x0)       
        var_d_0 = var_d_0.squeeze()
        S22 = p_vec @ var_d_0 @ p_vec                     

        _, var_d_a = model.posterior_derivative(xa)        
        var_d_a = var_d_a.squeeze()
        S44 = p_vec @ var_d_a @ p_vec                    

        '''
         THESIS 
         Description: Variance floor on diagonal S-terms.
          Floor scales with outputscale so it
          is GP-signal-relative. Gradient floor is outputscale/l^2
          (matches SE kernel derivative variance at prior).
          Design: simulates Mahsereci & Hennig's Wiener process which has
          non-zero variance everywhere along the path.
         '''
        if sigma_floor > 0.0:
            outputscale = float(model.covar_module.outputscale.detach())
            floor_var = (sigma_floor ** 2) * outputscale
            ls_mean = model.covar_module.base_kernel.lengthscale.mean().item()
            floor_grad_var = floor_var / (ls_mean ** 2)
            S11 = S11.clamp(min=floor_var)
            S33 = S33.clamp(min=floor_var)
            S22 = S22.clamp(min=floor_grad_var)
            S44 = S44.clamp(min=floor_grad_var)


        # Cross-covariances between function values
        S13 = posterior_cov(model, x0, xa)                

        # Cross-covariances between function value and gradient
        S12 = p_vec @ posterior_dcov_dx2(model, x0, x0)   
        S14 = p_vec @ posterior_dcov_dx2(model, x0, xa)   
        S23 = p_vec @ posterior_dcov_dx2(model, xa, x0)   
        S34 = p_vec @ posterior_dcov_dx2(model, xa, xa)  

        # Cross-covariance between gradients
        d2cov = posterior_d2cov_dx1dx2(model, x0, xa)     
        S24 = p_vec @ d2cov @ p_vec                       

    return {
        'S11': S11, 'S22': S22, 'S33': S33, 'S44': S44,
        'S13': S13,
        'S12': S12, 'S14': S14, 'S23': S23, 'S24': S24,
        'S34': S34,
    }



# Main line search utilities (shared by both variants)

def get_search_direction(
    model, theta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized search direction from posterior gradient mean.

    p = mean_d(theta) / ||mean_d(theta)||

    The search direction is the normalized posterior gradient. Since GIBO
    maximizes, ascending along this direction is the right choice.

    Args:
        model: DerivativeExactGPSEModel.
        theta: Current parameters, shape [1, D].

    Returns:
        p: Normalized search direction, shape [1, D].
        mean_d: Raw (unnormalized) posterior gradient mean, shape [1, D].
    """
    with torch.no_grad():
        mean_d, _ = model.posterior_derivative(theta)   
        norm = mean_d.norm()
        if norm < 1e-10:
            # Degenerate: gradient is zero -->no meaningful direction
            p = torch.zeros_like(mean_d)
        else:
            p = mean_d / norm                           
    return p, mean_d


def eval_phi(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the 1D surrogate phi(alpha) = mu_post(theta + alpha*p).

    Also computes the directional derivative phi'(alpha) and posterior
    variance sigma2(alpha). All values come from the GP posterior.

    Args:
        model: DerivativeExactGPSEModel.
        theta: Current parameters, shape [1, D].
        alpha: Step size candidate.
        p: Normalized search direction, shape [1, D].

    Returns:
        phi:       mu_post(theta + alpha*p), scalar tensor.
        phi_prime: p^T mean_d(theta + alpha*p).
        sigma2:    sigma2_post(theta + alpha*p.
    """
    with torch.no_grad():
        x_new = theta + alpha * p                              

        posterior = model.posterior(x_new)
        phi = posterior.mvn.mean.squeeze()                     
        sigma2 = posterior.mvn.variance.squeeze()              

        mean_d, _ = model.posterior_derivative(x_new)         
        phi_prime = (p * mean_d).sum()                         

    return phi, phi_prime, sigma2


def eval_phi_0(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate phi and phi' at alpha = 0 (at theta itself).

    Equivalent to eval_phi(model, theta, 0.0, p) but named separately
    for clarity

    Returns:
        phi_0:       mu_post(theta), scalar tensor.
        phi_prime_0: p^T mean_d(theta), scalar tensor.
        sigma2_0:    sigma2_post(theta), scalar tensor.
    """
    return eval_phi(model, theta, 0.0, p)



# THESIS ADD ON
# Description: Gradient SNR utility for ei_snr inner loop termination.
def compute_gradient_snr(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_prime_0: torch.Tensor = None,
) -> float:
    """Compute gradient Signal-To-Noise Ratio along search direction p.

    SNR = (p^T mean_d(theta))^2 / (p^T variance_d(theta) p)
        = phi'(0)^2 / S22

    Interpretation:
        SNR >= tau: gradient signal dominates uncertainty --> direction reliable
        SNR <  tau: gradient is noise-dominated --> collect more GI samples

    In the within-model setting, SNR grows as GI samples are added because
    variance_d(theta) shrinks while mean_d(theta) stabilises. (also early satisfied)

    Args:
        model: DerivativeExactGPSEModel with current training data.
        theta: Current parameters, shape [1, D].
        p: Normalized search direction, shape [1, D].
        phi_prime_0: Pre-computed p^T mean_d(theta) as scalar tensor.

    Returns:
        SNR as float. Returns inf if S22 < 1e-12 (gradient perfectly certain).
    """
    with torch.no_grad():
        mean_d, var_d = model.posterior_derivative(theta)  
        p_vec = p.squeeze()                                 
        var_d = var_d.squeeze()                             
        S22 = float(p_vec @ var_d @ p_vec)               

        if S22 < 1e-12:
            return float('inf')

        if phi_prime_0 is not None:
            signal = float(phi_prime_0) ** 2
        else:
            signal = float((p * mean_d).sum()) ** 2

    return signal / S22

