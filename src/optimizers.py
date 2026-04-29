from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List

from abc import ABC, abstractmethod

import numpy as np
import torch
import gpytorch
import botorch
from botorch.models import SingleTaskGP

from src.model import DerivativeExactGPSEModel
from src.environment_api import EnvironmentObjective
from src.acquisition_function import GradientInformation
from src.model import ExactGPSEModel, DerivativeExactGPSEModel

# ============================================================
# THESIS EXTENSION — BEGIN
# Description: Imports for adaptive inner loop termination variants
# ============================================================
from src.line_search.utils import get_search_direction, eval_phi_0
from src.line_search.prob_wolfe import find_alpha_star, compute_p_wolfe
from src.line_search.det_ei import (
    find_alpha_star_ei, check_det_wolfe,
    check_ei_pwolfe, check_ei_snr,
)
# ============================================================
# THESIS EXTENSION — END
# ============================================================


class AbstractOptimizer(ABC):
    """Abstract optimizer class.

    Sets a default optimizer interface.

    Attributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        param_args_ignore: Which parameters should not be optimized.
        optimizer_config: Configuration file for the optimizer.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        param_args_ignore: List[int] = None,
        **optimizer_config: Dict,
    ):
        """Inits the abstract optimizer."""
        # Optionally add batchsize to parameters.
        if len(params_init.shape) == 1:
            params_init = params_init.reshape(1, -1)
        self.params = params_init.clone()
        self.param_args_ignore = param_args_ignore
        self.objective = objective

    def __call__(self):
        """Call method of optimizers."""
        self.step()

    @abstractmethod
    def step(self) -> None:
        """One parameter update step."""
        pass


class RandomSearch(AbstractOptimizer):
    """Implementation of (augmented) random search.

    Method of the nips paper 'Simple random search of static linear policies is
    competitive for reinforcement learning'.

    Attributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        step_size: Step-size for parameter update, named alpha in the paper.
        samples_per_iteration: Number of random symmetric samples before
            parameter update, named N in paper.
        exploration_noise: Exploration distance from current parameters, nu in
            paper.
        standard_deviation_scaling: Scaling of the step-size with standard
            deviation of collected rewards, sigma_R in paper.
        num_top_directions: Number of directions that result in the largest
            rewards, b in paper.
        verbose: If True an output is logged.
        param_args_ignore: Which parameters should not be optimized.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        step_size: float,
        samples_per_iteration: int,
        exploration_noise: float,
        standard_deviation_scaling: bool = False,
        num_top_directions: Optional[int] = None,
        verbose: bool = True,
        param_args_ignore: List[int] = None,
    ):
        """Inits random search optimizer."""
        super(RandomSearch, self).__init__(params_init, objective, param_args_ignore)

        self.params_history_list = [self.params.clone()]
        self.step_size = step_size
        self.samples_per_iteration = samples_per_iteration
        self.exploration_noise = exploration_noise
        self._deltas = torch.empty(self.samples_per_iteration, self.params.shape[-1])

        # For augmented random search V1 and V2.
        self.standard_deviation_scaling = standard_deviation_scaling

        # For augmented random search V1-t and V2-t.
        if num_top_directions is None:
            num_top_directions = self.samples_per_iteration
        self.num_top_directions = num_top_directions

        self.verbose = verbose

    def step(self):
        # 1. Sample deltas.
        torch.randn(*self._deltas.shape, out=self._deltas)
        if self.param_args_ignore is not None:
            self._deltas[:, self.param_args_ignore] = 0.0
        # 2. Scale deltas.
        perturbations = self.exploration_noise * self._deltas
        # 3. Compute rewards
        rewards_plus = torch.tensor(
            [
                self.objective(self.params + perturbation)
                for perturbation in perturbations
            ]
        )
        rewards_minus = torch.tensor(
            [
                self.objective(self.params - perturbation)
                for perturbation in perturbations
            ]
        )
        if self.num_top_directions < self.samples_per_iteration:
            # 4. Using top performing directions.
            args_sorted = torch.argsort(
                torch.max(rewards_plus, rewards_minus), descending=True
            )
            args_relevant = args_sorted[: self.num_top_directions]
        else:
            args_relevant = slice(0, self.num_top_directions)
        if self.standard_deviation_scaling is not None:
            # 5. Perform standard deviation scaling.
            std_reward = torch.cat(
                [rewards_plus[args_relevant], rewards_minus[args_relevant]]
            ).std()
        else:
            std_reward = 1.0

        # 6. Update parameters.
        self.params.add_(
            (rewards_plus[args_relevant] - rewards_minus[args_relevant])
            @ self._deltas[args_relevant],
            alpha=self.step_size / (self.num_top_directions * std_reward),
        )

        # 7. Save new parameters.
        if (type(self.objective._func) is EnvironmentObjective) and (
            self.objective._func._manipulate_state is not None
        ):
            self.params_history_list.append(
                self.objective._func._manipulate_state.unnormalize_params(self.params)
            )
            # 8. Perform state normalization update.
            self.objective._func._manipulate_state.apply_update()
        else:
            self.params_history_list.append(self.params.clone())

        if self.verbose:
            print(f"Parameter {self.params.numpy()}.")
            print(
                f"Mean of (b) perturbation rewards {torch.mean(torch.cat([rewards_plus[args_relevant], rewards_minus[args_relevant]])) :.2f}."
            )
            if self.standard_deviation_scaling:
                print(f"Std of perturbation rewards {std_reward:.2f}.")


class CMAES(AbstractOptimizer):
    """CMA-ES: Evolutionary Strategy with Covariance Matrix Adaptation for
    nonlinear function optimization.

    Inspired by the matlab code of https://arxiv.org/abs/1604.00772.
    Hence this function does not implement negative weights, that is, w_i = 0 for i > mu.

    Attributes:
        params_init: Objective parameters initial value.
        objective: Objective function.
        sigma: Coordinate wise standard deviation (step-size).
        maximization: True if objective function is maximized, False if minimized.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        sigma: float = 0.5,
        maximization: bool = True,
        verbose: bool = True,
    ):
        """Inits CMA-ES optimizer."""
        super(CMAES, self).__init__(params_init, objective)

        self.params_history_list = [self.params.clone()]
        self.dim = self.params.shape[-1]
        
        self.xmean = self.params.clone().view(-1)
        self.maximization = maximization
        self.sigma = sigma

        # Strategy parameter setting: Selection.
        self.lambda_ = 4 + int(
            np.floor(3 * np.log(self.dim))
        )  # Population size, offspring number.
        self.mu = self.lambda_ // 2  # Number of parents/points for recombination.
        weights = np.log(self.mu + 0.5) - np.log(range(1, self.mu + 1))
        self.weights = torch.tensor(
            weights / sum(weights), dtype=torch.float32
        )  # Normalize recombination weights array.
        self.mueff = sum(self.weights) ** 2 / sum(
            self.weights ** 2
        )  # Variance-effective size of mu.

        # Strategy parameter setting: Adaption.
        self.cc = (4 + self.mueff / self.dim) / (
            self.dim + 4 + 2 * self.mueff / self.dim
        )  # Time constant for cumulation for C.
        self.cs = (self.mueff + 2) / (
            self.dim + self.mueff + 5
        )  # Time constant for cumulation for sigma-/step size control.
        self.c1 = 2 / (
            (self.dim + 1.3) ** 2 + self.mueff
        )  # Learning rate for rank-one update of C.
        self.cmu = (
            2
            * (self.mueff - 2 + 1 / self.mueff)
            / ((self.dim + 2) ** 2 + 2 * self.mueff / 2)
        )  # Learning rate for rank-mu update.
        self.damps = (
            1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        )  # Damping for sigma.

        # Initialize dynamic (internal) strategy parameters and constant.
        self.ps = torch.zeros(self.dim)  # Evolution path for sigma.
        self.pc = torch.zeros(self.dim)  # Evolution path for C.
        self.B = torch.eye(self.dim)
        self.D = torch.eye(
            self.dim
        )  # Eigendecomposition of C (pos. def.): B defines the coordinate system, diagonal matrix D the scaling.
        self.C = self.B @ self.D ** 2 @ self.D.transpose(0, 1)  # Covariance matrix.
        self.eigeneval = 0  # B and D updated at counteval == 0
        self.chiN = self.dim ** 0.5 * (
            1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2)
        )  # Expectation of ||N(0,I)|| == norm(randn(N,1))

        # Generation Loop.
        self.arz = torch.empty((self.dim, self.lambda_))
        self.arx = torch.empty((self.dim, self.lambda_))
        self.arfitness = torch.empty((self.lambda_))
        self.counteval = 0
        self.hs = 0

        self.verbose = verbose

    def step(self):

        # 1. Sampling and evaluating.
        for k in range(self.lambda_):
            # Reparameterization trick for samples.
            self.arz[:, k] = torch.randn(
                (self.dim)
            )  # Standard normally distributed vector.
            self.arx[:, k] = (
                self.xmean + self.sigma * self.B @ self.D @ self.arz[:, k]
            )  # Add mutation.
            self.arfitness[k] = self.objective(self.arx[:, k].unsqueeze(0))
            self.counteval += 1

        # 2. Sort solutions.
        args = torch.argsort(self.arfitness, descending=self.maximization)

        # 3. Update mean.
        self.xmean = self.arx[:, args[: self.mu]] @ self.weights  # Recombination.
        zmean = (
            self.arz[:, args[: self.mu]] @ self.weights
        )  # == D.inverse() @ B.transpose(0,1) * (xmean-xold)/sigma

        # 4. Update evolution paths.
        self.ps = (1 - self.cs) * self.ps + (
            np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        ) * (self.B @ zmean)

        if np.linalg.norm(self.ps) / (
            np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lambda_))
        ) < (1.4 + 2 / (self.dim + 1)):
            self.hs = 1

        self.pc = (1 - self.cc) * self.pc + self.hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * self.B @ self.D @ zmean

        # 5. Update covariance matrix.
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1
            * (
                self.pc.view(-1, 1) @ self.pc.view(-1, 1).transpose(0, 1)
                + (1 - self.hs) * self.cc * (2 - self.cc) * self.C
            )
            + self.cmu
            * (self.B @ self.D @ self.arz[:, args[: self.mu]])
            @ torch.diag(self.weights)
            @ (self.B @ self.D @ self.arz[:, args[: self.mu]]).transpose(0, 1)
        )

        # 6. Update step-size sigma.
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )

        # 7. Update B and D from C.
        if (
            self.counteval - self.eigeneval
            > self.lambda_ / (self.c1 + self.cmu) / self.dim / 10
        ):
            self.eigeneval = self.counteval
            self.C = torch.triu(self.C) + torch.triu(self.C, diagonal=1).transpose(
                0, 1
            )  # Enforce symmetry.
            D, self.B = torch.symeig(
                self.C, eigenvectors=True
            )  # Eigendecomposition, B == normalized eigenvectors.
            self.D = torch.diag(
                torch.sqrt(D.clamp_min(1e-20))
            )  # D contains standard deviations now.

        # Escape flat fitness, or better terminate?
        if self.arfitness[0] == self.arfitness[int(np.ceil(0.7 * self.lambda_)) - 1]:
            self.sigma *= np.exp(0.2 + self.cs / self.damps)

        self.params = self.arx[:, args[0]].view(
            1, -1
        )  # Return the best point of the last generation. Notice that xmean is expected to be even better.

        self.params_history_list.append(self.params.clone())

        if self.verbose:
            print(f"Parameter: {self.params.numpy()}.")
            print(f"Function value: {self.arfitness[args[0]]}.")
            print(f"Sigma: {self.sigma}.")


class VanillaBayesianOptimization(AbstractOptimizer):
    """Optimizer class for vanilla Bayesian optimization.

    Vanilla stands for the usage of a classic acquisition function like
    expected improvement.

    Atrributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        Model: Gaussian process model.
        model_config: Configuration dictionary for model.
        hyperparameter_config: Configuration dictionary for hyperparameters of
            Gaussian process model.
        acquisition_function: BoTorch acquisition function.
        acqf_config: Configuration dictionary acquisition function.
        optimize_acqf: Function that optimizes the acquisition function.
        optimize_acqf_config: Configuration dictionary for optimization of
            acquisition function.
        generate_initial_data: Function to generate initial data for Gaussian
            process model.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Callable[[torch.Tensor], torch.Tensor],
        Model,
        model_config: Dict,
        hyperparameter_config: Optional[Dict],
        acquisition_function,
        acqf_config: Dict,
        optimize_acqf: Callable,
        optimize_acqf_config: Dict[str, torch.Tensor],
        generate_initial_data=Optional[
            Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor]
        ],
        verbose=True,
    ):
        """Inits the vanilla BO optimizer."""
        super(VanillaBayesianOptimization, self).__init__(params_init, objective)

        # Parameter initialization.
        self.params_history_list = [self.params.clone()]
        self.D = self.params.shape[-1]

        # Initialization of training data.
        if generate_initial_data is None:
            train_x_init, train_y_init = torch.empty(0, self.D), torch.empty(0, 1)
        else:
            train_x_init, train_y_init = generate_initial_data(self.objective)

        # Add initialization parameter to training data.
        train_x_init = torch.cat([train_x_init, self.params])
        train_y_init = torch.cat(
            [train_y_init, self.objective(self.params).reshape(-1, 1)]
        )

        # Model initialization and optional hyperparameter settings.
        self.model = Model(train_x_init, train_y_init, **model_config)
        if hyperparameter_config["hypers"]:
            self.model.initialize(**hyperparameter_config["hypers"])
        if hyperparameter_config["no_noise_optimization"]:
            # Switch off the optimization of the noise parameter.
            self.model.likelihood.noise_covar.raw_noise.requires_grad = False

        self.optimize_hyperparamters = hyperparameter_config["optimize_hyperparameters"]

        # Acquistion function and its optimization properties.
        self.acquisition_function = acquisition_function
        self.acqf_config = acqf_config
        self.optimize_acqf = optimize_acqf
        self.optimize_acqf_config = optimize_acqf_config

        self.verbose = verbose

    def step(self) -> None:
        # Optionally optimize hyperparameters.
        if self.optimize_hyperparamters and self.model.train_targets.shape[0] > 20:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            )
            botorch.fit.fit_gpytorch_model(mll)

        # Optionally update best_f for acquistion function.
        if "best_f" in self.acqf_config.keys():
            self.acqf_config["best_f"] = self.model.train_targets.max()

        # Optimize acquistion function and get new observation.
        new_x = self.optimize_acqf(
            self.acquisition_function(self.model, **self.acqf_config),
            **self.optimize_acqf_config,
        )
        new_y = self.objective(new_x)
        self.params = new_x.clone()

        # Update training points.
        train_x = torch.cat([self.model.train_inputs[0], new_x])
        train_y = torch.cat([self.model.train_targets, new_y])
        self.model.set_train_data(inputs=train_x, targets=train_y, strict=False)

        self.params_history_list.append(self.params)

        if self.verbose:
            posterior = self.model.posterior(self.params)
            print(
                f"Parameter {self.params.numpy()} with mean {posterior.mvn.mean.item(): .2f} and variance {posterior.mvn.variance.item(): .2f} of the posterior of the GP model."
            )
            print(
                f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.detach().numpy()}, outputscale: {self.model.covar_module.outputscale.detach().numpy(): .2f},  noise {self.model.likelihood.noise.detach().numpy()}"
            )


class BayesianGradientAscent(AbstractOptimizer):
    """Optimizer for Bayesian gradient ascent.

    Also called gradient informative Bayesian optimization (GIBO).

    Attributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        max_samples_per_iteration: Maximum number of samples that are supplied
            by acquisition function before updating the parameters.
        OptimizerTorch: Torch optimizer to update parameters, e.g. SGD or Adam.
        optimizer_torch_config: Configuration dictionary for torch optimizer.
        lr_schedular: Optional learning rate schedular, mapping iterations to
            learning rates.
        Model: Gaussian process model, has to supply Jacobian information.
        model_config: Configuration dictionary for the Gaussian process model.
        hyperparameter_config: Configuration dictionary for hyperparameters of
            Gaussian process model.
        optimize_acqf: Function that optimizes the acquisition function.
        optimize_acqf_config: Configuration dictionary for optimization of
            acquisition function.
        bounds: Search bounds for optimization of acquisition function.
        delta: Defines search bounds for optimization of acquisition function
            indirectly by defining it within a distance of delta from the
            current parameter constellation.
        epsilon_diff_acq_value: Difference between acquisition values. Sampling
            of new data points with acquisition function stops when threshold of
            this epsilon value is reached.
        generate_initial_data: Function to generate initial data for Gaussian
            process model.
        normalize_gradient: Algorithmic extension, normalize the gradient
            estimate with its L2 norm and scale the remaining gradient direction
            with the trace of the lengthscale matrix.
        standard_deviation_scaling: Scale gradient with its variance, inspired
            by an augmentation of random search.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor], EnvironmentObjective],
        max_samples_per_iteration: int,
        OptimizerTorch: torch.optim.Optimizer,
        optimizer_torch_config: Optional[Dict],
        lr_schedular: Optional[Dict[int, int]],
        Model: DerivativeExactGPSEModel,
        model_config: Optional[
            Dict[
                str,
                Union[int, float, torch.nn.Module, gpytorch.priors.Prior],
            ]
        ],
        hyperparameter_config: Optional[Dict[str, bool]],
        optimize_acqf: Callable[[GradientInformation, torch.Tensor], torch.Tensor],
        optimize_acqf_config: Dict[str, Union[torch.Tensor, int, float]],
        bounds: Optional[torch.Tensor],
        delta: Optional[Union[int, float]],
        epsilon_diff_acq_value: Optional[Union[int, float]],
        generate_initial_data: Optional[
            Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor]
        ],
        normalize_gradient: bool = False,
        standard_deviation_scaling: bool = False,
        verbose: bool = True,
        
        # THESIS
        # Description: Parameters for adaptive inner loop termination
        inner_loop_mode: str = 'original',
        c1: float = 0.05,
        c2: float = 0.5,
        c_W: float = 0.3,
        alpha_max: Optional[float] = None,   #separate from GI delta; see det_ei branch
        min_samples_per_iteration: int = 1,
        sigma_floor: float = 0.1,            #ei_pwolfe: variance floor for p_Wolfe
        tau_snr: float = 1.0,                #ei_snr: gradient SNR threshold
        #THESIS EXTENSION END
    ) -> None:
        """Inits optimizer Bayesian gradient ascent."""
        super(BayesianGradientAscent, self).__init__(params_init, objective)

        self.normalize_gradient = normalize_gradient
        self.standard_deviation_scaling = standard_deviation_scaling

        # Parameter initialization.
        self.params_history_list = [self.params.clone()]
        self.params.grad = torch.zeros_like(self.params)
        self.D = self.params.shape[-1]

        # Torch optimizer initialization.
        self.optimizer_torch = OptimizerTorch([self.params], **optimizer_torch_config)
        self.lr_schedular = lr_schedular
        self.iteration = 0

        # Gradient certainty.
        self.epsilon_diff_acq_value = epsilon_diff_acq_value

        # Model initialization and optional hyperparameter settings.
        if (
            hasattr(self.objective._func, "_manipulate_state")
            and self.objective._func._manipulate_state is not None
        ):
            normalize = self.objective._func._manipulate_state.normalize_params
            unnormalize = self.objective._func._manipulate_state.unnormalize_params
        else:
            normalize = unnormalize = None
        self.model = Model(self.D, normalize, unnormalize, **model_config)
        # Initialization of training data.
        if generate_initial_data is not None:
            train_x_init, train_y_init = generate_initial_data(self.objective)
            self.model.append_train_data(train_x_init, train_y_init)

        if hyperparameter_config["hypers"]:
            hypers = dict(
                filter(
                    lambda item: item[1] is not None,
                    hyperparameter_config["hypers"].items(),
                )
            )
            self.model.initialize(**hypers)
        if hyperparameter_config["no_noise_optimization"]:
            # Switch off the optimization of the noise parameter.
            self.model.likelihood.noise_covar.raw_noise.requires_grad = False

        self.optimize_hyperparamters = hyperparameter_config["optimize_hyperparameters"]

        # Acquistion function and its optimization properties.
        self.acquisition_fcn = GradientInformation(self.model)
        self.optimize_acqf = lambda acqf, bounds: optimize_acqf(
            acqf, bounds, **optimize_acqf_config
        )
        self.bounds = bounds
        self.delta = delta
        self.update_bounds = self.bounds is None

        self.max_samples_per_iteration = max_samples_per_iteration
        self.verbose = verbose
        # THESIS 
        # Description: adaptive termination mode and hyperparameters.
        self.inner_loop_mode = inner_loop_mode
        self.c1 = c1
        self.c2 = c2
        self.c_W = c_W
        self.alpha_max = alpha_max #upper bound for line search. If None falls back to delta.
        self.min_samples_per_iteration = min_samples_per_iteration
        self.sigma_floor = sigma_floor
        self.tau_snr = tau_snr
        # Metrics from last step() call --> read by thesis experiment runner.
        self.last_step_info: dict = {}
        # THESIS EXTENSION END

    def step(self) -> None:
        # Sample with new params from objective and add this to train data.
        # Optionally forget old points (if N > N_max).
        f_params = self.objective(self.params)
        if self.verbose:
            print(f"Reward of parameters theta_(t-1): {f_params.item():.2f}.")
        self.model.append_train_data(self.params, f_params)

        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
            and self.objective._func._manipulate_state.apply_update() is not None
        ):
            self.objective._func._manipulate_state.apply_update()

        self.model.posterior(
            self.params
        )  # Call this to update prediction strategy of GPyTorch (get_L_lower, get_K_XX_inv)

        self.acquisition_fcn.update_theta_i(self.params)
        # Stay local around current parameters.
        if self.update_bounds:
            self.bounds = torch.tensor([[-self.delta], [self.delta]]) + self.params
        # Only optimize model hyperparameters if N >= N_max.
        if self.optimize_hyperparamters and (
            self.model.N >= self.model.N_max
        ):  # Adjust hyperparameters
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            )
            botorch.fit.fit_gpytorch_model(mll)
            self.model.posterior(
                self.params
            )  # Call this to update prediction strategy of GPyTorch.

        # THESIS EXTENSION
        # Description: Branch on inner_loop_mode for adaptive termination.
        if self.inner_loop_mode == 'original':
            #ORIGINAL GIBO CODE (unchanged) 
            acq_value_old = None
            _n_inner = 0 #change
            for i in range(self.max_samples_per_iteration):
                # Optimize acquistion function and get new observation.
                new_x, acq_value = self.optimize_acqf(self.acquisition_fcn, self.bounds)
                new_y = self.objective(new_x)

                # Update training points.
                self.model.append_train_data(new_x, new_y)

                if (
                    type(self.objective._func) is EnvironmentObjective
                    and self.objective._func._manipulate_state is not None
                    and self.objective._func._manipulate_state.apply_update() is not None
                ):
                    self.objective._func._manipulate_state.apply_update()

                self.model.posterior(self.params)
                self.acquisition_fcn.update_K_xX_dx()
                _n_inner += 1 #change

                # Stop sampling if differece of values of acquired points is smaller than a threshold.
                # Equivalent to: variance of gradient did not change larger than a threshold.
                if self.epsilon_diff_acq_value is not None:
                    if acq_value_old is not None:
                        diff = acq_value - acq_value_old
                        if diff < self.epsilon_diff_acq_value:
                            if self.verbose:
                                print(
                                    f"Stop sampling after {i+1} samples, since gradient certainty is {diff}."
                                )
                            break
                    acq_value_old = acq_value
           

        elif self.inner_loop_mode == 'prob_wolfe':
            '''
             Variant A: Probabilistic Wolfe inner loop 
             Design: p is fixed after the first GI sample, not before.
             Reason: before any GI samples, the SE kernel derivative at
             theta ( only training point so far) is zero, making
             the gradient direction degenerate. After one GI sample the
             posterior gradient at theta is non trivial.
             '''
            p_direction = None        
            phi_0 = None
            phi_prime_0 = None
            # THESIS EXTENSION
            # Description: Use alpha_max (separate from delta) as the upper bound for line search so that GI acquisition bounds are not affected when the line search range is changed.
            
            #ORIGINAL GIBO CODE
            # delta_val = float(self.delta) if self.delta is not None else 0.1
            alpha_max_val = (
                float(self.alpha_max) if self.alpha_max is not None
                else float(self.delta) if self.delta is not None
                else 0.1
            )
        
            alpha_candidate = alpha_max_val   # fallback: upper bound of search --> problem of first step explosion
            _wolfe_satisfied = False
            p_wolfe_val = 0.0  #default 
            _n_inner = 0

            for i in range(self.max_samples_per_iteration):
                # Sample
                new_x, acq_value = self.optimize_acqf(self.acquisition_fcn, self.bounds)
                new_y = self.objective(new_x)

                # Update GP 
                self.model.append_train_data(new_x, new_y)
                self.model.posterior(self.params)
                self.acquisition_fcn.update_K_xX_dx()
                _n_inner += 1

                # Fix search direction from first GI-informed posterior.
                # phi_0 and phi_prime_0 are computed so that direction and baseline are consistent with the same GP state.
                if p_direction is None:
                    p_direction, _ = get_search_direction(self.model, self.params)
                    phi_0, phi_prime_0, _ = eval_phi_0(
                        self.model, self.params, p_direction
                    )

                # compute alpha_candidate from updated GP posterior.
                alpha_candidate = find_alpha_star(
                    self.model, self.params, p_direction, delta=alpha_max_val # new delta here old is self.delta
                )

                # Check probabilistic Wolfe condition at alpha_candidate.
                p_wolfe_val = compute_p_wolfe(
                    self.model, self.params, alpha_candidate, p_direction,
                    phi_0=phi_0, phi_prime_0=phi_prime_0,
                    c1=self.c1, c2=self.c2,
                    sigma_floor=self.sigma_floor,
                )

                if self.verbose:
                    print(
                        f"  Prob-Wolfe iter {i+1}: alpha={alpha_candidate:.4f}, "
                        f"p_Wolfe={p_wolfe_val:.4f} (threshold={self.c_W})"
                    )

                # THESIS 
                # Description: min_samples_per_iteration --> enforce a minimum number of GI samples when posterior is barely informed
                if p_wolfe_val > self.c_W and _n_inner >= self.min_samples_per_iteration:
                    _wolfe_satisfied = True
                    if self.verbose:
                        print(
                            f"  Wolfe satisfied after {_n_inner} inner samples."
                        )
                    break
    

            if self.verbose and not _wolfe_satisfied:
                print(
                    f"  Prob-Wolfe: max_samples ({self.max_samples_per_iteration}) "
                    f"reached without satisfying Wolfe. Using alpha={alpha_candidate:.4f}."
                )

        elif self.inner_loop_mode == 'det_ei':
            '''
            Variant B: Deterministic Wolfe + EI inner loop
            Same lazy-direction design as Variant A: p is fixed after the
            first GI sample to avoid the zero gradient degeneracy at theta.
            eta = phi_0 serves as EI reference (current posterior mean at theta).
            '''
            p_direction = None
            phi_0 = None
            phi_prime_0 = None
            eta = None
            # ORIGINAL GIBO CODE 
            # delta_val = float(self.delta) if self.delta is not None else 0.1
            # END ORIGINAL GIBO CODE
            alpha_max_val = (
                float(self.alpha_max) if self.alpha_max is not None
                else float(self.delta) if self.delta is not None
                else 0.1
            )
            
            alpha_candidate = alpha_max_val
            _wolfe_satisfied = False
            armijo_ok = False             
            curvature_ok = False
            _n_inner = 0
            acq_value_old = None

            for i in range(self.max_samples_per_iteration):
                # Sample
                new_x, acq_value = self.optimize_acqf(self.acquisition_fcn, self.bounds)
                new_y = self.objective(new_x)

                #Update GP 
                self.model.append_train_data(new_x, new_y)
                self.model.posterior(self.params)
                self.acquisition_fcn.update_K_xX_dx()
                _n_inner += 1

                # Fix direction
                if p_direction is None:
                    p_direction, _ = get_search_direction(self.model, self.params)
                    phi_0, phi_prime_0, _ = eval_phi_0(
                        self.model, self.params, p_direction
                    )
                    eta = phi_0  #EI reference

                # compute alpha_candidate via EI maximization.
                alpha_candidate = find_alpha_star_ei(
                    self.model, self.params, p_direction, eta=eta, delta=alpha_max_val #also changed in det_ei
                )

                #Check strong Wolfe conditions
                armijo_ok, curvature_ok = check_det_wolfe(
                    self.model, self.params, alpha_candidate, p_direction,
                    phi_0=phi_0, phi_prime_0=phi_prime_0,
                    c1=self.c1, c2=self.c2,
                )

                if self.verbose:
                    print(
                        f"  Det-EI iter {i+1}: alpha={alpha_candidate:.4f}, "
                        f"Armijo={'OK' if armijo_ok else 'NO'}, "
                        f"Curvature={'OK' if curvature_ok else 'NO'}"
                    )
                '''
                THESIS  
                Description: epsilon_diff_acq_value termination for det_ei
                 mirrors the baseline criterion. Active only when
                 epsilon_diff_acq_value is set in config (not None).
                 Terminates when the GI acquisition value improvement falls
                 below the threshold--> gradient uncertainty reduction is
                 no longer meaningful.
                 Deterministic Wolfe check is kept for logging only (not used
                 for termination when epsilon_diff_acq_value is set).
                '''
                # OLD CODE
                # if armijo_ok and curvature_ok and _n_inner >= self.min_samples_per_iteration:
                #     _wolfe_satisfied = True
                #     break
                if self.epsilon_diff_acq_value is not None:
                    if acq_value_old is not None:
                        diff = acq_value - acq_value_old
                        if diff < self.epsilon_diff_acq_value:
                            _wolfe_satisfied = True  #reuse flag for logging
                            if self.verbose:
                                print(
                                    f"  Det-EI epsilon_diff satisfied after {_n_inner} "
                                    f"inner samples (diff={diff:.6f})."
                                )
                            break
                    acq_value_old = acq_value
                elif armijo_ok and curvature_ok and _n_inner >= self.min_samples_per_iteration:
                    # Fallback: use Wolfe when epsilon_diff_acq_value is None
                    _wolfe_satisfied = True
                    if self.verbose:
                        print(
                            f"  Det Wolfe satisfied after {_n_inner} inner samples."
                        )
                    break

            if self.verbose and not _wolfe_satisfied:
                print(
                    f"  Det-EI: max_samples ({self.max_samples_per_iteration}) "
                    f"reached without Wolfe/epsilon_diff. Using alpha={alpha_candidate:.4f}."
                )

        elif self.inner_loop_mode == 'ei_pwolfe':
            '''
             THESIS
             Description: Variant ei_pwolfe --> EI step size + probabilistic
               Wolfe termination. Identical inner loop structure to det_ei;
               only the termination check differs (check_ei_pwolfe with
               sigma_floor prevents p_Wolfe from collapsing near data).
            '''
            p_direction = None
            phi_0 = None
            phi_prime_0 = None
            eta = None
            alpha_max_val = (
                float(self.alpha_max) if self.alpha_max is not None
                else float(self.delta) if self.delta is not None
                else 0.1
            )
            alpha_candidate = alpha_max_val
            _wolfe_satisfied = False
            p_wolfe_val = 0.0
            armijo_ok = False
            curvature_ok = False
            _n_inner = 0

            for i in range(self.max_samples_per_iteration):
                new_x, acq_value = self.optimize_acqf(self.acquisition_fcn, self.bounds)
                new_y = self.objective(new_x)
                self.model.append_train_data(new_x, new_y)
                self.model.posterior(self.params)
                self.acquisition_fcn.update_K_xX_dx()
                _n_inner += 1

                if p_direction is None:
                    p_direction, _ = get_search_direction(self.model, self.params)
                    phi_0, phi_prime_0, _ = eval_phi_0(
                        self.model, self.params, p_direction
                    )
                    eta = phi_0

                (_wolfe_satisfied, alpha_candidate, 
                 p_wolfe_val, armijo_ok, curvature_ok,) = check_ei_pwolfe(
                                                          self.model, self.params, p_direction,
                                                          phi_0=phi_0, phi_prime_0=phi_prime_0, eta=eta,
                                                          delta=alpha_max_val, c1=self.c1, c2=self.c2,
                                                          c_W=self.c_W, sigma_floor=self.sigma_floor,
                                                        )

                if self.verbose:
                    print(
                        f"  EI-pWolfe iter {i+1}: alpha={alpha_candidate:.4f}, "
                        f"p_Wolfe={p_wolfe_val:.4f} (thr={self.c_W}), "
                        f"Armijo={'OK' if armijo_ok else 'NO'}, "
                        f"Curv={'OK' if curvature_ok else 'NO'}"
                    )

                if _wolfe_satisfied and _n_inner >= self.min_samples_per_iteration:
                    if self.verbose:
                        print(f"  EI-pWolfe satisfied after {_n_inner} samples.")
                    break

            if self.verbose and not _wolfe_satisfied:
                print(
                    f"  EI-pWolfe: max_samples ({self.max_samples_per_iteration}) "
                    f"reached. p_Wolfe={p_wolfe_val:.4f}, alpha={alpha_candidate:.4f}."
                )
    

        elif self.inner_loop_mode == 'ei_snr':
            '''
             THESIS
             Description: Variant ei_snr  EI step size + gradient SNR
                termination. Stops when SNR = phi'(0)^2/S22 >= tau_snr,
                meaning the gradient signal dominates its uncertainty.
            '''
            p_direction = None
            phi_0 = None
            phi_prime_0 = None
            eta = None
            alpha_max_val = (
                float(self.alpha_max) if self.alpha_max is not None
                else float(self.delta) if self.delta is not None
                else 0.1
            )
            alpha_candidate = alpha_max_val
            _snr_satisfied = False
            snr_val = 0.0
            armijo_ok = False
            curvature_ok = False
            _n_inner = 0

            for i in range(self.max_samples_per_iteration):
                new_x, acq_value = self.optimize_acqf(self.acquisition_fcn, self.bounds)
                new_y = self.objective(new_x)
                self.model.append_train_data(new_x, new_y)
                self.model.posterior(self.params)
                self.acquisition_fcn.update_K_xX_dx()
                _n_inner += 1

                if p_direction is None:
                    p_direction, _ = get_search_direction(self.model, self.params)
                    phi_0, phi_prime_0, _ = eval_phi_0(
                        self.model, self.params, p_direction
                    )
                    eta = phi_0

                (_snr_satisfied, alpha_candidate,
                  snr_val, armijo_ok, curvature_ok,) = check_ei_snr(
                                                       self.model, self.params, p_direction,
                                                       phi_0=phi_0, phi_prime_0=phi_prime_0, eta=eta,
                                                       delta=alpha_max_val, c1=self.c1, c2=self.c2,
                                                       tau_snr=self.tau_snr,
                                                      )
                if self.verbose:
                    snr_disp = f"{snr_val:.3f}" if snr_val != float('inf') else "inf"
                    print(
                        f"EI-SNR iter {i+1}: alpha={alpha_candidate:.4f}, "
                        f"SNR={snr_disp} (thr={self.tau_snr}), "
                        f"Armijo={'OK' if armijo_ok else 'NO'}, "
                        f"Curv={'OK' if curvature_ok else 'NO'}"
                    )

                if _snr_satisfied and _n_inner >= self.min_samples_per_iteration:
                    if self.verbose:
                        print(f"  EI-SNR satisfied after {_n_inner} samples.")
                    break

            if self.verbose and not _snr_satisfied:
                snr_disp = f"{snr_val:.3f}" if snr_val != float('inf') else "inf"
                print(
                    f"EI-SNR: max_samples ({self.max_samples_per_iteration}) "
                    f"reached. SNR={snr_disp}, alpha={alpha_candidate:.4f}."
                )
            

        # THESIS 
        # Description: Branch gradient step on inner_loop_mode
        if self.inner_loop_mode == 'original':
            # ORIGINAL GIBO CODE
            with torch.no_grad():
                self.optimizer_torch.zero_grad()
                mean_d, variance_d = self.model.posterior_derivative(self.params)
                params_grad = -mean_d.view(1, self.D)
                if self.normalize_gradient:
                    lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
                    params_grad = torch.nn.functional.normalize(params_grad) * lengthscale
                if self.standard_deviation_scaling:
                    params_grad = params_grad / torch.diag(variance_d.view(self.D, self.D))
                if self.lr_schedular:
                    lr = [v for k, v in self.lr_schedular.items() if k <= self.iteration][
                        -1
                    ]
                    self.params.grad[:] = lr * params_grad  # Define as gradient ascent.
                else:
                    self.params.grad[:] = params_grad  # Define as gradient ascent.
                self.optimizer_torch.step()
                self.iteration += 1

        elif self.inner_loop_mode in ('prob_wolfe', 'det_ei', 'ei_pwolfe', 'ei_snr'):
            #Variants A, B, ei_pwolfe, ei_snr: Direct update along p
            # Bypasses SGD entirely: theta_{t+1} = theta_t + alpha* * p
            with torch.no_grad():
                self.optimizer_torch.zero_grad()
                if p_direction is not None:
                    self.params.data += alpha_candidate * p_direction
                self.iteration += 1

        # THESIS EXTENSION
        # Description: Collect per-step metrics into last_step_info
        with torch.no_grad():
            _post = self.model.posterior(self.params)
            _sigma2 = _post.mvn.variance.squeeze().item()
            _mean_d, _ = self.model.posterior_derivative(self.params)
            _grad_norm = _mean_d.norm().item()

        if self.inner_loop_mode == 'original':
            self.last_step_info = {
                'mode': 'original',
                'n_inner_samples': _n_inner,
                'alpha': None,
                'grad_norm': _grad_norm,
                'sigma2': _sigma2,
            }
        elif self.inner_loop_mode == 'prob_wolfe':
            self.last_step_info = {
                'mode': 'prob_wolfe',
                'n_inner_samples': _n_inner,
                'alpha': float(alpha_candidate) if p_direction is not None else None,
                'p_wolfe': float(p_wolfe_val),
                'wolfe_satisfied': _wolfe_satisfied,
                'grad_norm': _grad_norm,
                'sigma2': _sigma2,
            }
        elif self.inner_loop_mode == 'det_ei':
            self.last_step_info = {
                'mode': 'det_ei',
                'n_inner_samples': _n_inner,
                'alpha': float(alpha_candidate) if p_direction is not None else None,
                'wolfe_satisfied': _wolfe_satisfied,
                'armijo_ok': armijo_ok,
                'curvature_ok': curvature_ok,
                'grad_norm': _grad_norm,
                'sigma2': _sigma2,
            }

        elif self.inner_loop_mode == 'ei_pwolfe':
            self.last_step_info = {
                'mode': 'ei_pwolfe',
                'n_inner_samples': _n_inner,
                'alpha': float(alpha_candidate) if p_direction is not None else None,
                'p_wolfe': float(p_wolfe_val),
                'wolfe_satisfied': _wolfe_satisfied,
                'armijo_ok': armijo_ok,
                'curvature_ok': curvature_ok,
                'grad_norm': _grad_norm,
                'sigma2': _sigma2,
            }
        elif self.inner_loop_mode == 'ei_snr':
            self.last_step_info = {
                'mode': 'ei_snr',
                'n_inner_samples': _n_inner,
                'alpha': float(alpha_candidate) if p_direction is not None else None,
                'snr_value': float(snr_val) if snr_val != float('inf') else None,
                'snr_satisfied': _snr_satisfied,
                'armijo_ok': armijo_ok,
                'curvature_ok': curvature_ok,
                'grad_norm': _grad_norm,
                'sigma2': _sigma2,
            }


        if (
            type(self.objective._func) is EnvironmentObjective
            and self.objective._func._manipulate_state is not None
        ):
            self.params_history_list.append(
                self.objective._func._manipulate_state.unnormalize_params(self.params)
            )
        else:
            self.params_history_list.append(self.params.clone())

        if self.verbose:
            posterior = self.model.posterior(self.params)
            print(
                f"theta_t: {self.params_history_list[-1].numpy()} predicted mean {posterior.mvn.mean.item(): .2f} and variance {posterior.mvn.variance.item(): .2f} of f(theta_i)."
            )
            print(
                f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.detach().numpy()}, outputscale: {self.model.covar_module.outputscale.detach().numpy()},  noise {self.model.likelihood.noise.detach().numpy()}"
            )
