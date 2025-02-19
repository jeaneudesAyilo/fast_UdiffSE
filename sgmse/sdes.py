"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""

import abc
import warnings

import numpy as np
from sgmse.util.tensors import batch_broadcast
import torch

from sgmse.util.registry import Registry
#from sgmse.sampling.schedulers import SchedulerRegistry

SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, device):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    @abc.abstractmethod
    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        # dt = 1 / self.N
        # drift, diffusion = self.sde(x, t, *args)
        # f = drift * dt
        # G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        # return f, G
        pass

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = (
                    rsde_parts["total_drift"],
                    rsde_parts["diffusion"],
                )
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, t, *args)
                score_drift = (
                    -sde_diffusion[:, None, None, None] ** 2
                    * score
                    * (0.5 if self.probability_flow else 1.0)
                )
                diffusion = (
                    torch.zeros_like(sde_diffusion)
                    if self.probability_flow
                    else sde_diffusion
                )
                total_drift = sde_drift + score_drift
                return {
                    "total_drift": total_drift,
                    "diffusion": diffusion,
                    "sde_drift": sde_drift,
                    "sde_diffusion": sde_diffusion,
                    "score_drift": score_drift,
                    "score": score,
                }

            def discretize(self, x, t, *args):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, *args)
                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, t, *args) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("ouve")
class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--sde-n",
            type=int,
            default=1000,
            help="The number of timesteps in the SDE discretization. 30 by default",
        )
        parser.add_argument(
            "--theta",
            type=float,
            default=1.5,
            help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.",
        )
        parser.add_argument(
            "--sigma-min",
            type=float,
            default=0.05,
            help="The minimum sigma to use. 0.05 by default.",
        )
        parser.add_argument(
            "--sigma-max",
            type=float,
            default=0.5,
            help="The maximum sigma to use. 0.5 by default.",
        )
        return parser

    def __init__(self, theta, sigma_min, sigma_max, N=1000, **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N
        self.__name__ = "ouve"

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        drift = self.theta * (-x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, t):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0  # + (1 - exp_interp) * y

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            / (theta + logsig)
        )

    def marginal_prob(self, x0, t):
        return self._mean(x0, t), self._std(t)

    def prior_sampling(self, shape, device):
        std = self._std(torch.ones((shape[0],)))
        x_T = (torch.randn(shape, dtype=torch.complex64) * std[:, None, None, None]).to(
            device
        )
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G


@SDERegistry.register("vpsde")
class VPSDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--sde-n",
            type=int,
            default=1000,
            help="The number of timesteps in the SDE discretization. 1000 by default",
        )
        parser.add_argument(
            "--beta-min", type=float, default=1e-4, help="The minimum beta to use."
        )
        parser.add_argument(
            "--beta-max", type=float, default=0.2, help="The maximum beta to use."
        )
        # parser.add_argument("--stiffness", type=float, default=1,
        #     help="The stiffness factor for the drift, to be multiplied by 0.5*beta(t). 1 by default.")
        return parser

    def __init__(self, beta_min=0.1, beta_max=20, N=1000, **ignored_kwargs):
        """Construct a Variance Preserving SDE.
        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps
        """
        super().__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.beta_tilde = self.discrete_betas * (
            (self.alphas - self.alphas_cumprod)
            / (self.alphas * (1 - self.alphas_cumprod))
        )
        self.__name__ = "vpsde"

    def copy(self):
        return VPSDE(self.beta_min, self.beta_max, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape, device):
        return torch.randn(*shape, dtype=torch.complex64).to(device)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


@SDERegistry.register("ouvp")
class OUVPSDE(SDE):
    # !!! We do not utilize this SDE in our works due to observed instabilities around t=0.2. !!!
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--sde-n",
            type=int,
            default=1000,
            help="The number of timesteps in the SDE discretization. 1000 by default",
        )
        parser.add_argument(
            "--beta-min", type=float, required=True, help="The minimum beta to use."
        )
        parser.add_argument(
            "--beta-max", type=float, required=True, help="The maximum beta to use."
        )
        parser.add_argument(
            "--stiffness",
            type=float,
            default=1,
            help="The stiffness factor for the drift, to be multiplied by 0.5*beta(t). 1 by default.",
        )
        return parser

    def __init__(self, beta_min, beta_max, stiffness=1, N=1000, **ignored_kwargs):
        """
        !!! We do not utilize this SDE in our works due to observed instabilities around t=0.2. !!!

        Construct an Ornstein-Uhlenbeck Variance Preserving SDE:

        dx = -1/2 * beta(t) * stiffness * (y-x) dt + sqrt(beta(t)) * dw

        with

        beta(t) = beta_min + t(beta_max - beta_min)

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        Args:
            beta_min: smallest sigma.
            beta_max: largest sigma.
            stiffness: stiffness factor of the drift. 1 by default.
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.stiffness = stiffness
        self.N = N

    def copy(self):
        return OUVPSDE(self.beta_min, self.beta_max, self.stiffness, N=self.N)

    @property
    def T(self):
        return 1

    def _beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x, t):
        drift = (
            0.5 * self.stiffness * batch_broadcast(self._beta(t), x) * (-x)
        )  # $changed batch_broadcast to x.shape which should be the same
        diffusion = torch.sqrt(self._beta(t))
        return drift, diffusion

    def _mean(self, x0, t):
        b0, b1, s = self.beta_min, self.beta_max, self.stiffness
        x0y_fac = torch.exp(-0.25 * s * t * (t * (b1 - b0) + 2 * b0))[
            :, None, None, None
        ]
        return 0 + x0y_fac * (x0 - 0)

    def _std(self, t):
        b0, b1, s = self.beta_min, self.beta_max, self.stiffness
        return (1 - torch.exp(-0.5 * s * t * (t * (b1 - b0) + 2 * b0))) / s

    def marginal_prob(self, x0, t):
        return self._mean(x0, t), self._std(t)

    def prior_sampling(self, shape, device):
        # if shape != y.shape:
        # warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((shape[0],)))
        x_T = (torch.randn(shape, dtype=torch.complex64) * std[:, None, None, None]).to(
            device
        )
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G


@SDERegistry.register("ve")
class VESDE(SDE):
    def __init__(self, sigma_min, sigma_max, N=1000, **ignored_kwargs):
        """Construct a Variance Exploding SDE.

        dx = sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N

    def copy(self):
        return VESDE(self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, *args, **kwargs):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return 0.0, diffusion

    def _mean(self, x0, t, *args, **kwargs):
        return x0

    def _std(self, t, *args, **kwargs):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, logsig = self.sigma_min, self.logsig
        return sigma_min * torch.sqrt(torch.exp(2 * logsig * t) - 1)

    def _inverse_std(self, sigma, *args, **kwargs):
        sigma_min, logsig = self.sigma_min, self.logsig
        return torch.log(sigma**2 / sigma_min**2 + 1) / (2 * logsig)

    def marginal_prob(self, x0, t, *args, **kwargs):
        return self._mean(x0, t), self._std(t)

    def prior_sampling(self, shape, y, unconditional_prior=True, **kwargs):
        if shape != y.shape:
            warnings.warn(
                f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape."
            )
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        std = std.view(std.size(0), *(1,) * (y.ndim - std.ndim))
        if unconditional_prior:
            return torch.randn_like(y) * std
        else:
            return y + torch.randn_like(y) * std

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for VE SDE not yet implemented!")

    def tweedie_from_score(self, score, x, t, *args):
        sigma = self._std(t)
        sigma = sigma.view(sigma.size(0), *(1,) * (score.ndim - sigma.ndim))
        return x + sigma**2 * score

    def score_from_tweedie(self, tweedie, x, t, *args):
        sigma = self._std(t)
        sigma = sigma.view(sigma.size(0), *(1,) * (tweedie.ndim - sigma.ndim))
        return 1 / sigma**2 * (tweedie - x)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--sde_n",
            type=int,
            default=1000,
            help="The number of timesteps in the SDE discretization. 1000 by default",
        )
        parser.add_argument(
            "--sigma_min", type=float, default=0.05, help="The minimum sigma to use."
        )
        parser.add_argument(
            "--sigma_max", type=float, default=0.5, help="The maximum sigma to use."
        )
        return parser

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G


@SDERegistry.register("edm")
class EDM(SDE):
    def __init__(self, sigma_min, sigma_max, rho, N=50, scheduler="edm", **kwargs):
        """Construct a Variance Exploding SDE.

        dx = sqrt( 2 sigma(t) ) dw

        with

        sigma(t) = t

        """
        super().__init__(N)
        self.N = N
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.scheduler = SchedulerRegistry.get_by_name(scheduler)(
            N=N, sigma_min=sigma_min, sigma_max=sigma_max, eps=0.0, rho=rho, **kwargs
        )

    def copy(self):
        return EDM(
            N=self.N, sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho
        )

    def sde(self, x, t, *args, **kwargs):
        diffusion = torch.sqrt(2 * t)
        return 0.0, diffusion

    def _mean(self, x0, t, *args, **kwargs):
        return x0

    def _std(self, t, *args, **kwargs):
        sigma = t
        return sigma

    def marginal_prob(self, x0, t, *args, **kwargs):
        return self._mean(x0, t), self._std(t)

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for VE SDE not yet implemented!")

    def tweedie_from_score(self, score, x, t, *args):
        sigma = self._std(t)
        sigma = sigma.view(sigma.size(0), *(1,) * (score.ndim - sigma.ndim))
        return x + sigma**2 * score

    def score_from_tweedie(self, tweedie, x, t, *args):
        sigma = self._std(t)
        sigma = sigma.view(sigma.size(0), *(1,) * (tweedie.ndim - sigma.ndim))
        return 1 / sigma**2 * (tweedie - x)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--sigma_min", type=float, default=1e-5, help="The minimum sigma to use."
        )
        parser.add_argument(
            "--sigma_max", type=float, default=150.0, help="The maximum sigma to use."
        )
        parser.add_argument("--rho", type=float, default=7.0)
        return parser
