#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[1]:


import jax
import jax.numpy as jnp
from jax import jit, vmap, random, lax
import jax.scipy as jsp
from jax.scipy.special import logsumexp
from abc import ABC, abstractmethod
from typing import Any, Dict
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, lax
import jax.scipy as jsp
from jax.scipy.special import logsumexp
from jax.experimental.ode import odeint
from abc import ABC, abstractmethod
from typing import Any, Callable
import matplotlib.pyplot as plt

from pathlib import PosixPath
import orbax.checkpoint as ocp

## Standard libraries
import os

import math
import time
import json
import numpy as np
import pandas as pd
from typing import Sequence
from functools import partial
import beartype

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgb
import matplotlib
import seaborn as sns

## Progress bar
from tqdm.notebook import tqdm

## JAX
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
from jax.scipy.special import gammaln

from jax import jit
from jax import lax

import flax
from flax import nnx
from flax.training import train_state, checkpoints
import optax

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from dynax import AbstractSystem
from dynax.custom_types import FloatScalarLike
from typing import Optional
from jax import random
import jax
import jax.numpy as jnp
from jax import lax, random, vmap
import equinox as eqx
from jaxtyping import Array, Float32 
from typing import Union
from functools import partial
from jaxtyping import Array, Float, PyTree

os.environ['JAX_PLATFORMS'] = 'cuda'
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
debug = False
key = jax.random.key(0)
key, subkey = jax.random.split(key)
rngs = nnx.Rngs(0)


# # Dynamical Systems

# In[2]:


class DynamicalSystem(ABC):
    state: Any

    @abstractmethod
    def flow(self, x: Array) -> Array:
        """
        Push given points forward along the flow of the dynamical system.
        """
        pass

    @abstractmethod
    def iterate(self) -> None:
        """
        Iterate the internal state of the dynamical system.
        """
        pass

    @abstractmethod
    def generate(self, key, batch_size):
        """
        Generate Points on the chaotic attractor of the set
        """
        pass


# ## Ikeda Map

# In[3]:


from jaxtyping import Float, Array, jaxtyped, Int
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
class IkedaSystem(nnx.Module, DynamicalSystem):
    u: nnx.Variable[Float[Array, ""]]
    state: nnx.Variable[Float[Array, "*batch 2"]]

    @jaxtyped(typechecker=typechecker)
    def __init__(self, state: Float[Array, "*batch 2"] = jnp.array([[1.25, 0.0]]), u: Float[Array, ""] = jnp.array(0.9)):
        self.u = nnx.Variable(u)
        self.state = nnx.Variable(state)

    @jaxtyped(typechecker=typechecker)
    @nnx.jit
    def flow(self, x: Float[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        x1, x2 = x[..., 0], x[..., 1]
        t = 0.4 - (6 / (1 + x1**2 + x2**2))
        sin_t, cos_t = jnp.sin(t), jnp.cos(t)
        x1_new = 1 + self.u * (x1 * cos_t - x2 * sin_t)
        x2_new = self.u * (x1 * sin_t + x2 * cos_t)
        return jnp.stack((x1_new, x2_new), axis=-1)

    def iterate(self):
        self.state = nnx.Variable(self.flow(self.state.value))

    @jaxtyped(typechecker=typechecker)
    @nnx.jit(static_argnames=['batch_size'])
    def generate(self, key, batch_size: int = 10**5) -> Array:
        def body_fn(i, val):
            return self.flow(val)
        initial_state = random.uniform(key, shape=(batch_size, 2), minval=-0.25, maxval=0.25)
        return lax.fori_loop(0, 15, body_fn, initial_state)

if debug:
    key, subkey = jax.random.split(key)
    ikeda = IkedaSystem(state=jnp.array([[1.0,2.0]]), u=jnp.array(0.9))
    batch = ikeda.generate(subkey)
    batch = ikeda.flow(batch)
    ikeda.iterate()
    print(ikeda.state.shape)



# In[4]:


@jaxtyped(typechecker=typechecker)
class MeasurementDevice(ABC):
    """
    Abstract base class for Measurement Devices in the filtering framework.
    """
    
    @abstractmethod
    def __call__(self, state: Float[Array, "..."], key: Optional[jax.random.PRNGKey] = None) -> Float[Array, "..."]:
        """
        Generate a measurement based on the current state.

        Parameters:
            state (Float[Array, "..."]): The current state of the dynamical system.
            key (Optional[jax.random.PRNGKey]): Random key for stochastic measurements.

        Returns:
            Float[Array, "..."]: The measurement derived from the state.
        """
        pass

class Distance(nnx.Module, MeasurementDevice):

    def __init__(self, covariance):
        self.covariance = nnx.Variable(covariance)

    @nnx.jit
    def __call__(self, state, key=None):
        perfect_measurement = jnp.linalg.norm(state)
        noise = 0 if key is None else jnp.sqrt(self.covariance.value) * jax.random.normal(key)
        return jnp.array([perfect_measurement + noise])

if debug:
    measurement_device = Distance(jnp.array(1/16))
    system = IkedaSystem()
    print(measurement_device(system.state))
    system.iterate()
    print(measurement_device(system.state).shape)


# ### Test Ikeda Map

# In[5]:


if debug:
    key, subkey = jax.random.split(key)
    ikeda = IkedaSystem()
    batch = ikeda.generate(subkey)
    batch = ikeda.flow(batch)
    ikeda.iterate()
    print(batch)


# ### Test Ikeda Measurement

# In[6]:


if debug:
    measurement_device = Distance(jnp.array(1/16))
    system = IkedaSystem()
    print(measurement_device(system.state))
    system.iterate()
    print(measurement_device(system.state).shape)


# ## Lorenz 63 (WIP)

# ## Lorenz 96 (WIP)

# # Normalizing Flow Based Discriminator

# In[7]:


# import jax.numpy as jnp
# import jax.random as random

# @partial(jax.jit, static_argnames=['mu', 'sigma', 'num_samples'])
# def sample_epanechnikov(key, mu, sigma, num_samples):
#     n = mu.shape[0]
#     key, subkey1, subkey2 = random.split(key, 3)

#     s = random.multivariate_normal(subkey1, jnp.zeros(n), jnp.eye(n), (num_samples,))
    
#     s_hat = jnp.sqrt(n + 4) * s / jnp.linalg.norm(s, axis=1, keepdims=True)
    
#     kappa = random.beta(subkey2, b=n / 2, a=2, shape=(num_samples,))[:, None]
#     scaled_s = kappa * s_hat

#     # Use symmetric sqrt (SVD)
#     eigvals, eigvecs = jnp.linalg.eigh(sigma)
#     sqrt_sigma = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T

#     samples = mu + (sqrt_sigma @ scaled_s.T).T
    
#     return samples

# @partial(jax.jit, static_argnames=['num_samples'])
# def sample_standard_epanechnikov(key, num_samples):
#     key, subkey1, subkey2 = random.split(key, 3)
#     s = random.normal(subkey1, (num_samples,2))
#     s_hat = jnp.sqrt(6) * s / jnp.linalg.norm(s, axis=1, keepdims=True)
#     print(jnp.linalg.norm(s, axis=1, keepdims=True).shape)
#     kappa = jnp.sqrt(random.beta(subkey2, b=2, a=1, shape=(num_samples,))[:, None])
#     samples = kappa * s_hat
#     return samples

# if debug:
#     #samples = sample_epanechnikov(key, mu=jnp.zeros(2), sigma=jnp.eye(2), num_samples=100_000)
#     samples = sample_standard_epanechnikov(key, num_samples=100_000)
#     print(samples.shape)
#     sns.jointplot(x=samples[:,0], y=samples[:, 1])


# In[8]:


# class DenseNetwork(nnx.Module):
#     hidden_dim: int
#     num_hidden_layers: int
#     output_dim: int

#     def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int, rngs: nnx.Rngs):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_hidden_layers = num_hidden_layers
#         self.output_dim = output_dim

#         self.layers = [nnx.Linear(self.input_dim, self.hidden_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)]        
#         for _ in range(self.num_hidden_layers):
#             self.layers.append(nnx.Linear(self.hidden_dim, self.hidden_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64))
#         self.output_layer = nnx.Linear(self.hidden_dim, self.output_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)

#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#             x = nnx.gelu(x)
#         x = self.output_layer(x)
#         return x

# class CouplingLayer(nnx.Module):
#     input_dim: int
#     hidden_dim: int
#     num_hidden_layers: int
#     swap: bool = False

#     def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, swap: bool, rngs: nnx.Rngs):
#         super().__init__()
#         self.input_dim = input_dim        
#         self.hidden_dim = hidden_dim
#         self.num_hidden_layers = num_hidden_layers
#         self.swap = swap

#         self.s_net = DenseNetwork(
#             input_dim=self.input_dim // 2,
#             hidden_dim=self.hidden_dim,
#             num_hidden_layers=self.num_hidden_layers,
#             output_dim=self.input_dim // 2,
#             rngs=rngs
#         )
#         self.t_net = DenseNetwork(
#             input_dim=self.input_dim // 2,
#             hidden_dim=self.hidden_dim,
#             num_hidden_layers=self.num_hidden_layers,
#             output_dim=self.input_dim // 2,
#             rngs=rngs
#         )

#     def __call__(self, x, reverse=False):
#         if self.swap:
#             x1, x2 = jnp.split(x, 2, axis=-1)
#             x1, x2 = x2, x1
#         else:
#             x1, x2 = jnp.split(x, 2, axis=-1)

#         s = self.s_net(x1)
#         #s = 5*jnp.tanh(s)
#         t = self.t_net(x1)

#         if reverse:
#             y2 = (x2 - t) * jnp.exp(-s)
#             log_det_jacobian = -jnp.sum(s,axis=-1)
#         else:
#             y2 = x2 * jnp.exp(s) + t
#             log_det_jacobian = jnp.sum(s,axis=-1)

#         if self.swap:
#             y = jnp.concatenate([y2, x1], axis=-1)
#         else:
#             y = jnp.concatenate([x1, y2], axis=-1)

#         return y, log_det_jacobian

# class InvertibleNN(nnx.Module):
#     input_dim: int = 2
#     num_coupling_layers: int = 5
#     hidden_dim: int = 128
#     num_hidden_layers: int = 4

#     def __init__(self, input_dim=2, num_coupling_layers=2, hidden_dim=32, num_hidden_layers=2, rngs=nnx.Rngs()):
#         super().__init__()
#         self.input_dim = input_dim
#         self.num_coupling_layers = num_coupling_layers
#         self.hidden_dim = hidden_dim
#         self.num_hidden_layers = num_hidden_layers

#         self.coupling_layers = []
#         for i in range(self.num_coupling_layers):
#             swap = i % 2 == 1
#             layer = CouplingLayer(
#                 input_dim=self.input_dim,
#                 hidden_dim=self.hidden_dim,
#                 num_hidden_layers=self.num_hidden_layers,
#                 swap=swap,
#                 rngs=rngs
#             )
#             self.coupling_layers.append(layer)

#     def __call__(self, x, reverse=False):
#         log_det_jacobian = 0
#         if not reverse:
#             for layer in self.coupling_layers:
#                 x, ldj = layer(x)
#                 log_det_jacobian += ldj
#         else:
#             for layer in reversed(self.coupling_layers):
#                 x, ldj = layer(x, reverse=True)
#                 log_det_jacobian += ldj
#         return x, log_det_jacobian


# In[9]:


# abstract_model = nnx.eval_shape(lambda: InvertibleNN(input_dim=2,
#                      num_coupling_layers=12, hidden_dim=170, num_hidden_layers=6, 
#                      rngs=nnx.Rngs(0)))


# graphdef, abstract_state = nnx.split(abstract_model)
# print('The abstract NNX state (all leaves are abstract arrays):')
# ckpt_dir = PosixPath('/home/zjabbar/koa_scratch/normalizing_flow_checkpoints/')
# checkpointer = ocp.StandardCheckpointer()
# state_restored = checkpointer.restore(ckpt_dir / 'model_state', abstract_state)
# model = nnx.merge(graphdef, state_restored)
# ikeda = IkedaSystem(u=0.9)
# batch = ikeda.generate(subkey, batch_size=10000)
# generated_data = model(sample_standard_epanechnikov(key, num_samples=batch.shape[0]), reverse=False)[0]
# plt.scatter(generated_data[:, 0], generated_data[:,1], c='red', alpha=0.15)
# plt.xlim(-0.5,2)
# plt.ylim(-2.5,1)
# plt.show()

# grid_spacing = 500
# x = jnp.linspace(-0.5, 2, grid_spacing)
# y = jnp.linspace(-2.5, 1, grid_spacing)
# XX, YY = jnp.meshgrid(x, y)
# grid = jnp.dstack([XX, YY])
# grid_points = grid.reshape(-1, 2)
# threshold = 0.01

# @jit
# def logpdf_epanechnikov(x, mu, sigma):
#     n = mu.shape[0]
#     x_centered = x - mu
#     # Compute the Mahalanobis distance
    
    
#     L = jnp.linalg.cholesky(sigma)
#     y = jnp.linalg.solve(L, x_centered.T).T
#     mahalanobis_dist = jnp.sum(y ** 2, axis=1)
    
#     #mahalanobis_dist = jnp.sum(x_centered * jnp.linalg.solve(sigma, x_centered.T).T, axis=1)
#     # Compute the logarithm of the constant term
#     log_constant = (
#         jnp.log(n + 2)
#         - jnp.log(2)
#         - (n / 2) * jnp.log(jnp.pi)
#         - gammaln(n / 2 + 1)
#         - (n / 2) * jnp.log(n + 4)
#     )
#     # Compute the logarithm of the density
#     log_density = jnp.where(
#         mahalanobis_dist <= (n + 4),
#         log_constant + jnp.log1p(-mahalanobis_dist / (n + 4)),
#         -jnp.inf  # Logarithm of zero for points outside the support
#     )
#     return log_density


# @jit
# def discriminator(points, threshold=0, normalizing_flow=model):
#     points = jnp.atleast_2d(points)
#     z, log_det_jacobian = model(points, reverse=True)
#     log_det_jacobian = jnp.nan_to_num(log_det_jacobian)
#     logpdf_values = logpdf_epanechnikov(z, jnp.zeros(2), jnp.eye(2))
#     total_logprob = logpdf_values + log_det_jacobian
#     p_x = jnp.exp(total_logprob)
#     if threshold is None:
#         return p_x
#     else:
#         return jnp.where(p_x > threshold, 1, 0)

# labels = discriminator(grid_points)
# labels_grid = labels.reshape(grid_spacing,grid_spacing)
# plt.figure(figsize=(8, 6))
# plt.contourf(XX, YY, labels_grid)
# plt.title('Discriminator Output')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


# # Filters

# In[10]:


class Filter(ABC):
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self, measurement):
        pass


# ## BRUF (WIP)

# ## Linearized EnKF (WIP)

# ## BRUEnKF (WIP)

# 
# ## EnGMF

# In[11]:


# prior_ensemble = jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init

# %timeit (1 / (prior_ensemble.shape[0] - 1)) * prior_ensemble.T @ (jnp.eye(prior_ensemble.shape[0]) - (1/prior_ensemble.shape[0]) * jnp.ones(shape=(prior_ensemble.shape[0], prior_ensemble.shape[0]))) @ prior_ensemble
# %timeit jnp.cov(prior_ensemble.T)

# f = jax.jit(jnp.cov)
# f(prior_ensemble.T)
# %timeit f(prior_ensemble.T)


# In[12]:


# prior_ensemble = jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init

# my_cov = (1 / (prior_ensemble.shape[0] - 1)) * prior_ensemble.T @ (jnp.eye(prior_ensemble.shape[0]) - (1/prior_ensemble.shape[0]) * jnp.ones(shape=(prior_ensemble.shape[0], prior_ensemble.shape[0]))) @ prior_ensemble
# jax_cov = jnp.cov(prior_ensemble.T)
# eps = jnp.finfo(jnp.float64).eps

# jnp.allclose(my_cov, jax_cov, atol=eps, rtol=0)


# In[13]:


# prior_ensemble = jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init

# diff_ensemble: Float[Array, "batch_dim state_dim"] = jax.random.uniform(key=subkey, shape=(201, 2), minval=-0.25, maxval=0.25) #init


# prior_ensemble.shape



# assert isinstance(prior_ensemble, Float[Array, "batch_dim state_dim"])


# In[14]:


# from jaxtyping import Float, Array, Int
# import jax.numpy as jnp
# from beartype.door import (
#      is_bearable,  # <-------- like "isinstance(...)"
#      die_if_unbearable,  # <-- like "assert isinstance(...)"
# )
# import jaxtyping
# # %load_ext jaxtyping
# # %jaxtyping.typechecker beartype.beartype  # or any other runtime type checker

# @jaxtyped(typechecker=typechecker)
# def f(z: Float[Array, "foo"]):
#     x: Float[Array, "my_size_of_3"] = jnp.array([1.0, 2.0, 3.0])
#     y: Float[Array, "my_size_of_4"] = jnp.array([1.0, 2.0, 3.0, 4.0])
#     assert isinstance(x, Float[Array, "foo"])
#     jaxtyping.print_bindings()

# f(jnp.array([2.0])) # Passes >:(


# In[15]:


# prior_ensemble = jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init

# system.flow(prior_ensemble).shape


# In[16]:


# ensemble_size = 20
# prior_ensemble = jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init

# B_j = jnp.cov(prior_ensemble.T)
# print(B_j.shape)
# x_j = jnp.array([1.25, 0])
# print(x_j.shape)
# H_j = jax.jacobian(measurement_device)(x_j)
# print(H_j.shape)
# G_j = B_j @ H_j.T @ jnp.linalg.inv(H_j @ B_j @ H_j.T + measurement_device.covariance)
# print(G_j.shape)
# print((jnp.linalg.inv(H_j @ B_j @ H_j.T + measurement_device.covariance)).shape)


# In[17]:


# jsp.stats.multivariate_normal.logpdf(jnp.zeros(2), jnp.zeros(2), jnp.eye(2))


# In[31]:


from jaxtyping import Key
import jaxtyping
from jax.experimental import checkify

class EnGMF(nnx.Module):

    @jaxtyped(typechecker=typechecker)
    def __init__(self, dynamical_system : DynamicalSystem, measurement_device: MeasurementDevice, state: Float[Array, "batch_dim state_dim"], bandwidth_factor: float) -> None:
        self.dynamical_system = dynamical_system
        self.measurement_device = measurement_device
        self.state = nnx.Variable(state)
        self.bandwidth_factor = bandwidth_factor
        self.covariances = None

    
    @nnx.jit(static_argnames=['debug'])
    @jaxtyped(typechecker=typechecker)
    def ensemble_gaussian_mixture_filter_update(self, point: Float[Array, "state_dim"], prior_mixture_covariance: Float[Array, "state_dim state_dim"], measurement: Float[Array, "measurement_dim"], debug: bool = False):
        measurement_jacobian = jax.jacfwd(self.measurement_device)(point)
        if debug:
            assert isinstance(measurement_jacobian, Float[Array, "measurement_dim state_dim"])
            assert not jnp.any(jnp.isnan(measurement_jacobian))

        
        kalman_gain = prior_mixture_covariance @ measurement_jacobian.T @ jnp.linalg.inv(measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + self.measurement_device.covariance)
        # assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
        # assert not jnp.any(jnp.isnan(kalman_gain))

        
        gaussian_mixture_covariance = (jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian) @ prior_mixture_covariance #+ 1e-10 * jnp.eye(point.shape[0])
        if debug:
            assert isinstance(gaussian_mixture_covariance, Float[Array, "state_dim state_dim"])
            jax.debug.callback(is_positive_definite, gaussian_mixture_covariance)
            jax.debug.callback(has_nan, gaussian_mixture_covariance)
        
        point = point - (kalman_gain @ jnp.atleast_2d(self.measurement_device(point) - measurement)).reshape(-1)
        # assert isinstance(point, Float[Array, "state_dim"])

       
        logposterior_weights = jsp.stats.multivariate_normal.logpdf(measurement, self.measurement_device(point), measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + self.measurement_device.covariance)
        # assert isinstance(logposterior_weights, Float[Array, ""])

        
        return point, logposterior_weights, gaussian_mixture_covariance

    @nnx.jit(static_argnames=['debug'])
    @jaxtyped(typechecker=typechecker)
    def update(self, key: Key[Array, ""], measurement: Float[Array, "measurement_dim"], debug: bool = False):
        key: Key[Array, ""]
        subkey: Key[Array, ""]
        subkeys: Key[Array, "{self.state.shape[0]}"]
        
        key, subkey, *subkeys = jax.random.split(key, 2 + self.state.shape[0])
        subkeys = jnp.array(subkeys)
        # assert isinstance(subkeys, Key[Array, "{self.state.shape[0]}"])

        emperical_covariance = jnp.cov(self.state.T)
        # jax.debug.callback(is_positive_definite, emperical_covariance)

        assert isinstance(emperical_covariance, Float[Array, "{self.state.shape[1]} {self.state.shape[1]}"])
        
        mixture_covariance = self.bandwidth_factor * emperical_covariance
        assert isinstance(mixture_covariance, Float[Array, "{self.state.shape[1]} {self.state.shape[1]}"])

        posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(self.ensemble_gaussian_mixture_filter_update, in_axes=(0, None, None, None))(self.state.value, mixture_covariance, measurement, debug)
        
        if debug:
            assert isinstance(posterior_ensemble, Float[Array, "{self.state.shape[0]} {self.state.shape[1]}"])
            assert isinstance(logposterior_weights, Float[Array, "{self.state.shape[0]}"])
            assert isinstance(posterior_covariances, Float[Array, "{self.state.shape[0]} {self.state.shape[1]} {self.state.shape[1]}"])
            jax.debug.callback(has_nan, posterior_covariances)

        
        # Scale Weights
        m = jnp.max(logposterior_weights)
        g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
        posterior_weights = jnp.exp(logposterior_weights - g)
        posterior_weights = (posterior_weights / jnp.sum(posterior_weights))
        #assert isinstance(posterior_weights, Float[Array, "{self.state.shape[0]}"])

        
        # Prevent Degenerate Particles
        variable = jax.random.choice(subkey, self.state.shape[0], shape=(self.state.shape[0],), p=posterior_weights)        
        posterior_ensemble = posterior_ensemble[variable, ...]
        posterior_covariances = posterior_covariances[variable, ...]
        #jax.debug.callback(has_nan, posterior_covariances)       
        
        posterior_samples = jax.vmap(lambda key, point, cov: jax.random.multivariate_normal(key, mean=point, cov=cov))(subkeys, posterior_ensemble, posterior_covariances)
        #assert isinstance(posterior_weights, Float[Array, "{self.state.shape[0]}"])

        
        self.covariances, self.state = nnx.Variable(posterior_covariances), nnx.Variable(posterior_samples)
        
    
    def iterate(self):
        self.state = nnx.Variable(self.dynamical_system.flow(self.state.value))



def is_positive_definite(M):
    try:
        jnp.linalg.cholesky(M)
        return True
    except:
        assert False, "COVARIANCE MATRIX IS NOT POSITIVE DEFINITE"

def has_nan(M):
    assert not jnp.any(jnp.isnan(M))
   

plot = False
key = jax.random.key(0)
key, subkey = jax.random.split(key)

measurement_device = Distance(jnp.array(1/16))
system = IkedaSystem()

if plot:
    key, subkey = jax.random.split(key)
    attractor = system.generate(subkey)

ensemble_size = 1000
silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study

key, subkey = jax.random.split(key)

prior_ensemble = system.generate(subkey, batch_size=ensemble_size) #jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init


filter = EnGMF(
    dynamical_system=system,
    measurement_device=measurement_device,
    bandwidth_factor=silverman_bandwidth,
    state = prior_ensemble,
)

burn_in_time = 100
measurement_time = 10*burn_in_time

covariances, states = [], []
for t in tqdm(range(burn_in_time + measurement_time), leave=False):
    prior_ensemble = filter.state
    
    key, subkey = jax.random.split(key)
    filter.update(subkey, measurement_device(system.state), debug=False)

    if plot:
        plt.scatter(attractor[:, 0], attractor[:, 1], c='blue', alpha=0.1, s=0.1)
        plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], alpha=0.8, s=10, c='purple', label='Prior')
        plt.scatter(filter.state[:, 0], filter.state[:, 1], alpha=0.8, s=10, c='yellow', label='Posterior')
        plt.scatter(system.state[0], system.state[1], c='lime', s=100, label='True')
        plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))
        plt.show()

    if t >= burn_in_time:
        # if plot:
        #     break
            
        states.append(system.state - jnp.mean(filter.state, axis = 0))
        cov = jnp.cov(filter.state.T)
        try:
            jnp.linalg.cholesky(cov)
            
        except:
            assert False, "COVARIANCE MATRIX IS NOT POSITIVE DEFINITE"
        covariances.append(cov)

    filter.iterate()
    system.iterate()
    

if len(states) != 0:
    e = jnp.expand_dims(jnp.array(states), -1)
    # assert isinstance(e, Float[Array, f"{measurement_time} 1 2 1"])
    P = jnp.expand_dims(jnp.array(covariances), 1)
    # assert isinstance(P, Float[Array, f"{measurement_time} 1 2 2"])
  
    rmse = jnp.mean(jnp.sqrt((1 / (e.shape[1] * e.shape[2] * e.shape[3])) * jnp.sum(e * e, axis=(1,2,3))))
    snees = (1 / e.size) * jnp.sum(jnp.swapaxes(e, -2, -1) @ jnp.linalg.inv(P) @ e)
    print(f"RMSE: {rmse}")
    print(f"SNEES: {snees}")

    
# see how conservitive the normalizing flow is:
# compute number of points in the attractor is in p> 0 on nf
# generate points in nf and see how many are on attractor


# In[33]:


e.shape


# In[ ]:


# key = jax.random.key(0)
# key, subkey = jax.random.split(key)

# measurement_device = Distance(jnp.array(1/16))
# system = IkedaSystem()
# ensemble_size = 400
# silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study

# key, subkey = jax.random.split(key)
# prior_ensemble = jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init

# plt.scatter(prior_ensemble[:, 0], prior_ensemble[:,1])

# for _ in range(12):
#     prior_ensemble = system.flow(prior_ensemble)

# plt.scatter(prior_ensemble[:, 0], prior_ensemble[:,1])


# In[ ]:


# plot = True
# key = jax.random.key(0)
# key, subkey = jax.random.split(key)

# measurement_device = Distance(jnp.array(1/16))
# system = IkedaSystem()

# if plot:
#     key, subkey = jax.random.split(key)
#     attractor = system.generate(subkey)

# ensemble_size = 25
# silverman_bandwidth = 2*(4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study

# key, subkey = jax.random.split(key)
# #prior_ensemble = system.state + jnp.sqrt(measurement_device.covariance) * jax.random.normal(key=subkey, shape=(ensemble_size, 2)) #init

# prior_ensemble = jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init

# for _ in range(12):
#     prior_ensemble = system.flow(prior_ensemble)

# filter = EnGMF(
#     dynamical_system=system,
#     measurement_device=measurement_device,
#     bandwidth_factor=silverman_bandwidth,
#     state = prior_ensemble,
# )

# burn_in_time = 2
# measurement_time = 2*burn_in_time

# covariances, states = [], []
# for t in tqdm(range(burn_in_time + measurement_time), leave=False):
#     prior_ensemble = filter.state
    
#     key, subkey = jax.random.split(key)
#     filter.update(subkey, measurement_device(system.state))

#     if plot:
#         plt.scatter(attractor[:, 0], attractor[:, 1], c='blue', alpha=0.1, s=0.1)
#         plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], alpha=0.8, s=10, c='purple', label='Prior')
#         plt.scatter(filter.state[:, 0], filter.state[:, 1], alpha=0.8, s=10, c='yellow', label='Posterior')
#         plt.scatter(system.state[0], system.state[1], c='lime', s=100, label='True')
#         plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))
#         plt.show()

    
#     if t >= burn_in_time:
#         # if plot:
#         #     break
            
#         states.append(system.state - filter.state)
        
#         emperical_covariance = (1 / (filter.state.shape[0] - 1)) * filter.state.T @ (jnp.eye(filter.state.shape[0]) - (1/filter.state.shape[0]) * jnp.ones(shape=(filter.state.shape[0], filter.state.shape[0]))) @ filter.state

#         covariances.append(emperical_covariance)

    
        
    
#     filter.iterate()
#     system.iterate()
    


# if len(states) != 0:
#     x = jnp.expand_dims(jnp.array(states), -1) # shape = (measurement_time, ensemble_size, state_dim, 1)
#     P = jnp.array(covariances) # shape = (measurement_time, ensemble_size, state_dim, state_dim)
    
#     rmse = jnp.sqrt((1 / x.size) * jnp.sum(x ** 2))
#     snees = (1 / x.size) * jnp.sum(jnp.swapaxes(x, -2, -1) @ jnp.linalg.inv(jnp.expand_dims(P, 1)) @ x)
#     #snees = (1 / x.size) * jnp.sum(jnp.swapaxes(x, -2, -1) @ jnp.linalg.inv(P) @ x)
    
#     print(rmse, snees)

# # 0.07095652325533497 5.178030147006457
# # 


# In[ ]:


# TODO: Compute Covariance Correctly without NaN
# Implement Ikeda discriminator, backwards dynamics then check chp 16 
# Log scale ensemble size vs RMSE SNEES
# Use log linspace


# ## Discriminator Augmented EnGMF

# In[ ]:


# class DEnGMF(nnx.Module):

#     def __init__(self, dynamical_system, measurement_device, discriminator, state, bandwidth_factor):
#         self.dynamical_system = dynamical_system
#         self.measurement_device = measurement_device
#         self.discriminator = discriminator
#         self.state = nnx.Variable(state)
#         self.bandwidth_factor = bandwidth_factor
#         self.covariances = None

#     @nnx.jit
#     def ensemble_gaussian_mixture_filter_update(self, point, prior_mixture_covariance, measurement):
#         measurement_jacobian = jax.jacfwd(self.measurement_device)(point)
#         kalman_gain = prior_mixture_covariance @ measurement_jacobian.T * (1 / (measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + self.measurement_device.covariance))
#         gaussian_mixture_covariance = (jnp.eye(point.shape[-1]) - kalman_gain @ measurement_jacobian.T) @ prior_mixture_covariance
#         point = point - kalman_gain * (self.measurement_device(point) - measurement)
#         logposterior_weights = jsp.stats.norm.logpdf(measurement, self.measurement_device(point), measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + self.measurement_device.covariance)
#         return point, logposterior_weights, gaussian_mixture_covariance

#     def gauss_sample_discriminator(self, key, points, covs, num_samples_per_attempt=10, max_attempts=5):
#         batch_size = points.shape[0]
#         state_dim = points.shape[1]
    
#         accepted_samples = jnp.zeros_like(points)
#         has_accepted = jnp.zeros((batch_size,), dtype=bool)
    
#         def attempt_fn(carry, _):
#             key, accepted_samples, has_accepted = carry
#             key, *subkeys = random.split(key, batch_size + 1)
#             subkeys = jnp.array(subkeys)
    
#             def sample_and_discriminate(subkey, point, cov, accepted):
#                 samples = random.multivariate_normal(subkey, mean=point, cov=cov, shape=(num_samples_per_attempt,))
#                 disc_results = vmap(self.discriminator)(samples)
#                 has_valid = jnp.any(disc_results)
#                 idx = jnp.argmax(disc_results)
#                 accepted_sample = samples[idx]
#                 return accepted_sample, has_valid
    
#             # Apply to all points
#             new_samples, new_accepted = jax.vmap(sample_and_discriminate)(
#                 subkeys, points, covs, has_accepted
#             )
    
#             # Update accepted samples and flags
#             accepted_samples = jnp.where(
#                 has_accepted[:, None], accepted_samples, new_samples
#             )
#             has_accepted = has_accepted | new_accepted
    
#             return (key, accepted_samples, has_accepted), None
    
#         # Run the loop
#         (key, accepted_samples, has_accepted), _ = jax.lax.scan(
#             attempt_fn, (key, accepted_samples, has_accepted), xs=None, length=max_attempts
#         )
    
#         # Handle points without valid samples
#         final_samples = jnp.where(
#             has_accepted[:, None], accepted_samples, points
#         )
    
#         return final_samples


    
#     @nnx.jit
#     def _update(self, key, measurement):
#         key, subkey, *subkeys = jax.random.split(key, 2 + self.state.shape[0])
#         subkeys = jnp.array(subkeys)

#         emperical_covariance = (1 / (self.state.shape[0] - 1)) * self.state.T @ (jnp.eye(self.state.shape[0]) - (1/self.state.shape[0]) * jnp.ones(shape=(self.state.shape[0], self.state.shape[0]))) @ self.state
#         mixture_covariance = self.bandwidth_factor * emperical_covariance
#         posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(self.ensemble_gaussian_mixture_filter_update, in_axes=(0, None, None))(self.state.value, mixture_covariance, measurement)

#         # Scale Weights
#         m = jnp.max(logposterior_weights)
#         g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
#         posterior_weights = jnp.exp(logposterior_weights - g)
#         posterior_weights = (posterior_weights / jnp.sum(posterior_weights)).flatten()
        
#         # Prevent Degenerate Particles
#         variable = jax.random.choice(subkey, self.state.shape[0], shape=(self.state.shape[0],), p=posterior_weights)        
#         posterior_ensemble = posterior_ensemble[variable, ...]
#         posterior_covariances = posterior_covariances[variable, ...]

        
#         posterior_samples = self.gauss_sample_discriminator(subkeys[0], posterior_ensemble, posterior_covariances)
        
#         #posterior_samples = jax.vmap(self.gauss_sample_discriminator)(subkeys, posterior_ensemble, posterior_covariances)
#         return posterior_covariances, posterior_samples
    
#     def update(self, key, measurement):
#         covariances, state = self._update(key, measurement)
#         self.covariances, self.state = nnx.Variable(covariances), nnx.Variable(state)

#     def iterate(self):
#         self.state = nnx.Variable(self.dynamical_system.flow(self.state))


# plot = False
# key = jax.random.key(0)
# key, subkey = jax.random.split(key)

# measurement_device = Distance(jnp.array(1/16))
# system = IkedaSystem()

# if plot:
#     key, subkey = jax.random.split(key)
#     attractor = system.generate(subkey)

# ensemble_size = 1_000
# silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study

# key, subkey = jax.random.split(key)
# prior_ensemble = system.state + jnp.sqrt(measurement_device.covariance) * jax.random.normal(key=subkey, shape=(ensemble_size, 2)) #init

# filter = DEnGMF(
#     dynamical_system=system,
#     measurement_device=measurement_device,
#     discriminator=discriminator,
#     bandwidth_factor=silverman_bandwidth,
#     state = prior_ensemble,
# )

# burn_in_time = 100
# measurement_time = 10*burn_in_time

# covariances, states = [], []
# for t in tqdm(range(burn_in_time + measurement_time), leave=False):
#     prior_ensemble = filter.state
    
#     key, subkey = jax.random.split(key)
#     filter.update(subkey, measurement_device(system.state))

#     if plot:
#         plt.scatter(attractor[:, 0], attractor[:, 1], c='blue', alpha=0.1, s=0.1)
#         plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], alpha=0.35, s=10, c='purple', label='Prior')
#         plt.scatter(filter.state[:, 0], filter.state[:, 1], alpha=0.35, s=10, c='yellow', label='Posterior')
#         plt.scatter(system.state[0], system.state[1], c='lime', s=100, label='True')
#         plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))
#         plt.show()

    
#     if t >= burn_in_time:
#         if plot:
#             break
#         covariances.append(filter.covariances)
#         states.append(system.state - filter.state)
    
#     filter.iterate()
#     system.iterate()


# if len(states) != 0:
#     x = jnp.expand_dims(jnp.array(states), -1) # shape = (measurement_time, ensemble_size, state_dim, 1)
#     P = jnp.array(covariances) # shape = (measurement_time, ensemble_size, state_dim, state_dim)
    
#     rmse = jnp.sqrt((1 / x.size) * jnp.sum(x ** 2))
#     snees = (1 / x.size) * jnp.sum(jnp.swapaxes(x, -2, -1) @ jnp.linalg.inv(P) @ x)
    
#     print(rmse, snees)


# In[ ]:


# class DEnGMF(nnx.Module):

#     def __init__(self, dynamical_system, measurement_device, discriminator, state, bandwidth_factor):
#         self.dynamical_system = dynamical_system
#         self.measurement_device = measurement_device
#         self.discriminator = discriminator
#         self.state = nnx.Variable(state)
#         self.bandwidth_factor = bandwidth_factor
#         self.covariances = None

#     @nnx.jit
#     def ensemble_gaussian_mixture_filter_update(self, point, prior_mixture_covariance, measurement):
#         measurement_jacobian = jax.jacfwd(self.measurement_device)(point)
#         kalman_gain = prior_mixture_covariance @ measurement_jacobian.T * (1 / (measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + self.measurement_device.covariance))
#         gaussian_mixture_covariance = (jnp.eye(point.shape[-1]) - kalman_gain @ measurement_jacobian.T) @ prior_mixture_covariance
#         point = point - kalman_gain * (self.measurement_device(point) - measurement)
#         logposterior_weights = jsp.stats.norm.logpdf(measurement, self.measurement_device(point), measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + self.measurement_device.covariance)
#         return point, logposterior_weights, gaussian_mixture_covariance

#     @nnx.jit
#     def gauss_sample_discriminator(self, key, points, covs, num_samples_per_attempt=10, max_attempts=5):
#         batch_size = points.shape[0]
#         state_dim = points.shape[1]
    
#         accepted_samples = jnp.zeros_like(points)
#         has_accepted = jnp.zeros((batch_size,), dtype=bool)
    
#         def attempt_fn(carry, _):
#             key, accepted_samples, has_accepted = carry
#             key, *subkeys = random.split(key, batch_size + 1)
#             subkeys = jnp.array(subkeys)
    
#             def sample_and_discriminate(subkey, point, cov, accepted):
#                 samples = random.multivariate_normal(subkey, mean=point, cov=cov, shape=(num_samples_per_attempt,))
#                 disc_results = vmap(self.discriminator)(samples)
#                 has_valid = jnp.any(disc_results)
#                 idx = jnp.argmax(disc_results)
#                 accepted_sample = samples[idx]
#                 return accepted_sample, has_valid
    
#             # Apply to all points
#             new_samples, new_accepted = jax.vmap(sample_and_discriminate)(
#                 subkeys, points, covs, has_accepted
#             )
    
#             # Update accepted samples and flags
#             accepted_samples = jnp.where(
#                 has_accepted[:, None], accepted_samples, new_samples
#             )
#             has_accepted = has_accepted | new_accepted
    
#             return (key, accepted_samples, has_accepted), None
    
#         # Run the loop
#         (key, accepted_samples, has_accepted), _ = jax.lax.scan(
#             attempt_fn, (key, accepted_samples, has_accepted), xs=None, length=max_attempts
#         )
    
#         # Handle points without valid samples
#         final_samples = jnp.where(
#             has_accepted[:, None], accepted_samples, points
#         )
    
#         return final_samples


    
#     @nnx.jit
#     def _update(self, key, measurement):
#         key, subkey, *subkeys = jax.random.split(key, 2 + self.state.shape[0])
#         subkeys = jnp.array(subkeys)

#         emperical_covariance = (1 / (self.state.shape[0] - 1)) * self.state.T @ (jnp.eye(self.state.shape[0]) - (1/self.state.shape[0]) * jnp.ones(shape=(self.state.shape[0], self.state.shape[0]))) @ self.state
#         mixture_covariance = self.bandwidth_factor * emperical_covariance
#         posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(self.ensemble_gaussian_mixture_filter_update, in_axes=(0, None, None))(self.state.value, mixture_covariance, measurement)

#         # Scale Weights
#         m = jnp.max(logposterior_weights)
#         g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
#         posterior_weights = jnp.exp(logposterior_weights - g)
#         posterior_weights = (posterior_weights / jnp.sum(posterior_weights)).flatten()
        
#         # Prevent Degenerate Particles
#         variable = jax.random.choice(subkey, self.state.shape[0], shape=(self.state.shape[0],), p=posterior_weights)        
#         posterior_ensemble = posterior_ensemble[variable, ...]
#         posterior_covariances = posterior_covariances[variable, ...]

        
#         posterior_samples = self.gauss_sample_discriminator(subkeys[0], posterior_ensemble, posterior_covariances)
        
#         #posterior_samples = jax.vmap(self.gauss_sample_discriminator)(subkeys, posterior_ensemble, posterior_covariances)
#         return posterior_covariances, posterior_samples
    
#     def update(self, key, measurement):
#         covariances, state = self._update(key, measurement)
#         self.covariances, self.state = nnx.Variable(covariances), nnx.Variable(state)

#     def iterate(self):
#         self.state = nnx.Variable(self.dynamical_system.flow(self.state))


# plot = False
# key = jax.random.key(0)
# key, subkey = jax.random.split(key)

# measurement_device = Distance(jnp.array(1/16))
# system = IkedaSystem()

# if plot:
#     key, subkey = jax.random.split(key)
#     attractor = system.generate(subkey)

# ensemble_size = 1_000
# silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study

# key, subkey = jax.random.split(key)
# prior_ensemble = system.state + jnp.sqrt(measurement_device.covariance) * jax.random.normal(key=subkey, shape=(ensemble_size, 2)) #init

# filter = DEnGMF(
#     dynamical_system=system,
#     measurement_device=measurement_device,
#     discriminator=discriminator,
#     bandwidth_factor=silverman_bandwidth,
#     state = prior_ensemble,
# )

# burn_in_time = 100
# measurement_time = 10*burn_in_time

# covariances, states = [], []
# for t in tqdm(range(burn_in_time + measurement_time), leave=False):
#     prior_ensemble = filter.state
    
#     key, subkey = jax.random.split(key)
#     filter.update(subkey, measurement_device(system.state))

#     if plot:
#         plt.scatter(attractor[:, 0], attractor[:, 1], c='blue', alpha=0.1, s=0.1)
#         plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], alpha=0.35, s=10, c='purple', label='Prior')
#         plt.scatter(filter.state[:, 0], filter.state[:, 1], alpha=0.35, s=10, c='yellow', label='Posterior')
#         plt.scatter(system.state[0], system.state[1], c='lime', s=100, label='True')
#         plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))
#         plt.show()

    
#     if t >= burn_in_time:
#         if plot:
#             break
#         covariances.append(filter.covariances)
#         states.append(system.state - filter.state)
    
#     filter.iterate()
#     system.iterate()


# if len(states) != 0:
#     x = jnp.expand_dims(jnp.array(states), -1) # shape = (measurement_time, ensemble_size, state_dim, 1)
#     P = jnp.array(covariances) # shape = (measurement_time, ensemble_size, state_dim, state_dim)
    
#     rmse = jnp.sqrt((1 / x.size) * jnp.sum(x ** 2))
#     snees = (1 / x.size) * jnp.sum(jnp.swapaxes(x, -2, -1) @ jnp.linalg.inv(P) @ x)
    
#     print(rmse, snees)


# In[ ]:


# class DEnGMF(nnx.Module):

#     def __init__(self, dynamical_system, measurement_device, discriminator, state, bandwidth_factor):
#         self.dynamical_system = dynamical_system
#         self.measurement_device = measurement_device
#         self.discriminator = discriminator
#         self.state = nnx.Variable(state)
#         self.bandwidth_factor = bandwidth_factor
#         self.covariances = None

#     @nnx.jit
#     def ensemble_gaussian_mixture_filter_update(self, point, prior_mixture_covariance, measurement):
#         measurement_jacobian = jax.jacfwd(self.measurement_device)(point)
#         kalman_gain = prior_mixture_covariance @ measurement_jacobian.T * (1 / (measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + self.measurement_device.covariance))
#         gaussian_mixture_covariance = (jnp.eye(point.shape[-1]) - kalman_gain @ measurement_jacobian.T) @ prior_mixture_covariance
#         point = point - kalman_gain * (self.measurement_device(point) - measurement)
#         logposterior_weights = jsp.stats.norm.logpdf(measurement, self.measurement_device(point), measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + self.measurement_device.covariance)
#         return point, logposterior_weights, gaussian_mixture_covariance

#     @nnx.jit
#     def gauss_sample_discriminator(self, key, point, cov):
#         key, subkey = jax.random.split(key)
        
#         def cond_fn(val):
#             return jnp.any(self.discriminator(jax.random.multivariate_normal(val, mean=point, cov=cov)))

#         def body_fn(val):
#             key, subkey = jax.random.split(val)
#             return subkey

#         correct_key = lax.while_loop(cond_fn, body_fn, subkey)
        
#         return jax.random.multivariate_normal(correct_key, mean=point, cov=cov)
    

#     @nnx.jit
#     def _update(self, key, measurement):
#         key, subkey, *subkeys = jax.random.split(key, 2 + self.state.shape[0])
#         subkeys = jnp.array(subkeys)

#         emperical_covariance = (1 / (self.state.shape[0] - 1)) * self.state.T @ (jnp.eye(self.state.shape[0]) - (1/self.state.shape[0]) * jnp.ones(shape=(self.state.shape[0], self.state.shape[0]))) @ self.state
#         mixture_covariance = self.bandwidth_factor * emperical_covariance
#         posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(self.ensemble_gaussian_mixture_filter_update, in_axes=(0, None, None))(self.state.value, mixture_covariance, measurement)

#         # Scale Weights
#         m = jnp.max(logposterior_weights)
#         g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
#         posterior_weights = jnp.exp(logposterior_weights - g)
#         posterior_weights = (posterior_weights / jnp.sum(posterior_weights)).flatten()
        
#         # Prevent Degenerate Particles
#         variable = jax.random.choice(subkey, self.state.shape[0], shape=(self.state.shape[0],), p=posterior_weights)        
#         posterior_ensemble = posterior_ensemble[variable, ...]
#         posterior_covariances = posterior_covariances[variable, ...]
        
#         posterior_samples = jax.vmap(self.gauss_sample_discriminator)(subkeys, posterior_ensemble, posterior_covariances)
#         return posterior_covariances, posterior_samples
    
#     def update(self, key, measurement):
#         covariances, state = self._update(key, measurement)
#         self.covariances, self.state = nnx.Variable(covariances), nnx.Variable(state)

#     def iterate(self):
#         self.state = nnx.Variable(self.dynamical_system.flow(self.state))


# plot = False
# key = jax.random.key(0)
# key, subkey = jax.random.split(key)

# measurement_device = Distance(jnp.array(1/16))
# system = IkedaSystem()

# if plot:
#     key, subkey = jax.random.split(key)
#     attractor = system.generate(subkey)

# ensemble_size = 1_000
# silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study

# key, subkey = jax.random.split(key)
# prior_ensemble = system.state + jnp.sqrt(measurement_device.covariance) * jax.random.normal(key=subkey, shape=(ensemble_size, 2)) #init

# filter = DEnGMF(
#     dynamical_system=system,
#     measurement_device=measurement_device,
#     discriminator=discriminator,
#     bandwidth_factor=silverman_bandwidth,
#     state = prior_ensemble,
# )

# burn_in_time = 100
# measurement_time = 10*burn_in_time

# covariances, states = [], []
# for t in tqdm(range(burn_in_time + measurement_time), leave=False):
#     prior_ensemble = filter.state
    
#     key, subkey = jax.random.split(key)
#     filter.update(subkey, measurement_device(system.state))

#     if plot:
#         plt.scatter(attractor[:, 0], attractor[:, 1], c='blue', alpha=0.1, s=0.1)
#         plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], alpha=0.35, s=10, c='purple', label='Prior')
#         plt.scatter(filter.state[:, 0], filter.state[:, 1], alpha=0.35, s=10, c='yellow', label='Posterior')
#         plt.scatter(system.state[0], system.state[1], c='lime', s=100, label='True')
#         plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))
#         plt.show()

    
#     if t >= burn_in_time:
#         if plot:
#             break
#         covariances.append(filter.covariances)
#         states.append(system.state - filter.state)
    
#     filter.iterate()
#     system.iterate()
    


# if len(states) != 0:
#     x = jnp.expand_dims(jnp.array(states), -1) # shape = (measurement_time, ensemble_size, state_dim, 1)
#     P = jnp.array(covariances) # shape = (measurement_time, ensemble_size, state_dim, state_dim)
    
#     rmse = jnp.sqrt((1 / x.size) * jnp.sum(x ** 2))
#     snees = (1 / x.size) * jnp.sum(jnp.swapaxes(x, -2, -1) @ jnp.linalg.inv(P) @ x)
    
#     print(rmse, snees)


# # Results

# In[ ]:


def run_engmf(ensemble_size):
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    
    measurement_device = Distance(jnp.array(1/16))
    system = IkedaSystem()
    
    if plot:
        key, subkey = jax.random.split(key)
        attractor = system.generate(subkey)
    
    silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study
    
    key, subkey = jax.random.split(key)
    
    prior_ensemble = system.generate(subkey, batch_size=ensemble_size) #jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init
    
    
    filter = EnGMF(
        dynamical_system=system,
        measurement_device=measurement_device,
        bandwidth_factor=silverman_bandwidth,
        state = prior_ensemble,
    )
    
    burn_in_time = 100
    measurement_time = 10*burn_in_time
    
    covariances, states = [], []
    for t in tqdm(range(burn_in_time + measurement_time), leave=False):
        prior_ensemble = filter.state
        
        key, subkey = jax.random.split(key)
        filter.update(subkey, measurement_device(system.state))
    
        if plot:
            plt.scatter(attractor[:, 0], attractor[:, 1], c='blue', alpha=0.1, s=0.1)
            plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], alpha=0.8, s=10, c='purple', label='Prior')
            plt.scatter(filter.state[:, 0], filter.state[:, 1], alpha=0.8, s=10, c='yellow', label='Posterior')
            plt.scatter(system.state[0], system.state[1], c='lime', s=100, label='True')
            plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))
            plt.show()
    
        
        if t >= burn_in_time:
            # if plot:
            #     break
                
            states.append(system.state - filter.state)
            cov = jnp.cov(filter.state.T)
            try:
                jnp.linalg.cholesky(cov)
                
            except:
                assert False, "COVARIANCE MATRIX IS NOT POSITIVE DEFINITE"
            covariances.append(cov)
    
        filter.iterate()
        system.iterate()
        
    
    
    if len(states) != 0:
        e = jnp.expand_dims(jnp.array(states), -1)
        # assert isinstance(e, Float[Array, f"{measurement_time} {ensemble_size} 2 1"])
        P = jnp.expand_dims(jnp.array(covariances), 1)
        # assert isinstance(P, Float[Array, f"{measurement_time} 1 2 2"])
      
        rmse = jnp.mean(jnp.sqrt((1 / (e.shape[1] * e.shape[2] * e.shape[3])) * jnp.sum(e * e, axis=(1,2,3))))
        snees = (1 / e.size) * jnp.sum(jnp.swapaxes(e, -2, -1) @ jnp.linalg.inv(P) @ e)
        print(f"RMSE: {rmse}")
        print(f"SNEES: {snees}")

        return rmse, snees
        
    # see how conservitive the normalizing flow is:
    # compute number of points in the attractor is in p> 0 on nf
    # generate points in nf and see how many are on attractor

for ensemble_size in jnp.geomspace(10, 100_000, num=40, dtype=int):
    print("Ensemble Size: {ensemble_size}")
    run_engmf(int(ensemble_size))

run_engmf(int(5 * 10 ** 4))

# jax.vmap(run_engmf)(jnp.geomspace(10, 1000, num=20, dtype=int))


# ## Results Discriminator based EnGMF

# Make plots of SNEES and RMSe 
# Implement Normalizing Flow using Discriminator
# True Discriminator Ikeda Map

# In[ ]:


# 


# for bandwidth_factor in np.linspace(0.01, 0.13, 10):
    
#     plot = False
#     key = jax.random.key(0)
#     key, subkey = jax.random.split(key)
    
#     measurement_device = Distance(jnp.array(1/16))
#     system = IkedaSystem()
    
#     if plot:
#         key, subkey = jax.random.split(key)
#         attractor = system.generate(subkey)
    
#     ensemble_size = 10_000
#     silverman_bandwidth = bandwidth_factor #(4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study
    
#     key, subkey = jax.random.split(key)
#     prior_ensemble = system.state + jnp.sqrt(measurement_device.covariance) * jax.random.normal(key=subkey, shape=(ensemble_size, 2)) #init
    
#     filter = DEnGMF(
#         dynamical_system=system,
#         measurement_device=measurement_device,
        
#         bandwidth_factor=silverman_bandwidth,
#         state = prior_ensemble,
#     )
    
#     burn_in_time = 100
#     measurement_time = 10*burn_in_time
    
#     covariances, states = [], []
#     for t in tqdm(range(burn_in_time + measurement_time), leave=False):
#         prior_ensemble = filter.state
        
#         key, subkey = jax.random.split(key)
#         filter.update(subkey, measurement_device(system.state))
    
#         if plot:
#             plt.scatter(attractor[:, 0], attractor[:, 1], c='blue', alpha=0.1, s=0.1)
#             plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], alpha=0.35, s=10, c='purple', label='Prior')
#             plt.scatter(filter.state[:, 0], filter.state[:, 1], alpha=0.35, s=10, c='yellow', label='Posterior')
#             plt.scatter(system.state[0], system.state[1], c='lime', s=100, label='True')
#             plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))
#             plt.show()
    
        
#         if t >= burn_in_time:
#             if plot:
#                 break
#             covariances.append(filter.covariances)
#             states.append(system.state - filter.state)
        
#         filter.iterate()
#         system.iterate()
        
    
    
#     if len(states) != 0:
#         x = jnp.expand_dims(jnp.array(states), -1) # shape = (measurement_time, ensemble_size, state_dim, 1)
#         P = jnp.array(covariances) # shape = (measurement_time, ensemble_size, state_dim, state_dim)
        
#         rmse = jnp.sqrt((1 / x.size) * jnp.sum(x ** 2))
#         snees = (1 / x.size) * jnp.sum(jnp.swapaxes(x, -2, -1) @ jnp.linalg.inv(P) @ x)
        
#         print(bandwidth_factor, rmse, snees)


# 0.0 nan nan
# 
# 0.014444444444444446 0.05334210752012345 32.073140328723404
# 
# 0.02888888888888889 0.056609108643425055 16.13561766538242
# 
# 0.043333333333333335 0.05990563647788256 10.879738437247315
# 
# 0.05777777777777778 0.06373132281416906 8.247150365313095
# 
# 0.07222222222222223 0.06790614053438164 6.683052676391782
# 
# 0.08666666666666667 0.07243906662868915 5.659756733998286
# 
# 0.10111111111111112 0.07777917782030805 4.92905444085195
# 
# 0.11555555555555556 0.08359115188530089 4.394020263142449
# 
# 0.13 0.09034964532778407 3.999876545061847
