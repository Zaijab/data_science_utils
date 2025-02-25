%cd ~/koa_scratch/data_science_utils
#!rm -r /home/zjabbar/koa_scratch/jax_cache/*

import os
#os.environ["JAX_LOG_COMPILES"] = "1"
#os.environ["TF_CUDA_COMPUTE_CAPABILITIES"] = "7.0"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"


import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_compilation_cache_dir", "/home/zjabbar/koa_scratch/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")



import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from data_science_utils.dynamical_systems import IkedaSystem, flow, generate, ikeda_attractor_discriminator

key = jax.random.key(0)
key, subkey = jax.random.split(key)
ikeda = IkedaSystem(u=0.9)
batch = generate(subkey)
batch = flow(batch)

grid_spacing = 500
x = jnp.linspace(-0.5, 2, grid_spacing)
y = jnp.linspace(-2.5, 1, grid_spacing)
XX, YY = jnp.meshgrid(x, y)
grid = jnp.dstack([XX, YY])
grid_points = grid.reshape(-1, 2)
threshold = 0.01

import matplotlib.pyplot as plt

labels = ikeda_attractor_discriminator(grid_points, ninverses=10)
labels_grid = labels.reshape(grid_spacing,grid_spacing)
plt.figure(figsize=(8, 6))
plt.contourf(XX, YY, labels_grid)
plt.title('Discriminator Output')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


import json
import math
import os
import time
from functools import partial
from typing import Sequence

## JAX
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from IPython.display import set_matplotlib_formats
from jax import jit, lax, random
from jax.scipy.special import gammaln
from jax.scipy.stats import norm
from matplotlib.colors import to_rgb
from tqdm.notebook import tqdm

debug = False
key = jax.random.key(0)
key, subkey = jax.random.split(key)

from data_science_utils.dynamical_systems import flow, generate

class DenseNetwork(nnx.Module):
    hidden_dim: int
    num_hidden_layers: int
    output_dim: int

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim

        self.layers = [nnx.Linear(self.input_dim, self.hidden_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)]        
        for _ in range(self.num_hidden_layers):
            self.layers.append(nnx.Linear(self.hidden_dim, self.hidden_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64))
        self.output_layer = nnx.Linear(self.hidden_dim, self.output_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nnx.gelu(x)
        x = self.output_layer(x)
        return x

class CouplingLayer(nnx.Module):
    input_dim: int
    hidden_dim: int
    num_hidden_layers: int
    swap: bool = False

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, swap: bool, rngs: nnx.Rngs):
        super().__init__()
        self.input_dim = input_dim        
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.swap = swap

        self.s_net = DenseNetwork(
            input_dim=self.input_dim // 2,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            output_dim=self.input_dim // 2,
            rngs=rngs
        )
        self.t_net = DenseNetwork(
            input_dim=self.input_dim // 2,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            output_dim=self.input_dim // 2,
            rngs=rngs
        )

    def __call__(self, x, reverse=False):
        if self.swap:
            x1, x2 = jnp.split(x, 2, axis=-1)
            x1, x2 = x2, x1
        else:
            x1, x2 = jnp.split(x, 2, axis=-1)

        s = self.s_net(x1)
        #s = 5*jnp.tanh(s)
        t = self.t_net(x1)

        if reverse:
            y2 = (x2 - t) * jnp.exp(-s)
            log_det_jacobian = -jnp.sum(s,axis=-1)
        else:
            y2 = x2 * jnp.exp(s) + t
            log_det_jacobian = jnp.sum(s,axis=-1)

        if self.swap:
            y = jnp.concatenate([y2, x1], axis=-1)
        else:
            y = jnp.concatenate([x1, y2], axis=-1)

        return y, log_det_jacobian

class InvertibleNN(nnx.Module):
    input_dim: int = 2
    num_coupling_layers: int = 5
    hidden_dim: int = 128
    num_hidden_layers: int = 4

    def __init__(self, input_dim=2, num_coupling_layers=2, hidden_dim=32, num_hidden_layers=2, rngs=nnx.Rngs()):
        super().__init__()
        self.input_dim = input_dim
        self.num_coupling_layers = num_coupling_layers
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.coupling_layers = []
        for i in range(self.num_coupling_layers):
            swap = i % 2 == 1
            layer = CouplingLayer(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_hidden_layers=self.num_hidden_layers,
                swap=swap,
                rngs=rngs
            )
            self.coupling_layers.append(layer)

    def __call__(self, x, reverse=False):
        log_det_jacobian = 0
        if not reverse:
            for layer in self.coupling_layers:
                x, ldj = layer(x)
                log_det_jacobian += ldj
        else:
            for layer in reversed(self.coupling_layers):
                x, ldj = layer(x, reverse=True)
                log_det_jacobian += ldj
        return x, log_det_jacobian
