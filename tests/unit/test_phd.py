import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Key

from data_science_utils.dynamical_systems import Vicsek
from data_science_utils.filters.abc import AbstractFilter
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from data_science_utils.statistics import GMM


class AbstractMultiTargetFilter(eqx.Module, strict=True):

    @abc.abstractmethod
    def update(
        self,
        key: Key[Array, "..."],
        prior_ensemble: Float[Array, "batch_size state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_size state_dim"]:
        raise NotImplementedError


class PHDFilter(AbstractMultiTargetFilter):
    dynamical_system: object
    measurement_system: object
    survival_probability: float
    birth_gmm: GMM
    tracking_multi_gmm: object
    max_components: int

    birth_components_per_step: int
    birth_weight: float

    def predict(self, states, mask, key):
        """Predict step for CV model with filter-level birth/death."""

        survival_key, dynamical_system_key, birth_key = jax.random.split(key, 3)

        # Survival
        survived = jax.random.bernoulli(
            survival_key, shape=(self.max_particles,), p=self.survival_probability
        )
        mask = mask & (survived)

        # Of the points that survive, propogate them forward
        states, mask = dynamical_system.forward(states, mask, dynamical_system_key)

        # Birth 10 point from the Gaussian
        birth_sample = birth_gmm.sample(birth_key)

    def update(
        self,
        key: Key[Array, "..."],
        prior_ensemble: Float[Array, "state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
        mask: Bool[Array, "max_measurements"],
    ) -> Float[Array, "batch_size state_dim"]:
        return jnp.array()


key = jax.random.key(0)
system = Vicsek(n_particles=10)
key, subkey = jax.random.split(key)
true_state = system.initial_state(subkey)
initial_belief = GMM(
    means=jnp.array([[0.0, 0.0]]),
    covs=jnp.array([[[1.0, 0.0], [0.0, 1.0]]]),
    weights=jnp.array([1e-16]),
)

for _ in range(10):
    # Births
    # Add 10 more Gaussian Terms

    # Gate Measurements

    # Update
    #

    key, subkey = jax.random.split(key)
    true_state = system.flow(jax.random.key(10), 0.0, 1.0, true_state)
    break
