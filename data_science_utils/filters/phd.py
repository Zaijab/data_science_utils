from data_science_utils.filters import AbstractFilter


class PHDFilter(AbstractFilter):
    pass


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


if __name__ == "__main__":

    stochastic_filter = PHDFilter()
