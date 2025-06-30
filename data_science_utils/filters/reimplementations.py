import jax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

@jax.jit
@jax.vmap
def sample_gaussian_mixture(key, point, cov):
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


class EnGMF(eqx.Module, strict=True):

    dynamical_system: AbstractDynamicalSystem
    measurement_system: AbstractMeasurementSystem
    sampling_function: Callable = jax.tree_util.Partial(sample_gaussian_mixture)
    ensemble_size = 100
    debug: bool = False

    def silverman_kde_estimate(self, means):
        weights = jnp.ones(self.ensemble_size) / self.ensemble_size
        n, d = self.ensemble_size, self.dynamical_system.dimension
        silverman_beta = (((4) / (d + 2)) ** ((2) / (d + 4))) #* (n ** ((-2) / (d + 4)))
        covs = jnp.tile(silverman_beta * jnp.cov(means.T), reps=(self.ensemble_size, 1, 1))
        components = distrax.MultivariateNormalFullCovariance(loc=means, covariance_matrix=covs)
        return distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(probs=weights),
            components_distribution=components
        )
        

    @eqx.filter_vmap
    def initialize(
        self,
        key: Key[Array, "..."],
        initial_belief: distrax.Distribution
    ) -> distrax.Distribution:
        """
        Discretize our PDF into an empirical distribution.
        EnGMF propagates particles in place of a parameter.
        """
        means = initial_belief.sample(seed=key, sample_shape=(self.ensemble_size,))
        return self.silverman_kde_estimate(means)    
    

    def predict(
        self,
        key: Key[Array, "..."],
        posterior_distribution: distrax.Distribution,
        measurement: Float[Array, "measurement_dim"],
    ) -> distrax.Distribution:
        new_means = eqx.filter_vmap(dynamical_system.flow)(0.0, 1.0, posterior_belief.components_distribution.loc)
        return self.silverman_kde_estimate(new_means)


    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, ""],
        prior_belief: distrax.Distribution,
        measurement: Float[Array, "measurement_dim"],
    ) -> Float[Array, "batch_size state_dim"]:
        subkey: Key[Array, ""]
        subkeys: Key[Array, "batch_dim"]
        prior_ensemble = prior_belief.components_distribution.loc

        key, subkey, *subkeys = jax.random.split(key, 2 + prior_ensemble.shape[0])
        subkeys = jnp.array(subkeys)

        if self.debug:
            assert isinstance(subkeys, Key[Array, " batch_dim"])

        bandwidth = (
            (4) / (prior_ensemble.shape[0] * (prior_ensemble.shape[-1] + 2))
        ) ** ((2) / (prior_ensemble.shape[-1] + 4))
        emperical_covariance = jnp.cov(prior_ensemble.T)  # + 1e-8 * jnp.eye(2)

        state_dim = emperical_covariance.shape[0]
        i_indices = jnp.arange(state_dim)[:, None]
        j_indices = jnp.arange(state_dim)[None, :]
        distances = jnp.abs(i_indices - j_indices)

        # Gaussian localization with radius L
        L = 3.0  # or 4.0
        rho = jnp.exp(-(distances**2) / (2 * L**2))

        emperical_covariance = emperical_covariance * rho

        if self.debug:
            jax.debug.callback(is_positive_definite, emperical_covariance)
            assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])

        mixture_covariance = bandwidth * emperical_covariance
        if self.debug:
            assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

        posterior_ensemble, posterior_covariances, logposterior_weights = jax.vmap(
            self.update_point,
            in_axes=(0, None, None),
        )(
            prior_ensemble,
            mixture_covariance,
            measurement,
        )

        if self.debug:
            assert isinstance(posterior_ensemble, Float[Array, "batch_dim state_dim"])
            assert isinstance(logposterior_weights, Float[Array, "batch_dim"])
            assert isinstance(
                posterior_covariances, Float[Array, "batch_dim state_dim state_dim"]
            )
            jax.self.debug.callback(has_nan, posterior_covariances)

        # Scale Weights
        m = jnp.max(logposterior_weights)
        g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
        posterior_weights = jnp.exp(logposterior_weights - g)
        posterior_weights = posterior_weights / jnp.sum(posterior_weights)
        if self.debug:
            assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        # Prevent Degenerate Particles
        variable = jax.random.choice(
            subkey,
            prior_ensemble.shape[0],
            shape=(prior_ensemble.shape[0],),
            p=posterior_weights,
        )
        posterior_ensemble = posterior_ensemble[variable, ...]
        posterior_covariances = posterior_covariances[variable, ...]

        if self.debug:
            jax.self.debug.callback(has_nan, posterior_covariances)

        posterior_samples = self.sampling_function(
            subkeys, posterior_ensemble, posterior_covariances
        )
        if self.debug:
            assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        
        return distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(probs=posterior_weights),
            components_distribution=distrax.MultivariateNormalFullCovariance(loc=posterior_samples, covariance_matrix=posterior_covariances)
        )


    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update_point(
        self,
        point: Float[Array, "state_dim"], # x_{k|k-1}^(i)
        prior_mixture_covariance: Float[Array, "state_dim state_dim"], # \hat{P}_{k|k-1}^(i)
        measurement: Float[Array, "measurement_dim"], # z
    ):
        ### (eq. 21)
        # H_{k}^{(i)} = \frac{\partial h}{\partial x} (x_{k|k-1}^(i))
        measurement_jacobian = jax.jacfwd(self.measurement_system)(point)

        if self.debug:
            assert isinstance(
                measurement_jacobian, Float[Array, "measurement_dim state_dim"]
            ), measurement_jacobian.shape
            
        ### (eq. 19)
        # S_k^(i) = H_k^(i) P_{k | k - 1}^(i) H_k^(i) + R

        innovation_cov = (
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + self.measurement_system.covariance
        )
        innovation_cov = (innovation_cov + innovation_cov.T) / 2  # Symmetrize
        if self.debug:
            assert isinstance(innovation_cov, Float[Array, "measurement_dim measurement_dim"])

        ### (eq. 18)

        # K_k^(i) = P H.T S^(-1)
        kalman_gain = jax.scipy.linalg.solve(innovation_cov, measurement_jacobian @ prior_mixture_covariance).T

        if self.debug:
            assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
            # jax.debug.print("Hello {}", jnp.allclose(kalman_gain_unstable, kalman_gain))

        ### (eq. 17)
        
        # \hat{P}_{k | k}^{(i)} = \hat{P}_{k | k - 1}^{(i)} - K_{k}^{(i)} H_{k}^{(i)} \hat{P}_{k | k - 1}^{(i)}
        # We may, of course, factor to the right
        # \hat{P}_{k | k}^{(i)} = ( I - K_{k}^{(i)} H_{k}^{(i)} ) \hat{P}_{k | k - 1}^{(i)}
        posterior_covariance = (
            jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
        ) @ prior_mixture_covariance

        if self.debug:
            assert isinstance(
                posterior_covariance, Float[Array, "state_dim state_dim"]
            )

        ### (eq. 16)
        
        # \hat{x}_{k | k}^{(i)} = \hat{x}_{k | k - 1}^{(i)} + K_{k}^{(i)} ( z - h(\hat{x}_{k | k - 1}^{(i)}))
        posterior_point = point + kalman_gain @ (measurement - self.measurement_system(point))

        if self.debug:
            assert isinstance(point, Float[Array, "state_dim"])
            assert self.measurement_system(point).shape == measurement.shape
            assert posterior_point.shape == point.shape

        ### (eq. 22)
        # \xi_{k}^{(i)} = N(z; \hat{x}_{k | k - 1}^{(i)}, S_{k}^{(i)})
        logposterior_weight = jsp.stats.multivariate_normal.logpdf(
            measurement,
            mean=self.measurement_system(point),
            cov=innovation_cov
        )

        if self.debug:
            assert isinstance(logposterior_weight, Float[Array, ""])

        return posterior_point, posterior_covariance, logposterior_weight

key = jax.random.key(0)
key, subkey = jax.random.split(key)
initial_belief = distrax.MultivariateNormalFullCovariance(dynamical_system.initial_state(), jnp.eye(6))

# f_{0 | 0}(X | Z_0)
key, subkey = jax.random.split(key)
posterior_belief = stochastic_filter.initialize(subkey, initial_belief)

# f_{1 | 0}(X | Z_0)
key, subkey = jax.random.split(key)
prior_belief = stochastic_filter.predict(subkey, posterior_belief, None)

# f_{1 | 1}(X | Z_1)
key, update_key, measurement_key = jax.random.split(key, 3)
measurement = measurement_system(true_state, measurement_key)
posterior_belief = stochastic_filter.update(update_key, prior_belief, measurement)
