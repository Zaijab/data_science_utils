from data_science_utils.filtering import ensemble_gaussian_mixture_filter_update_ensemble
from data_science_utils.measurement_functions import norm_measurement

debug = False
ensemble_sizes = range(3, 21, 1)

key = jax.random.key(42)
key, ensemble_key = jax.random.split(key)

measurement_covariance = jnp.array([[1.0]])
measurement_device = norm_measurement

true_state = jnp.array([1.25, 0])
prior_state = jnp.array([1.25, 0])
ensemble = prior_state + (1 / 16) * jax.random.normal(subkey, shape=(num_samples, 2))

for _ in range(10):
    key, subkey = jax.random.split(key)
    measurement = measurement_device

    num_samples = 100
    inflation_factor = 1.01
    key, subkey = jax.random.split(key)


    mean = jnp.mean(ensemble, axis=0)
    ensemble = mean + inflation_factor * (ensemble - mean)
    ensemble_covariance = jnp.cov(ensemble.T)

    @jit
    def update_ensemble_point(point, key=key):
        key, subkey = jax.random.split(key)
        ensemble_measurement = measure(point) + measurement_covariance * jax.random.normal(subkey)
        measurement_jacobian = jax.grad(measure)(point)
        innovation_covariance = measurement_jacobian @ ensemble_covariance @ measurement_jacobian.T + measurement_covariance
        kalman_gain = ensemble_covariance @ measurement_jacobian.T * ((innovation_covariance) ** (-1))
        point = point + kalman_gain * (measurement - ensemble_measurement)
        return point

    key, subkey = jax.random.split(key)
    ensemble = jax.vmap(update_ensemble_point)(ensemble)

    true_state = ikeda_update(true_state)
    prior_state = ikeda_update(prior_state)

    key, subkey = jax.random.split(key)
    true_measurement = measure(true_state) + 1/16 * jax.random.normal(subkey)
