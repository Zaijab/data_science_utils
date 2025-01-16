import jax
import jax.numpy as jnp

from tqdm import tqdm
from data_science_utils.dynamical_systems import IkedaSystem
from data_science_utils.filtering import EnGMF, DEnGMF
from data_science_utils.measurement_functions import Distance

plot = False
key = jax.random.key(0)
key, subkey = jax.random.split(key)
measurement_device = Distance(jnp.array([[1 / 32]]))
system = IkedaSystem()

if plot:
    import matplotlib.pyplot as plt
    key, subkey = jax.random.split(key)
    attractor = system.generate(subkey)

ensemble_size = 1000
silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2 + 4))

key, subkey = jax.random.split(key)
prior_ensemble = jax.random.multivariate_normal(key=subkey, shape=(ensemble_size,), mean=system.state, cov=1/16 * jnp.eye(2))
#prior_ensemble = jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25)


filter = DEnGMF(
    dynamical_system=system,
    measurement_device=measurement_device,
    bandwidth_factor=silverman_bandwidth,
    state=prior_ensemble,
    ninverses=7,
)

burn_in_time = 100
measurement_time = 10 * burn_in_time
debug = False
covariances, states = [], []
for t in tqdm(range(burn_in_time + measurement_time), leave=False):

    prior_ensemble = filter.state

    key, subkey = jax.random.split(key)
    # with jax.disable_jit():
    filter.update(subkey, measurement_device(system.state), debug=debug)
        
    if plot:
        plt.scatter(attractor[:, 0], attractor[:, 1], c='blue', alpha=0.1, s=0.1)
        plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], alpha=0.8, s=10, c='purple', label='Prior')
        plt.scatter(filter.state[:, 0], filter.state[:, 1], alpha=0.8, s=10, c='yellow', label='Posterior')
        plt.scatter(system.state[0, 0], system.state[0, 1], c='lime', s=100, label='True')
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
        plt.show()

    if t >= burn_in_time:
        if plot:
            break

        states.append(system.state - jnp.mean(filter.state, axis=0))
        cov = jnp.cov(filter.state.T)

        if debug:
            try:
                jnp.linalg.cholesky(cov)
            except e:
                assert False, "COVARIANCE MATRIX IS NOT POSITIVE DEFINITE"

        covariances.append(cov)

    filter.predict()
    system.iterate()


if len(states) != 0:
    e = jnp.expand_dims(jnp.array(states), -1)
    if debug:
        assert isinstance(e, Float[Array, f"{measurement_time} 1 2 1"])
    P = jnp.expand_dims(jnp.array(covariances), 1)
    if debug:
        assert isinstance(P, Float[Array, f"{measurement_time} 1 2 2"])

    rmse = jnp.mean(jnp.sqrt((1 / (e.shape[1] * e.shape[2] * e.shape[3])) * jnp.sum(e * e, axis=(1, 2, 3))))
    snees = (1 / e.size) * jnp.sum(jnp.swapaxes(e, -2, -1) @ jnp.linalg.inv(P) @ e)

    print(f'{str(type(filter)).split(".")[-1][:-2]}: Batch {filter.state.shape[0]} / Time {measurement_time}')
    print(f"RMSE: {rmse}")
    print(f"SNEES: {snees}")


# DEnGMF 10
# RMSE: 0.13522035761087647
# SNEES: 0.7933309550001773

# DEnGMF 100
# RMSE: 0.0859875482169743
# SNEES: 0.2685743363547288

# DEnGMF (Bypass 10%)
# RMSE: 0.0696469824359378

# EnGMF 1000 Batch / 5000 Measurement
# RMSE: 0.06726956850487631
# SNEES: 0.17278851467324655
