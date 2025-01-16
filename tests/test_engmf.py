import jax
import jax.numpy as jnp
import jax.scipy as jsp
from data_science_utils.dynamical_systems.ikeda import IkedaSystem
from data_science_utils.measurement_functions.norms import Distance
from data_science_utils.filtering.engmf import EnGMF

debug = True
plot = False
key = jax.random.key(0)
key, subkey = jax.random.split(key)

# Initial Conditions
ensemble_size = 1000
silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study
key, subkey = jax.random.split(key)
prior_ensemble = system.generate(subkey, batch_size=ensemble_size) #jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init

# Surrogate System
system = IkedaSystem()

# Measurement
measurement_device = Distance(jnp.array([[1/16]]))

# Filtering Process
filter = EnGMF(
    dynamical_system=system,
    measurement_device=measurement_device,
    bandwidth_factor=silverman_bandwidth,
    state = prior_ensemble,
)


if plot:
    key, subkey = jax.random.split(key)
    attractor = system.generate(subkey)


burn_in_time = 10
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

        states.append(system.state - jnp.mean(filter.state, axis=0))
        cov = jnp.cov(filter.state.T)
        print(cov)
        if debug:
            try:
                jnp.linalg.cholesky(cov)
            except:
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

    rmse = jnp.mean(jnp.sqrt((1 / (e.shape[1] * e.shape[2] * e.shape[3])) * jnp.sum(e * e, axis=(1,2,3))))
    snees = (1 / e.size) * jnp.sum(jnp.swapaxes(e, -2, -1) @ jnp.linalg.inv(P) @ e)
    print(f"RMSE: {rmse}")
    print(f"SNEES: {snees}")
