

@jit
def update_ensemble_point(point, key=key):
    key, subkey = jax.random.split(key)
    ensemble_measurement = measure(point) + measurement_covariance * jax.random.normal(subkey)
    measurement_jacobian = jax.grad(measure)(point)
    innovation_covariance = measurement_jacobian @ ensemble_covariance @ measurement_jacobian.T + measurement_covariance
    kalman_gain = ensemble_covariance @ measurement_jacobian.T * ((innovation_covariance) ** (-1))
    point = point + kalman_gain * (measurement - ensemble_measurement)
    return point

true_state = jnp.array([1.25, 0])
prior_state = jnp.array([1.25, 0])

for _ in range(10):
    key, subkey = jax.random.split(key)
    measurement = measure(true_state) + 1/16 * jax.random.normal(subkey)

    num_samples = 100
    inflation_factor = 1.01
    key, subkey = jax.random.split(key)
    ensemble = prior_state + 1/16 * jax.random.normal(subkey, shape=(num_samples,2))
    
    mean = jnp.mean(ensemble, axis=0)
    ensemble = mean + inflation_factor * (ensemble - mean)
    
    ensemble_covariance = (1 / (num_samples - 1)) * (ensemble - mean).T @ (ensemble - mean)

    @jit
    def update_ensemble_point(point, key=key):
        
        key, subkey = jax.random.split(key)
        ensemble_measurement = measure(point) + measurement_covariance * jax.random.normal(subkey)
        
        measurement_jacobian = jax.grad(measure)(point)
        
        innovation_covariance = measurement_jacobian @ ensemble_covariance @ measurement_jacobian.T + measurement_covariance
        kalman_gain = ensemble_covariance @ measurement_jacobian.T  * ((innovation_covariance) ** (-1))
        point = point + kalman_gain * (measurement - ensemble_measurement)
        return point

    test_ensemble = ensemble
    ensemble = jax.vmap(update_ensemble_point)(ensemble);key, subkey = jax.random.split(key)


    plot_ikeda_points([
        {'x':0,'y':0,'c':'red','label':'Measurement', 's':100},
        {'x':jnp.mean(ensemble[..., 0]), 'y':jnp.mean(ensemble[...,1]), 'c':'purple', 'label':'Estimate', 's':100},
        {'x':true_state[0], 'y':true_state[1], 'c':'lime', 'label':'True State', 'label':'True State', 's':100},
    ])
    plt.show()
    
    true_state = ikeda_update(true_state)
    prior_state = ikeda_update(prior_state)
    true_measurement = measure(true_state) + 1/16 * jax.random.normal(subkey);key, subkey = jax.random.split(key)
