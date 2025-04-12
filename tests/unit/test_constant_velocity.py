from data_science_utils.dynamical_systems.constant_velocity import CVModel

# # Initialize model
cv_model = CVModel()

print(cv_model.process_noise_matrix.shape)

# # Single state prediction
state = jnp.array([[0.0], [1.0], [1.0], [-4.0], [1.0], [1.0]])
key = jax.random.key(0)
next_state = cv_model.flow(state, key)
print(next_state)
