class RandomWalk(eqx.Module):

    def random_walk_per_state(self, state, mask, key):
        state = state + jax.random.multivariate_normal(
            key, mean=jnp.zeros(state.shape[0]), cov=jnp.eye(state.shape[0])
        )
        return state, mask

    def forward(self, state, mask, key):
        # No special birth / death from the system itself

        state, mask = eqx.filter_vmap(self.random_walk_per_state)(
            state, mask, jax.random.split(key, mask.shape[0])
        )

        return state, mask


if __name__ == "__main__":
    dynamical_system = RandomWalk()
    key = jax.random.key(0)
    state = jnp.array([[0, 1, 2], [2, 1, 1], [0, 0, 0]], dtype=jnp.float64)
    mask = jnp.array([True, True, False])
    state, mask = dynamical_system.forward(state, mask, key)

    print(state)
