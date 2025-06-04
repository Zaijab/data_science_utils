from data_science_utils.dynamical_systems import RandomWalk


def test_random_walk_api() -> None:
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    system = RandomWalk()
    initial_state = system.initial_state(subkey)

    system.flow(0.0, 10.0, initial_state, subkey)

    key = jax.random.key(1000)
    key, subkey = jax.random.split(key)
    system = RandomWalk(1)
    state = system.initial_state(subkey)

    key, subkey = jax.random.split(key)
    from diffrax import SaveAt

    orbit = system.orbit(subkey, 0.0, 100.0, state, SaveAt(ts=jnp.arange(0, 100)))
    print(orbit)
    plt.plot(orbit)

    trajectory = []
    for _ in range(100):
        key, subkey = jax.random.split(key)
        state = system.forward(state, subkey)
        trajectory.append(state)
    with jnp.printoptions(precision=4, suppress=True):
        print(state)

    trajectory = jnp.stack(trajectory)[:, 0]

    print(trajectory)

    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("2D Random Walk Trajectory")
    plt.grid(True)
    plt.axis("equal")
    plt.plot(trajectory, marker="o", markersize=2, linewidth=1)


def test_random_walk_plot() -> None:
    pass
