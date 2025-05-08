def plot_update(prior_ensemble, posterior_ensemble, true_state):
    print(prior_ensemble.shape)
    print(posterior_ensemble.shape)
    print(true_state.shape)
    print(jnp.mean(prior_ensemble, axis=0).shape)
    print(true_state - jnp.mean(prior_ensemble, axis=0))
    prior_error = jnp.sqrt(
        jnp.mean((true_state - jnp.mean(prior_ensemble, axis=0)) ** 2)
    )
    posterior_error = jnp.sqrt(
        jnp.mean((true_state - jnp.mean(posterior_ensemble, axis=0)) ** 2)
    )
    plt.title(f"Prior Error {prior_error:.4f} | Posterior Error {posterior_error:.4f}")
    plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], c="red", label="Prior")
    plt.scatter(
        posterior_ensemble[:, 0], posterior_ensemble[:, 1], c="blue", label="Posterior"
    )
    plt.scatter(true_state[..., 0], true_state[..., 1], c="green", label="True")
    plt.legend(bbox_to_anchor=(1.3, 1), loc="upper right")
    plt.show()
