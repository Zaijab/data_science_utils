import jax
from functools import partial
from jax import lax
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker


#@jaxtyped(typechecker=typechecker)
#@partial(jax.jit, static_argnames=["ninverses"])
def rejection_sample_single(
    rng: Key[Array, ""],
    mean: Float[Array, "state_dim"],
    cov: Float[Array, "state_dim state_dim"],
    generator,
    discriminator,
    ninverses: int = 10,
    u: float = 0.9
) -> Float[Array, "state_dim"]:
    """
    Samples from the multivariate normal defined by (mean, cov) until
    ikeda_attractor_discriminator(candidate) is True. Returns the first
    candidate that passes.
    """

    def cond_fun(carry):
        # carry = (rng, candidate, accepted_bool)
        return ~carry[2]

    def body_fun(carry):
        rng, sample, accepted = carry
        rng, subkey = jax.random.split(rng)
        candidate = generator(subkey, mean, cov)
        # If candidate passes, store it
        pass_sample = discriminator(candidate, ninverses, u)
        sample = jnp.where(pass_sample, candidate, sample)
        accepted = accepted | pass_sample
        return (rng, sample, accepted)

    # Initialize with zero vector and False acceptance
    carry_init = (rng, jnp.zeros_like(mean), False)
    _, final_sample, _ = lax.while_loop(cond_fun, body_fun, carry_init)
    return final_sample


#@partial(jax.jit, static_argnames=["ninverses"])
def rejection_sample_batch(
    rng: jax.random.PRNGKey,
    means: Float[Array, "batch state_dim"],
    covs: Float[Array, "batch state_dim state_dim"],
    generator,
    discriminator,
    ninverses: int = 10,
    u: float = 0.9
) -> Float[Array, "batch state_dim"]:
    """
    For each row in 'means' and 'covs', run rejection_sample_single.
    """
    batch_size = means.shape[0]
    subkeys = jax.random.split(rng, batch_size)
    return jax.vmap(
        lambda sk, m, c: rejection_sample_single(sk, m, c, generator, discriminator, ninverses, u)
    )(subkeys, means, covs)

# For simplicity, define a generator (Gaussian) and a simple discriminator (circle boundary).
def sample_multivariate_normal(rng, mean, cov):
    return jax.random.multivariate_normal(rng, mean, cov)

def circle_discriminator(candidate, ninverses, u):
    # Pass if the distance from origin is less than 1.0
    return jnp.linalg.norm(candidate) < 1.0

# Boolean test 1: Single-sample rejection ensures each sample is inside the circle.
def test_single_rejection_all_pass():
    rng = jax.random.PRNGKey(0)
    mean = jnp.zeros(2)
    cov = jnp.eye(2)
    sample = rejection_sample_single(
        rng, mean, cov, sample_multivariate_normal, circle_discriminator
    )
    # Assert that the single sample passes the discriminator.
    assert circle_discriminator(sample, 10, 0.9), "Sample did not pass circle discriminator."

# Boolean test 2: Batch sampling ensures all returned samples pass the circle discriminator.
def test_batch_rejection_all_pass():
    rng = jax.random.PRNGKey(1)
    batch_size = 100
    means = jnp.zeros((batch_size, 2))
    covs = jnp.broadcast_to(jnp.eye(2), (batch_size, 2, 2))
    samples = rejection_sample_batch(
        rng, means, covs, sample_multivariate_normal, circle_discriminator
    )
    passes = jax.vmap(circle_discriminator, in_axes=(0, None, None))(samples, 10, 0.9)
    assert jnp.all(passes), "Not all samples passed circle discriminator."

# Visual test 1: Scatter plot showing raw vs. accepted samples for single-sample rejection.
def visualize_single_rejection():
    rng = jax.random.PRNGKey(2)
    mean = jnp.zeros(2)
    cov = jnp.eye(2)

    # Generate samples without rejection
    rng_gen, rng_rej = jax.random.split(rng)
    raw_samples = jax.vmap(lambda k: sample_multivariate_normal(k, mean, cov))(
        jax.random.split(rng_gen, 1000)
    )

    # Generate samples with rejection
    accepted_samples = jax.vmap(
        lambda k: rejection_sample_single(
            k, mean, cov, sample_multivariate_normal, circle_discriminator
        )
    )(jax.random.split(rng_rej, 1000))

    # Plot raw samples (color by pass/fail) and accepted samples (green).
    pass_inds = jax.vmap(circle_discriminator, in_axes=(0, None, None))(raw_samples, 10, 0.9)
    plt.figure(figsize=(6,6))
    plt.scatter(raw_samples[~pass_inds,0], raw_samples[~pass_inds,1], color='red', alpha=0.5, label='Raw fail')
    plt.scatter(raw_samples[pass_inds,0], raw_samples[pass_inds,1], color='blue', alpha=0.5, label='Raw pass')
    plt.scatter(accepted_samples[:,0], accepted_samples[:,1], color='green', alpha=0.8, label='Accepted')
    plt.legend()
    plt.title("Raw vs. Rejected vs. Accepted Samples")
    plt.axis('equal')
    plt.show()

# Visual test 2: Discriminator boundary in red/blue for each point in a grid.
def visualize_discriminator_boundary():
    # Evaluate circle_discriminator over a grid in [-2,2] x [-2,2].
    xx = jnp.linspace(-2, 2, 200)
    yy = jnp.linspace(-2, 2, 200)
    X, Y = jnp.meshgrid(xx, yy)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    Z = jax.vmap(circle_discriminator, in_axes=(0, None, None))(grid_points, 10, 0.9)
    Z = Z.reshape(X.shape)

    plt.figure(figsize=(6,6))
    plt.contourf(X, Y, Z, levels=[-0.5, 0.5, 1.5], colors=["red", "blue"], alpha=0.5)
    plt.title("Discriminator Output (Red=0, Blue=1)")
    plt.axis('equal')
    plt.show()


# Example calls to run tests and see visual results:
if __name__ == "__main__":
    test_single_rejection_all_pass()
    test_batch_rejection_all_pass()
    visualize_single_rejection()
    visualize_discriminator_boundary()
