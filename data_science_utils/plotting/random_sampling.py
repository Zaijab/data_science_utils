from data_science_utils.statistics import rejection_sample_batch

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
