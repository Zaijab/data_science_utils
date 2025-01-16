import jax
from beartype import beartype as typechecker
from jaxtyping import jaxtyped, Float, Array
from functools import partial
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from dataclasses import dataclass
from typing import Optional
import equinox as eqx
from jax import lax
import matplotlib.pyplot as plt
from data_science_utils.dynamical_systems.ikeda import IkedaSystem
from beartype import beartype as typechecker

from jaxtyping import jaxtyped, Float, Array, Key, Bool
from functools import partial


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["u", "ninverses"])
def ikeda_attractor_discriminator(
        x: Float[Array, "*batch 2"],
        ninverses: int = 10,
        u: float = 0.9
) -> Bool[Array, "*batch"]:
    """
    Returns a boolean array indicating which points in x are on the Ikeda map's attractor.
    Replicates the logic of the MATLAB 'onIkedaAttractor' function using repeated inverse iterations.
    """

    @jax.jit
    def ikeda_inv(zp1: Float[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """
        Inverts the Ikeda map via a fixed number of Newton's method steps and normalizes
        to preserve the radius as in the MATLAB code.
        """

        # Unshift and unscale
        zn = (zp1 - jnp.array([1.0, 0.0])) / u

        def newton_iteration(zi, _):
            xi, yi = zi[..., 0], zi[..., 1]

            opx2y2 = 1.0 + xi**2 + yi**2
            ti = 0.4 - 6.0 / opx2y2
            cti, sti = jnp.cos(ti), jnp.sin(ti)

            dti_dx = 12.0 * xi / (opx2y2**2)
            dti_dy = 12.0 * yi / (opx2y2**2)

            # Jacobian terms
            # J = [[J11, J12],
            #      [J21, J22]]
            # Each is shape (...,) for batch
            J11 = cti - (yi * cti + xi * sti) * dti_dx
            J12 = -sti - (yi * cti + xi * sti) * dti_dy
            J21 = sti + (xi * cti - yi * sti) * dti_dx
            J22 = cti + (xi * cti - yi * sti) * dti_dy

            # Residual c = zn - forward(zi)
            # forward(zi) = [xi*cos(ti)-yi*sin(ti), xi*sin(ti)+yi*cos(ti)]
            c0 = zn[..., 0] - (xi * cti - yi * sti)
            c1 = zn[..., 1] - (xi * sti + yi * cti)

            # Solve 2x2 system J * dZi = c
            detJ = J11 * J22 - J12 * J21
            dx0 = ( J22 * c0 - J12 * c1) / detJ
            dx1 = (-J21 * c0 + J11 * c1) / detJ
            zi_next = jnp.stack([xi + dx0, yi + dx1], axis=-1)

            # Enforce the same radius as zn
            zn_norm = jnp.linalg.norm(zn, axis=-1, keepdims=True)
            zi_norm = jnp.linalg.norm(zi_next, axis=-1, keepdims=True)
            zi_next = jnp.where(zi_norm > 0, zn_norm * zi_next / zi_norm, zi_next)

            return zi_next, None

        # Initialize with zn and do 12 Newton steps
        zi_final, _ = lax.scan(newton_iteration, zn, None, length=12)
        return zi_final

    def apply_inverse_n_times(x_):
        def body_fn(_, state):
            return ikeda_inv(state)
        return lax.fori_loop(0, ninverses, body_fn, x_)

    x_inv = apply_inverse_n_times(x)
    threshold = jnp.sqrt(1.0 / (1.0 - u))
    return jnp.linalg.norm(x_inv, axis=-1) < threshold


def apply_fun(params, x):
    return params[0] * x**2 + params[1] * x


def sample_fun(rng, params, num_samples, ymax):
    def rejection_sample(args):
        rng, all_x, i = args
        rng, split_rng = jax.random.split(rng)
        x = jax.random.uniform(split_rng, minval=0, maxval=1, shape=(1,))
        rng, split_rng = jax.random.split(rng)
        y = jax.random.uniform(split_rng, minval=0, maxval=ymax, shape=(1,))
        passed = (y < apply_fun(params, x)).astype(bool)
        all_x = all_x.at[i].add((passed * x)[0])
        i = i + passed[0]
        return rng, all_x, i

    all_x = jnp.zeros(num_samples)
    _, all_x, _ = jax.lax.while_loop(lambda i: i[2] < num_samples, rejection_sample, (rng, all_x, 0))
    return all_x


debug = False


@jaxtyped(typechecker=typechecker)
@jax.jit
def norm_measurement(covariance: Float[Array, "measurement_dim measurement_dim"], state: Float[Array, "*batch_dim state_dim"], key: Optional[Key[Array, "1"]] = None) -> Float[Array, "measurement_dim"]:
    perfect_measurement = jnp.linalg.norm(state)
    noise = 0 if key is None else jnp.sqrt(covariance.value) * jax.random.normal(key)
    return jnp.array([perfect_measurement + noise])


@jaxtyped(typechecker=typechecker)
class Distance(eqx.Module):
    covariance: Float[Array, "measurement_dim measurement_dim"]

    def __call__(self, state, key=None):
        return norm_measurement(state=state, key=key, covariance=self.covariance)


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["debug"])
def ensemble_gaussian_mixture_filter_update_point(
        point: Float[Array, "state_dim"],
        prior_mixture_covariance: Float[Array, "state_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_device,
        debug: bool = False
):
    measurement_jacobian = jax.jacfwd(measurement_device)(point)
    if debug:
        assert isinstance(measurement_jacobian, Float[Array, "measurement_dim state_dim"])
        assert not jnp.any(jnp.isnan(measurement_jacobian))

    kalman_gain = prior_mixture_covariance @ measurement_jacobian.T @ jnp.linalg.inv(measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + measurement_device.covariance)
    if debug:
        assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
        assert not jnp.any(jnp.isnan(kalman_gain))

    gaussian_mixture_covariance = (jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian) @ prior_mixture_covariance #+ 1e-10 * jnp.eye(point.shape[0])
    if debug:
        assert isinstance(gaussian_mixture_covariance, Float[Array, "state_dim state_dim"])
        jax.debug.callback(is_positive_definite, gaussian_mixture_covariance)
        jax.debug.callback(has_nan, gaussian_mixture_covariance)

    point = point - (kalman_gain @ jnp.atleast_2d(measurement_device(point) - measurement)).reshape(-1)
    if debug:
        assert isinstance(point, Float[Array, "state_dim"])

    logposterior_weights = jsp.stats.multivariate_normal.logpdf(measurement, measurement_device(point), measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + measurement_device.covariance)
    if debug:
        assert isinstance(logposterior_weights, Float[Array, ""])

    return point, logposterior_weights, gaussian_mixture_covariance


@jax.jit
def sample_multivariate_normal(key, point, cov):
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["debug"])
def ensemble_gaussian_mixture_filter_update_ensemble(
        state: Float[Array, "batch_dim state_dim"],
        bandwidth_factor: float,
        key: Key[Array, ""],
        measurement: Float[Array, "measurement_dim"],
        measurement_device,
        debug: bool = False
):
    key: Key[Array, ""]
    subkey: Key[Array, ""]
    subkeys: Key[Array, "batch_dim"]

    key, subkey, *subkeys = jax.random.split(key, 2 + state.shape[0])
    subkeys = jnp.array(subkeys)
    if debug:
        assert isinstance(subkeys, Key[Array, "{self.state.shape[0]}"])

    emperical_covariance = jnp.cov(state.T)
    if debug:
        jax.debug.callback(is_positive_definite, emperical_covariance)
        assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])

    mixture_covariance = bandwidth_factor * emperical_covariance
    if debug:
        assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

    posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(ensemble_gaussian_mixture_filter_update_point, in_axes=(0, None, None, None, None))(state, mixture_covariance, measurement, measurement_device, debug)

    if debug:
        assert isinstance(posterior_ensemble, Float[Array, "{self.state.shape[0]} {self.state.shape[1]}"])
        assert isinstance(logposterior_weights, Float[Array, "{self.state.shape[0]}"])
        assert isinstance(posterior_covariances, Float[Array, "{self.state.shape[0]} {self.state.shape[1]} {self.state.shape[1]}"])
        jax.debug.callback(has_nan, posterior_covariances)


    # Scale Weights
    m = jnp.max(logposterior_weights)
    g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
    posterior_weights = jnp.exp(logposterior_weights - g)
    posterior_weights = (posterior_weights / jnp.sum(posterior_weights))
    if debug:
        assert isinstance(posterior_weights, Float[Array, "{self.state.shape[0]}"])


    # Prevent Degenerate Particles
    variable = jax.random.choice(subkey, state.shape[0], shape=(state.shape[0],), p=posterior_weights)
    posterior_ensemble = posterior_ensemble[variable, ...]
    posterior_covariances = posterior_covariances[variable, ...]
    if debug:
        jax.debug.callback(has_nan, posterior_covariances)


    posterior_samples = jax.vmap(sample_multivariate_normal)(subkeys, posterior_ensemble, posterior_covariances)
    if debug:
        assert isinstance(posterior_weights, Float[Array, "{self.state.shape[0]}"])

    return posterior_covariances, posterior_samples


@dataclass
class EnGMF:
    dynamical_system: object
    measurement_device: object
    state: Float[Array, "batch_dim state_dim"]
    bandwidth_factor: float

    def update(self, key, measurement, debug):
        self.covariences, self.state = ensemble_gaussian_mixture_filter_update_ensemble(state=self.state,
                                                                                        bandwidth_factor=self.bandwidth_factor,
                                                                                        measurement_device=self.measurement_device,
                                                                                        key=key,
                                                                                        measurement=measurement,
                                                                                        debug=debug)

    def predict(self):
        self.state = self.dynamical_system.flow(self.state)


# ---------------------------------------------------------------------
# Generator: samples from a multivariate Normal, given mean & covariance
# ---------------------------------------------------------------------
@jax.jit
def sample_multivariate_normal(
    rng: jax.random.PRNGKey,
    mean: Float[Array, "state_dim"],
    cov: Float[Array, "state_dim state_dim"]
) -> Float[Array, "state_dim"]:
    """
    rng: a JAX PRNGKey
    mean: 1D array of shape [state_dim]
    cov: 2D array of shape [state_dim, state_dim]
    """
    return jax.random.multivariate_normal(rng, mean=mean, cov=cov)

# ---------------------------------------------------------------------
# Discriminator: checks if a point is on the Ikeda map's attractor
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnames=["ninverses"])
def ikeda_attractor_discriminator(
    x: Float[Array, "state_dim"],
    ninverses: int = 10,
    u: float = 0.9
) -> Bool[Array, ""]:
    """
    Returns True if 'x' lies on the Ikeda attractor, computed via ninverses
    inverse iterations. For batch usage, vmap over x.
    """
    def ikeda_inv(zp1):
        """
        Single inverse iteration: Newton's method to invert Ikeda forward step,
        then radius normalization to match the input radius.
        """
        # Shift/unscale
        zn = (zp1 - jnp.array([1.0, 0.0])) / u

        def newton_iteration(zi, _):
            xi, yi = zi[0], zi[1]
            r2 = xi**2 + yi**2
            opx2y2 = 1.0 + r2
            ti = 0.4 - 6.0 / opx2y2
            cti, sti = jnp.cos(ti), jnp.sin(ti)

            # Partial derivatives of ti wrt (x, y)
            dti_dx = 12.0 * xi / (opx2y2**2)
            dti_dy = 12.0 * yi / (opx2y2**2)

            # Jacobian
            J11 = cti - (yi * cti + xi * sti) * dti_dx
            J12 = -sti - (yi * cti + xi * sti) * dti_dy
            J21 = sti + (xi * cti - yi * sti) * dti_dx
            J22 = cti + (xi * cti - yi * sti) * dti_dy
            detJ = J11 * J22 - J12 * J21

            # Residual
            c0 = zn[0] - (xi * cti - yi * sti)
            c1 = zn[1] - (xi * sti + yi * cti)

            dx0 = (J22 * c0 - J12 * c1) / detJ
            dx1 = (-J21 * c0 + J11 * c1) / detJ

            zi_next = jnp.array([xi + dx0, yi + dx1])
            # Enforce same radius as zn
            zn_norm = jnp.linalg.norm(zn)
            zi_norm = jnp.linalg.norm(zi_next)
            zi_next = jnp.where(
                zi_norm > 0,
                zi_next * (zn_norm / zi_norm),
                zi_next
            )
            return zi_next, None

        zi_final, _ = lax.scan(newton_iteration, zn, None, length=12)
        return zi_final

    def body_fn(_, state):
        return ikeda_inv(state)

    # Apply ninverses inverse iterations
    x_inv = lax.fori_loop(0, ninverses, body_fn, x)
    threshold = jnp.sqrt(1.0 / (1.0 - u))
    return (jnp.linalg.norm(x_inv) < threshold)

# ---------------------------------------------------------------------
# Single-sample Rejection: repeatedly draw until point passes discriminator
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnames=["ninverses"])
def rejection_sample_single(
    rng: jax.random.PRNGKey,
    mean: Float[Array, "state_dim"],
    cov: Float[Array, "state_dim state_dim"],
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
        candidate = sample_multivariate_normal(subkey, mean, cov)
        # If candidate passes, store it
        pass_sample = ikeda_attractor_discriminator(candidate, ninverses, u)
        sample = jnp.where(pass_sample, candidate, sample)
        accepted = accepted | pass_sample
        return (rng, sample, accepted)

    # Initialize with zero vector and False acceptance
    carry_init = (rng, jnp.zeros_like(mean), False)
    _, final_sample, _ = lax.while_loop(cond_fun, body_fun, carry_init)
    return final_sample

# ---------------------------------------------------------------------
# Batch Rejection: vmap across multiple means/covariances
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnames=["ninverses"])
def rejection_sample_batch(
    rng: jax.random.PRNGKey,
    means: Float[Array, "batch state_dim"],
    covs: Float[Array, "batch state_dim state_dim"],
    ninverses: int = 10,
    u: float = 0.9
) -> Float[Array, "batch state_dim"]:
    """
    For each row in 'means' and 'covs', run rejection_sample_single.
    """
    batch_size = means.shape[0]
    subkeys = jax.random.split(rng, batch_size)
    return jax.vmap(
        lambda sk, m, c: rejection_sample_single(sk, m, c, ninverses, u)
    )(subkeys, means, covs)


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["debug"])
def discriminator_ensemble_gaussian_mixture_filter_update_ensemble(
        state: Float[Array, "batch_dim state_dim"],
        bandwidth_factor: float,
        key: Key[Array, ""],
        measurement: Float[Array, "measurement_dim"],
        measurement_device,
        debug: bool = False
):
    key: Key[Array, ""]
    subkey: Key[Array, ""]
    subkeys: Key[Array, "batch_dim"]

    key, subkey, *subkeys = jax.random.split(key, 2 + state.shape[0])
    subkeys = jnp.array(subkeys)
    if debug:
        assert isinstance(subkeys, Key[Array, "{self.state.shape[0]}"])

    emperical_covariance = jnp.cov(state.T)
    if debug:
        jax.debug.callback(is_positive_definite, emperical_covariance)
        assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])

    mixture_covariance = bandwidth_factor * emperical_covariance
    if debug:
        assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

    posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(ensemble_gaussian_mixture_filter_update_point, in_axes=(0, None, None, None, None))(state, mixture_covariance, measurement, measurement_device, debug)

    if debug:
        assert isinstance(posterior_ensemble, Float[Array, "{self.state.shape[0]} {self.state.shape[1]}"])
        assert isinstance(logposterior_weights, Float[Array, "{self.state.shape[0]}"])
        assert isinstance(posterior_covariances, Float[Array, "{self.state.shape[0]} {self.state.shape[1]} {self.state.shape[1]}"])
        jax.debug.callback(has_nan, posterior_covariances)


    # Scale Weights
    m = jnp.max(logposterior_weights)
    g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
    posterior_weights = jnp.exp(logposterior_weights - g)
    posterior_weights = (posterior_weights / jnp.sum(posterior_weights))
    if debug:
        assert isinstance(posterior_weights, Float[Array, "{self.state.shape[0]}"])


    # Prevent Degenerate Particles
    variable = jax.random.choice(subkey, state.shape[0], shape=(state.shape[0],), p=posterior_weights)
    posterior_ensemble = posterior_ensemble[variable, ...]
    posterior_covariances = posterior_covariances[variable, ...]
    if debug:
        jax.debug.callback(has_nan, posterior_covariances)


    posterior_samples = rejection_sample_batch(key, posterior_ensemble, posterior_covariances, ninverses=15, u=0.9)

    #posterior_samples = jax.vmap(sample_multivariate_normal)(subkeys, posterior_ensemble, posterior_covariances)
    if debug:
        assert isinstance(posterior_weights, Float[Array, "{self.state.shape[0]}"])

    return posterior_covariances, posterior_samples


@dataclass
class DEnGMF:
    dynamical_system: object
    measurement_device: object
    state: Float[Array, "batch_dim state_dim"]
    bandwidth_factor: float

    def update(self, key, measurement, debug):
        self.covariences, self.state = discriminator_ensemble_gaussian_mixture_filter_update_ensemble(state=self.state,
                                                                                                      bandwidth_factor=self.bandwidth_factor,
                                                                                                      measurement_device=self.measurement_device,
                                                                                                      key=key,
                                                                                                      measurement=measurement,
                                                                                                      debug=debug)

    def predict(self):
        self.state = self.dynamical_system.flow(self.state)



# TESTING
def is_positive_definite(M):
    try:
        jnp.linalg.cholesky(M)
        return True
    except:
        assert False, "COVARIANCE MATRIX IS NOT POSITIVE DEFINITE"

def has_nan(M):
    assert not jnp.any(jnp.isnan(M))


plot = True
key = jax.random.key(0)
key, subkey = jax.random.split(key)

measurement_device = Distance(jnp.array([[1/32]]))
system = IkedaSystem()

if plot:
    key, subkey = jax.random.split(key)
    attractor = system.generate(subkey)

ensemble_size = 1000
silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2+4)) # Likely going to have a parameter study

key, subkey = jax.random.split(key)

#prior_ensemble = system.generate(subkey, batch_size=ensemble_size)
prior_ensemble = jax.random.uniform(key=subkey, shape=(ensemble_size, 2), minval=-0.25, maxval=0.25) #init


filter = DEnGMF(
    dynamical_system=system,
    measurement_device=measurement_device,
    bandwidth_factor=silverman_bandwidth,
    state = prior_ensemble,
)



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
        plt.scatter(system.state[0, 0], system.state[0, 1], c='lime', s=100, label='True')
        plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))
        plt.show()

    if t >= burn_in_time:
        if plot:
            break

        states.append(system.state - jnp.mean(filter.state, axis=0))
        cov = jnp.cov(filter.state.T)

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
