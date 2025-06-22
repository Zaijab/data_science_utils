import jax.numpy as jnp

from data_science_utils.models.kmeans import kmeans
from data_science_utils.statistics import GMM


def state_extraction_from_gmm(key, gmm: GMM):
    points = intensity_function.means
    estimated_cardinality = int(jnp.ceil(jnp.sum(intensity_function.weights)))
    return kmeans(key, points, estimated_cardinality).centroids

key = jax.random.key(0)
state_extraction_from_gmm(key, intensity_function)
