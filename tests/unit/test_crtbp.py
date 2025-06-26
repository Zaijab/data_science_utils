import matplotlib.pyplot as plt
import pandas as pd
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from mpl_toolkits.mplot3d import Axes3D

from data_science_utils.dynamical_systems.crtbp import CR3BP
from data_science_utils.measurement_systems import RangeSensor
from data_science_utils.statistics import GMM

key = jax.random.key(10)
key, subkey = jax.random.split(key)

dynamical_system = CR3BP()
measurement_system = Radar()

# prior_ensemble = jnp.asarray(pd.read_csv('./cache/InitialStates.csv', header=None).values)
prior_ensemble = jnp.asarray(pd.read_csv('./cache/PropagatedStates.csv', header=None).values)

# zain_propogated_ensmeble = eqx.filter_vmap(dynamical_system.flow)(0.0, 0.75103, prior_ensemble)

# print(prior_ensemble.shape)
# print(propogated_ensemble.shape)
# print(zain_propogated_ensmeble.shape)

bandwidth = (
    (4) / (prior_ensemble.shape[0] * (prior_ensemble.shape[-1] + 2))
) ** ((2) / (prior_ensemble.shape[-1] + 4))
emperical_covariance = jnp.cov(prior_ensemble.T)
mixture_covariance = bandwidth * emperical_covariance
weights = 1 / prior_ensemble.shape[0]

target_components = prior_ensemble.shape[0]
new_gmm = GMM(
    means=prior_ensemble,
    covs=jnp.tile(mixture_covariance, (target_components, 1, 1)),
    weights=jnp.full(target_components, 1 / target_components)
)
import numpy as np
import os
import numpy as np

# Create directory if it doesn't exist
os.makedirs('.cache/gmm', exist_ok=True)


np.savetxt('cache/gmm/gmm_weights.csv', np.array(new_gmm.weights), delimiter=',')
np.savetxt('cache/gmm/gmm_means.csv', np.array(new_gmm.means), delimiter=',')

# Save each covariance matrix separately
covs = np.array(new_gmm.covs)
for i in range(covs.shape[0]):
    np.savetxt(f'gmm_cov_{i:03d}.csv', covs[i], delimiter=',')

# Save metadata
np.savetxt('cache/gmm/gmm_info.csv', [covs.shape[0], covs.shape[1], covs.shape[2]], 
           delimiter=',', fmt='%d')





def plot_points():
    # Extract first 3 components (positions)
    positions = batch[:, :3]

    # Create 3D plot  
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

