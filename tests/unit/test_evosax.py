import jax
import jax.numpy as jnp
from evosax import CMA_ES

key = jax.random.key(0)
key, subkey = jax.random.split(key)
key, key_init, key_ask, key_eval, key_tell = jax.random.split(key, 5)

ensemble_size = 10

# Instantiate the evolution strategy instance
strategy = CMA_ES(num_dims=ensemble_size**2, popsize=10)

# Get default hyperparameters (e.g. lrate, etc.)
es_params = strategy.default_params


# Initialize the strategy
rng = jax.random.key(0)
state = strategy.initialize(rng, es_params)

# Have a look at the hyperparameters (change if desired)
es_params


a = jnp.ones(ensemble_size) / ensemble_size
b = jax.random.uniform(subkey, shape=(ensemble_size,))
b = b / jnp.sum(b)


@jax.vmap
def transport_matrix_fitness(T, a=a, b=b):
    T = T.reshape(ensemble_size, ensemble_size)
    a_diff = jnp.sum(T, axis=1) - a
    b_diff = jnp.sum(T, axis=0) - b

    return jnp.sum((a_diff**2) + (b_diff**2))


# Ask for a set of candidate solutions to evaluate
x, state = strategy.ask(rng, state, es_params)
# Evaluate the population members
fitness = transport_matrix_fitness(x)
# Update the evolution strategy
state = strategy.tell(x, fitness, state, es_params)
state


# Jittable logging helper
num_gens = 500
es_logging = ESLog(num_dims=100, num_generations=num_gens, top_k=3, maximize=False)
log = es_logging.initialize()

rng = jax.random.key(100)
state = es.initialize(rng, es_params)
for i in range(num_gens):
    rng, rng_ask = jax.random.split(rng)
    x, state = es.ask(rng_ask, state, es_params)
    fitness = transport_matrix_fitness(x)
    state = es.tell(x, fitness, state, es_params)
    log = es_logging.update(log, x, fitness)

es_logging.plot(log, "CMA-ES")
