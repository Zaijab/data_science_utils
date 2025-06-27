@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def ospa_metric(
    X: Float[Array, "m state_dim"],
    Y: Float[Array, "n state_dim"],
    c: float = 100.0,
    p: float = 2.0,
) -> tuple[
    float | Float[Array, ""], float | Float[Array, ""], float | Float[Array, ""]
]:
    """
    X: estimated states
    Y: ground truth states
    c: cut-off parameter
    p: order parameter
    """
    m, n = X.shape[0], Y.shape[0]

    # Empty set cases
    if m == 0 and n == 0:
        return 0.0, 0.0, 0.0
    if m == 0:
        # d_p^(c)(∅, Y) = c
        return c, 0.0, c
    if n == 0:
        # d_p^(c)(X, ∅) = c
        return c, c, 0.0

    # Compute distance matrix D_{ij} = ||x_i - y_j||
    D = jnp.sqrt(jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))

    # Apply cut-off: D^(c)_{ij} = min(D_{ij}, c)
    D_cut = jnp.minimum(D, c)

    # Cost matrix for assignment: C_{ij} = (D^(c)_{ij})^p
    C = D_cut**p

    # Solve assignment problem: π* = argmin_π Σ C_{i,π(i)}
    row_ind, col_ind = optax.assignment.hungarian_algorithm(C)

    # Assignment cost: Σ_{i=1}^{min(m,n)} (d^(c)(x_i, y_{π*(i)}))^p
    assignment_cost = C[row_ind, col_ind].sum()

    # Localization error: (1/min(m,n) * assignment_cost)^(1/p)
    e_loc = (assignment_cost / jnp.minimum(m, n)) ** (1 / p)

    # Cardinality error: c * |m - n| / max(m,n)
    e_card = c * jnp.abs(m - n) / jnp.maximum(m, n)

    # Total OSPA: ((assignment_cost + c^p * |m-n|) / max(m,n))^(1/p)
    ospa = ((assignment_cost + c**p * jnp.abs(m - n)) / jnp.maximum(m, n)) ** (1 / p)

    return ospa, e_loc, e_card
