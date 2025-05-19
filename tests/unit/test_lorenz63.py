import equinox as eqx
import jax
import jax.numpy as jnp
from data_science_utils.dynamical_systems import Lorenz63
from diffrax import ConstantStepSize, ODETerm, SaveAt, Tsit5, diffeqsolve
from jaxtyping import Array, Float


def test_lorenz_63_integrate_single_trajectory() -> None:
    system = Lorenz63()
    x = system.initial_state
    term = ODETerm(system.vector_field)
    solver = Tsit5()
    dt = 0.01
    t1 = 10

    sol_save_last = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t1,
        dt0=dt,
        y0=x,
        stepsize_controller=ConstantStepSize(),
        saveat=SaveAt(t1=True),
        max_steps=100_000,
    )


def test_lorenz_63_integrate_single_trajectory_save_multiple_time() -> None:
    system = Lorenz63()
    x = system.initial_state
    term = ODETerm(system.vector_field)
    solver = Tsit5()
    dt = 0.01
    t1 = 10

    ts = jnp.linspace(0, t1, 500)
    sol_save_all = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t1,
        dt0=dt,
        y0=x,
        stepsize_controller=ConstantStepSize(),
        saveat=SaveAt(ts=ts),
        max_steps=100_000,
    )
    assert sol_save_all is not None


def test_lorenz_63_integrate_batch_trajectory_save_multiple_time() -> None:
    system = Lorenz63()
    batch = jax.random.normal(jax.random.key(0), shape=(10, 3))

    term = ODETerm(system.vector_field)
    solver = Tsit5()
    dt = 0.01
    t1 = 10

    def solve_trajectory(x: Float[Array, "..."]) -> Float[Array, "..."]:
        return diffeqsolve(
            term,
            solver,
            t0=0,
            t1=t1,
            dt0=dt,
            y0=x,
            stepsize_controller=ConstantStepSize(),
            saveat=SaveAt(t1=True),
            max_steps=100_000,
        ).ys[0]

    assert batch.shape == eqx.filter_vmap(solve_trajectory)(batch).shape
