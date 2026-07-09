"""AMAGOLD and SGHMC kernels, faithful to the original code.

- ``amagold_kernel``  simulation/amagold.m (leapfrog with amortized M-H)
- ``sghmc_kernel``    simulation/sghmc.m   (the SGHMC baseline used by the paper)

Both operate on the simulation parameterization: momentum p ~ N(0, 1) is
resampled per call, position increments are p * dt, friction C enters as
beta = C / 2 with injected noise sigma = sqrt(2 dt C). Stochastic gradients
are ``grad_u(key, x)`` (fresh noise per evaluation).

The BNN form of AMAGOLD (persistent momentum, minibatch leapfrog, full-data
M-H) lives in :mod:`amagold_jax.bnn.train`; the SGHMC baseline there also has
a blackjax-backed variant (blackjax has no AMAGOLD kernel).
"""

import jax
import jax.numpy as jnp


def amagold_kernel(u_fn, grad_u, *, dt, nstep, C, mh=True):
    """One AMAGOLD outer step (simulation/amagold.m). Returns step(key, x) -> (x, accepted).

    Leapfrog with a semi-implicit friction update
        p' = ((1 - dt beta) p - dt gradU + N(0, 2 dt C)) / (1 + dt beta),
    beta = C/2, and an amortized M-H correction that accumulates the kinetic
    correction rho = sum_i gradU_i (p_i + p_{i-1}) dt / 2 along the path and
    accepts with probability exp(U_old - U_new + rho).
    """
    sigma = jnp.sqrt(2.0 * dt * C)
    beta = 0.5 * C

    def step(key, x):
        key_p, key_steps, key_mh = jax.random.split(key, 3)
        p = jax.random.normal(key_p, jnp.shape(x))
        old_x = x
        old_energy = u_fn(x)

        x = x + p * dt / 2.0

        def leapfrog(carry, inp):
            x, p, rho = carry
            i, subkey = inp
            k_grad, k_noise = jax.random.split(subkey)
            x = jnp.where(i > 0, x + p * dt, x)
            p_old = p
            grad_x = grad_u(k_grad, x)
            p = ((1.0 - dt * beta) * p - grad_x * dt
                 + jax.random.normal(k_noise) * sigma) / (1.0 + dt * beta)
            rho = rho + grad_x * (p + p_old) * dt / 2.0
            return (x, p, rho), None

        (x, p, rho), _ = jax.lax.scan(
            leapfrog, (x, p, jnp.zeros_like(u_fn(x))),
            (jnp.arange(nstep), jax.random.split(key_steps, nstep)))
        x = x + p * dt / 2.0

        if not mh:
            return x, jnp.asarray(True)
        new_energy = u_fn(x)
        accept = jnp.exp(old_energy - new_energy + rho) >= jax.random.uniform(key_mh)
        return jnp.where(accept, x, old_x), accept

    return step


def sghmc_kernel(grad_u, *, dt, nstep, C):
    """One SGHMC step of nstep inner updates (simulation/sghmc.m). step(key, x) -> x."""
    sigma = jnp.sqrt(2.0 * dt * C)

    def step(key, x):
        key_p, key_steps = jax.random.split(key)
        p = jax.random.normal(key_p, jnp.shape(x))

        def inner(carry, subkey):
            x, p = carry
            k_grad, k_noise = jax.random.split(subkey)
            p = p - grad_u(k_grad, x) * dt - p * C * dt + jax.random.normal(k_noise) * sigma
            x = x + p * dt
            return (x, p), None

        (x, _), _ = jax.lax.scan(inner, (x, p), jax.random.split(key_steps, nstep))
        return x

    return step
