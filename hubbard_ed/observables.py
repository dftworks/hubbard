"""Expectation values in the site basis.

All routines accept an arbitrary nonzero state vector and normalize it before
forming expectation values.  The spin convention is
``S_i^z = (n_{i,up} - n_{i,down}) / 2``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import HubbardBasis
from .operators import apply_spin_hop

Spin = Literal["up", "down"]


def _normalized_state(vector: ArrayLike, basis: HubbardBasis) -> NDArray[np.complex128]:
    state = np.asarray(vector, dtype=complex)
    if state.ndim != 1 or state.shape != (basis.dimension,):
        raise ValueError(f"state vector must have shape ({basis.dimension},)")
    norm = np.linalg.norm(state)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("state vector must have finite, nonzero norm")
    return state / norm


def _probabilities(vector: ArrayLike, basis: HubbardBasis) -> NDArray[np.float64]:
    state = _normalized_state(vector, basis)
    return np.abs(state) ** 2


def _site_occupations(basis: HubbardBasis) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    up = np.empty((basis.dimension, basis.L), dtype=float)
    down = np.empty_like(up)
    for row, (up_bits, down_bits) in enumerate(basis):
        for site in range(basis.L):
            up[row, site] = (up_bits >> site) & 1
            down[row, site] = (down_bits >> site) & 1
    return up, down


def local_double_occupancy(
    vector: ArrayLike, basis: HubbardBasis
) -> NDArray[np.float64]:
    """Return ``<n_i_up n_i_down>`` for every site."""

    probabilities = _probabilities(vector, basis)
    up, down = _site_occupations(basis)
    return probabilities @ (up * down)


def total_double_occupancy(vector: ArrayLike, basis: HubbardBasis) -> float:
    """Return ``sum_i <n_i_up n_i_down>``."""

    return float(np.sum(local_double_occupancy(vector, basis)))


def double_occupancy_per_site(vector: ArrayLike, basis: HubbardBasis) -> float:
    """Return the site average ``D/L``."""

    return total_double_occupancy(vector, basis) / basis.L


def local_charge(vector: ArrayLike, basis: HubbardBasis) -> NDArray[np.float64]:
    """Return ``<n_i_up + n_i_down>`` for every site."""

    probabilities = _probabilities(vector, basis)
    up, down = _site_occupations(basis)
    return probabilities @ (up + down)


def local_magnetization(vector: ArrayLike, basis: HubbardBasis) -> NDArray[np.float64]:
    """Return ``<n_i_up - n_i_down>`` for every site (twice ``<S_i^z>``)."""

    probabilities = _probabilities(vector, basis)
    up, down = _site_occupations(basis)
    return probabilities @ (up - down)


def spin_z_correlation_matrix(
    vector: ArrayLike, basis: HubbardBasis, *, connected: bool = False
) -> NDArray[np.float64]:
    """Return all ``<S_i^z S_j^z>`` correlations.

    If ``connected`` is true, ``<S_i^z><S_j^z>`` is subtracted.
    """

    probabilities = _probabilities(vector, basis)
    up, down = _site_occupations(basis)
    spin_z = 0.5 * (up - down)
    correlations = (spin_z * probabilities[:, None]).T @ spin_z
    if connected:
        means = probabilities @ spin_z
        correlations -= np.outer(means, means)
    return correlations


def spin_z_correlation(
    vector: ArrayLike, basis: HubbardBasis, i: int, j: int
) -> float:
    """Return ``<S_i^z S_j^z>``."""

    _validate_site(basis, i)
    _validate_site(basis, j)
    return float(spin_z_correlation_matrix(vector, basis)[i, j])


def charge_correlation_matrix(
    vector: ArrayLike, basis: HubbardBasis
) -> NDArray[np.float64]:
    """Return connected charge correlations ``<n_i n_j>-<n_i><n_j>``."""

    probabilities = _probabilities(vector, basis)
    up, down = _site_occupations(basis)
    charge = up + down
    means = probabilities @ charge
    return (charge * probabilities[:, None]).T @ charge - np.outer(means, means)


def charge_correlation(
    vector: ArrayLike, basis: HubbardBasis, i: int, j: int
) -> float:
    """Return ``<n_i n_j> - <n_i><n_j>``."""

    _validate_site(basis, i)
    _validate_site(basis, j)
    return float(charge_correlation_matrix(vector, basis)[i, j])


def _validate_site(basis: HubbardBasis, site: int) -> None:
    if isinstance(site, bool) or not isinstance(site, int) or not 0 <= site < basis.L:
        raise ValueError(f"site must satisfy 0 <= site < {basis.L}")


def one_body_density_matrix(
    vector: ArrayLike, basis: HubbardBasis, spin: Spin
) -> NDArray[np.complex128]:
    """Return ``gamma[i,j] = <c_i,spin^dagger c_j,spin>``.

    The trace equals ``N_up`` or ``N_down`` for a normalized fixed-sector
    state.  Off-diagonal signs follow the same global convention as the
    Hamiltonian.
    """

    if spin not in ("up", "down"):
        raise ValueError("spin must be 'up' or 'down'")
    state_vector = _normalized_state(vector, basis)
    density_matrix = np.zeros((basis.L, basis.L), dtype=complex)
    for column, state in enumerate(basis):
        if state_vector[column] == 0:
            continue
        for i in range(basis.L):
            for j in range(basis.L):
                result = apply_spin_hop(state, basis.L, i, j, spin)
                if result is None:
                    continue
                target, sign = result
                row = basis.state_index(target)
                density_matrix[i, j] += (
                    np.conjugate(state_vector[row])
                    * sign
                    * state_vector[column]
                )
    return density_matrix


def momentum_distribution(
    vector: ArrayLike,
    basis: HubbardBasis,
    spin: Spin | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return momenta and ``n(k)`` on ``k=2*pi*m/L``.

    ``n_sigma(k) = L^-1 sum_ij exp(i*k*(i-j)) gamma_sigma[i,j]``.
    With ``spin=None`` (the default), the two spin distributions are summed.
    This Fourier-grid definition is most physically natural for periodic
    chains, but is also a well-defined diagnostic for open chains.
    """

    if spin is None:
        density = one_body_density_matrix(vector, basis, "up")
        density += one_body_density_matrix(vector, basis, "down")
    else:
        density = one_body_density_matrix(vector, basis, spin)
    momenta = 2.0 * np.pi * np.arange(basis.L) / basis.L
    sites = np.arange(basis.L)
    values = np.empty(basis.L, dtype=float)
    for index, momentum in enumerate(momenta):
        phase = np.exp(1j * momentum * (sites[:, None] - sites[None, :]))
        values[index] = float(np.real(np.sum(phase * density) / basis.L))
    return momenta, values


def charge_structure_factor(
    vector: ArrayLike, basis: HubbardBasis
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``N(q)=L^-1 sum_ij exp(iq(i-j)) C_charge(i,j)``."""

    return _structure_factor(charge_correlation_matrix(vector, basis), basis.L)


def spin_structure_factor(
    vector: ArrayLike, basis: HubbardBasis, *, connected: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``S(q)=L^-1 sum_ij exp(iq(i-j)) <S_i^z S_j^z>``.

    Pass ``connected=True`` to subtract ``<S_i^z><S_j^z>`` first.
    """

    correlations = spin_z_correlation_matrix(vector, basis, connected=connected)
    return _structure_factor(correlations, basis.L)


def _structure_factor(
    correlations: NDArray[np.float64], L: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    momenta = 2.0 * np.pi * np.arange(L) / L
    sites = np.arange(L)
    values = np.empty(L, dtype=float)
    for index, momentum in enumerate(momenta):
        phase = np.exp(1j * momentum * (sites[:, None] - sites[None, :]))
        values[index] = float(np.real(np.sum(phase * correlations) / L))
    return momenta, values

