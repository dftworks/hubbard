"""Analytic and noninteracting reference results used for validation."""

from __future__ import annotations

from itertools import combinations
from math import cos, hypot, pi

import numpy as np
from numpy.typing import NDArray

from .basis import validate_sector
from .hamiltonian import Boundary, validate_boundary


def two_site_ground_energy(U: float, t: float = 1.0) -> float:
    """Exact half-filled two-site singlet ground-state energy."""

    return 0.5 * U - 0.5 * hypot(U, 4.0 * t)


def two_site_singlet_triplet_gap(U: float, t: float = 1.0) -> float:
    """Exact triplet-minus-singlet gap for repulsive ``U``."""

    return -two_site_ground_energy(U, t)


def single_particle_energies(
    L: int, *, t: float = 1.0, boundary: Boundary = "open"
) -> NDArray[np.float64]:
    """Return sorted one-particle tight-binding energies.

    Open chains have ``epsilon_m = -2t cos(m*pi/(L+1))``.  Periodic chains
    with at least three sites have ``epsilon_m = -2t cos(2*pi*m/L)``.  Under
    this package's unique-edge convention, ``L=2`` instead has energies
    ``(-|t|, |t|)``; ``L=1`` has one zero-energy orbital.
    """

    validate_sector(L, 0, 0)
    validate_boundary(boundary)
    if L == 1:
        return np.zeros(1, dtype=float)
    if boundary == "open":
        energies = [-2.0 * t * cos(m * pi / (L + 1)) for m in range(1, L + 1)]
    elif L == 2:
        energies = [-abs(t), abs(t)]
    else:
        energies = [-2.0 * t * cos(2.0 * pi * m / L) for m in range(L)]
    return np.sort(np.asarray(energies, dtype=float))


def _occupation_sums(energies: NDArray[np.float64], particles: int) -> list[float]:
    return [
        float(sum(energies[index] for index in occupied))
        for occupied in combinations(range(len(energies)), particles)
    ]


def noninteracting_sector_energies(
    L: int,
    n_up: int,
    n_down: int,
    *,
    t: float = 1.0,
    boundary: Boundary = "open",
) -> NDArray[np.float64]:
    """Return the complete sorted ``U=0`` many-body spectrum in a sector."""

    validate_sector(L, n_up, n_down)
    one_body = single_particle_energies(L, t=t, boundary=boundary)
    up_sums = _occupation_sums(one_body, n_up)
    down_sums = _occupation_sums(one_body, n_down)
    return np.sort(
        np.asarray([up + down for up in up_sums for down in down_sums], dtype=float)
    )


def noninteracting_ground_energy(
    L: int,
    n_up: int,
    n_down: int,
    *,
    t: float = 1.0,
    boundary: Boundary = "open",
) -> float:
    """Return the ``U=0`` ground-state energy from occupied orbitals."""

    energies = single_particle_energies(L, t=t, boundary=boundary)
    return float(np.sum(energies[:n_up]) + np.sum(energies[:n_down]))
