"""Exact transformations between orthonormal one-particle bases."""

from __future__ import annotations

from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import validate_sector
from .hamiltonian import validate_hopping_matrix
from .interactions import onsite_interaction_tensor, validate_interaction_tensor

Matrix: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
Tensor: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
UNITARITY_TOLERANCE = 1e-12


def _square_size(name: str, array: ArrayLike) -> int:
    candidate = np.asarray(array)
    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if candidate.shape[0] < 1:
        raise ValueError(f"{name} must have positive dimension")
    return int(candidate.shape[0])


def validate_unitary(rotation: ArrayLike, L: int) -> Matrix:
    """Validate and copy a unitary orbital-rotation matrix.

    Columns of ``rotation`` are the new orbitals expressed in the old basis.
    Thus ``c_i = sum_a rotation[i,a] d_a``.  The returned array is read-only.
    """

    validate_sector(L, 0, 0)
    candidate = np.asarray(rotation)
    if candidate.shape != (L, L):
        raise ValueError(f"rotation must have shape ({L}, {L})")
    if not np.issubdtype(candidate.dtype, np.number):
        raise TypeError("rotation must contain numeric values")
    dtype = np.result_type(candidate.dtype, np.float64)
    matrix = np.array(candidate, dtype=dtype, copy=True)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must contain only finite values")

    identity = np.eye(L, dtype=matrix.dtype)
    gram = matrix.conj().T @ matrix
    if not np.allclose(
        gram,
        identity,
        rtol=UNITARITY_TOLERANCE,
        atol=UNITARITY_TOLERANCE,
    ):
        maximum_error = float(np.max(np.abs(gram - identity)))
        raise ValueError(
            "rotation must be unitary; maximum |R^dagger R-I| is "
            f"{maximum_error:.3e}"
        )
    matrix.flags.writeable = False
    return matrix


def rotate_one_body(hopping_matrix: ArrayLike, rotation: ArrayLike) -> Matrix:
    """Return ``R^dagger h R`` in the rotated orbital basis."""

    L = _square_size("rotation", rotation)
    matrix = validate_hopping_matrix(hopping_matrix, L)
    unitary = validate_unitary(rotation, L)
    transformed = unitary.conj().T @ matrix @ unitary
    return validate_hopping_matrix(transformed, L)


def rotate_interaction_tensor(
    interaction_tensor: ArrayLike, rotation: ArrayLike
) -> Tensor:
    """Transform an up-down tensor into the rotated orbital basis.

    For ``c_i = sum_a R[i,a] d_a``, the transformed tensor is

    ``V'[a,b,c,d] = sum_ijkl R*[i,a] R[j,b] R*[k,c] R[l,d] V[i,j,k,l]``.
    """

    L = _square_size("rotation", rotation)
    tensor = validate_interaction_tensor(interaction_tensor, L)
    unitary = validate_unitary(rotation, L)
    transformed = np.einsum(
        "ia,jb,kc,ld,ijkl->abcd",
        unitary.conj(),
        unitary,
        unitary.conj(),
        unitary,
        tensor,
        optimize=True,
    )
    return validate_interaction_tensor(transformed, L)


def rotate_integrals(
    hopping_matrix: ArrayLike,
    interaction_tensor: ArrayLike,
    rotation: ArrayLike,
) -> tuple[Matrix, Tensor]:
    """Transform consistent one- and two-body integrals with one unitary."""

    return (
        rotate_one_body(hopping_matrix, rotation),
        rotate_interaction_tensor(interaction_tensor, rotation),
    )


def rotate_hubbard_integrals(
    hopping_matrix: ArrayLike, U: Real, rotation: ArrayLike
) -> tuple[Matrix, Tensor]:
    """Rotate a one-body matrix and its local Hubbard interaction exactly."""

    L = _square_size("rotation", rotation)
    return rotate_integrals(
        hopping_matrix,
        onsite_interaction_tensor(L, U),
        rotation,
    )
