"""Ground-state and low-energy solving through SciPy's Lanczos interface."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

from .basis import DEFAULT_MAX_DIMENSION, HubbardBasis, estimate_dimension
from .hamiltonian import (
    Boundary,
    HoppingMatrixHamiltonian,
    HubbardHamiltonian,
)

DEFAULT_EIGSH_TOLERANCE = 1e-10
_MAX_DENSE_FALLBACK_DIMENSION = 64

StateVector: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
StateMatrix: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class GroundStateResult:
    """Lowest eigenpairs and diagnostics returned by the solver functions."""

    energy: float
    state: StateVector
    basis: HubbardBasis
    residual_norm: float
    excited_energies: tuple[float, ...]
    eigenvalues: NDArray[np.float64]
    eigenvectors: StateMatrix
    residual_norms: NDArray[np.float64]


def _dense_lowest(
    hamiltonian: HoppingMatrixHamiltonian, n_eigenvalues: int
) -> tuple[NDArray[np.float64], StateMatrix]:
    dense = hamiltonian.to_sparse(max_dimension=None).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return eigenvalues[:n_eigenvalues], eigenvectors[:, :n_eigenvalues]


def solve_hamiltonian(
    hamiltonian: HoppingMatrixHamiltonian,
    *,
    n_eigenvalues: int = 1,
    tolerance: float = DEFAULT_EIGSH_TOLERANCE,
    maxiter: int | None = None,
    matrix_free: bool = True,
) -> GroundStateResult:
    """Solve a constructed fixed-sector Hamiltonian for its lowest states.

    The default route passes the on-the-fly ``LinearOperator`` to
    :func:`scipy.sparse.linalg.eigsh`.  Real and complex Hermitian hopping
    matrices are both supported.  Tiny sectors and complete-spectrum requests
    use a dense fallback because ARPACK requires ``k < dimension``.  Residuals
    are always recomputed with the matrix-free action.
    """

    if not isinstance(hamiltonian, HoppingMatrixHamiltonian):
        raise TypeError("hamiltonian must be a HoppingMatrixHamiltonian")
    dimension = hamiltonian.basis.dimension
    if isinstance(n_eigenvalues, bool) or not isinstance(n_eigenvalues, int):
        raise TypeError("n_eigenvalues must be an integer")
    if not 1 <= n_eigenvalues <= dimension:
        raise ValueError(
            f"n_eigenvalues must satisfy 1 <= n_eigenvalues <= {dimension}"
        )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or tolerance <= 0
        or not np.isfinite(tolerance)
    ):
        raise ValueError("tolerance must be a finite positive real number")
    if not isinstance(matrix_free, bool):
        raise TypeError("matrix_free must be a boolean")

    use_dense = dimension <= 4 or n_eigenvalues == dimension
    if use_dense:
        if dimension > _MAX_DENSE_FALLBACK_DIMENSION:
            raise ValueError(
                "requesting the complete spectrum would require an unsafe dense "
                f"matrix of dimension {dimension}; request fewer eigenvalues"
            )
        eigenvalues, eigenvectors = _dense_lowest(hamiltonian, n_eigenvalues)
    else:
        operator = (
            hamiltonian.aslinearoperator()
            if matrix_free
            else hamiltonian.to_sparse()
        )
        # A deterministic random start avoids accidentally selecting only one
        # symmetry block, which can happen with a constant initial vector.
        rng = np.random.default_rng(1729)
        v0: StateVector = rng.normal(size=dimension)
        if np.issubdtype(hamiltonian.dtype, np.complexfloating):
            v0 = v0 + 1j * rng.normal(size=dimension)
        eigenvalues, eigenvectors = eigsh(
            operator,
            k=n_eigenvalues,
            which="SA",
            tol=float(tolerance),
            maxiter=maxiter,
            v0=v0,
        )
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    eigenvectors = np.asarray(eigenvectors)
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    values = np.asarray(eigenvalues, dtype=float)
    residuals = np.asarray(
        [
            np.linalg.norm(
                hamiltonian.matvec(eigenvectors[:, index])
                - energy * eigenvectors[:, index]
            )
            for index, energy in enumerate(values)
        ],
        dtype=float,
    )
    return GroundStateResult(
        energy=float(values[0]),
        state=eigenvectors[:, 0].copy(),
        basis=hamiltonian.basis,
        residual_norm=float(residuals[0]),
        excited_energies=tuple(float(value) for value in values[1:]),
        eigenvalues=values,
        eigenvectors=eigenvectors,
        residual_norms=residuals,
    )


def solve_ground_state(
    *,
    L: int,
    n_up: int,
    n_down: int,
    t: Real = 1.0,
    U: Real = 0.0,
    boundary: Boundary = "open",
    n_eigenvalues: int = 1,
    tolerance: float = DEFAULT_EIGSH_TOLERANCE,
    maxiter: int | None = None,
    matrix_free: bool = True,
    max_dimension: int | None = DEFAULT_MAX_DIMENSION,
) -> GroundStateResult:
    """Construct and solve a nearest-neighbor one-dimensional Hubbard chain.

    The default ARPACK tolerance is ``1e-10``.  Call
    :func:`hubbard_ed.basis.estimate_dimension` before solving when exposing
    this function in an interactive workflow.
    """

    estimate_dimension(L, n_up, n_down)
    basis = HubbardBasis(L, n_up, n_down, max_dimension=max_dimension)
    hamiltonian = HubbardHamiltonian(basis, t=t, U=U, boundary=boundary)
    return solve_hamiltonian(
        hamiltonian,
        n_eigenvalues=n_eigenvalues,
        tolerance=tolerance,
        maxiter=maxiter,
        matrix_free=matrix_free,
    )
