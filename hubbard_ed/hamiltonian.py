"""Sparse and matrix-free Hubbard Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Iterator, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator

from .basis import BasisState, HubbardBasis
from .operators import Spin, apply_spin_hop

Boundary = Literal["open", "periodic"]
Matrix: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]

# A conservative default for explicit CSR construction.  Matrix-free solving
# has a separate limit inherited from HubbardBasis.
DEFAULT_MAX_CSR_DIMENSION = 250_000


def validate_boundary(boundary: str) -> Boundary:
    if boundary not in ("open", "periodic"):
        raise ValueError("boundary must be 'open' or 'periodic'")
    return boundary


def _validate_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def nearest_neighbor_bonds(L: int, boundary: Boundary) -> tuple[tuple[int, int], ...]:
    """Return unique undirected nearest-neighbor bonds.

    For two sites, open and periodic chains both contain the single bond
    ``(0, 1)``.  This unique-edge convention avoids doubling the hopping and is
    the convention used by the two-site analytic reference.
    """

    validate_boundary(boundary)
    bonds = [(site, site + 1) for site in range(L - 1)]
    if boundary == "periodic" and L > 2:
        bonds.append((L - 1, 0))
    return tuple(bonds)


def double_occupancy_count(state: BasisState) -> int:
    """Return the number of sites occupied by both spins."""

    return (state[0] & state[1]).bit_count()


@dataclass(frozen=True, slots=True)
class HubbardHamiltonian:
    """Hubbard Hamiltonian tied to a particular fixed-sector basis.

    ``matvec`` applies the Hamiltonian directly from bit operations without
    storing a sparse matrix.  ``to_sparse`` constructs the equivalent CSR
    representation from the same hopping primitive.
    """

    basis: HubbardBasis
    t: float = 1.0
    U: float = 0.0
    boundary: Boundary = "open"

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _validate_real("t", self.t))
        object.__setattr__(self, "U", _validate_real("U", self.U))
        object.__setattr__(self, "boundary", validate_boundary(self.boundary))

    @property
    def shape(self) -> tuple[int, int]:
        return self.basis.dimension, self.basis.dimension

    @property
    def bonds(self) -> tuple[tuple[int, int], ...]:
        return nearest_neighbor_bonds(self.basis.L, self.boundary)

    def _hopping_targets(
        self, state: BasisState
    ) -> Iterator[tuple[BasisState, float]]:
        for site_a, site_b in self.bonds:
            for spin in ("up", "down"):
                typed_spin: Spin = spin
                for destination, source in ((site_a, site_b), (site_b, site_a)):
                    result = apply_spin_hop(
                        state,
                        self.basis.L,
                        destination,
                        source,
                        typed_spin,
                    )
                    if result is not None:
                        target, sign = result
                        yield target, -self.t * sign

    def matvec(self, vector: ArrayLike) -> Matrix:
        """Compute ``H @ vector`` without materializing ``H``."""

        x = np.asarray(vector)
        if x.ndim != 1 or x.shape[0] != self.basis.dimension:
            raise ValueError(
                f"vector must have shape ({self.basis.dimension},), got {x.shape}"
            )
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("vector must contain numeric values")
        dtype = np.result_type(x.dtype, np.float64)
        y = np.zeros(self.basis.dimension, dtype=dtype)

        for column, state in enumerate(self.basis):
            amplitude = x[column]
            if amplitude == 0:
                continue
            y[column] += self.U * double_occupancy_count(state) * amplitude
            for target, matrix_element in self._hopping_targets(state):
                y[self.basis.state_index(target)] += matrix_element * amplitude
        return y

    def aslinearoperator(self) -> LinearOperator:
        """Expose the matrix-free action as a SciPy ``LinearOperator``."""

        return LinearOperator(
            shape=self.shape,
            matvec=self.matvec,
            rmatvec=self.matvec,
            dtype=np.dtype(np.float64),
        )

    def to_sparse(
        self, *, max_dimension: int | None = DEFAULT_MAX_CSR_DIMENSION
    ) -> csr_matrix:
        """Construct the Hamiltonian as a real symmetric CSR matrix."""

        dimension = self.basis.dimension
        if max_dimension is not None and dimension > max_dimension:
            raise ValueError(
                f"CSR construction for dimension {dimension:,} exceeds the "
                f"configured limit {max_dimension:,}; use the matrix-free action "
                "or raise max_dimension explicitly after estimating memory"
            )

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for column, state in enumerate(self.basis):
            diagonal = self.U * double_occupancy_count(state)
            if diagonal != 0.0:
                rows.append(column)
                columns.append(column)
                values.append(diagonal)
            for target, matrix_element in self._hopping_targets(state):
                rows.append(self.basis.state_index(target))
                columns.append(column)
                values.append(matrix_element)

        matrix = coo_matrix(
            (values, (rows, columns)), shape=self.shape, dtype=np.float64
        ).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix


def build_hamiltonian(
    basis: HubbardBasis,
    *,
    t: Real = 1.0,
    U: Real = 0.0,
    boundary: Boundary = "open",
    max_dimension: int | None = DEFAULT_MAX_CSR_DIMENSION,
) -> csr_matrix:
    """Convenience wrapper returning a CSR Hubbard Hamiltonian."""

    return HubbardHamiltonian(basis, t=t, U=U, boundary=boundary).to_sparse(
        max_dimension=max_dimension
    )


def hamiltonian_action(
    basis: HubbardBasis,
    vector: ArrayLike,
    *,
    t: Real = 1.0,
    U: Real = 0.0,
    boundary: Boundary = "open",
) -> Matrix:
    """Convenience wrapper for a one-off matrix-free action ``H @ vector``."""

    return HubbardHamiltonian(basis, t=t, U=U, boundary=boundary).matvec(vector)
