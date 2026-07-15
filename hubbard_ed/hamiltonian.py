"""Sparse and matrix-free Hubbard Hamiltonians."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Iterator, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator

from .basis import BasisState, HubbardBasis, validate_sector
from .interactions import validate_interaction_tensor
from .operators import Spin, apply_spin_hop, apply_up_down_term

Boundary = Literal["open", "periodic"]
Matrix: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
Scalar: TypeAlias = float | complex

# A conservative default for explicit CSR construction.  Matrix-free solving
# has a separate limit inherited from HubbardBasis.
DEFAULT_MAX_CSR_DIMENSION = 250_000
DEFAULT_MAX_TENSOR_ACTIONS = 5_000_000
HERMITICITY_TOLERANCE = 1e-12


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

    validate_sector(L, 0, 0)
    validate_boundary(boundary)
    bonds = [(site, site + 1) for site in range(L - 1)]
    if boundary == "periodic" and L > 2:
        bonds.append((L - 1, 0))
    return tuple(bonds)


def nearest_neighbor_hopping_matrix(
    L: int, *, t: Real = 1.0, boundary: Boundary = "open"
) -> NDArray[np.float64]:
    """Return ``h`` for the chain kinetic term ``sum_ij h_ij c_i^dagger c_j``."""

    validate_sector(L, 0, 0)
    hopping = _validate_real("t", t)
    matrix = np.zeros((L, L), dtype=float)
    for site_a, site_b in nearest_neighbor_bonds(L, boundary):
        matrix[site_a, site_b] = -hopping
        matrix[site_b, site_a] = -hopping
    return matrix


def double_occupancy_count(state: BasisState) -> int:
    """Return the number of sites occupied by both spins."""

    return (state[0] & state[1]).bit_count()


def validate_hopping_matrix(
    hopping_matrix: ArrayLike, L: int
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Validate, Hermitian-symmetrize, and copy a one-body matrix."""

    candidate = np.asarray(hopping_matrix)
    if candidate.shape != (L, L):
        raise ValueError(f"hopping_matrix must have shape ({L}, {L})")
    if not np.issubdtype(candidate.dtype, np.number):
        raise TypeError("hopping_matrix must contain numeric values")
    dtype = np.result_type(candidate.dtype, np.float64)
    matrix = np.array(candidate, dtype=dtype, copy=True)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hopping_matrix must contain only finite values")
    if not np.allclose(
        matrix,
        matrix.conj().T,
        rtol=HERMITICITY_TOLERANCE,
        atol=HERMITICITY_TOLERANCE,
    ):
        maximum_error = float(np.max(np.abs(matrix - matrix.conj().T)))
        raise ValueError(
            "hopping_matrix must be Hermitian; maximum |h-h^dagger| is "
            f"{maximum_error:.3e}"
        )
    # Accepted roundoff-level asymmetry is projected away so rmatvec=matvec and
    # Hermitian eigensolvers remain exact properties of the stored operator.
    matrix = 0.5 * (matrix + matrix.conj().T)
    matrix.flags.writeable = False
    return matrix


class HoppingMatrixHamiltonian:
    """Fixed-sector Hubbard Hamiltonian for an arbitrary one-body matrix.

    The convention is

    ``H = sum_ij,sigma h[i,j] c_i,sigma^dagger c_j,sigma
         + U sum_i n_i,up n_i,down``.

    ``h`` must be finite and Hermitian, may be real or complex, and may contain
    diagonal onsite terms.  It is copied and made read-only at construction.
    The same matrix is applied to both spins.  Every nonzero matrix entry is
    evaluated through :func:`hubbard_ed.operators.apply_spin_hop`, preserving
    the package's global fermionic convention.
    """

    def __init__(
        self,
        basis: HubbardBasis,
        hopping_matrix: ArrayLike,
        *,
        U: Real = 0.0,
    ) -> None:
        if not isinstance(basis, HubbardBasis):
            raise TypeError("basis must be a HubbardBasis")
        self.basis = basis
        self.U = _validate_real("U", U)
        self.hopping_matrix = validate_hopping_matrix(hopping_matrix, basis.L)
        self.dtype = np.dtype(np.result_type(self.hopping_matrix.dtype, np.float64))
        self._hopping_terms: tuple[tuple[int, int, Scalar], ...] = tuple(
            (int(destination), int(source), self.hopping_matrix[destination, source].item())
            for destination, source in zip(*np.nonzero(self.hopping_matrix))
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.basis.dimension, self.basis.dimension

    def _hopping_targets(
        self, state: BasisState
    ) -> Iterator[tuple[BasisState, Scalar]]:
        spins: tuple[Spin, Spin] = ("up", "down")
        for destination, source, matrix_element in self._hopping_terms:
            for spin in spins:
                result = apply_spin_hop(
                    state,
                    self.basis.L,
                    destination,
                    source,
                    spin,
                )
                if result is not None:
                    target, sign = result
                    yield target, matrix_element * sign

    def _column_targets(
        self, state: BasisState
    ) -> Iterator[tuple[BasisState, Scalar]]:
        diagonal = self.U * double_occupancy_count(state)
        if diagonal != 0.0:
            yield state, diagonal
        yield from self._hopping_targets(state)

    def matvec(self, vector: ArrayLike) -> Matrix:
        """Compute ``H @ vector`` without materializing ``H``."""

        x = np.asarray(vector)
        if x.ndim != 1 or x.shape[0] != self.basis.dimension:
            raise ValueError(
                f"vector must have shape ({self.basis.dimension},), got {x.shape}"
            )
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("vector must contain numeric values")
        dtype = np.result_type(x.dtype, self.dtype)
        y = np.zeros(self.basis.dimension, dtype=dtype)

        for column, state in enumerate(self.basis):
            amplitude = x[column]
            if amplitude == 0:
                continue
            for target, matrix_element in self._column_targets(state):
                y[self.basis.state_index(target)] += matrix_element * amplitude
        return y

    def aslinearoperator(self) -> LinearOperator:
        """Expose the Hermitian matrix-free action as a SciPy ``LinearOperator``."""

        return LinearOperator(
            shape=self.shape,
            matvec=self.matvec,
            rmatvec=self.matvec,
            dtype=self.dtype,
        )

    def to_sparse(
        self, *, max_dimension: int | None = DEFAULT_MAX_CSR_DIMENSION
    ) -> csr_matrix:
        """Construct the Hamiltonian as a Hermitian CSR matrix."""

        dimension = self.basis.dimension
        if max_dimension is not None and dimension > max_dimension:
            raise ValueError(
                f"CSR construction for dimension {dimension:,} exceeds the "
                f"configured limit {max_dimension:,}; use the matrix-free action "
                "or raise max_dimension explicitly after estimating memory"
            )

        rows: list[int] = []
        columns: list[int] = []
        values: list[Scalar] = []
        for column, state in enumerate(self.basis):
            for target, matrix_element in self._column_targets(state):
                rows.append(self.basis.state_index(target))
                columns.append(column)
                values.append(matrix_element)

        matrix = coo_matrix(
            (values, (rows, columns)), shape=self.shape, dtype=self.dtype
        ).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix


class TwoBodyTensorHamiltonian(HoppingMatrixHamiltonian):
    """Hamiltonian with an arbitrary spin-conserving up-down interaction.

    The interaction convention is

    ``sum_abcd V[a,b,c,d] c_a,up^dagger c_b,up``
    ``c_c,down^dagger c_d,down``.

    The tensor must be finite and Hermitian under
    ``V[a,b,c,d] = conj(V[b,a,d,c])``.  Only its exact nonzero entries are
    stored as action terms.  The default work guard estimates the number of
    attempted one- and two-body terms in one matrix-vector product.
    """

    def __init__(
        self,
        basis: HubbardBasis,
        hopping_matrix: ArrayLike,
        interaction_tensor: ArrayLike,
        *,
        max_action_terms: int | None = DEFAULT_MAX_TENSOR_ACTIONS,
    ) -> None:
        super().__init__(basis, hopping_matrix, U=0.0)
        self.interaction_tensor = validate_interaction_tensor(
            interaction_tensor, basis.L
        )
        self.dtype = np.dtype(
            np.result_type(
                self.hopping_matrix.dtype,
                self.interaction_tensor.dtype,
                np.float64,
            )
        )
        self._interaction_terms: tuple[
            tuple[int, int, int, int, Scalar], ...
        ] = tuple(
            (
                int(creation_up),
                int(annihilation_up),
                int(creation_down),
                int(annihilation_down),
                self.interaction_tensor[
                    creation_up,
                    annihilation_up,
                    creation_down,
                    annihilation_down,
                ].item(),
            )
            for creation_up, annihilation_up, creation_down, annihilation_down in zip(
                *np.nonzero(self.interaction_tensor)
            )
        )
        self.interaction_term_count = len(self._interaction_terms)
        self.estimated_action_terms = self.basis.dimension * (
            2 * len(self._hopping_terms) + self.interaction_term_count
        )

        if max_action_terms is not None:
            if isinstance(max_action_terms, bool) or not isinstance(
                max_action_terms, int
            ):
                raise TypeError("max_action_terms must be an integer or None")
            if max_action_terms < 1:
                raise ValueError("max_action_terms must be positive or None")
            if self.estimated_action_terms > max_action_terms:
                raise ValueError(
                    f"estimated tensor action work {self.estimated_action_terms:,} "
                    f"exceeds the configured limit {max_action_terms:,}; use a "
                    "sparser tensor, a smaller sector, or raise the limit "
                    "explicitly after benchmarking"
                )

    def _column_targets(
        self, state: BasisState
    ) -> Iterator[tuple[BasisState, Scalar]]:
        yield from super()._column_targets(state)
        for (
            creation_up,
            annihilation_up,
            creation_down,
            annihilation_down,
            matrix_element,
        ) in self._interaction_terms:
            result = apply_up_down_term(
                state,
                self.basis.L,
                creation_up,
                annihilation_up,
                creation_down,
                annihilation_down,
            )
            if result is not None:
                target, sign = result
                yield target, matrix_element * sign


class HubbardHamiltonian(HoppingMatrixHamiltonian):
    """Nearest-neighbor one-dimensional Hubbard Hamiltonian.

    This backwards-compatible specialization constructs the chain hopping
    matrix and delegates all action to :class:`HoppingMatrixHamiltonian`.
    """

    def __init__(
        self,
        basis: HubbardBasis,
        t: Real = 1.0,
        U: Real = 0.0,
        boundary: Boundary = "open",
    ) -> None:
        self.t = _validate_real("t", t)
        self.boundary = validate_boundary(boundary)
        self._bonds = nearest_neighbor_bonds(basis.L, self.boundary)
        super().__init__(
            basis,
            nearest_neighbor_hopping_matrix(
                basis.L, t=self.t, boundary=self.boundary
            ),
            U=U,
        )

    @property
    def bonds(self) -> tuple[tuple[int, int], ...]:
        return self._bonds


def build_hamiltonian(
    basis: HubbardBasis,
    *,
    t: Real = 1.0,
    U: Real = 0.0,
    boundary: Boundary = "open",
    max_dimension: int | None = DEFAULT_MAX_CSR_DIMENSION,
) -> csr_matrix:
    """Convenience wrapper returning a CSR nearest-neighbor Hamiltonian."""

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
    """Convenience wrapper for a one-off nearest-neighbor action ``H @ vector``."""

    return HubbardHamiltonian(basis, t=t, U=U, boundary=boundary).matvec(vector)
