import numpy as np
import pytest

from hubbard_ed.basis import HubbardBasis
from hubbard_ed.hamiltonian import (
    HoppingMatrixHamiltonian,
    HubbardHamiltonian,
    nearest_neighbor_hopping_matrix,
)
from hubbard_ed.lanczos import solve_hamiltonian


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_nearest_neighbor_specialization_matches_generic(boundary: str) -> None:
    basis = HubbardBasis(5, 2, 1)
    hopping = nearest_neighbor_hopping_matrix(5, t=0.73, boundary=boundary)
    generic = HoppingMatrixHamiltonian(basis, hopping, U=2.4)
    chain = HubbardHamiltonian(basis, t=0.73, U=2.4, boundary=boundary)

    np.testing.assert_array_equal(chain.hopping_matrix, hopping)
    np.testing.assert_array_equal(generic.to_sparse().toarray(), chain.to_sparse().toarray())

    vector = np.random.default_rng(19).normal(size=len(basis))
    np.testing.assert_allclose(generic.matvec(vector), chain.matvec(vector), atol=0.0)


def test_arbitrary_matrix_sparse_and_matrix_free_agree() -> None:
    basis = HubbardBasis(4, 2, 2)
    hopping = np.asarray(
        [
            [0.2, -1.0, 0.3j, 0.0],
            [-1.0, -0.1, -0.7, 0.15],
            [-0.3j, -0.7, 0.4, -0.8],
            [0.0, 0.15, -0.8, -0.25],
        ],
        dtype=complex,
    )
    hamiltonian = HoppingMatrixHamiltonian(basis, hopping, U=1.9)
    sparse = hamiltonian.to_sparse()
    rng = np.random.default_rng(88)
    vector = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))

    np.testing.assert_allclose(sparse.toarray(), sparse.toarray().conj().T, atol=1e-14)
    np.testing.assert_allclose(
        sparse @ vector, hamiltonian.matvec(vector), rtol=1e-13, atol=1e-13
    )


def test_noninteracting_spectrum_with_onsite_and_long_range_terms() -> None:
    basis = HubbardBasis(3, 1, 1)
    hopping = np.asarray(
        [
            [0.4, -1.0, -0.25],
            [-1.0, -0.3, -0.8],
            [-0.25, -0.8, 0.1],
        ]
    )
    hamiltonian = HoppingMatrixHamiltonian(basis, hopping, U=0.0)
    numerical = np.linalg.eigvalsh(hamiltonian.to_sparse().toarray())
    orbitals = np.linalg.eigvalsh(hopping)
    expected = np.sort(np.asarray([up + down for up in orbitals for down in orbitals]))
    np.testing.assert_allclose(numerical, expected, rtol=1e-12, atol=1e-12)


def test_complex_flux_iterative_solver() -> None:
    L = 4
    basis = HubbardBasis(L, 2, 1)
    hopping = np.zeros((L, L), dtype=complex)
    phase = np.exp(1j * 0.61 / L)
    for source in range(L):
        destination = (source + 1) % L
        hopping[destination, source] = -phase
        hopping[source, destination] = -phase.conjugate()

    hamiltonian = HoppingMatrixHamiltonian(basis, hopping, U=1.7)
    exact = np.linalg.eigvalsh(hamiltonian.to_sparse().toarray())[:4]
    matrix_free = solve_hamiltonian(
        hamiltonian, n_eigenvalues=4, matrix_free=True
    )
    sparse = solve_hamiltonian(
        hamiltonian, n_eigenvalues=4, matrix_free=False
    )

    np.testing.assert_allclose(matrix_free.eigenvalues, exact, atol=1e-10)
    np.testing.assert_allclose(sparse.eigenvalues, exact, atol=1e-10)
    assert np.iscomplexobj(matrix_free.state)
    assert np.all(matrix_free.residual_norms < 1e-8)
    assert np.all(sparse.residual_norms < 1e-8)


def test_hopping_matrix_is_copied_and_read_only() -> None:
    basis = HubbardBasis(2, 1, 0)
    source = np.asarray([[0.0, -1.0], [-1.0, 0.0]])
    hamiltonian = HoppingMatrixHamiltonian(basis, source)
    source[0, 1] = 99.0
    assert hamiltonian.hopping_matrix[0, 1] == -1.0
    with pytest.raises(ValueError, match="read-only"):
        hamiltonian.hopping_matrix[0, 1] = 2.0


def test_roundoff_level_asymmetry_is_projected_away() -> None:
    basis = HubbardBasis(2, 1, 0)
    hopping = np.asarray([[0.0, 1.0 + 5e-14j], [1.0, 0.0]], dtype=complex)
    hamiltonian = HoppingMatrixHamiltonian(basis, hopping)
    np.testing.assert_array_equal(
        hamiltonian.hopping_matrix, hamiltonian.hopping_matrix.conj().T
    )


@pytest.mark.parametrize(
    ("matrix", "error"),
    [
        (np.zeros((2, 3)), "shape"),
        (np.asarray([[0.0, 1.0], [0.0, 0.0]]), "Hermitian"),
        (np.asarray([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
        (np.asarray([[0.0, "x"], ["x", 0.0]]), "numeric"),
    ],
)
def test_invalid_hopping_matrices(matrix: np.ndarray, error: str) -> None:
    basis = HubbardBasis(2, 1, 1)
    with pytest.raises((TypeError, ValueError), match=error):
        HoppingMatrixHamiltonian(basis, matrix)
