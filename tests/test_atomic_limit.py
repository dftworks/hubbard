import numpy as np

from hubbard_ed.basis import HubbardBasis
from hubbard_ed.hamiltonian import HubbardHamiltonian, double_occupancy_count


def test_atomic_limit_diagonal_energies() -> None:
    basis = HubbardBasis(5, 3, 2)
    U = -2.75
    hamiltonian = HubbardHamiltonian(basis, t=0.0, U=U, boundary="periodic")
    matrix = hamiltonian.to_sparse().toarray()
    expected = np.asarray([U * double_occupancy_count(state) for state in basis])
    np.testing.assert_array_equal(matrix, np.diag(expected))


def test_atomic_limit_matrix_free() -> None:
    basis = HubbardBasis(4, 2, 2)
    U = 3.2
    vector = np.linspace(-1.0, 1.0, len(basis))
    result = HubbardHamiltonian(basis, t=0.0, U=U).matvec(vector)
    expected = np.asarray(
        [U * double_occupancy_count(state) * value for state, value in zip(basis, vector)]
    )
    np.testing.assert_allclose(result, expected, atol=0.0)
