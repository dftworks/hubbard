import numpy as np
import pytest

from hubbard_ed.basis import HubbardBasis
from hubbard_ed.hamiltonian import HubbardHamiltonian
from hubbard_ed.lanczos import solve_ground_state


@pytest.mark.parametrize("boundary", ["open", "periodic"])
@pytest.mark.parametrize("sector", [(4, 2, 2), (5, 2, 1), (3, 0, 2)])
def test_hermiticity(boundary: str, sector: tuple[int, int, int]) -> None:
    basis = HubbardBasis(*sector)
    matrix = HubbardHamiltonian(basis, t=0.7, U=-1.3, boundary=boundary).to_sparse()
    difference = matrix - matrix.getH()
    assert np.linalg.norm(difference.toarray()) < 1e-14


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_sparse_and_matrix_free_agree(boundary: str) -> None:
    basis = HubbardBasis(5, 2, 3)
    hamiltonian = HubbardHamiltonian(basis, t=1.2, U=3.7, boundary=boundary)
    sparse = hamiltonian.to_sparse()
    rng = np.random.default_rng(814)
    for vector in (
        rng.normal(size=len(basis)),
        rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis)),
    ):
        np.testing.assert_allclose(
            sparse @ vector, hamiltonian.matvec(vector), rtol=1e-13, atol=1e-13
        )


def test_periodic_boundary_hopping_sign_appears_in_matrix() -> None:
    basis = HubbardBasis(4, 2, 0)
    matrix = HubbardHamiltonian(basis, t=2.0, boundary="periodic").to_sparse()
    source = basis.state_index((0b1010, 0))
    target = basis.state_index((0b0011, 0))
    # The operator sign is -1, so the Hamiltonian matrix element -t * sign is +t.
    assert matrix[target, source] == pytest.approx(2.0)
    assert matrix[source, target] == pytest.approx(2.0)


def test_two_site_periodic_chain_uses_one_unique_bond() -> None:
    basis = HubbardBasis(2, 1, 0)
    open_matrix = HubbardHamiltonian(basis, t=1.0, boundary="open").to_sparse()
    periodic_matrix = HubbardHamiltonian(
        basis, t=1.0, boundary="periodic"
    ).to_sparse()
    np.testing.assert_array_equal(open_matrix.toarray(), periodic_matrix.toarray())


def test_invalid_vector_and_boundary() -> None:
    basis = HubbardBasis(2, 1, 1)
    with pytest.raises(ValueError, match="boundary"):
        HubbardHamiltonian(basis, boundary="wrapped")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        HubbardHamiltonian(basis).matvec(np.ones(len(basis) + 1))


def test_sparse_and_matrix_free_eigensolvers_agree() -> None:
    parameters = dict(
        L=4,
        n_up=2,
        n_down=2,
        t=0.85,
        U=2.3,
        boundary="periodic",
        n_eigenvalues=3,
    )
    matrix_free = solve_ground_state(**parameters, matrix_free=True)
    sparse = solve_ground_state(**parameters, matrix_free=False)
    np.testing.assert_allclose(
        matrix_free.eigenvalues, sparse.eigenvalues, rtol=1e-12, atol=1e-12
    )
    assert np.all(matrix_free.residual_norms < 1e-8)
    assert np.all(sparse.residual_norms < 1e-8)
