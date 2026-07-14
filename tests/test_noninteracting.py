import numpy as np
import pytest

from hubbard_ed.analytic import noninteracting_sector_energies
from hubbard_ed.basis import HubbardBasis
from hubbard_ed.hamiltonian import HubbardHamiltonian
from hubbard_ed.lanczos import solve_ground_state


@pytest.mark.parametrize(
    ("L", "n_up", "n_down", "boundary"),
    [
        (2, 1, 1, "open"),
        (2, 1, 0, "periodic"),
        (4, 2, 1, "open"),
        (4, 2, 1, "periodic"),
        (5, 1, 2, "periodic"),
    ],
)
def test_complete_noninteracting_spectrum(
    L: int, n_up: int, n_down: int, boundary: str
) -> None:
    basis = HubbardBasis(L, n_up, n_down)
    matrix = HubbardHamiltonian(basis, t=0.83, U=0.0, boundary=boundary).to_sparse()
    numerical = np.linalg.eigvalsh(matrix.toarray())
    analytic = noninteracting_sector_energies(
        L, n_up, n_down, t=0.83, boundary=boundary
    )
    np.testing.assert_allclose(numerical, analytic, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_noninteracting_iterative_ground_state_residual(boundary: str) -> None:
    result = solve_ground_state(
        L=6,
        n_up=3,
        n_down=2,
        t=1.0,
        U=0.0,
        boundary=boundary,
        n_eigenvalues=3,
    )
    expected = noninteracting_sector_energies(6, 3, 2, boundary=boundary)[:3]
    np.testing.assert_allclose(result.eigenvalues, expected, atol=1e-10)
    assert np.all(result.residual_norms < 1e-8)

