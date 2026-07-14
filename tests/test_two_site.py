import numpy as np
import pytest

from hubbard_ed.analytic import two_site_ground_energy
from hubbard_ed.lanczos import solve_ground_state


@pytest.mark.parametrize(
    ("U", "t"), [(0.0, 1.0), (1.0, 0.7), (4.0, 1.0), (12.0, 1.5), (-3.0, 0.8)]
)
def test_two_site_ground_state(U: float, t: float) -> None:
    result = solve_ground_state(
        L=2, n_up=1, n_down=1, t=t, U=U, boundary="open", n_eigenvalues=2
    )
    assert result.energy == pytest.approx(two_site_ground_energy(U, t), abs=1e-12)
    assert np.linalg.norm(result.state) == pytest.approx(1.0, abs=1e-14)
    assert np.all(result.residual_norms < 1e-12)


@pytest.mark.parametrize("U", [50.0, 100.0, 200.0])
def test_large_u_singlet_triplet_splitting(U: float) -> None:
    t = 1.0
    result = solve_ground_state(
        L=2, n_up=1, n_down=1, t=t, U=U, n_eigenvalues=2
    )
    splitting = result.eigenvalues[1] - result.eigenvalues[0]
    assert splitting == pytest.approx(4.0 * t * t / U, rel=2e-3)
    assert np.all(result.residual_norms < 1e-11)

