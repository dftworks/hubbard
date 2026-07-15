"""Step 5: compute low-energy states and verify their residuals."""

import numpy as np

from hubbard_ed import solve_ground_state
from hubbard_ed.analytic import two_site_ground_energy


def main() -> None:
    result = solve_ground_state(
        L=6,
        n_up=3,
        n_down=3,
        t=1.0,
        U=4.0,
        boundary="open",
        n_eigenvalues=3,
    )
    print(f"dimension: {len(result.basis)}")
    print(f"lowest energies: {result.eigenvalues}")
    print(f"residual norms:  {result.residual_norms}")
    assert np.all(result.residual_norms < 1e-8)

    two_site = solve_ground_state(L=2, n_up=1, n_down=1, t=1.0, U=4.0)
    exact = two_site_ground_energy(U=4.0, t=1.0)
    print(f"two-site |E_ED-E_exact| = {abs(two_site.energy - exact):.3e}")
    assert abs(two_site.energy - exact) < 1e-12


if __name__ == "__main__":
    main()
