"""Step 6: measure diagonal, off-diagonal, and Fourier observables."""

import numpy as np

from hubbard_ed import solve_ground_state
from hubbard_ed.observables import (
    charge_correlation,
    double_occupancy_per_site,
    local_charge,
    one_body_density_matrix,
    spin_structure_factor,
    spin_z_correlation,
    total_double_occupancy,
)


def main() -> None:
    result = solve_ground_state(L=6, n_up=3, n_down=3, U=4.0)
    psi, basis = result.state, result.basis

    charge = local_charge(psi, basis)
    density_up = one_body_density_matrix(psi, basis, "up")
    momenta, spin_factor = spin_structure_factor(psi, basis)
    pi_index = basis.L // 2

    print(f"total double occupancy: {total_double_occupancy(psi, basis):.9f}")
    print(f"double occupancy/site:  {double_occupancy_per_site(psi, basis):.9f}")
    print(f"local charge:           {np.array2string(charge, precision=6)}")
    print(f"<Sz_0 Sz_1>:            {spin_z_correlation(psi, basis, 0, 1):.9f}")
    print(f"connected Cn(0,1):      {charge_correlation(psi, basis, 0, 1):.9f}")
    print(f"Tr(gamma_up):           {np.trace(density_up).real:.9f}")
    print(f"S(q=pi):                {spin_factor[pi_index]:.9f}")

    np.testing.assert_allclose(np.sum(charge), 6.0, atol=1e-12)
    assert np.isclose(np.trace(density_up), 3.0)
    assert np.isclose(momenta[pi_index], np.pi)


if __name__ == "__main__":
    main()
