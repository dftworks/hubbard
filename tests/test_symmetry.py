import numpy as np

from hubbard_ed.basis import HubbardBasis
from hubbard_ed.hamiltonian import HubbardHamiltonian
from hubbard_ed.lanczos import solve_ground_state
from hubbard_ed.observables import local_charge, spin_z_correlation_matrix


def test_spin_exchange_symmetry_of_spectrum() -> None:
    up_down = HubbardBasis(5, 3, 1)
    down_up = HubbardBasis(5, 1, 3)
    spectrum_a = np.linalg.eigvalsh(
        HubbardHamiltonian(up_down, t=0.9, U=2.1, boundary="periodic")
        .to_sparse()
        .toarray()
    )
    spectrum_b = np.linalg.eigvalsh(
        HubbardHamiltonian(down_up, t=0.9, U=2.1, boundary="periodic")
        .to_sparse()
        .toarray()
    )
    np.testing.assert_allclose(spectrum_a, spectrum_b, atol=1e-12)


def test_open_chain_ground_state_has_reflection_symmetry() -> None:
    result = solve_ground_state(L=6, n_up=3, n_down=3, U=4.0, boundary="open")
    charge = local_charge(result.state, result.basis)
    spin = spin_z_correlation_matrix(result.state, result.basis)
    np.testing.assert_allclose(charge, charge[::-1], atol=1e-10)
    np.testing.assert_allclose(spin, spin[::-1, ::-1], atol=1e-10)
    assert result.residual_norm < 1e-8

