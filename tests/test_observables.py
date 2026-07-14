import numpy as np
import pytest

from hubbard_ed.lanczos import solve_ground_state
from hubbard_ed.observables import (
    charge_correlation_matrix,
    charge_structure_factor,
    double_occupancy_per_site,
    local_charge,
    local_double_occupancy,
    local_magnetization,
    momentum_distribution,
    one_body_density_matrix,
    spin_structure_factor,
    spin_z_correlation_matrix,
    total_double_occupancy,
)


@pytest.fixture(scope="module")
def ground_state():
    return solve_ground_state(L=4, n_up=2, n_down=2, U=3.0, n_eigenvalues=1)


def test_number_sum_rules(ground_state) -> None:
    charge = local_charge(ground_state.state, ground_state.basis)
    magnetization = local_magnetization(ground_state.state, ground_state.basis)
    assert np.sum(charge) == pytest.approx(4.0, abs=1e-12)
    assert np.sum(magnetization) == pytest.approx(0.0, abs=1e-12)


def test_double_occupancy_normalizations(ground_state) -> None:
    local = local_double_occupancy(ground_state.state, ground_state.basis)
    total = total_double_occupancy(ground_state.state, ground_state.basis)
    assert total == pytest.approx(np.sum(local), abs=1e-14)
    assert double_occupancy_per_site(
        ground_state.state, ground_state.basis
    ) == pytest.approx(total / 4.0, abs=1e-14)


def test_correlation_matrices_are_symmetric(ground_state) -> None:
    spin = spin_z_correlation_matrix(ground_state.state, ground_state.basis)
    charge = charge_correlation_matrix(ground_state.state, ground_state.basis)
    np.testing.assert_allclose(spin, spin.T, atol=1e-14)
    np.testing.assert_allclose(charge, charge.T, atol=1e-14)
    # Fixed total particle number makes the connected charge q=0 mode vanish.
    assert np.sum(charge) == pytest.approx(0.0, abs=1e-12)


def test_one_body_density_matrix_sum_rules(ground_state) -> None:
    for spin in ("up", "down"):
        density = one_body_density_matrix(
            ground_state.state, ground_state.basis, spin
        )
        np.testing.assert_allclose(density, density.conj().T, atol=1e-13)
        assert np.trace(density).real == pytest.approx(2.0, abs=1e-12)


def test_fourier_observable_sum_rules(ground_state) -> None:
    _, momentum = momentum_distribution(ground_state.state, ground_state.basis)
    _, charge = charge_structure_factor(ground_state.state, ground_state.basis)
    _, spin = spin_structure_factor(ground_state.state, ground_state.basis)
    assert np.sum(momentum) == pytest.approx(4.0, abs=1e-12)
    assert charge[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(charge > -1e-12)
    assert np.all(spin > -1e-12)


def test_observables_reject_zero_vector(ground_state) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        local_charge(np.zeros(len(ground_state.basis)), ground_state.basis)

