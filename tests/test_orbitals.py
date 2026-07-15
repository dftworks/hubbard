import numpy as np
import pytest

from hubbard_ed import (
    HoppingMatrixHamiltonian,
    HubbardBasis,
    TwoBodyTensorHamiltonian,
    nearest_neighbor_hopping_matrix,
    onsite_interaction_tensor,
    rotate_hubbard_integrals,
    rotate_integrals,
    rotate_interaction_tensor,
    rotate_one_body,
    solve_hamiltonian,
    validate_unitary,
)


def _random_unitary(L: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(L, L)) + 1j * rng.normal(size=(L, L))
    unitary, _ = np.linalg.qr(raw)
    return unitary


def _complex_one_body(L: int) -> np.ndarray:
    matrix = nearest_neighbor_hopping_matrix(L, t=0.8, boundary="periodic")
    matrix = np.asarray(matrix, dtype=complex)
    matrix[0, 1] += 0.17j
    matrix[1, 0] -= 0.17j
    matrix += np.diag(np.linspace(-0.2, 0.3, L))
    return matrix


def test_validate_unitary_copies_and_makes_read_only() -> None:
    source = _random_unitary(3, seed=11)
    expected = source.copy()
    rotation = validate_unitary(source, 3)
    source[...] = 0.0

    np.testing.assert_allclose(rotation, expected, rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="read-only"):
        rotation[0, 0] = 2.0


@pytest.mark.parametrize(
    ("rotation", "error"),
    [
        (np.zeros((2, 3)), "shape"),
        (np.diag([1.0, 2.0]), "unitary"),
        (np.full((2, 2), np.nan), "finite"),
        (np.full((2, 2), "x"), "numeric"),
    ],
)
def test_validate_unitary_rejects_invalid_inputs(
    rotation: np.ndarray, error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        validate_unitary(rotation, 2)


def test_identity_rotation_leaves_integrals_unchanged() -> None:
    L = 3
    hopping = _complex_one_body(L)
    tensor = onsite_interaction_tensor(L, U=2.7)
    identity = np.eye(L)

    rotated_h = rotate_one_body(hopping, identity)
    rotated_v = rotate_interaction_tensor(tensor, identity)
    np.testing.assert_allclose(rotated_h, hopping)
    np.testing.assert_allclose(rotated_v, tensor)
    assert not rotated_h.flags.writeable
    assert not rotated_v.flags.writeable


def test_rotating_then_rotating_back_recovers_integrals() -> None:
    L = 3
    hopping = _complex_one_body(L)
    tensor = onsite_interaction_tensor(L, U=-1.4)
    rotation = _random_unitary(L, seed=29)

    rotated_h, rotated_v = rotate_integrals(hopping, tensor, rotation)
    recovered_h, recovered_v = rotate_integrals(
        rotated_h, rotated_v, rotation.conj().T
    )

    np.testing.assert_allclose(recovered_h, hopping, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(recovered_v, tensor, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("n_up", "n_down"), [(1, 1), (2, 1), (2, 2)])
def test_exact_rotation_preserves_complete_many_body_spectrum(
    n_up: int, n_down: int
) -> None:
    L = 4
    U = 3.2
    basis = HubbardBasis(L, n_up, n_down)
    hopping = _complex_one_body(L)
    rotation = _random_unitary(L, seed=43)
    rotated_h, rotated_v = rotate_hubbard_integrals(hopping, U, rotation)

    site_hamiltonian = TwoBodyTensorHamiltonian(
        basis, hopping, onsite_interaction_tensor(L, U)
    )
    rotated_hamiltonian = TwoBodyTensorHamiltonian(
        basis, rotated_h, rotated_v
    )
    site_spectrum = np.linalg.eigvalsh(site_hamiltonian.to_sparse().toarray())
    rotated_spectrum = np.linalg.eigvalsh(
        rotated_hamiltonian.to_sparse().toarray()
    )

    np.testing.assert_allclose(
        rotated_spectrum, site_spectrum, rtol=1e-11, atol=1e-11
    )


def test_rotated_matrix_free_solver_has_small_residual() -> None:
    L = 4
    basis = HubbardBasis(L, 2, 2)
    hopping = _complex_one_body(L)
    rotation = _random_unitary(L, seed=71)
    rotated_h, rotated_v = rotate_hubbard_integrals(hopping, 4.0, rotation)
    hamiltonian = TwoBodyTensorHamiltonian(basis, rotated_h, rotated_v)

    result = solve_hamiltonian(hamiltonian, n_eigenvalues=3)
    sparse = hamiltonian.to_sparse()
    exact = np.linalg.eigvalsh(sparse.toarray())[:3]
    rng = np.random.default_rng(97)
    vector = rng.normal(size=basis.dimension) + 1j * rng.normal(
        size=basis.dimension
    )

    np.testing.assert_allclose(result.eigenvalues, exact, atol=1e-10)
    np.testing.assert_allclose(
        hamiltonian.matvec(vector), sparse @ vector, rtol=1e-12, atol=1e-12
    )
    assert np.all(result.residual_norms < 1e-8)


def test_rotating_only_one_body_part_changes_interacting_model() -> None:
    L = 4
    U = 3.2
    basis = HubbardBasis(L, 2, 2)
    hopping = _complex_one_body(L)
    rotation = _random_unitary(L, seed=101)
    rotated_h, rotated_v = rotate_hubbard_integrals(hopping, U, rotation)

    exact = TwoBodyTensorHamiltonian(basis, rotated_h, rotated_v)
    incorrectly_local = HoppingMatrixHamiltonian(basis, rotated_h, U=U)
    exact_spectrum = np.linalg.eigvalsh(exact.to_sparse().toarray())
    incorrect_spectrum = np.linalg.eigvalsh(
        incorrectly_local.to_sparse().toarray()
    )

    assert np.max(np.abs(exact_spectrum - incorrect_spectrum)) > 1e-3
