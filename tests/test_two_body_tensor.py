import numpy as np
import pytest

from hubbard_ed import (
    HoppingMatrixHamiltonian,
    HubbardBasis,
    TwoBodyTensorHamiltonian,
    onsite_interaction_tensor,
    solve_hamiltonian,
)
from hubbard_ed.interactions import validate_interaction_tensor
from hubbard_ed.operators import apply_up_down_term


def _random_hermitian_tensor(L: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(L, L, L, L)) + 1j * rng.normal(
        size=(L, L, L, L)
    )
    return 0.5 * (raw + raw.conj().transpose(1, 0, 3, 2))


def test_local_operator_term_is_double_occupancy() -> None:
    occupied = (0b101, 0b001)
    assert apply_up_down_term(occupied, 3, 0, 0, 0, 0) == (occupied, 1)
    assert apply_up_down_term(occupied, 3, 2, 2, 2, 2) is None


def test_onsite_tensor_entries_and_immutability() -> None:
    tensor = onsite_interaction_tensor(4, U=2.5)
    expected = np.zeros((4, 4, 4, 4))
    sites = np.arange(4)
    expected[sites, sites, sites, sites] = 2.5
    np.testing.assert_array_equal(tensor, expected)
    with pytest.raises(ValueError, match="read-only"):
        tensor[0, 0, 0, 0] = 1.0


@pytest.mark.parametrize("U", [-2.3, 0.0, 4.1])
def test_local_tensor_matches_existing_hamiltonian(U: float) -> None:
    basis = HubbardBasis(4, 2, 2)
    hopping = np.asarray(
        [
            [0.2, -1.0 + 0.1j, 0.0, -0.3j],
            [-1.0 - 0.1j, -0.1, -0.8, 0.0],
            [0.0, -0.8, 0.3, -0.6 + 0.2j],
            [0.3j, 0.0, -0.6 - 0.2j, 0.0],
        ],
        dtype=complex,
    )
    existing = HoppingMatrixHamiltonian(basis, hopping, U=U)
    tensor = TwoBodyTensorHamiltonian(
        basis, hopping, onsite_interaction_tensor(4, U)
    )

    np.testing.assert_allclose(
        tensor.to_sparse().toarray(),
        existing.to_sparse().toarray(),
        rtol=1e-13,
        atol=1e-13,
    )
    rng = np.random.default_rng(41)
    vector = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    np.testing.assert_allclose(
        tensor.matvec(vector),
        existing.matvec(vector),
        rtol=1e-13,
        atol=1e-13,
    )


def test_arbitrary_tensor_is_hermitian_and_matrix_free_agrees() -> None:
    basis = HubbardBasis(3, 1, 1)
    hopping = np.asarray(
        [[0.1, -1.0j, -0.2], [1.0j, 0.0, -0.7], [-0.2, -0.7, -0.1]]
    )
    tensor = _random_hermitian_tensor(3, seed=53)
    hamiltonian = TwoBodyTensorHamiltonian(basis, hopping, tensor)
    sparse = hamiltonian.to_sparse()
    rng = np.random.default_rng(87)
    vector = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))

    np.testing.assert_allclose(
        sparse.toarray(), sparse.toarray().conj().T, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        sparse @ vector, hamiltonian.matvec(vector), rtol=1e-13, atol=1e-13
    )


def test_tensor_iterative_solver_residuals() -> None:
    basis = HubbardBasis(4, 2, 1)
    hopping = np.diag(np.linspace(-0.2, 0.2, 4))
    tensor = np.zeros((4, 4, 4, 4), dtype=complex)
    tensor[0, 1, 2, 3] = 0.3j
    tensor[1, 0, 3, 2] = -0.3j
    tensor += onsite_interaction_tensor(4, 1.7)
    hamiltonian = TwoBodyTensorHamiltonian(basis, hopping, tensor)
    result = solve_hamiltonian(hamiltonian, n_eigenvalues=3)
    exact = np.linalg.eigvalsh(hamiltonian.to_sparse().toarray())[:3]

    np.testing.assert_allclose(result.eigenvalues, exact, atol=1e-10)
    assert np.all(result.residual_norms < 1e-8)


def test_tensor_is_copied_projected_and_read_only() -> None:
    source = np.zeros((2, 2, 2, 2), dtype=complex)
    source[0, 1, 0, 1] = 1.0 + 5e-14j
    source[1, 0, 1, 0] = 1.0
    tensor = validate_interaction_tensor(source, 2)
    source[...] = 99.0

    np.testing.assert_array_equal(tensor, tensor.conj().transpose(1, 0, 3, 2))
    assert tensor[0, 1, 0, 1] != 99.0
    with pytest.raises(ValueError, match="read-only"):
        tensor[0, 0, 0, 0] = 1.0


@pytest.mark.parametrize(
    ("tensor", "error"),
    [
        (np.zeros((2, 2, 2)), "shape"),
        (
            np.pad(np.ones((1, 1, 1, 1)), ((0, 1), (0, 1), (0, 1), (1, 0))),
            "Hermiticity",
        ),
        (np.full((2, 2, 2, 2), np.nan), "finite"),
        (np.full((2, 2, 2, 2), "x"), "numeric"),
    ],
)
def test_invalid_interaction_tensors(tensor: np.ndarray, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        validate_interaction_tensor(tensor, 2)


def test_tensor_action_work_guard() -> None:
    basis = HubbardBasis(4, 2, 2)
    hopping = np.zeros((4, 4))
    tensor = np.ones((4, 4, 4, 4))
    with pytest.raises(ValueError, match="estimated tensor action work"):
        TwoBodyTensorHamiltonian(
            basis, hopping, tensor, max_action_terms=9_000
        )
