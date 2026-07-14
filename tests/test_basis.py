from math import comb

import pytest

from hubbard_ed.basis import HubbardBasis, estimate_dimension


@pytest.mark.parametrize(
    ("L", "n_up", "n_down"),
    [(1, 0, 0), (2, 1, 1), (4, 2, 1), (6, 3, 3), (8, 2, 5)],
)
def test_basis_dimension(L: int, n_up: int, n_down: int) -> None:
    basis = HubbardBasis(L, n_up, n_down)
    assert len(basis) == comb(L, n_up) * comb(L, n_down)
    assert len(basis) == estimate_dimension(L, n_up, n_down)
    assert len(set(basis)) == len(basis)


def test_state_index_round_trip() -> None:
    basis = HubbardBasis(5, 2, 3)
    for index, state in enumerate(basis):
        assert basis.state_index(state) == index
        assert basis.index_state(index) == state
        assert state[0].bit_count() == 2
        assert state[1].bit_count() == 3


@pytest.mark.parametrize(
    "args",
    [(0, 0, 0), (4, -1, 1), (4, 5, 1), (4, 1, 5), (4.0, 1, 1)],
)
def test_invalid_sector_parameters(args: tuple[object, object, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        HubbardBasis(*args)  # type: ignore[arg-type]


def test_dimension_guard() -> None:
    with pytest.raises(ValueError, match="dimension 400"):
        HubbardBasis(6, 3, 3, max_dimension=399)


def test_missing_state_lookup() -> None:
    basis = HubbardBasis(3, 1, 1)
    with pytest.raises(KeyError, match="not in this particle-number sector"):
        basis.state_index((0b011, 0b001))

