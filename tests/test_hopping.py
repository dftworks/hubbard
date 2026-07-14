from collections.abc import Callable

import pytest

from hubbard_ed.basis import BasisState
from hubbard_ed.operators import (
    OperatorResult,
    apply_annihilation,
    apply_creation,
    apply_spin_hop,
)


def _compose(
    state: BasisState,
    first: Callable[[BasisState], OperatorResult | None],
    second: Callable[[BasisState], OperatorResult | None],
) -> OperatorResult | None:
    first_result = first(state)
    if first_result is None:
        return None
    intermediate, sign_1 = first_result
    second_result = second(intermediate)
    if second_result is None:
        return None
    final, sign_2 = second_result
    return final, sign_1 * sign_2


def _as_amplitudes(*terms: OperatorResult | None) -> dict[BasisState, int]:
    result: dict[BasisState, int] = {}
    for term in terms:
        if term is not None:
            state, amplitude = term
            result[state] = result.get(state, 0) + amplitude
    return {state: value for state, value in result.items() if value}


@pytest.mark.parametrize("L", [1, 2, 3])
def test_creation_annihilation_anticommutator(L: int) -> None:
    """Verify {c_p, c_q^dagger} = delta_pq on the complete Fock space."""

    for up_bits in range(1 << L):
        for down_bits in range(1 << L):
            state = (up_bits, down_bits)
            for p in range(2 * L):
                for q in range(2 * L):
                    c_p = lambda s, p=p: apply_annihilation(s, L, p)
                    cd_q = lambda s, q=q: apply_creation(s, L, q)
                    anticommutator = _as_amplitudes(
                        _compose(state, cd_q, c_p),
                        _compose(state, c_p, cd_q),
                    )
                    expected = {state: 1} if p == q else {}
                    assert anticommutator == expected


def test_annihilation_anticommutator() -> None:
    """Verify {c_p, c_q} = 0, including operators of different spin."""

    L = 3
    state = (0b111, 0b111)
    for p in range(2 * L):
        for q in range(2 * L):
            c_p = lambda s, p=p: apply_annihilation(s, L, p)
            c_q = lambda s, q=q: apply_annihilation(s, L, q)
            assert _as_amplitudes(
                _compose(state, c_q, c_p),
                _compose(state, c_p, c_q),
            ) == {}


def test_down_spin_sign_includes_all_up_orbitals() -> None:
    # Creating 0-down crosses both occupied up orbitals in the global ordering.
    assert apply_creation((0b101, 0), 3, 3) == ((0b101, 0b001), 1)
    assert apply_creation((0b001, 0), 3, 3) == ((0b001, 0b001), -1)


def test_nearest_neighbor_and_periodic_hopping_signs() -> None:
    # An adjacent hop has no occupied orbital strictly between source and target.
    assert apply_spin_hop((0b0010, 0), 4, 2, 1, "up") == ((0b0100, 0), 1)

    # The periodic 3 -> 0 hop crosses occupied site 1, hence has sign -1.
    state = (0b1010, 0b0011)
    assert apply_spin_hop(state, 4, 0, 3, "up") == ((0b0011, 0b0011), -1)

    # For down spin the up block is crossed twice and cancels; the sign is set
    # by occupied down orbitals between sites 0 and 3.
    assert apply_spin_hop(state, 4, 3, 0, "down") == ((0b1010, 0b1010), -1)

