"""Elementary fermionic operators with one documented global convention.

The spin orbitals are ordered as

``(0 up, 1 up, ..., L-1 up, 0 down, ..., L-1 down)``.

Acting with an operator on orbital ``p`` therefore contributes the
Jordan-Wigner sign ``(-1)**m``, where ``m`` is the number of occupied orbitals
strictly before ``p`` in this ordering.  This rule is used for every hop,
including the bond crossing a periodic boundary.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from .basis import BasisState

OperatorResult: TypeAlias = tuple[BasisState, int]
Spin = Literal["up", "down"]


def _validate_state(state: BasisState, L: int) -> None:
    if L < 1:
        raise ValueError("L must be at least 1")
    if len(state) != 2:
        raise ValueError("state must be an (up_bits, down_bits) pair")
    mask = (1 << L) - 1
    if any(isinstance(bits, bool) or not isinstance(bits, int) for bits in state):
        raise TypeError("state bit strings must be integers")
    if state[0] < 0 or state[1] < 0 or state[0] & ~mask or state[1] & ~mask:
        raise ValueError("state has occupied bits outside the lattice")


def spin_orbital(L: int, site: int, spin: Spin) -> int:
    """Map a site and spin label to the global spin-orbital index."""

    if isinstance(site, bool) or not isinstance(site, int):
        raise TypeError("site must be an integer")
    if not 0 <= site < L:
        raise ValueError(f"site must satisfy 0 <= site < {L}")
    if spin == "up":
        return site
    if spin == "down":
        return L + site
    raise ValueError("spin must be 'up' or 'down'")


def _combined_bits(state: BasisState, L: int) -> int:
    return state[0] | (state[1] << L)


def occupation(state: BasisState, L: int, orbital: int) -> int:
    """Return 0 or 1 for a global spin orbital."""

    _validate_state(state, L)
    if not 0 <= orbital < 2 * L:
        raise ValueError(f"orbital must satisfy 0 <= orbital < {2 * L}")
    return (_combined_bits(state, L) >> orbital) & 1


def apply_annihilation(
    state: BasisState, L: int, orbital: int
) -> OperatorResult | None:
    """Apply ``c_orbital`` and return ``(new_state, sign)``, or ``None``."""

    if occupation(state, L, orbital) == 0:
        return None
    combined = _combined_bits(state, L)
    sign = -1 if (combined & ((1 << orbital) - 1)).bit_count() % 2 else 1
    combined ^= 1 << orbital
    mask = (1 << L) - 1
    return (combined & mask, combined >> L), sign


def apply_creation(
    state: BasisState, L: int, orbital: int
) -> OperatorResult | None:
    """Apply ``c_orbital^dagger`` and return ``(new_state, sign)``, or ``None``."""

    if occupation(state, L, orbital) == 1:
        return None
    combined = _combined_bits(state, L)
    sign = -1 if (combined & ((1 << orbital) - 1)).bit_count() % 2 else 1
    combined |= 1 << orbital
    mask = (1 << L) - 1
    return (combined & mask, combined >> L), sign


def apply_hop(
    state: BasisState, L: int, destination: int, source: int
) -> OperatorResult | None:
    """Apply ``c_destination^dagger c_source`` using right-to-left action."""

    removed = apply_annihilation(state, L, source)
    if removed is None:
        return None
    intermediate, sign_1 = removed
    added = apply_creation(intermediate, L, destination)
    if added is None:
        return None
    final_state, sign_2 = added
    return final_state, sign_1 * sign_2


def apply_spin_hop(
    state: BasisState,
    L: int,
    destination_site: int,
    source_site: int,
    spin: Spin,
) -> OperatorResult | None:
    """Apply a number-conserving hop between two sites for one spin."""

    return apply_hop(
        state,
        L,
        spin_orbital(L, destination_site, spin),
        spin_orbital(L, source_site, spin),
    )
