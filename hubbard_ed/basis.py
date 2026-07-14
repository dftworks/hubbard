"""Fixed-particle-number Fock bases represented by pairs of bit strings."""

from __future__ import annotations

from itertools import combinations, product
from math import comb
from typing import Iterator, TypeAlias

BasisState: TypeAlias = tuple[int, int]

# Constructing a basis much larger than this is already impractical in pure
# Python.  Callers may choose a smaller limit, or explicitly raise it after
# estimating the memory required by their calculation.
DEFAULT_MAX_DIMENSION = 2_000_000


def _validate_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def validate_sector(L: int, n_up: int, n_down: int) -> None:
    """Validate lattice size and particle numbers for a Hubbard sector."""

    for name, value in (("L", L), ("n_up", n_up), ("n_down", n_down)):
        _validate_integer(name, value)
    if L < 1:
        raise ValueError("L must be at least 1")
    if not 0 <= n_up <= L:
        raise ValueError("n_up must satisfy 0 <= n_up <= L")
    if not 0 <= n_down <= L:
        raise ValueError("n_down must satisfy 0 <= n_down <= L")


def estimate_dimension(L: int, n_up: int, n_down: int) -> int:
    """Return ``binom(L, n_up) * binom(L, n_down)`` after validation."""

    validate_sector(L, n_up, n_down)
    return comb(L, n_up) * comb(L, n_down)


def fixed_weight_bits(L: int, particles: int) -> tuple[int, ...]:
    """Return all length-``L`` bit strings with the requested popcount."""

    return tuple(
        sum(1 << site for site in occupied_sites)
        for occupied_sites in combinations(range(L), particles)
    )


class HubbardBasis:
    """Basis for one fixed ``(N_up, N_down)`` particle-number sector.

    States are ordered deterministically: the up-spin bit string is the outer
    index and the down-spin bit string is the inner index.  No physical result
    depends on this enumeration order.
    """

    def __init__(
        self,
        L: int,
        n_up: int,
        n_down: int,
        *,
        max_dimension: int | None = DEFAULT_MAX_DIMENSION,
    ) -> None:
        validate_sector(L, n_up, n_down)
        dimension = estimate_dimension(L, n_up, n_down)
        if max_dimension is not None:
            _validate_integer("max_dimension", max_dimension)
            if max_dimension < 1:
                raise ValueError("max_dimension must be positive or None")
            if dimension > max_dimension:
                raise ValueError(
                    f"sector dimension {dimension:,} exceeds the configured "
                    f"limit {max_dimension:,}; estimate memory first and raise "
                    "max_dimension explicitly only if the calculation is safe"
                )

        self.L = L
        self.n_up = n_up
        self.n_down = n_down
        self.dimension = dimension

        up_states = fixed_weight_bits(L, n_up)
        down_states = fixed_weight_bits(L, n_down)
        self._states: tuple[BasisState, ...] = tuple(product(up_states, down_states))
        self._state_to_index = {state: index for index, state in enumerate(self._states)}

    def __len__(self) -> int:
        return self.dimension

    def __iter__(self) -> Iterator[BasisState]:
        return iter(self._states)

    def __getitem__(self, index: int) -> BasisState:
        return self.index_state(index)

    def index_state(self, index: int) -> BasisState:
        """Return the state at ``index``."""

        return self._states[index]

    def state_index(self, state: BasisState) -> int:
        """Return the basis index of ``state``, raising ``KeyError`` if absent."""

        try:
            return self._state_to_index[state]
        except KeyError as exc:
            raise KeyError(f"state {state!r} is not in this particle-number sector") from exc

    def __repr__(self) -> str:
        return (
            f"HubbardBasis(L={self.L}, n_up={self.n_up}, "
            f"n_down={self.n_down}, dimension={self.dimension})"
        )

