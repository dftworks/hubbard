"""Step 2: inspect signs from the global spin-orbital ordering."""

from hubbard_ed.operators import (
    apply_annihilation,
    apply_creation,
    apply_spin_hop,
)


def main() -> None:
    # Ordering for L=3: (0 up, 1 up, 2 up, 0 down, 1 down, 2 down).
    L = 3
    state = (0b001, 0b000)
    first_down = L
    created = apply_creation(state, L, first_down)
    print(f"create 0-down in {state}: {created}")
    assert created == ((0b001, 0b001), -1)

    restored = apply_annihilation(created[0], L, first_down)
    print(f"annihilate it again:      {restored}")
    assert restored == (state, -1)
    assert created[1] * restored[1] == 1

    # A periodic hop crosses occupied sites and need not have sign +1.
    periodic_state = (0b1010, 0)
    periodic_hop = apply_spin_hop(periodic_state, 4, 0, 3, "up")
    print(f"periodic 3 -> 0 hop:      {periodic_hop}")
    assert periodic_hop == ((0b0011, 0), -1)


if __name__ == "__main__":
    main()
