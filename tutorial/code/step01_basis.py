"""Step 1: enumerate one fixed particle-number sector."""

from math import comb

from hubbard_ed import HubbardBasis


def bits(value: int, L: int) -> str:
    """Display sites from L-1 down to 0, as in an ordinary binary literal."""

    return format(value, f"0{L}b")


def main() -> None:
    L, n_up, n_down = 4, 2, 1
    basis = HubbardBasis(L, n_up, n_down)
    expected = comb(L, n_up) * comb(L, n_down)

    print(f"sector: L={L}, N_up={n_up}, N_down={n_down}")
    print(f"dimension: {len(basis)} (analytic: {expected})")
    print("first six states:")
    for index, (up_bits, down_bits) in enumerate(basis):
        if index == 6:
            break
        state = (up_bits, down_bits)
        print(
            f"  {index:2d}: up={bits(up_bits, L)} "
            f"down={bits(down_bits, L)} lookup={basis.state_index(state)}"
        )

    assert len(basis) == expected
    assert all(up.bit_count() == n_up for up, _ in basis)
    assert all(down.bit_count() == n_down for _, down in basis)


if __name__ == "__main__":
    main()
