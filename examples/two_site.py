"""Solve the half-filled two-site problem and compare with the exact energy."""

from hubbard_ed.analytic import two_site_ground_energy
from hubbard_ed.lanczos import solve_ground_state
from hubbard_ed.observables import double_occupancy_per_site


def main() -> None:
    t = 1.0
    U = 4.0
    result = solve_ground_state(
        L=2,
        n_up=1,
        n_down=1,
        t=t,
        U=U,
        boundary="open",
        n_eigenvalues=2,
    )
    exact = two_site_ground_energy(U, t)
    print(f"basis dimension:          {len(result.basis)}")
    print(f"ED ground-state energy:  {result.energy:.12f}")
    print(f"analytic energy:         {exact:.12f}")
    print(f"absolute error:          {abs(result.energy - exact):.3e}")
    print(
        "double occupancy/site: "
        f"{double_occupancy_per_site(result.state, result.basis):.12f}"
    )
    print(f"residual norm:           {result.residual_norm:.3e}")


if __name__ == "__main__":
    main()

