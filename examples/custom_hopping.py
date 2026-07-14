"""Solve a Hubbard model with complex flux and next-nearest-neighbor hopping."""

from __future__ import annotations

import numpy as np

from hubbard_ed import HubbardBasis, HoppingMatrixHamiltonian, solve_hamiltonian
from hubbard_ed.observables import double_occupancy_per_site, local_charge


def main() -> None:
    L = 6
    t = 1.0
    t_next = 0.2
    U = 4.0
    flux = 0.4

    # h[i,j] multiplies c_i^dagger c_j.  A total Peierls phase `flux` is
    # distributed uniformly around the ring.
    hopping = np.zeros((L, L), dtype=complex)
    phase = np.exp(1j * flux / L)
    for source in range(L):
        destination = (source + 1) % L
        hopping[destination, source] += -t * phase
        hopping[source, destination] += -t * phase.conjugate()

    # Add the six unique next-nearest-neighbor edges on the L=6 ring.
    for site_a, site_b in ((0, 2), (1, 3), (2, 4), (3, 5), (4, 0), (5, 1)):
        hopping[site_a, site_b] += -t_next
        hopping[site_b, site_a] += -t_next

    # Diagonal entries are spin-independent onsite potentials.
    hopping[np.diag_indices(L)] = np.linspace(-0.1, 0.1, L)

    basis = HubbardBasis(L, n_up=3, n_down=3)
    hamiltonian = HoppingMatrixHamiltonian(basis, hopping, U=U)
    result = solve_hamiltonian(hamiltonian)

    print(f"basis dimension:          {len(basis):,}")
    error = np.max(np.abs(hopping - hopping.conj().T))
    print(f"Hermiticity error:        {error:.3e}")
    print(f"ground-state energy:      {result.energy:.12f}")
    print(
        "double occupancy/site: "
        f"{double_occupancy_per_site(result.state, result.basis):.12f}"
    )
    print(f"total charge:             {np.sum(local_charge(result.state, basis)):.12f}")
    print(f"residual norm:            {result.residual_norm:.3e}")


if __name__ == "__main__":
    main()
