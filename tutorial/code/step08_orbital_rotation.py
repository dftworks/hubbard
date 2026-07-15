"""Step 8: rotate both one- and two-body terms at finite interaction."""

import numpy as np

from hubbard_ed import (
    HubbardBasis,
    TwoBodyTensorHamiltonian,
    nearest_neighbor_hopping_matrix,
    onsite_interaction_tensor,
    rotate_hubbard_integrals,
)


def main() -> None:
    L = 4
    U = 3.0
    basis = HubbardBasis(L, n_up=2, n_down=2)
    hopping = nearest_neighbor_hopping_matrix(L, boundary="open")

    rng = np.random.default_rng(17)
    raw = rng.normal(size=(L, L)) + 1j * rng.normal(size=(L, L))
    rotation, _ = np.linalg.qr(raw)
    rotated_h, rotated_v = rotate_hubbard_integrals(hopping, U, rotation)

    original = TwoBodyTensorHamiltonian(
        basis, hopping, onsite_interaction_tensor(L, U)
    )
    rotated = TwoBodyTensorHamiltonian(basis, rotated_h, rotated_v)
    original_energies = np.linalg.eigvalsh(original.to_sparse().toarray())
    rotated_energies = np.linalg.eigvalsh(rotated.to_sparse().toarray())
    error = np.max(np.abs(original_energies - rotated_energies))

    print(f"basis dimension:         {basis.dimension}")
    print(f"site interaction terms:  {original.interaction_term_count}")
    print(f"rotated terms:           {rotated.interaction_term_count}")
    print(f"maximum spectrum error:  {error:.3e}")
    assert error < 1e-10


if __name__ == "__main__":
    main()
