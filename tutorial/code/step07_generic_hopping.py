"""Step 7: solve a model defined by a complex Hermitian one-body matrix."""

import numpy as np

from hubbard_ed import HubbardBasis, HoppingMatrixHamiltonian, solve_hamiltonian


def main() -> None:
    L = 4
    flux = 0.6
    hopping = np.zeros((L, L), dtype=complex)
    phase = np.exp(1j * flux / L)

    for source in range(L):
        destination = (source + 1) % L
        hopping[destination, source] = -phase
        hopping[source, destination] = -phase.conjugate()

    # Add a real next-nearest-neighbor edge and onsite potentials.
    hopping[0, 2] = hopping[2, 0] = -0.2
    hopping[np.diag_indices(L)] = [-0.1, 0.0, 0.0, 0.1]

    basis = HubbardBasis(L, n_up=2, n_down=2)
    hamiltonian = HoppingMatrixHamiltonian(basis, hopping, U=2.5)
    result = solve_hamiltonian(hamiltonian, n_eigenvalues=2)

    hermiticity_error = np.max(np.abs(hopping - hopping.conj().T))
    print(f"Hermiticity error: {hermiticity_error:.3e}")
    print(f"lowest energies:   {result.eigenvalues}")
    print(f"residual norms:    {result.residual_norms}")
    print(f"complex state:     {np.iscomplexobj(result.state)}")

    assert hermiticity_error == 0.0
    assert np.all(result.residual_norms < 1e-8)


if __name__ == "__main__":
    main()
