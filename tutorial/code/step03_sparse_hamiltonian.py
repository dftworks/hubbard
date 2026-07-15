"""Step 3: construct and inspect a sparse Hubbard Hamiltonian."""

import numpy as np

from hubbard_ed import HubbardBasis, HubbardHamiltonian


def main() -> None:
    basis = HubbardBasis(L=4, n_up=2, n_down=2)
    hamiltonian = HubbardHamiltonian(
        basis, t=1.0, U=4.0, boundary="periodic"
    )
    matrix = hamiltonian.to_sparse()
    antihermitian = matrix - matrix.getH()
    error = 0.0 if antihermitian.nnz == 0 else np.max(np.abs(antihermitian.data))

    print(f"shape: {matrix.shape}")
    print(f"stored nonzeros: {matrix.nnz}")
    print(f"Hermiticity error: {error:.3e}")
    print(f"first eight diagonal entries: {matrix.diagonal()[:8]}")

    assert matrix.shape == (len(basis), len(basis))
    assert error < 1e-14


if __name__ == "__main__":
    main()
