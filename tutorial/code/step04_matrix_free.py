"""Step 4: compare CSR multiplication with the matrix-free action."""

import numpy as np

from hubbard_ed import HubbardBasis, HubbardHamiltonian


def main() -> None:
    basis = HubbardBasis(L=5, n_up=2, n_down=2)
    hamiltonian = HubbardHamiltonian(basis, t=0.8, U=3.0, boundary="open")
    sparse = hamiltonian.to_sparse()

    rng = np.random.default_rng(1234)
    vector = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    explicit = sparse @ vector
    matrix_free = hamiltonian.matvec(vector)
    error = np.linalg.norm(explicit - matrix_free)

    print(f"dimension: {len(basis)}")
    print(f"||H_CSR x - H_free x|| = {error:.3e}")
    assert np.allclose(explicit, matrix_free, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    main()
