"""Verify an exact finite-U orbital rotation against the site-basis model."""

import numpy as np

from hubbard_ed import (
    HoppingMatrixHamiltonian,
    HubbardBasis,
    TwoBodyTensorHamiltonian,
    nearest_neighbor_hopping_matrix,
    onsite_interaction_tensor,
    rotate_hubbard_integrals,
    solve_hamiltonian,
)


def random_unitary(L: int, seed: int) -> np.ndarray:
    """Return a deterministic Haar-compatible unitary from a complex QR."""

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(L, L)) + 1j * rng.normal(size=(L, L))
    unitary, _ = np.linalg.qr(raw)
    return unitary


def spectrum(hamiltonian: TwoBodyTensorHamiltonian) -> np.ndarray:
    """Return the complete spectrum; this example deliberately stays tiny."""

    return np.linalg.eigvalsh(hamiltonian.to_sparse().toarray())


def main() -> None:
    L = 4
    U = 4.0
    basis = HubbardBasis(L, n_up=2, n_down=2)
    print(f"basis dimension:                 {basis.dimension}")

    hopping = nearest_neighbor_hopping_matrix(L, t=1.0, boundary="open")
    site_tensor = onsite_interaction_tensor(L, U)
    rotation = random_unitary(L, seed=2026)
    rotated_h, rotated_v = rotate_hubbard_integrals(hopping, U, rotation)

    site_model = TwoBodyTensorHamiltonian(basis, hopping, site_tensor)
    rotated_model = TwoBodyTensorHamiltonian(basis, rotated_h, rotated_v)
    # This tempting construction is not the same model at finite U.
    one_body_only = HoppingMatrixHamiltonian(basis, rotated_h, U=U)

    site_energies = spectrum(site_model)
    rotated_energies = spectrum(rotated_model)
    incorrect_energies = np.linalg.eigvalsh(
        one_body_only.to_sparse().toarray()
    )
    result = solve_hamiltonian(rotated_model, n_eigenvalues=3)

    unitarity_error = np.max(
        np.abs(rotation.conj().T @ rotation - np.eye(L))
    )
    spectrum_error = np.max(np.abs(rotated_energies - site_energies))
    incorrect_error = np.max(np.abs(incorrect_energies - site_energies))
    print(f"unitarity error:                 {unitarity_error:.3e}")
    print(f"site-basis ground energy:        {site_energies[0]:.12f}")
    print(f"rotated-basis ground energy:     {result.energy:.12f}")
    print(f"complete-spectrum error:         {spectrum_error:.3e}")
    print(f"one-body-only spectrum error:    {incorrect_error:.3e}")
    print(f"rotated interaction terms:       {rotated_model.interaction_term_count}")
    print(f"ground-state residual:           {result.residual_norm:.3e}")

    assert spectrum_error < 1e-10
    assert incorrect_error > 1e-3
    assert result.residual_norm < 1e-8


if __name__ == "__main__":
    main()
