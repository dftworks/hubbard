"""Exact diagonalization tools for the one-dimensional Hubbard model."""

from .basis import BasisState, HubbardBasis, estimate_dimension
from .hamiltonian import (
    HoppingMatrixHamiltonian,
    HubbardHamiltonian,
    TwoBodyTensorHamiltonian,
    build_hamiltonian,
    nearest_neighbor_hopping_matrix,
)
from .interactions import onsite_interaction_tensor
from .lanczos import GroundStateResult, solve_ground_state, solve_hamiltonian
from .observables import (
    charge_correlation,
    double_occupancy_per_site,
    local_charge,
    local_magnetization,
    spin_z_correlation,
    total_double_occupancy,
)

__all__ = [
    "BasisState",
    "HubbardBasis",
    "HoppingMatrixHamiltonian",
    "HubbardHamiltonian",
    "TwoBodyTensorHamiltonian",
    "GroundStateResult",
    "build_hamiltonian",
    "charge_correlation",
    "double_occupancy_per_site",
    "estimate_dimension",
    "local_charge",
    "local_magnetization",
    "nearest_neighbor_hopping_matrix",
    "onsite_interaction_tensor",
    "solve_ground_state",
    "solve_hamiltonian",
    "spin_z_correlation",
    "total_double_occupancy",
]
