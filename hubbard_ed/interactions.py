"""Interaction tensors for spin-conserving up-down two-body terms."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import validate_sector

InteractionTensor: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
INTERACTION_HERMITICITY_TOLERANCE = 1e-12


def _validate_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_interaction_tensor(
    interaction_tensor: ArrayLike, L: int
) -> InteractionTensor:
    """Validate and copy an up-down interaction tensor.

    ``V[a,b,c,d]`` multiplies
    ``c_a,up^dagger c_b,up c_c,down^dagger c_d,down``.  Hermiticity therefore
    requires ``V[a,b,c,d] = conj(V[b,a,d,c])``.  Roundoff-level asymmetry is
    projected away, and the returned tensor is read-only.
    """

    validate_sector(L, 0, 0)
    candidate = np.asarray(interaction_tensor)
    expected_shape = (L, L, L, L)
    if candidate.shape != expected_shape:
        raise ValueError(f"interaction_tensor must have shape {expected_shape}")
    if not np.issubdtype(candidate.dtype, np.number):
        raise TypeError("interaction_tensor must contain numeric values")
    dtype = np.result_type(candidate.dtype, np.float64)
    tensor = np.array(candidate, dtype=dtype, copy=True)
    if not np.all(np.isfinite(tensor)):
        raise ValueError("interaction_tensor must contain only finite values")

    adjoint = tensor.conj().transpose(1, 0, 3, 2)
    if not np.allclose(
        tensor,
        adjoint,
        rtol=INTERACTION_HERMITICITY_TOLERANCE,
        atol=INTERACTION_HERMITICITY_TOLERANCE,
    ):
        maximum_error = float(np.max(np.abs(tensor - adjoint)))
        raise ValueError(
            "interaction_tensor must satisfy V[a,b,c,d] = "
            "conj(V[b,a,d,c]); maximum Hermiticity error is "
            f"{maximum_error:.3e}"
        )

    tensor = 0.5 * (tensor + adjoint)
    tensor.flags.writeable = False
    return tensor


def onsite_interaction_tensor(L: int, U: Real) -> NDArray[np.float64]:
    """Return the local Hubbard tensor with ``V[i,i,i,i] = U``."""

    validate_sector(L, 0, 0)
    interaction = _validate_real("U", U)
    tensor = np.zeros((L, L, L, L), dtype=float)
    sites = np.arange(L)
    tensor[sites, sites, sites, sites] = interaction
    tensor.flags.writeable = False
    return tensor
