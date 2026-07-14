# `hubbard-ed`

A compact, readable exact-diagonalization reference for the spin-1/2 Hubbard
model on a one-dimensional chain.  The implementation uses only integer bit
operations, NumPy, and SciPy; it deliberately does not depend on a specialized
many-body package.

## Model and sector

The Hamiltonian is

\[
H = -t \sum_{\langle i,j\rangle,\sigma}
    (c^\dagger_{i\sigma}c_{j\sigma} + c^\dagger_{j\sigma}c_{i\sigma})
    + U \sum_i n_{i\uparrow}n_{i\downarrow}.
\]

Calculations stay in a fixed `(N_up, N_down)` sector.  A basis state is the
pair `(up_bits, down_bits)`, where bit `i` is the occupation of site `i`.  The
dimension is

\[
\binom{L}{N_\uparrow}\binom{L}{N_\downarrow}.
\]

`HubbardBasis` provides deterministic iteration and constant-time lookup in
both directions.

## Fermionic convention

There is one global spin-orbital ordering:

```text
(0 up, 1 up, ..., L-1 up, 0 down, 1 down, ..., L-1 down).
```

Creation or annihilation at orbital `p` has sign `(-1)^m`, where `m` is the
number of occupied orbitals before `p` in this ordering.  A hop
`c_p^dagger c_q` applies annihilation first and creation second, retaining both
signs.  Consequently, hopping across the periodic boundary is not assumed to
have sign `+1`; its sign depends on the intervening occupations.

Nearest-neighbor bonds are unique undirected graph edges, and both hopping
directions are applied.  In particular, the two-site periodic chain contains
one `(0, 1)` edge, just like the open chain, rather than a doubled edge.  This
matches the two-site analytic formula used in the tests.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The core dependencies are NumPy and SciPy.  Matplotlib is optional and used
only by the scan plots.

## Basic use

```python
from hubbard_ed import solve_ground_state
from hubbard_ed.observables import double_occupancy_per_site

result = solve_ground_state(
    L=6,
    n_up=3,
    n_down=3,
    t=1.0,
    U=4.0,
    boundary="open",
)

print(result.energy)
print(result.residual_norm)
print(double_occupancy_per_site(result.state, result.basis))
```

`solve_ground_state` uses `scipy.sparse.linalg.eigsh` and a matrix-free
Hamiltonian by default.  Its ARPACK tolerance is `1e-10`; the reported
residual is recomputed as `||H psi - E psi||`.  Set `n_eigenvalues` to return
low-lying excited energies and vectors, or `matrix_free=False` to use an
explicit CSR matrix.

The package also exposes `HubbardHamiltonian.to_sparse()`, `matvec()`, and
`aslinearoperator()` separately, making it straightforward to test new state
representations against the same reference action.

## Arbitrary hopping matrices

`HoppingMatrixHamiltonian` implements

\[
H(h)=\sum_{ij,\sigma}h_{ij}c^\dagger_{i\sigma}c_{j\sigma}
     +U\sum_i n_{i\uparrow}n_{i\downarrow}.
\]

Here `h[i,j]` is the coefficient of `c_i^dagger c_j`.  The matrix must have
shape `(L,L)`, contain finite values, and be Hermitian to numerical tolerance
`1e-12`.  It may be real or complex, its diagonal represents spin-independent
onsite potentials, and the same matrix is used for both spins.  The constructor
copies the matrix and stores it read-only so later caller mutation cannot
change a calculation.

```python
import numpy as np

from hubbard_ed import (
    HubbardBasis,
    HoppingMatrixHamiltonian,
    nearest_neighbor_hopping_matrix,
    solve_hamiltonian,
)

L = 6
basis = HubbardBasis(L, 3, 3)
h = nearest_neighbor_hopping_matrix(L, t=1.0, boundary="open")
h[0, 2] = h[2, 0] = -0.2        # next-nearest-neighbor edge
h[np.diag_indices(L)] = 0.1      # onsite potential

hamiltonian = HoppingMatrixHamiltonian(basis, h, U=4.0)
result = solve_hamiltonian(hamiltonian)
```

Both `to_sparse()` and `matvec()` evaluate every nonzero `h[i,j]` through the
same tested fermionic hopping primitive.  Complex Peierls phases and iterative
complex-Hermitian solving are supported.  See `examples/custom_hopping.py` for
a flux-threaded ring with next-nearest-neighbor hopping.

An important orbital-rotation limitation is deliberate: rotating the
single-particle basis also transforms the onsite Hubbard interaction into a
general four-index two-body tensor.  Passing only a rotated `h` while retaining
onsite `U` therefore describes a different interacting model, except at
`U=0` or when the rotation preserves the local interaction.  A future general
two-body operator is required for exact wavelet, natural-orbital, or learned
orbital representations of the full interacting Hamiltonian.

## Observables and normalization

The observable module implements:

- total, local, and site-averaged double occupancy;
- local charge `<n_i>` and magnetization `<n_i_up-n_i_down>`;
- `<S_i^z S_j^z>` with `S_i^z=(n_i_up-n_i_down)/2`;
- connected charge correlation `<n_i n_j>-<n_i><n_j>`;
- `gamma_sigma[i,j]=<c_i,sigma^dagger c_j,sigma>`;
- momentum distributions on `k=2*pi*m/L`;
- charge and longitudinal-spin structure factors.

Input states are normalized internally.  The momentum convention is

\[
n_\sigma(k)=\frac{1}{L}\sum_{ij}e^{ik(i-j)}
\langle c^\dagger_{i\sigma}c_{j\sigma}\rangle.
\]

The charge structure factor uses the connected charge correlation and a
factor `1/L`.  The spin structure factor uses `<S_i^z S_j^z>/L` by default and
has a `connected=True` option.

## Validation and examples

Run the complete suite with:

```bash
python -m pytest
```

The tests cover basis dimensions, fermionic anticommutators, explicit periodic
hopping signs, Hermiticity, the analytic two-site energy, the full `U=0`
many-body spectrum for both boundaries (including degeneracies), the atomic
limit, large-`U` superexchange, symmetry checks, CSR versus matrix-free action,
observable sum rules, arbitrary real and complex hopping matrices, and
eigenpair residuals.

Run the small examples with:

```bash
python examples/two_site.py
python examples/chain_scan.py --L 6
python examples/chain_scan.py --L 8 --save-dir plots
python examples/custom_hopping.py
```

The scan prints `U/t`, `E0`, `E0/L`, double occupancy per site, and residual
norm for `U/t = 0, 1, 2, 4, 8, 16`.  Plot files are generated only when
`--save-dir` is supplied; generated images are not tracked.

## Practical limits

Exact diagonalization grows combinatorially.  The code prints the dimension in
the scan example and refuses basis dimensions above 2,000,000 or explicit CSR
matrices above 250,000 by default.  These are safety guards, not promises of
good performance.

At half filling, representative dimensions are:

| `L` | `(N_up,N_down)` | dimension |
|---:|---:|---:|
| 6 | (3,3) | 400 |
| 8 | (4,4) | 4,900 |
| 10 | (5,5) | 63,504 |
| 12 | (6,6) | 853,776 |

`L=8` is quick, and matrix-free `L=10` is a reasonable upper target for this
clear pure-Python reference.  `L=12` is generally too slow and memory-heavy
because the basis tuple and reverse-lookup dictionary must still be stored.
Matrix-free Lanczos avoids storing all CSR nonzeros and is therefore the path
to the largest feasible sectors, but it does not remove basis storage or the
cost of repeated Python-level hopping operations.

A dense generic hopping matrix has up to `L^2` nonzero terms per spin, so its
matrix-free action can be substantially slower than the nearest-neighbor chain
even at the same Hilbert-space dimension.  Sparse physical hopping graphs are
therefore preferable when using this reference implementation.

The basis, elementary operators, Hamiltonian graph construction, and solver
are separate modules so future orbital transformations, arbitrary hopping
graphs, natural orbitals, entanglement tools, and variational wavefunctions can
be compared with this reference without changing its core conventions.
