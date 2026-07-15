# Hubbard ED tutorial

This directory contains a chaptered LaTeX tutorial and a sequence of runnable
Python programs.  The chapters follow the implementation in increasing order
of complexity:

1. fixed-sector bit-string bases;
2. fermionic creation, annihilation, and hopping signs;
3. sparse Hamiltonian assembly;
4. matrix-free action and Lanczos solving;
5. observables;
6. analytic and limiting-case validation;
7. arbitrary real or complex hopping matrices;
8. exact finite-interaction orbital rotations and spectral invariance;
9. research extensions and their consistency requirements.

## Run the code

From the repository root, install the project and run every tutorial program:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

for script in tutorial/code/step*.py; do
    python "$script"
done
```

Alternatively, from this directory run `make code` after creating the root
virtual environment.

## Build the document

The document uses standard LaTeX packages and `listings`; it does not require
Pygments or `--shell-escape`.

```bash
cd tutorial
latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -outdir=build main.tex
```

The resulting PDF is `tutorial/build/main.pdf`.  Run `make clean` to remove the
generated build directory.  Generated PDF and auxiliary files are not tracked.

The code listings in the PDF are included directly from `tutorial/code/`, so
the manuscript and executable examples cannot silently drift apart.
