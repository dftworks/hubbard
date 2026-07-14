"""Scan interaction strength for an open, half-filled Hubbard chain."""

from __future__ import annotations

import argparse
from pathlib import Path

from hubbard_ed.basis import estimate_dimension
from hubbard_ed.lanczos import solve_ground_state
from hubbard_ed.observables import double_occupancy_per_site, spin_z_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=6, help="even chain length (default: 6)")
    parser.add_argument("--t", type=float, default=1.0, help="hopping amplitude")
    parser.add_argument(
        "--boundary", choices=("open", "periodic"), default="open"
    )
    parser.add_argument(
        "--plot", action="store_true", help="display the three scan plots"
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="save PNG plots in this directory instead of displaying them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.L < 2 or args.L % 2:
        raise SystemExit("--L must be a positive even integer of at least 2")
    if args.t == 0.0:
        raise SystemExit("--t must be nonzero because this example scans U/t")

    n_up = n_down = args.L // 2
    dimension = estimate_dimension(args.L, n_up, n_down)
    print(
        f"L={args.L}, N_up=N_down={n_up}, boundary={args.boundary}, "
        f"dimension={dimension:,}"
    )
    print(" U/t       E0              E0/L            D/L             residual")
    print("-----  --------------  --------------  --------------  ------------")

    ratios = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    energies: list[float] = []
    doubles: list[float] = []
    nearest_spin: list[float] = []
    end_spin: list[float] = []
    for ratio in ratios:
        result = solve_ground_state(
            L=args.L,
            n_up=n_up,
            n_down=n_down,
            t=args.t,
            U=ratio * args.t,
            boundary=args.boundary,
        )
        double = double_occupancy_per_site(result.state, result.basis)
        nearest = spin_z_correlation(result.state, result.basis, 0, 1)
        end_to_end = spin_z_correlation(
            result.state, result.basis, 0, args.L - 1
        )
        energies.append(result.energy)
        doubles.append(double)
        nearest_spin.append(nearest)
        end_spin.append(end_to_end)
        print(
            f"{ratio:5.1f}  {result.energy:14.9f}  "
            f"{result.energy / args.L:14.9f}  {double:14.9f}  "
            f"{result.residual_norm:12.3e}"
        )

    if args.plot or args.save_dir is not None:
        plot_scan(
            ratios,
            energies,
            doubles,
            nearest_spin,
            end_spin,
            save_dir=args.save_dir,
        )


def plot_scan(
    ratios: list[float],
    energies: list[float],
    doubles: list[float],
    nearest_spin: list[float],
    end_spin: list[float],
    *,
    save_dir: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "plotting requires the optional dependency: pip install -e '.[plots]'"
        ) from exc

    figures = []
    for y, ylabel, title, filename in (
        (energies, r"$E_0$", "Ground-state energy", "ground_state_energy.png"),
        (doubles, r"$D/L$", "Double occupancy", "double_occupancy.png"),
    ):
        figure, axis = plt.subplots()
        axis.plot(ratios, y, "o-")
        axis.set(xlabel=r"$U/t$", ylabel=ylabel, title=title)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figures.append((figure, filename))

    figure, axis = plt.subplots()
    axis.plot(ratios, nearest_spin, "o-", label=r"$\langle S_0^z S_1^z\rangle$")
    axis.plot(ratios, end_spin, "s-", label=r"$\langle S_0^z S_{L-1}^z\rangle$")
    axis.set(
        xlabel=r"$U/t$", ylabel="spin correlation", title="Selected spin correlations"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figures.append((figure, "spin_correlations.png"))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for figure, filename in figures:
            figure.savefig(save_dir / filename, dpi=160)
            plt.close(figure)
        print(f"saved plots to {save_dir}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

