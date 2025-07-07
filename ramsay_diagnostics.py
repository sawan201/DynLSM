#!/usr/bin/env python3
"""diagnostics.py – interactive and static diagnostics for DRUMS

Run from the command line, e.g.:

    python diagnostics.py --file sim_run.npz \
                         --burn 1000       \
                         --actor 2         \
                         --time 0          \
                         --stride 10       \
                         --interactive

*If run inside a Jupyter notebook, simply:* ::

    from diagnostics import interactive_latent_trajectory
    fig = interactive_latent_trajectory("sim_run.npz", burn=1000, actor=2, time=0, stride=10)
    fig.show()
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# I/O helper
# -----------------------------------------------------------------------------

def _load_npz(path: str | Path) -> Any:
    """Load an npz file and raise a helpful error if key is missing."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Could not find NPZ file at {path!s}")
    return np.load(path, allow_pickle=False)

# -----------------------------------------------------------------------------
# Interactive 3‑D latent trajectory
# -----------------------------------------------------------------------------

def interactive_latent_trajectory(
    file: str | Path = "sim_run.npz",
    *,
    burn: int = 0,
    actor: int = 0,
    time: int = 0,
    stride: int = 1,
) -> "go.Figure":
    """Return a Plotly Figure showing the 3‑D MCMC path of one latent point.

    Parameters
    ----------
    file : str or Path
        Path to the ``.npz`` file produced by the sampler (must contain
        ``X_chain`` – shape ``(ns, T, n, p)``).
    burn : int, default 0
        Number of initial iterations to discard.
    actor : int, default 0
        Index ``i`` of the actor whose position we follow.
    time : int, default 0
        Time slice ``t`` whose latent position is tracked across draws.
    stride : int, default 1
        Keep every *stride*‑th sample to thin the path for readability.

    Returns
    -------
    plotly.graph_objects.Figure
        Ready‑to‑show Figure. Call ``fig.show()`` or ``pio.write_html``.
    """
    data = _load_npz(file)

    try:
        X_chain = data["X_chain"]  # (ns, T, n, p)
    except KeyError as exc:
        raise KeyError(".npz does not contain 'X_chain'.") from exc

    ns, T, n, p = X_chain.shape
    if actor >= n or time >= T:
        raise IndexError(f"actor index {actor} or time {time} out of range (n={n}, T={T}).")

    # slice & thin
    xyz = X_chain[burn::stride, time, actor, :]  # (n_keep, p)
    if xyz.shape[1] != 2:
        raise ValueError("Latent space must have p=2 dimensions for 3‑D trajectory plot.")

    iters = np.arange(burn, burn + stride * xyz.shape[0], stride)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=iters,
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(width=2),
            )
        ]
    )

    fig.update_layout(
        title=f"Latent trajectory – actor {actor}, time {time}",
        scene=dict(
            xaxis_title="latent dim 1",
            yaxis_title="latent dim 2",
            zaxis_title="iteration",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    return fig

# -----------------------------------------------------------------------------
# Static matplotlib helper (optional)
# -----------------------------------------------------------------------------

def plot_latent_trajectory_matplotlib(
    file: str | Path = "sim_run.npz",
    *,
    burn: int = 0,
    actor: int = 0,
    time: int = 0,
    stride: int = 1,
    save: str | None = None,
    show: bool = True,
) -> None:
    """Matplotlib fallback – 3‑D static plot."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3‑D proj.)

    data = _load_npz(file)
    X_chain = data["X_chain"]
    xyz = X_chain[burn::stride, time, actor, :]
    iters = np.arange(burn, burn + stride * xyz.shape[0], stride)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], iters, lw=0.8)
    ax.set_xlabel("latent dim 1")
    ax.set_ylabel("latent dim 2")
    ax.set_zlabel("iteration")
    ax.set_title(f"Latent trajectory – actor {actor}, time {time}")

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive diagnostics for DRUMS sampler")
    ap.add_argument("--file", type=str, default="sim_run.npz", help="path to .npz chain file")
    ap.add_argument("--burn", type=int, default=0, help="burn‑in sweeps to discard")
    ap.add_argument("--actor", type=int, default=0, help="actor index i")
    ap.add_argument("--time", type=int, default=0, help="time slice t")
    ap.add_argument("--stride", type=int, default=1, help="thin every k‑th sample")
    ap.add_argument("--interactive", action="store_true", help="launch Plotly figure in browser")
    ap.add_argument("--save", type=str, default=None, metavar="OUT",
                    help="save plot to PNG / HTML instead of showing")
    return ap.parse_args()


def _main() -> None:
    args = _parse_args()

    if args.interactive:
        fig = interactive_latent_trajectory(
            args.file, burn=args.burn, actor=args.actor, time=args.time, stride=args.stride
        )
        if args.save:
            if args.save.endswith(".html"):
                pio.write_html(fig, args.save, include_plotlyjs="cdn")
            else:  # PNG, SVG, … (needs kaleido)
                pio.write_image(fig, args.save)
        else:
            fig.show()
    else:
        plot_latent_trajectory_matplotlib(
            args.file,
            burn=args.burn,
            actor=args.actor,
            time=args.time,
            stride=args.stride,
            save=args.save,
            show=args.save is None,
        )


if __name__ == "__main__":
    _main()
