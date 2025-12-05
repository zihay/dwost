"""
2D Flow Streamline Visualization Module

This module provides functionality for visualizing 2D flow fields using
streamlines and gradient field visualization.
"""

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from diff_wost.core.fwd import Array2, dr
from diff_wost.solvers.wost_grad import WoStGrad
from diff_wost.utils.plot import find_paths
from examples.example_2d.scene import (
    FlowScene,
    make_rounded_square_scene,
    make_square_scene,
)
from examples.example_2d.trace import StreamlineTracer

CUR_DIR = Path(__file__).parent


def save_streamlines(streamlines, scene, outdir: Path, prefix: str = "streamline"):
    """Save streamlines visualization to files."""
    # Ensure directory exists
    outdir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy array if it's not already
    streamlines_array = np.array(streamlines)

    # Create figure
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(streamlines_array[:, 0], streamlines_array[:, 1], "k-", linewidth=0.4)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis("off")
    plt.gca().set_aspect("equal")  # Equal aspect ratio

    # Plot scene boundaries
    vertices = scene.vertices.numpy().T
    indices = scene.indices.numpy().T
    paths = find_paths(indices)
    for path in paths:
        plt.plot(vertices[path, 0], vertices[path, 1], "k--", linewidth=1.0)

    # Save as PNG and PDF
    plt.savefig(outdir / f"{prefix}.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.savefig(outdir / f"{prefix}.pdf", bbox_inches="tight", pad_inches=0, dpi=300)

    # Save raw data
    np.save(outdir / f"{prefix}.npy", streamlines_array)

    # Close figure to free memory
    plt.close()


def trace_streamline(
    outdir: Path = Path("out"),
    scene: FlowScene = make_square_scene(),
    niter: int = 100,
    step_size: float = 0.02,
    nwalks: int = 1,
    nsubwalks: int = 100,
    nsteps: int = 200,
    samples_file: str = "samples.txt",
) -> None:
    """
    Trace streamlines.

    Args:
        outdir: Output directory for results
        niter: Number of iterations
        step_size: Size of each step along streamlines
        nwalks: Number of walks for the solver
        nsubwalks: Number of subwalks for the solver
        nsteps: Number of steps for the solver
        save_interval: How often to save intermediate results (every N iterations)
        samples_file: File containing initial sample points
    """
    start_time = time.time()
    print(f"Starting streamline test with {niter} iterations...")

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Initialize scene and points
    pts = np.loadtxt(samples_file)
    pts = Array2(pts.T)
    inside = scene.inside(pts)
    pts = dr.gather(Array2, pts, dr.compress(inside))

    solver = WoStGrad(
        nwalks=1,
        nsubwalks=nsubwalks,
        nsteps=nsteps,
        ignore_boundary=True,  # zero neumann boundary
        ignore_silhouette=True,  # zero neumann silhouette
    )

    # Create tracer
    tracer = StreamlineTracer(
        scene=scene, solver=solver, step_size=step_size, nwalks=nwalks
    )

    # Initialize streamlines with starting points
    streamlines = [pts.numpy()]

    # Main iteration loop with progress tracking
    for i in tqdm(range(niter), desc="Tracing streamlines"):
        # Advance streamlines
        p = tracer.next_position(pts, i)
        dr.eval(p)
        streamlines.append(p.numpy())
        pts = p
        # Save at specified intervals
        save_streamlines(streamlines, scene, outdir)

    # Save final result at the top level
    save_streamlines(streamlines, scene, outdir)

    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    trace_streamline(
        outdir=Path("out/example_2d"),
        scene=make_rounded_square_scene(),
        nwalks=100,
        nsubwalks=100,
        nsteps=500,
        niter=100,
        step_size=0.01,
        samples_file=CUR_DIR / "samples.txt",
    )
