"""
3D Flow Streamline Visualization Module

This module provides functionality for visualizing 3D flow fields using
streamlines and gradient field visualization.
"""

import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from diff_wost.core.fwd import Array3, dr
from diff_wost.solvers.wost_grad_3d import WoStGrad3D
from examples.example_3d.scene import TestScene3D, make_test_scene
from examples.example_3d.trace import StreamlineTracer

CUR_DIR = Path(__file__).parent


def read_points(file_path: str) -> np.ndarray:
    """
    Read 3D points from an OBJ file.

    Args:
        file_path: Path to the OBJ file containing vertex data

    Returns:
        numpy array of shape (N, 3) containing 3D points
    """
    points = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
    return np.array(points)


def save_streamlines(streamlines: list, outdir: Path, prefix: str = "streamline"):
    """Save streamlines data to files."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy array
    streamlines_array = np.array(streamlines)

    # Save raw data as numpy file
    np.save(outdir / f"{prefix}.npy", streamlines_array)

    # Also save as OBJ for visualization
    with open(outdir / f"{prefix}.obj", "w") as f:
        f.write("# Streamlines\n")
        vertex_idx = 1
        for i in range(streamlines_array.shape[1]):  # For each point
            # Write vertices for this streamline
            for j in range(streamlines_array.shape[0]):  # For each timestep
                x, y, z = streamlines_array[j, i, :]
                f.write(f"v {x} {y} {z}\n")

            # Write line segments for this streamline
            nsteps = streamlines_array.shape[0]
            if nsteps > 1:
                indices = " ".join(str(vertex_idx + k) for k in range(nsteps))
                f.write(f"l {indices}\n")
            vertex_idx += nsteps


def trace_streamline(
    outdir: Path = Path("out"),
    scene: TestScene3D = None,
    niter: int = 100,
    step_size: float = 0.1,
    nwalks: int = 100,
    nsubwalks: int = 1,
    nsteps: int = 100,
    min_R: float = 0.5,
    samples_file: str = "samples.obj",
) -> None:
    """
    Trace streamlines in 3D.

    Args:
        outdir: Output directory for results
        scene: The 3D scene to trace in
        niter: Number of iterations
        step_size: Size of each step along streamlines
        nwalks: Number of walks for the tracer
        nsubwalks: Number of subwalks for the solver
        nsteps: Number of steps for the solver
        min_R: Minimum radius for the solver
        samples_file: File containing initial sample points (OBJ format)
    """
    start_time = time.time()
    print(f"Starting 3D streamline tracing with {niter} iterations...")

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Create scene if not provided
    if scene is None:
        scene = make_test_scene()

    # Load initial points
    pts = read_points(samples_file)
    pts = Array3(pts.T)

    # Filter to keep only inside points
    inside = scene.inside(pts)
    pts = dr.gather(Array3, pts, dr.compress(inside))
    print(f"Loaded {dr.width(pts)} points inside the scene")

    # Create solver
    solver = WoStGrad3D(
        nwalks=1,
        nsubwalks=nsubwalks,
        nsteps=nsteps,
        min_R=min_R,
    )

    # Create tracer
    tracer = StreamlineTracer(
        scene=scene, solver=solver, step_size=step_size, nwalks=nwalks
    )

    # Initialize streamlines with starting points
    streamlines = [pts.numpy().T]

    # Main iteration loop
    for i in tqdm(range(niter), desc="Tracing streamlines"):
        # Advance streamlines
        p = tracer.next_position(pts, seed=i)
        dr.eval(p)
        streamlines.append(p.numpy().T)
        pts = p

        # Save at intervals
        if (i + 1) % 10 == 0:
            save_streamlines(streamlines, outdir)

    # Save final result
    save_streamlines(streamlines, outdir)

    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    trace_streamline(
        outdir=Path("out/example_3d"),
        scene=make_test_scene(),
        nwalks=100,
        nsubwalks=30,
        nsteps=200,
        niter=100,
        step_size=0.2,
        min_R=5e-1,
        samples_file=CUR_DIR / "samples.obj",
    )
