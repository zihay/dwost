from typing import Any

import numpy as np
import trimesh
from matplotlib import patches


def find_paths(indices: list[tuple[int, int]]) -> list[list[int]]:
    """Find connected paths in a directed graph.

    Args:
        indices: List of directed edges [(src, dst), ...]

    Returns:
        List of vertex paths
    """
    # Build graph structure
    vertices = set()
    forward_edges = {}
    backward_edges = {}

    for src, dst in indices:
        vertices.update([src, dst])
        forward_edges[src] = dst
        backward_edges[dst] = src

    visited = set()
    paths = []

    def trace_path(start: int) -> list[int] | None:
        """Trace path from start vertex in both directions."""
        if start in visited:
            return None

        # Forward traversal
        forward_path = [start]
        visited.add(start)
        current = start

        while current in forward_edges:
            next_v = forward_edges[current]
            if next_v in visited:
                forward_path.append(next_v)  # Close cycle
                break
            visited.add(next_v)
            forward_path.append(next_v)
            current = next_v

        # Backward traversal
        backward_path = []
        while current in backward_edges:
            prev_v = backward_edges[current]
            if prev_v in visited:
                break
            visited.add(prev_v)
            backward_path.append(prev_v)
            current = prev_v

        # Combine paths
        return backward_path[::-1] + forward_path

    # Collect all paths
    for vertex in vertices:
        if path := trace_path(vertex):  # Using assignment expression (Python 3.8+)
            paths.append(path)

    return paths


def slice_mesh(
    vertices: np.ndarray, indices: np.ndarray, z: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Slice a 3D mesh with a horizontal plane at z-coordinate.

    Args:
        vertices: Mesh vertices (n, 3)
        indices: Mesh face indices
        z: Z-coordinate of the slicing plane

    Returns:
        (boundary_vertices, boundary_indices) or (None, None) if no intersection
    """
    # Create mesh and define slicing plane
    mesh = trimesh.Trimesh(vertices=vertices, faces=indices)
    plane_normal = np.array([0, 0, 1])
    plane_origin = np.array([0, 0, z])

    # Perform the slice
    slice_result = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if not slice_result:
        return None, None

    # Extract boundary
    boundary_vertices = slice_result.vertices
    boundary_indices = extract_boundary_indices(slice_result.entities)

    return boundary_vertices, boundary_indices


def extract_boundary_indices(boundary_entities: list[Any]) -> np.ndarray:
    """Extract boundary indices from mesh section entities.

    Args:
        boundary_entities: Boundary entities from trimesh section

    Returns:
        Array of point index pairs
    """
    indices = []
    for entity in boundary_entities:
        points = entity.points
        # Create pairs of consecutive points
        indices.extend([points[i : i + 2] for i in range(len(points) - 1)])

    return np.array(indices)


def draw_square(ax, center, half_size, color=(0, 0, 0), thickness=1, linestyle="solid"):
    """
    Draws a square on the Matplotlib axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to draw on.
        center (tuple): (x, y) coordinates of the square's center in pixels.
        size (float): Length of one side of the square.
        color (tuple): Color of the square in RGB format (0-1).
        thickness (float): Thickness of the square's edges.
    """
    xy = (center[0] - half_size, center[1] - half_size)
    square = patches.Rectangle(
        xy,
        half_size * 2,
        half_size * 2,
        linewidth=thickness,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(square)


def draw_rectangle(
    ax,
    center,
    v_half_size,
    h_half_size,
    color=(0, 0, 0),
    thickness=1,
    linestyle="solid",
):
    xy = (center[0] - h_half_size, center[1] - v_half_size)
    rectangle = patches.Rectangle(
        xy,
        h_half_size * 2,
        v_half_size * 2,
        linewidth=thickness,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rectangle)


def draw_dot(
    ax,
    center,
    radius,
    stroke_color=(1, 0, 0),
    fill_color=(1, 1, 1),
    thickness=1,
    zorder=0,
):
    """
    Draws a dot with a stroke and fill on the Matplotlib axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to draw on.
        center (tuple): (x, y) coordinates of the dot's center in pixels.
        radius (float): Radius of the dot.
        stroke_color (tuple): Color of the stroke in RGB format (0-1).
        fill_color (tuple): Color of the fill in RGB format (0-1).
    """
    # Stroke (outer circle)
    stroke_circle = patches.Circle(
        center,
        radius,
        edgecolor=stroke_color,
        facecolor=fill_color,
        linewidth=thickness,
        fill=True,
        zorder=zorder,
    )
    ax.add_patch(stroke_circle)

    # # Fill (inner circle)
    # fill_circle = patches.Circle(
    #     center, radius - 2, edgecolor='none', facecolor=fill_color
    # )
    # ax.add_patch(fill_circle)


def draw_line(ax, start, end, color="white", thickness=1, linestyle="solid", zorder=0):
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=thickness,
        linestyle=linestyle,
        zorder=zorder,
    )
