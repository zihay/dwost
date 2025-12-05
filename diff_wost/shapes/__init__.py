"""Shape primitives and spatial data structures."""

from .bvh import BVH, BoundingBox, PointBVH
from .bvh3d import BVH3D, BoundingBox3D, Point3DBVH, Segment3DBVH
from .line_segment import LineSegment, LineSegment3D
from .mesh import Mesh, Scene3D
from .plane import Plane
from .point import Point2D, Point3D
from .polyline import Polyline, Scene2D
from .primitive import BoundaryType
from .silhouette_edge import SilhouetteEdge
from .silhouette_vertex import SilhouetteVertex
from .snch import SNCH
from .snch3d import SNCH3D
from .triangle import Triangle

__all__ = [
    # Primitives
    "BoundaryType",
    "Point2D",
    "Point3D",
    "LineSegment",
    "LineSegment3D",
    "Triangle",
    "Plane",
    # Scenes
    "Polyline",
    "Scene2D",
    "Mesh",
    "Scene3D",
    # BVH
    "BVH",
    "BoundingBox",
    "PointBVH",
    "BVH3D",
    "BoundingBox3D",
    "Segment3DBVH",
    "Point3DBVH",
    # SNCH
    "SNCH",
    "SNCH3D",
    # Silhouette
    "SilhouetteEdge",
    "SilhouetteVertex",
]
