#pragma once

#include <bounding_volume.h>
#include <fwd.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace diff_wost {

// 3D silhouette edge primitive: models an edge AB shared by up to two
// triangles (A, B, C) and (D, A, B).
struct SilhouetteEdge {
    // members
    Vector3 a = Vector3::Zero(); // first vertex of the edge
    Vector3 b = Vector3::Zero(); // second vertex of the edge
    Vector3 c = Vector3::Zero(); // third vertex of triangle (A, B, C)
    Vector3 d = Vector3::Zero(); // third vertex of triangle (D, A, B)

    // Vertex indices [c, a, b, d] (mirroring the SoA used in the original code).
    int indices[4] = { -1, -1, -1, -1 };

    // Application-defined edge index
    int pIndex = -1;

    SilhouetteEdge() = default;

    SilhouetteEdge(const Vector3 &a_,
                   const Vector3 &b_,
                   const Vector3 &c_,
                   const Vector3 &d_,
                   int            idxC,
                   int            idxA,
                   int            idxB,
                   int            idxD,
                   int            pIndex_ = -1)
        : a(a_), b(b_), c(c_), d(d_), pIndex(pIndex_) {
        indices[0] = idxC;
        indices[1] = idxA;
        indices[2] = idxB;
        indices[3] = idxD;
    }

    // returns bounding box of endpoints a and b
    BoundingBox<3> boundingBox() const {
        BoundingBox<3> box(a);
        box.expandToInclude(b);
        return box;
    }

    // returns centroid (midpoint of the edge)
    Vector<3> centroid() const {
        return (a + b) * 0.5f;
    }

    // returns edge length (used as "surface area" for BVH heuristics)
    float surfaceArea() const {
        return (b - a).norm();
    }

    // checks whether silhouette has adjacent face
    bool hasFace(int fIndex) const {
        // By convention, indices[0] and indices[3] encode the presence
        // of the two incident faces.
        return fIndex == 0 ? indices[0] != -1 : indices[3] != -1;
    }

    // returns normal of adjacent face
    Vector<3> normal(int fIndex, bool normalize = false) const {
        if (fIndex == 0) {
            Vector3 n = (b - a).cross(c - a);
            return normalize ? n.normalized() : n;
        } else {
            Vector3 n = (d - a).cross(b - a);
            return normalize ? n.normalized() : n;
        }
    }

    // returns averaged normalized silhouette normal
    Vector<3> normal() const {
        Vector3 n = Vector3::Zero();
        if (hasFace(0))
            n += normal(0, false);
        if (hasFace(1))
            n += normal(1, false);
        return n.normalized();
    }

    // get and set index
    int  getIndex() const { return pIndex; }
    void setIndex(int index) { pIndex = index; }
};

// Helper that classifies whether an edge is a silhouette edge given the
// two adjacent normals and view direction (adapted from FCPW).
inline bool isSilhouetteEdge(const Vector3 &pa,
                             const Vector3 &pb,
                             const Vector3 &n0,
                             const Vector3 &n1,
                             const Vector3 &viewDir,
                             float          d,
                             bool           flipNormalOrientation,
                             float          precision) {
    float sign = flipNormalOrientation ? 1.0f : -1.0f;

    // edge is a silhouette if it is concave and the query point lies on the edge
    if (d <= precision) {
        Vector3 edgeDir = (pb - pa).normalized();
        float   signedDihedralAngle =
            std::atan2(edgeDir.dot(n0.cross(n1)), n0.dot(n1));
        return sign * signedDihedralAngle > precision;
    }

    // edge is a silhouette if the query point lies on the halfplane defined
    // by an adjacent triangle and the other triangle is backfacing
    Vector3 viewDirUnit = viewDir / d;
    float   dot0        = viewDirUnit.dot(n0);
    float   dot1        = viewDirUnit.dot(n1);

    bool isZeroDot0 = std::fabs(dot0) <= precision;
    if (isZeroDot0)
        return sign * dot1 > precision;

    bool isZeroDot1 = std::fabs(dot1) <= precision;
    if (isZeroDot1)
        return sign * dot0 > precision;

    // edge is a silhouette if an adjacent triangle is frontfacing w.r.t. the
    // query point and the other triangle is backfacing
    return dot0 * dot1 < 0.0f;
}

// SoA Structure
struct SilhouetteEdgeSoA {
    std::vector<Vector3> a;
    std::vector<Vector3> b;
    std::vector<Vector3> c;
    std::vector<Vector3> d;
    std::vector<int>     indices[4];
    std::vector<int>     pIndex;

    void resize(size_t n) {
        a.resize(n);
        b.resize(n);
        c.resize(n);
        d.resize(n);
        for (int k = 0; k < 4; ++k)
            indices[k].resize(n);
        pIndex.resize(n);
    }

    void clear() {
        a.clear();
        b.clear();
        c.clear();
        d.clear();
        for (int k = 0; k < 4; ++k)
            indices[k].clear();
        pIndex.clear();
    }

    bool hasFace(int i, int fIndex) const {
        return fIndex == 0 ? indices[0][i] != -1 : indices[3][i] != -1;
    }

    Vector3 normal(int i, int fIndex, bool normalize = false) const {
        if (fIndex == 0) {
            Vector3 n = (b[i] - a[i]).cross(c[i] - a[i]);
            return normalize ? n.normalized() : n;
        } else {
            Vector3 n = (d[i] - a[i]).cross(b[i] - a[i]);
            return normalize ? n.normalized() : n;
        }
    }

    Vector3 normal(int i) const {
        Vector3 n = Vector3::Zero();
        if (hasFace(i, 0))
            n += normal(i, 0, false);
        if (hasFace(i, 1))
            n += normal(i, 1, false);
        return n.normalized();
    }
};

} // namespace diff_wost
