#pragma once

#include <bounding_volume.h>
#include <fwd.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace diff_wost {

// 2D silhouette vertex primitive: models a polyline vertex with neighbors a,b,c.
struct SilhouetteVertex {
    // members
    Vector2 a          = Vector2::Zero();
    Vector2 b          = Vector2::Zero();
    Vector2 c          = Vector2::Zero();
    int     indices[3] = { -1, -1, -1 }; // [prev, current, next] vertex indices
    int     pIndex     = -1;             // application-defined index

    SilhouetteVertex() = default;

    SilhouetteVertex(const Vector2 &a_,
                     const Vector2 &b_,
                     const Vector2 &c_,
                     int            idxPrev,
                     int            idxCenter,
                     int            idxNext,
                     int            pIndex_ = -1)
        : a(a_), b(b_), c(c_), pIndex(pIndex_) {
        indices[0] = idxPrev;
        indices[1] = idxCenter;
        indices[2] = idxNext;
    }

    // returns bounding box around the central vertex b
    BoundingBox<2> boundingBox() const {
        return BoundingBox<2>(b);
    }

    // returns centroid (the central vertex)
    Vector<2> centroid() const {
        return b;
    }

    // silhouettes are zero-area primitives
    float surfaceArea() const {
        return 0.0f;
    }

    // checks whether silhouette has adjacent face
    bool hasFace(int fIndex) const {
        return fIndex == 0 ? indices[2] != -1 : indices[0] != -1;
    }

    // returns normal of adjacent face (2D outward normal of polyline segment)
    Vector<2> normal(int fIndex, bool normalize = true) const {
        const Vector2 &pa = (fIndex == 0) ? b : a;
        const Vector2 &pb = (fIndex == 0) ? c : b;

        Vector2 s = pb - pa;
        Vector2 n(s[1], -s[0]);

        return normalize ? n.normalized() : n;
    }

    // returns averaged normalized silhouette normal
    Vector<2> normal() const {
        Vector2 n = Vector2::Zero();
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

// Helper that classifies whether a vertex is a silhouette point given adjacent
// normals and view direction (matches the Python `is_silhouette` logic).
inline bool isSilhouetteVertex(const Vector2 &n0,
                               const Vector2 &n1,
                               const Vector2 &viewDir,
                               float          d,
                               bool           flipNormalOrientation,
                               float          precision) {
    float sign = flipNormalOrientation ? 1.0f : -1.0f;

    // vertex is a silhouette point if it is concave and the query point lies on the vertex
    if (d <= precision) {
        float det = n0[0] * n1[1] - n0[1] * n1[0];
        return sign * det > precision;
    }

    // vertex is a silhouette point if the query point lies on the halfplane
    // defined by an adjacent line segment and the other segment is backfacing
    Vector2 viewDirUnit = viewDir / d;
    float   dot0        = viewDirUnit.dot(n0);
    float   dot1        = viewDirUnit.dot(n1);

    bool isZeroDot0 = std::fabs(dot0) <= precision;
    if (isZeroDot0)
        return sign * dot1 > precision;

    bool isZeroDot1 = std::fabs(dot1) <= precision;
    if (isZeroDot1)
        return sign * dot0 > precision;

    // vertex is a silhouette point if an adjacent line segment is frontfacing
    // w.r.t. the query point and the other segment is backfacing
    return dot0 * dot1 < 0.0f;
}

// SoA Structure
struct SilhouetteVertexSoA {
    std::vector<Vector2> a;
    std::vector<Vector2> b;
    std::vector<Vector2> c;
    std::vector<int>     indices[3];
    std::vector<int>     pIndex;

    void resize(size_t n) {
        a.resize(n);
        b.resize(n);
        c.resize(n);
        for (int k = 0; k < 3; ++k)
            indices[k].resize(n);
        pIndex.resize(n);
    }

    void clear() {
        a.clear();
        b.clear();
        c.clear();
        for (int k = 0; k < 3; ++k)
            indices[k].clear();
        pIndex.clear();
    }

    bool hasFace(int i, int fIndex) const {
        return fIndex == 0 ? indices[2][i] != -1 : indices[0][i] != -1;
    }

    Vector2 normal(int i, int fIndex, bool normalize = true) const {
        const Vector2 &pa = (fIndex == 0) ? b[i] : a[i];
        const Vector2 &pb = (fIndex == 0) ? c[i] : b[i];

        Vector2 s = pb - pa;
        Vector2 n(s[1], -s[0]);

        return normalize ? n.normalized() : n;
    }

    Vector2 normal(int i) const {
        Vector2 n = Vector2::Zero();
        if (hasFace(i, 0))
            n += normal(i, 0, false);
        if (hasFace(i, 1))
            n += normal(i, 1, false);
        return n.normalized();
    }
};

} // namespace diff_wost
