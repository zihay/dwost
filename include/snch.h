#pragma once

#include <bounding_volume.h>
#include <bvh.h>
#include <fwd.h>
#include <primitive.h>
#include <silhouette_edge.h>
#include <silhouette_vertex.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace diff_wost {

template <size_t DIM>
struct SnchNode {
    BoundingBox<DIM>  box;
    BoundingCone<DIM> cone;
    union {
        int referenceOffset;
        int secondChildOffset;
    };
    int silhouetteReferenceOffset;
    int nReferences;
    int nSilhouetteReferences;

    SnchNode()
        : referenceOffset(0), silhouetteReferenceOffset(0), nReferences(0),
          nSilhouetteReferences(0) {}
};

template <size_t DIM>
struct SnchNodeSoA {
    BoundingBoxSoA<DIM>  box;
    BoundingConeSoA<DIM> cone;
    std::vector<int>     referenceOffset; // Also acts as secondChildOffset
    std::vector<int>     silhouetteReferenceOffset;
    std::vector<int>     nReferences;
    std::vector<int>     nSilhouetteReferences;

    void resize(size_t n) {
        box.resize(n);
        cone.resize(n);
        referenceOffset.resize(n);
        silhouetteReferenceOffset.resize(n);
        nReferences.resize(n);
        nSilhouetteReferences.resize(n);
    }

    void clear() {
        box.clear();
        cone.clear();
        referenceOffset.clear();
        silhouetteReferenceOffset.clear();
        nReferences.clear();
        nSilhouetteReferences.clear();
    }
};

template <size_t DIM, typename Primitive, typename Silhouette>
class SNCH {
public:
    // Connectivity type: Vector2i for 2D (vertex indices), Vector3i for 3D (edge indices)
    using Connectivity = typename std::conditional<DIM == 2, Vector2i, Vector3i>::type;

    SNCH(const BVH<DIM, Primitive>             &bvh,
         const std::vector<Silhouette>         &silhouettes,
         const std::vector<Connectivity>       &connectivity,
         const std::function<bool(float, int)> &ignoreSilhouette = {});

    // Convenience constructor from vertices and indices
    SNCH(const std::vector<Vector<DIM>>        &vertices,
         const std::vector<Connectivity>       &indices,
         const std::function<bool(float, int)> &ignoreSilhouette = {});

    const std::vector<SnchNode<DIM>> &nodes() const { return m_flat_tree; }
    const std::vector<Silhouette *>  &silhouetteRefs() const {
        return m_silhouette_refs;
    }

    // Accessors for owned data (populated when using convenience constructor)
    const std::vector<Primitive> &primitives() const {
        if (m_owned_bvh)
            return m_owned_bvh->primitives();
        static const std::vector<Primitive> empty;
        return empty;
    }

    const std::vector<Silhouette> &silhouettes() const {
        return m_owned_silhouettes;
    }

    void convertToSoA(SnchNodeSoA<DIM> &nodesSoA) const {
        nodesSoA.resize(m_flat_tree.size());
        for (size_t i = 0; i < m_flat_tree.size(); ++i) {
            nodesSoA.box.pMin[i]                  = m_flat_tree[i].box.pMin;
            nodesSoA.box.pMax[i]                  = m_flat_tree[i].box.pMax;
            nodesSoA.cone.axis[i]                 = m_flat_tree[i].cone.axis;
            nodesSoA.cone.halfAngle[i]            = m_flat_tree[i].cone.halfAngle;
            nodesSoA.cone.radius[i]               = m_flat_tree[i].cone.radius;
            nodesSoA.referenceOffset[i]           = m_flat_tree[i].referenceOffset;
            nodesSoA.silhouetteReferenceOffset[i] = m_flat_tree[i].silhouetteReferenceOffset;
            nodesSoA.nReferences[i]               = m_flat_tree[i].nReferences;
            nodesSoA.nSilhouetteReferences[i]     = m_flat_tree[i].nSilhouetteReferences;
        }
    }

private:
    void build(const BVH<DIM, Primitive> &bvh);

    void assignGeometricDataToNodes(
        const std::vector<Primitive>          &primitives,
        const std::vector<Silhouette>         &silhouettes,
        const std::vector<Connectivity>       &connectivity,
        const std::function<bool(float, int)> &ignoreSilhouette);

    void computeBoundingConesRecursive(
        const std::vector<Silhouette *> &silhouetteRefs,
        const std::vector<Vector<DIM>>  &silhouetteNormals,
        const std::vector<Vector<DIM>>  &silhouetteFaceNormals,
        std::vector<SnchNode<DIM>>      &flatTree,
        int                              start,
        int                              end);

    std::vector<SnchNode<DIM>> m_flat_tree;
    std::vector<Silhouette *>  m_silhouette_refs;

    // Owned data for convenience constructor
    std::unique_ptr<BVH<DIM, Primitive>> m_owned_bvh;
    std::vector<Silhouette>              m_owned_silhouettes;
    std::vector<Connectivity>            m_owned_connectivity;

    // Topology helpers
    void buildTopology(const std::vector<Vector<DIM>>  &vertices,
                       const std::vector<Connectivity> &indices,
                       std::vector<Silhouette>         &silhouettes,
                       std::vector<Connectivity>       &connectivity);
};

// Implementation details

template <size_t DIM, typename Primitive, typename Silhouette>
SNCH<DIM, Primitive, Silhouette>::SNCH(
    const BVH<DIM, Primitive>             &bvh,
    const std::vector<Silhouette>         &silhouettes,
    const std::vector<Connectivity>       &connectivity,
    const std::function<bool(float, int)> &ignoreSilhouette) {
    build(bvh);
    assignGeometricDataToNodes(bvh.primitives(), silhouettes, connectivity,
                               ignoreSilhouette);
}

template <size_t DIM, typename Primitive, typename Silhouette>
void SNCH<DIM, Primitive, Silhouette>::build(const BVH<DIM, Primitive> &bvh) {
    const auto &bvhNodes = bvh.nodes();
    m_flat_tree.resize(bvhNodes.size());

    for (size_t i = 0; i < bvhNodes.size(); ++i) {
        m_flat_tree[i].box                       = bvhNodes[i].box;
        m_flat_tree[i].referenceOffset           = bvhNodes[i].referenceOffset;
        m_flat_tree[i].nReferences               = bvhNodes[i].nReferences;
        m_flat_tree[i].nSilhouetteReferences     = 0;
        m_flat_tree[i].silhouetteReferenceOffset = 0;
        // Cone initialized to default (invalid)
    }
}

// 2D Specialization
template <>
inline void SNCH<2, LineSegment, SilhouetteVertex>::assignGeometricDataToNodes(
    const std::vector<LineSegment>        &primitives,
    const std::vector<SilhouetteVertex>   &silhouettes,
    const std::vector<Vector2i>           &connectivity,
    const std::function<bool(float, int)> &ignoreSilhouette) {
    Vector2              zero = Vector2::Zero();
    std::vector<Vector2> silhouetteRefNormals, silhouetteRefFaceNormals;

    for (int i = 0; i < (int) m_flat_tree.size(); i++) {
        SnchNode<2>                  &node(m_flat_tree[i]);
        std::unordered_map<int, bool> seenVertex;
        int                           start = (int) m_silhouette_refs.size();

        for (int j = 0; j < node.nReferences; j++) { // leaf node if nReferences > 0
            int         referenceIndex = node.referenceOffset + j;
            const auto &lineSegment    = primitives[referenceIndex];
            // Use original index to look up connectivity
            int primIndex = lineSegment.getIndex();

            // Connectivity stores indices into 'silhouettes' array
            const Vector2i &indices = connectivity[primIndex];

            for (int k = 0; k < 2; k++) {
                int vIndex = indices[k];
                // Access silhouette by index
                const SilhouetteVertex *silhouetteVertex = &silhouettes[vIndex];

                if (seenVertex.find(vIndex) == seenVertex.end()) {
                    seenVertex[vIndex]          = true;
                    bool    ignore              = false;
                    bool    hasFace0            = silhouetteVertex->hasFace(0);
                    bool    hasFace1            = silhouetteVertex->hasFace(1);
                    bool    hasTwoAdjacentFaces = hasFace0 && hasFace1;
                    Vector2 n0                  = hasFace0 ? silhouetteVertex->normal(0, true) : zero;
                    Vector2 n1                  = hasFace1 ? silhouetteVertex->normal(1, true) : zero;
                    Vector2 n                   = silhouetteVertex->normal();

                    if (hasTwoAdjacentFaces && ignoreSilhouette) {
                        float det = n0[0] * n1[1] - n0[1] * n1[0];
                        ignore    = ignoreSilhouette(det, primIndex);
                    }

                    if (!ignore) {
                        // Store pointer to the silhouette in the original array
                        // We need to const_cast because m_silhouette_refs stores non-const pointers
                        // but we are reading from const input.
                        // In a real scenario, we might want to store copies or handle constness better.
                        // For now, assuming silhouettes lifetime is managed externally and exceeds SNCH.
                        m_silhouette_refs.emplace_back(const_cast<SilhouetteVertex *>(silhouetteVertex));
                        silhouetteRefNormals.emplace_back(n);
                        silhouetteRefFaceNormals.emplace_back(n0);
                        silhouetteRefFaceNormals.emplace_back(n1);
                    }
                }
            }
        }

        int end                        = (int) m_silhouette_refs.size();
        node.silhouetteReferenceOffset = start;
        node.nSilhouetteReferences     = end - start;
    }

    computeBoundingConesRecursive(m_silhouette_refs, silhouetteRefNormals,
                                  silhouetteRefFaceNormals, m_flat_tree, 0,
                                  (int) m_flat_tree.size());
}

// 3D Specialization
template <>
inline void SNCH<3, Triangle, SilhouetteEdge>::assignGeometricDataToNodes(
    const std::vector<Triangle>           &primitives,
    const std::vector<SilhouetteEdge>     &silhouettes,
    const std::vector<Vector3i>           &connectivity,
    const std::function<bool(float, int)> &ignoreSilhouette) {
    Vector3              zero = Vector3::Zero();
    std::vector<Vector3> silhouetteRefNormals, silhouetteRefFaceNormals;

    for (int i = 0; i < (int) m_flat_tree.size(); i++) {
        SnchNode<3>                  &node(m_flat_tree[i]);
        std::unordered_map<int, bool> seenEdge;
        int                           start = (int) m_silhouette_refs.size();

        for (int j = 0; j < node.nReferences; j++) { // leaf node if nReferences > 0
            int         referenceIndex = node.referenceOffset + j;
            const auto &triangle       = primitives[referenceIndex];
            int         primIndex      = triangle.getIndex();

            const Vector3i &indices = connectivity[primIndex];

            for (int k = 0; k < 3; k++) {
                int                   eIndex         = indices[k];
                const SilhouetteEdge *silhouetteEdge = &silhouettes[eIndex];

                if (seenEdge.find(eIndex) == seenEdge.end()) {
                    seenEdge[eIndex]            = true;
                    bool    ignore              = false;
                    bool    hasFace0            = silhouetteEdge->hasFace(0);
                    bool    hasFace1            = silhouetteEdge->hasFace(1);
                    bool    hasTwoAdjacentFaces = hasFace0 && hasFace1;
                    Vector3 n0                  = hasFace0 ? silhouetteEdge->normal(0, true) : zero;
                    Vector3 n1                  = hasFace1 ? silhouetteEdge->normal(1, true) : zero;
                    Vector3 n                   = silhouetteEdge->normal();

                    if (hasTwoAdjacentFaces && ignoreSilhouette) {
                        // Need soup positions to compute edge direction for dihedral angle
                        // But SilhouetteEdge stores points a and b directly.
                        Vector3 edgeDir = (silhouetteEdge->b - silhouetteEdge->a).normalized();
                        float   dihedralAngle =
                            std::atan2(edgeDir.dot(n0.cross(n1)), n0.dot(n1));
                        ignore = ignoreSilhouette(dihedralAngle, primIndex);
                    }

                    if (!ignore) {
                        m_silhouette_refs.emplace_back(const_cast<SilhouetteEdge *>(silhouetteEdge));
                        silhouetteRefNormals.emplace_back(n);
                        silhouetteRefFaceNormals.emplace_back(n0);
                        silhouetteRefFaceNormals.emplace_back(n1);
                    }
                }
            }
        }

        int end                        = (int) m_silhouette_refs.size();
        node.silhouetteReferenceOffset = start;
        node.nSilhouetteReferences     = end - start;
    }

    computeBoundingConesRecursive(m_silhouette_refs, silhouetteRefNormals,
                                  silhouetteRefFaceNormals, m_flat_tree, 0,
                                  (int) m_flat_tree.size());
}

template <size_t DIM, typename Primitive, typename Silhouette>
void SNCH<DIM, Primitive, Silhouette>::computeBoundingConesRecursive(
    const std::vector<Silhouette *> &silhouetteRefs,
    const std::vector<Vector<DIM>>  &silhouetteNormals,
    const std::vector<Vector<DIM>>  &silhouetteFaceNormals,
    std::vector<SnchNode<DIM>>      &flatTree,
    int                              start,
    int                              end) {
    BoundingCone<DIM> cone;
    SnchNode<DIM>    &node(flatTree[start]);
    Vector<DIM>       centroid = node.box.centroid();

    // compute bounding cone axis
    bool anySilhouetteRefs               = false;
    bool silhouettesHaveTwoAdjacentFaces = true;
    for (int i = start; i < end; i++) {
        SnchNode<DIM> &childNode(flatTree[i]);

        for (int j = 0; j < childNode.nSilhouetteReferences;
             j++) { // is leaf if nSilhouetteReferences > 0
            int               referenceIndex = childNode.silhouetteReferenceOffset + j;
            const Silhouette *silhouette     = silhouetteRefs[referenceIndex];

            cone.axis += silhouetteNormals[referenceIndex];
            cone.radius =
                std::max(cone.radius, (silhouette->centroid() - centroid).norm());
            silhouettesHaveTwoAdjacentFaces = silhouettesHaveTwoAdjacentFaces &&
                                              silhouette->hasFace(0) &&
                                              silhouette->hasFace(1);
            anySilhouetteRefs = true;
        }
    }

    // compute bounding cone angle
    if (!anySilhouetteRefs) {
        node.cone.halfAngle = -M_PI;

    } else if (!silhouettesHaveTwoAdjacentFaces) {
        node.cone.halfAngle = M_PI;

    } else {
        float axisNorm = cone.axis.norm();
        if (axisNorm > epsilon) {
            cone.axis /= axisNorm;
            cone.halfAngle = 0.0f;

            for (int i = start; i < end; i++) {
                SnchNode<DIM> &childNode(flatTree[i]);

                for (int j = 0; j < childNode.nSilhouetteReferences;
                     j++) { // is leaf if nSilhouetteReferences > 0
                    int               referenceIndex = childNode.silhouetteReferenceOffset + j;
                    const Silhouette *silhouette     = silhouetteRefs[referenceIndex];

                    for (int k = 0; k < 2; k++) {
                        const Vector<DIM> &n =
                            silhouetteFaceNormals[2 * referenceIndex + k];
                        float angle = std::acos(
                            std::max(-1.0f, std::min(1.0f, cone.axis.dot(n))));
                        cone.halfAngle = std::max(cone.halfAngle, angle);
                    }
                }
            }

            node.cone = cone;
        }
    }

    // recurse on children
    if (node.nReferences == 0) { // not a leaf
        computeBoundingConesRecursive(silhouetteRefs, silhouetteNormals,
                                      silhouetteFaceNormals, flatTree, start + 1,
                                      start + node.secondChildOffset);
        computeBoundingConesRecursive(silhouetteRefs, silhouetteNormals,
                                      silhouetteFaceNormals, flatTree,
                                      start + node.secondChildOffset, end);
    }
}

// Implementation of convenience constructor
template <size_t DIM, typename Primitive, typename Silhouette>
SNCH<DIM, Primitive, Silhouette>::SNCH(
    const std::vector<Vector<DIM>>        &vertices,
    const std::vector<Connectivity>       &indices,
    const std::function<bool(float, int)> &ignoreSilhouette) {
    // 1. Build BVH
    m_owned_bvh = std::make_unique<BVH<DIM, Primitive>>(vertices, indices);

    // 2. Build Silhouettes and Connectivity
    buildTopology(vertices, indices, m_owned_silhouettes, m_owned_connectivity);

    // 3. Build SNCH
    build(*m_owned_bvh);
    assignGeometricDataToNodes(m_owned_bvh->primitives(), m_owned_silhouettes,
                               m_owned_connectivity, ignoreSilhouette);
}

// Topology Builder Specializations

// 2D Specialization
template <>
inline void SNCH<2, LineSegment, SilhouetteVertex>::buildTopology(
    const std::vector<Vector2>    &vertices,
    const std::vector<Vector2i>   &indices,
    std::vector<SilhouetteVertex> &silhouettes,
    std::vector<Vector2i>         &connectivity) {
    // 1. Build Adjacency
    std::vector<std::vector<int>> adj(vertices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        int u = indices[i][0];
        int v = indices[i][1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // 2. Build Silhouettes (Vertices)
    silhouettes.reserve(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        int idxPrev = -1;
        int idxNext = -1;

        if (adj[i].size() >= 1)
            idxPrev = adj[i][0];
        if (adj[i].size() >= 2)
            idxNext = adj[i][1];

        Vector2 a = (idxPrev != -1) ? vertices[idxPrev] : Vector2::Zero();
        Vector2 b = vertices[i];
        Vector2 c = (idxNext != -1) ? vertices[idxNext] : Vector2::Zero();

        silhouettes.emplace_back(a, b, c, idxPrev, (int) i, idxNext, (int) i);
    }

    // 3. Build Connectivity
    connectivity = indices;
}

// 3D Specialization
template <>
inline void SNCH<3, Triangle, SilhouetteEdge>::buildTopology(
    const std::vector<Vector3>  &vertices,
    const std::vector<Vector3i> &indices,
    std::vector<SilhouetteEdge> &silhouettes,
    std::vector<Vector3i>       &connectivity) {
    std::map<std::pair<int, int>, int> edgeMap;
    connectivity.resize(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        const Vector3i &tri = indices[i];
        for (int k = 0; k < 3; ++k) {
            int u = tri[k];
            int v = tri[(k + 1) % 3];
            int w = tri[(k + 2) % 3];

            std::pair<int, int> key = (u < v) ? std::make_pair(u, v) : std::make_pair(v, u);

            if (edgeMap.find(key) == edgeMap.end()) {
                int edgeIdx  = (int) silhouettes.size();
                edgeMap[key] = edgeIdx;

                SilhouetteEdge edge;
                edge.a          = vertices[u];
                edge.b          = vertices[v];
                edge.c          = vertices[w];
                edge.indices[0] = w;
                edge.indices[1] = u;
                edge.indices[2] = v;
                edge.indices[3] = -1;
                edge.pIndex     = edgeIdx;

                silhouettes.push_back(edge);
                connectivity[i][k] = edgeIdx;
            } else {
                int             edgeIdx = edgeMap[key];
                SilhouetteEdge &edge    = silhouettes[edgeIdx];
                edge.d                  = vertices[w];
                edge.indices[3]         = w;
                connectivity[i][k]      = edgeIdx;
            }
        }
    }
}

} // namespace diff_wost
