#pragma once

#include <bounding_volume.h>
#include <fwd.h>
#include <primitive.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

// Local BVH scaffold used in this project. This is a
// minimal, self-consistent header that compiles and can
// be extended with a full implementation later.

namespace diff_wost {

enum class CostHeuristic {
    LongestAxisCenter,
    SurfaceArea,
    OverlapSurfaceArea,
    Volume,
    OverlapVolume
};

template <size_t DIM>
struct BVHNode {
    BoundingBox<DIM> box;
    union {
        int referenceOffset;
        int secondChildOffset;
    };
    int nReferences;
};

template <size_t DIM>
struct BVHNodeSoA {
    BoundingBoxSoA<DIM> box;
    std::vector<int>    referenceOffset; // Also acts as secondChildOffset
    std::vector<int>    nReferences;

    void resize(size_t n) {
        box.resize(n);
        referenceOffset.resize(n);
        nReferences.resize(n);
    }

    void clear() {
        box.clear();
        referenceOffset.clear();
        nReferences.clear();
    }
};

template <size_t DIM, typename Primitive>
class BVH {
    static constexpr int ROOT_PARENT   = 0xfffffffc;
    static constexpr int UNTOUCHED     = 0xffffffff;
    static constexpr int TOUCHED_TWICE = 0xfffffffd;

public:
    BVH(const std::vector<Primitive> &primitives_,
        CostHeuristic                 costHeuristic_ = CostHeuristic::SurfaceArea,
        bool packLeaves_ = false, int leafSize_ = 4, int nBuckets_ = 8);

    // Convenience constructors for owned-vertex primitives
    template <typename T = Primitive,
              typename std::enable_if<
                  (DIM == 2) && std::is_same<T, LineSegment>::value,
                  int>::type = 0>
    BVH(const std::vector<Vector2>  &vertices,
        const std::vector<Vector2i> &indices,
        CostHeuristic                costHeuristic_ = CostHeuristic::SurfaceArea,
        bool packLeaves_ = false, int leafSize_ = 4, int nBuckets_ = 8);

    template <
        typename T = Primitive,
        typename std::enable_if<
            (DIM == 3) && std::is_same<T, Triangle>::value, int>::type = 0>
    BVH(const std::vector<Vector3>  &vertices,
        const std::vector<Vector3i> &indices,
        CostHeuristic                costHeuristic_ = CostHeuristic::SurfaceArea,
        bool packLeaves_ = false, int leafSize_ = 4, int nBuckets_ = 8);

    // DIM=3, Primitive=LineSegment3: convenience ctor from vertex array and
    // index pairs
    template <typename T = Primitive,
              typename std::enable_if<
                  (DIM == 3) && std::is_same<T, LineSegment3>::value,
                  int>::type = 0>
    BVH(const std::vector<Vector3>  &vertices,
        const std::vector<Vector2i> &indices,
        CostHeuristic                costHeuristic_ = CostHeuristic::SurfaceArea,
        bool packLeaves_ = false, int leafSize_ = 4, int nBuckets_ = 8);

    // DIM=2, Primitive=Point2: convenience ctor from points
    template <
        typename T                         = Primitive,
        typename std::enable_if<(DIM == 2) && std::is_same<T, Point2>::value,
                                int>::type = 0>
    BVH(const std::vector<Vector2> &points,
        CostHeuristic               costHeuristic_ = CostHeuristic::SurfaceArea,
        bool packLeaves_ = false, int leafSize_ = 4, int nBuckets_ = 8);

    // DIM=3, Primitive=Point3: convenience ctor from points
    template <
        typename T                         = Primitive,
        typename std::enable_if<(DIM == 3) && std::is_same<T, Point3>::value,
                                int>::type = 0>
    BVH(const std::vector<Vector3> &points,
        CostHeuristic               costHeuristic_ = CostHeuristic::SurfaceArea,
        bool packLeaves_ = false, int leafSize_ = 4, int nBuckets_ = 8);

    ~BVH() = default;

    void build();

    void buildRecursive(std::vector<BoundingBox<DIM>> &referenceBoxes,
                        std::vector<Vector<DIM>>      &referenceCentroids,
                        std::vector<BVHNode<DIM>> &buildNodes, int parent,
                        int start, int end, int depth);

    float computeSplitCost(const BoundingBox<DIM> &boxLeft,
                           const BoundingBox<DIM> &boxRight,
                           int                     nReferencesLeft,
                           int                     nReferencesRight,
                           int                     depth) const;

    float computeObjectSplit(const BoundingBox<DIM>              &nodeBoundingBox,
                             const BoundingBox<DIM>              &nodeCentroidBox,
                             const std::vector<BoundingBox<DIM>> &referenceBoxes,
                             const std::vector<Vector<DIM>>      &referenceCentroids,
                             int depth, int nodeStart, int nodeEnd, int &splitDim,
                             float &splitCoord);

    int performObjectSplit(int start, int end, int splitDim, float splitCoord,
                           std::vector<BoundingBox<DIM>> &referenceBoxes,
                           std::vector<Vector<DIM>>      &referenceCentroids);

    // Accessors for Python bindings
    inline const std::vector<Primitive> &primitives() const {
        return m_primitives;
    }
    inline const std::vector<BVHNode<DIM>> &nodes() const { return m_flat_tree; }

    // SoA Conversion
    void convertToSoA(BVHNodeSoA<DIM> &nodesSoA) const {
        nodesSoA.resize(m_flat_tree.size());
        for (size_t i = 0; i < m_flat_tree.size(); ++i) {
            nodesSoA.box.pMin[i]        = m_flat_tree[i].box.pMin;
            nodesSoA.box.pMax[i]        = m_flat_tree[i].box.pMax;
            nodesSoA.referenceOffset[i] = m_flat_tree[i].referenceOffset;
            nodesSoA.nReferences[i]     = m_flat_tree[i].nReferences;
        }
    }

    template <typename PrimitiveSoA>
    void convertPrimitivesToSoA(PrimitiveSoA &primsSoA) const {
        primsSoA.resize(m_primitives.size());
        for (size_t i = 0; i < m_primitives.size(); ++i) {
            const auto &prim = m_primitives[i];
            if constexpr (std::is_same_v<Primitive, LineSegment>) {
                primsSoA.a[i]      = prim.a;
                primsSoA.b[i]      = prim.b;
                primsSoA.pIndex[i] = prim.pIndex;
            } else if constexpr (std::is_same_v<Primitive, Triangle>) {
                primsSoA.a[i]      = prim.a;
                primsSoA.b[i]      = prim.b;
                primsSoA.c[i]      = prim.c;
                primsSoA.pIndex[i] = prim.pIndex;
            } else if constexpr (std::is_same_v<Primitive, LineSegment3>) {
                primsSoA.a[i]      = prim.a;
                primsSoA.b[i]      = prim.b;
                primsSoA.pIndex[i] = prim.pIndex;
            } else if constexpr (std::is_same_v<Primitive, Point2>) {
                primsSoA.p[i]      = prim.p;
                primsSoA.radius[i] = prim.radius;
                primsSoA.pIndex[i] = prim.pIndex;
            } else if constexpr (std::is_same_v<Primitive, Point3>) {
                primsSoA.p[i]      = prim.p;
                primsSoA.radius[i] = prim.radius;
                primsSoA.pIndex[i] = prim.pIndex;
            }
        }
    }

private:
    // configuration
    CostHeuristic          m_heuristic   = CostHeuristic::SurfaceArea;
    bool                   m_pack_leaves = false;
    int                    m_leaf_size   = 4;
    int                    m_n_buckets   = 8;
    std::vector<Primitive> m_primitives;

    // derived parameters
    int m_depth_guess;

    // Flat tree storage used during build
    std::vector<BVHNode<DIM>> m_flat_tree;

    // Working sets
    std::vector<std::pair<BoundingBox<DIM>, int>> m_buckets;
    std::vector<std::pair<BoundingBox<DIM>, int>> m_right_buckets;

    // Statistics
    size_t m_max_depth = 0;
    size_t m_n_nodes   = 0;
    size_t m_n_leaves  = 0;
};

template <size_t DIM, typename Primitive>
inline BVH<DIM, Primitive>::BVH(const std::vector<Primitive> &primitives_,
                                CostHeuristic costHeuristic_, bool packLeaves_,
                                int leafSize_, int nBuckets_)
    : m_heuristic(costHeuristic_), m_primitives(primitives_),
      m_pack_leaves(packLeaves_), m_leaf_size(leafSize_),
      m_n_buckets(nBuckets_),
      m_depth_guess(primitives_.empty()
                        ? 0
                        : static_cast<int>(std::log2(primitives_.size()))) {
    // Ensure bucket storage is initialized
    m_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    m_right_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    build();
}

// DIM=2, Primitive=LineSegment
template <size_t DIM, typename Primitive>
template <
    typename T,
    typename std::enable_if<
        (DIM == 2) && std::is_same<T, LineSegment>::value, int>::type>
inline BVH<DIM, Primitive>::BVH(const std::vector<Vector2>  &vertices,
                                const std::vector<Vector2i> &indices,
                                CostHeuristic costHeuristic_, bool packLeaves_,
                                int leafSize_, int nBuckets_)
    : m_heuristic(costHeuristic_), m_pack_leaves(packLeaves_),
      m_leaf_size(leafSize_), m_n_buckets(nBuckets_) {
    m_primitives.reserve(indices.size());
    for (int i = 0; i < (int) indices.size(); i++) {
        const Vector2i &f = indices[i];
        m_primitives.emplace_back(vertices[f.x()], vertices[f.y()]);
        m_primitives.back().setIndex(i);
    }

    m_depth_guess = m_primitives.empty()
                        ? 0
                        : static_cast<int>(std::log2(m_primitives.size()));
    m_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    m_right_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    build();
}

// DIM=3, Primitive=Triangle
template <size_t DIM, typename Primitive>
template <typename T,
          typename std::enable_if<
              (DIM == 3) && std::is_same<T, Triangle>::value, int>::type>
inline BVH<DIM, Primitive>::BVH(const std::vector<Vector3>  &vertices,
                                const std::vector<Vector3i> &indices,
                                CostHeuristic costHeuristic_, bool packLeaves_,
                                int leafSize_, int nBuckets_)
    : m_heuristic(costHeuristic_), m_pack_leaves(packLeaves_),
      m_leaf_size(leafSize_), m_n_buckets(nBuckets_) {
    m_primitives.reserve(indices.size());
    for (int i = 0; i < (int) indices.size(); i++) {
        const Vector3i &f = indices[i];
        m_primitives.emplace_back(vertices[f.x()], vertices[f.y()],
                                  vertices[f.z()]);
        m_primitives.back().setIndex(i);
    }

    m_depth_guess = m_primitives.empty()
                        ? 0
                        : static_cast<int>(std::log2(m_primitives.size()));
    m_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    m_right_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    build();
}

// DIM=3, Primitive=LineSegment3
template <size_t DIM, typename Primitive>
template <
    typename T,
    typename std::enable_if<
        (DIM == 3) && std::is_same<T, LineSegment3>::value, int>::type>
inline BVH<DIM, Primitive>::BVH(const std::vector<Vector3>  &vertices,
                                const std::vector<Vector2i> &indices,
                                CostHeuristic costHeuristic_, bool packLeaves_,
                                int leafSize_, int nBuckets_)
    : m_heuristic(costHeuristic_), m_pack_leaves(packLeaves_),
      m_leaf_size(leafSize_), m_n_buckets(nBuckets_) {
    m_primitives.reserve(indices.size());
    for (int i = 0; i < (int) indices.size(); i++) {
        const Vector2i &e = indices[i];
        m_primitives.emplace_back(vertices[e.x()], vertices[e.y()]);
        m_primitives.back().setIndex(i);
    }

    m_depth_guess = m_primitives.empty()
                        ? 0
                        : static_cast<int>(std::log2(m_primitives.size()));
    m_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    m_right_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    build();
}

// DIM=2, Primitive=Point2
template <size_t DIM, typename Primitive>
template <typename T,
          typename std::enable_if<
              (DIM == 2) && std::is_same<T, Point2>::value, int>::type>
inline BVH<DIM, Primitive>::BVH(const std::vector<Vector2> &points,
                                CostHeuristic costHeuristic_, bool packLeaves_,
                                int leafSize_, int nBuckets_)
    : m_heuristic(costHeuristic_), m_pack_leaves(packLeaves_),
      m_leaf_size(leafSize_), m_n_buckets(nBuckets_) {
    m_primitives.reserve(points.size());
    for (int i = 0; i < (int) points.size(); i++) {
        m_primitives.emplace_back(points[i]);
        m_primitives.back().setIndex(i);
    }

    m_depth_guess = m_primitives.empty()
                        ? 0
                        : static_cast<int>(std::log2(m_primitives.size()));
    m_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    m_right_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    build();
}

// DIM=3, Primitive=Point3
template <size_t DIM, typename Primitive>
template <typename T,
          typename std::enable_if<
              (DIM == 3) && std::is_same<T, Point3>::value, int>::type>
inline BVH<DIM, Primitive>::BVH(const std::vector<Vector3> &points,
                                CostHeuristic costHeuristic_, bool packLeaves_,
                                int leafSize_, int nBuckets_)
    : m_heuristic(costHeuristic_), m_pack_leaves(packLeaves_),
      m_leaf_size(leafSize_), m_n_buckets(nBuckets_) {
    m_primitives.reserve(points.size());
    for (int i = 0; i < (int) points.size(); i++) {
        m_primitives.emplace_back(points[i]);
        m_primitives.back().setIndex(i);
    }

    m_depth_guess = m_primitives.empty()
                        ? 0
                        : static_cast<int>(std::log2(m_primitives.size()));
    m_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    m_right_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    build();
}

template <size_t DIM, typename Primitive>
inline void BVH<DIM, Primitive>::build() {
    // Reset statistics for rebuilds
    m_n_nodes   = 0;
    m_n_leaves  = 0;
    m_max_depth = 0;

    if (m_primitives.empty()) {
        return;
    }

    // Precompute bounding boxes and centroids
    const size_t                  nReferences = m_primitives.size();
    std::vector<BoundingBox<DIM>> referenceBoxes(nReferences);
    std::vector<Vector<DIM>>      referenceCentroids(nReferences);

    m_flat_tree.clear();
    m_flat_tree.reserve(nReferences * 2);

    for (size_t i = 0; i < nReferences; i++) {
        referenceBoxes[i]     = m_primitives[i].boundingBox();
        referenceCentroids[i] = m_primitives[i].centroid();
    }

    // Ensure bucket storage is initialized for rebuilds
    if (m_buckets.size() != static_cast<size_t>(m_n_buckets)) {
        m_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
        m_right_buckets.assign(m_n_buckets, std::make_pair(BoundingBox<DIM>(), 0));
    }

    // Build tree recursively (flat representation)
    // ROOT_PARENT is used as a sentinel parent index for the root
    buildRecursive(referenceBoxes, referenceCentroids, m_flat_tree, ROOT_PARENT,
                   0, static_cast<int>(nReferences), 0);

    // Clear working sets to save memory, but keep capacity for potential rebuilds
    // Actually, for safety in rebuilds, we should probably just leave them or ensure they are resized next time.
    // The previous code cleared them, which caused the bug if we didn't resize.
    // Since we added the resize check above, we can clear them here.
    m_buckets.clear();
    m_right_buckets.clear();
}

template <size_t DIM, typename Primitive>
inline void BVH<DIM, Primitive>::buildRecursive(
    std::vector<BoundingBox<DIM>> &referenceBoxes,
    std::vector<Vector<DIM>>      &referenceCentroids,
    std::vector<BVHNode<DIM>>     &buildNodes,
    int                            parent,
    int                            start,
    int                            end,
    int                            depth) {
    m_max_depth = std::max<size_t>(static_cast<size_t>(depth), m_max_depth);

    // add node to tree
    BVHNode<DIM> node;
    int          currentNodeIndex = static_cast<int>(m_n_nodes);
    int          nReferences      = end - start;

    m_n_nodes++;

    // calculate the bounding box for this node
    BoundingBox<DIM> bb, bc;
    for (int p = start; p < end; p++) {
        bb.expandToInclude(referenceBoxes[p]);
        bc.expandToInclude(referenceCentroids[p]);
    }

    node.box = bb;

    // if the number of references at this point is less than the leaf
    // size, then this will become a leaf
    static constexpr int MaxDepth = 64;
    if (nReferences <= m_leaf_size || depth == MaxDepth - 2) {
        node.referenceOffset = start;
        node.nReferences     = nReferences;
        m_n_leaves++;

    } else {
        node.secondChildOffset = UNTOUCHED;
        node.nReferences       = 0;
    }

    buildNodes.emplace_back(node);

    // child touches parent...
    // special case: don't do this for the root
    if (parent != ROOT_PARENT) {
        buildNodes[parent].secondChildOffset--;

        // when this is the second touch, this is the right child;
        // the right child sets up the offset for the flat tree
        if (buildNodes[parent].secondChildOffset == TOUCHED_TWICE) {
            buildNodes[parent].secondChildOffset = m_n_nodes - 1 - parent;
        }
    }

    // if this is a leaf, no need to subdivide
    if (node.nReferences > 0)
        return;

    // compute object split
    int   splitDim;
    float splitCoord;
    float splitCost =
        computeObjectSplit(bb, bc, referenceBoxes, referenceCentroids, depth,
                           start, end, splitDim, splitCoord);
    (void) splitCost; // suppress unused warning

    // partition the list of references on split
    int mid = performObjectSplit(start, end, splitDim, splitCoord, referenceBoxes,
                                 referenceCentroids);

    // push left and right children
    buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
                   currentNodeIndex, start, mid, depth + 1);
    buildRecursive(referenceBoxes, referenceCentroids, buildNodes,
                   currentNodeIndex, mid, end, depth + 1);
}

template <size_t DIM, typename Primitive>
inline float BVH<DIM, Primitive>::computeSplitCost(
    const BoundingBox<DIM> &boxLeft,
    const BoundingBox<DIM> &boxRight,
    int                     nReferencesLeft,
    int                     nReferencesRight,
    int                     depth) const {
    float cost = std::numeric_limits<float>::max();
    if (m_pack_leaves && depth > 0 && ((float) m_depth_guess / depth) < 1.5f &&
        nReferencesLeft % m_leaf_size != 0 &&
        nReferencesRight % m_leaf_size != 0) {
        return cost;
    }

    if (m_heuristic == CostHeuristic::SurfaceArea) {
        cost = nReferencesLeft * boxLeft.surfaceArea() +
               nReferencesRight * boxRight.surfaceArea();

    } else if (m_heuristic == CostHeuristic::OverlapSurfaceArea) {
        // set the cost to be negative if the left and right boxes don't overlap at
        // all
        BoundingBox<DIM> boxIntersected = boxLeft.intersect(boxRight);
        float            saLeft         = boxLeft.surfaceArea();
        float            saRight        = boxRight.surfaceArea();

        float invSaLeft  = (saLeft > 1e-6f) ? 1.0f / saLeft : 0.0f;
        float invSaRight = (saRight > 1e-6f) ? 1.0f / saRight : 0.0f;

        cost = (nReferencesLeft * invSaRight +
                nReferencesRight * invSaLeft) *
               std::abs(boxIntersected.surfaceArea());
        if (!boxIntersected.isValid())
            cost *= -1;

    } else if (m_heuristic == CostHeuristic::Volume) {
        cost = nReferencesLeft * boxLeft.volume() +
               nReferencesRight * boxRight.volume();

    } else if (m_heuristic == CostHeuristic::OverlapVolume) {
        // set the cost to be negative if the left and right boxes don't overlap at
        // all
        BoundingBox<DIM> boxIntersected = boxLeft.intersect(boxRight);
        float            volLeft        = boxLeft.volume();
        float            volRight       = boxRight.volume();

        float invVolLeft  = (volLeft > 1e-6f) ? 1.0f / volLeft : 0.0f;
        float invVolRight = (volRight > 1e-6f) ? 1.0f / volRight : 0.0f;

        cost = (nReferencesLeft * invVolRight +
                nReferencesRight * invVolLeft) *
               std::abs(boxIntersected.volume());
        if (!boxIntersected.isValid())
            cost *= -1;
    }

    return cost;
}

template <size_t DIM, typename Primitive>
inline float BVH<DIM, Primitive>::computeObjectSplit(
    const BoundingBox<DIM>              &nodeBoundingBox,
    const BoundingBox<DIM>              &nodeCentroidBox,
    const std::vector<BoundingBox<DIM>> &referenceBoxes,
    const std::vector<Vector<DIM>>      &referenceCentroids,
    int                                  depth,
    int                                  nodeStart,
    int                                  nodeEnd,
    int                                 &splitDim,
    float                               &splitCoord) {
    (void) depth; // unused here; may be used in advanced heuristics
    float splitCost = std::numeric_limits<float>::max();
    splitDim        = -1;
    splitCoord      = 0.0f;

    if (m_heuristic != CostHeuristic::LongestAxisCenter) {
        Vector<DIM> extent = nodeBoundingBox.extent();

        // find the best split across all dimensions
        for (size_t dim = 0; dim < DIM; dim++) {
            // ignore flat dimension
            if (extent[dim] < 1e-6f)
                continue;

            // bin references into buckets
            float bucketWidth =
                extent[dim] / static_cast<float>(m_n_buckets);
            for (int b = 0; b < m_n_buckets; b++) {
                m_buckets[b].first  = BoundingBox<DIM>();
                m_buckets[b].second = 0;
            }

            for (int p = nodeStart; p < nodeEnd; p++) {
                int bucketIndex = static_cast<int>(
                    (referenceCentroids[p][dim] - nodeBoundingBox.pMin[dim]) /
                    bucketWidth);
                bucketIndex = std::clamp(bucketIndex, 0, m_n_buckets - 1);
                m_buckets[bucketIndex].first.expandToInclude(referenceBoxes[p]);
                m_buckets[bucketIndex].second += 1;
            }

            // sweep right to left to build right bucket bounding boxes
            BoundingBox<DIM> boxRefRight;
            for (int b = m_n_buckets - 1; b > 0; b--) {
                boxRefRight.expandToInclude(m_buckets[b].first);
                m_right_buckets[b].first  = boxRefRight;
                m_right_buckets[b].second = m_buckets[b].second;
                if (b != m_n_buckets - 1)
                    m_right_buckets[b].second += m_right_buckets[b + 1].second;
            }

            // evaluate bucket split costs
            BoundingBox<DIM> boxRefLeft;
            int              nReferencesLeft = 0;
            for (int b = 1; b < static_cast<int>(m_n_buckets); b++) {
                boxRefLeft.expandToInclude(m_buckets[b - 1].first);
                nReferencesLeft += m_buckets[b - 1].second;

                if (nReferencesLeft > 0 && m_right_buckets[b].second > 0) {
                    float cost = computeSplitCost(boxRefLeft, m_right_buckets[b].first,
                                                  nReferencesLeft,
                                                  m_right_buckets[b].second, depth);

                    if (cost < splitCost) {
                        splitCost  = cost;
                        splitDim   = static_cast<int>(dim);
                        splitCoord = nodeBoundingBox.pMin[dim] + b * bucketWidth;
                    }
                }
            }
        }
    }

    // if no split dimension was chosen, fallback to LongestAxisCenter heuristic
    if (splitDim == -1) {
        splitDim = nodeCentroidBox.maxDimension();
        splitCoord =
            (nodeCentroidBox.pMin[splitDim] + nodeCentroidBox.pMax[splitDim]) *
            0.5f;
    }

    return splitCost;
}

template <size_t DIM, typename Primitive>
inline int BVH<DIM, Primitive>::performObjectSplit(
    int nodeStart, int nodeEnd, int splitDim, float splitCoord,
    std::vector<BoundingBox<DIM>> &referenceBoxes,
    std::vector<Vector<DIM>>      &referenceCentroids) {
    int mid = nodeStart;
    for (int i = nodeStart; i < nodeEnd; i++) {
        if (referenceCentroids[i][splitDim] < splitCoord) {
            std::swap(m_primitives[i], m_primitives[mid]);
            std::swap(referenceBoxes[i], referenceBoxes[mid]);
            std::swap(referenceCentroids[i], referenceCentroids[mid]);
            mid++;
        }
    }

    // if we get a bad split, just choose the center...
    if (mid == nodeStart || mid == nodeEnd) {
        mid = nodeStart + (nodeEnd - nodeStart) / 2;

        // ensure the number of primitives in one branch is a multiple of the leaf
        // size
        if (m_pack_leaves) {
            while ((mid - nodeStart) % m_leaf_size != 0 && mid < nodeEnd)
                mid++;
            if (mid == nodeEnd)
                mid = nodeStart + (nodeEnd - nodeStart) / 2;
        }
    }

    return mid;
}
} // namespace diff_wost