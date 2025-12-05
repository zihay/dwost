// Python bindings for the local BVH scaffold in `include/bvh.h` using nanobind.
//
// The bindings expose:
// - CostHeuristic enum
// - BVHNode<DIM> as BVHNode2 / BVHNode3
// - Concrete BVH instantiations for common primitives:
//     * BVH<2, LineSegment>  -> BVH2LineSegment
//     * BVH<3, Triangle>     -> BVH3Triangle
//     * BVH<2, Point2>       -> BVH2Point
//     * BVH<3, Point3>       -> BVH3Point

#include <bvh.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using diff_wost::BoundingBox;
using diff_wost::BVH;
using diff_wost::BVHNode;
using diff_wost::CostHeuristic;
using diff_wost::LineSegment;
using diff_wost::LineSegment3;
using diff_wost::Point2;
using diff_wost::Point3;
using diff_wost::Triangle;

// Helper to bind BVHNode<DIM>
template <size_t DIM>
static void bind_bvh_node(nb::module_ &m, const char *name) {
    using Node = BVHNode<DIM>;

    nb::class_<Node>(m, name)
        .def(nb::init<>())
        .def_rw("box", &Node::box,
                "Axis-aligned bounding box of this node")
        .def_rw("reference_offset", &Node::referenceOffset,
                "Offset into the primitive index range for leaf nodes")
        .def_rw("second_child_offset", &Node::secondChildOffset,
                "Relative offset to the second child node for inner nodes")
        .def_rw("n_references", &Node::nReferences,
                "Number of primitives referenced by this node (leaf only)")
        .def(
            "is_leaf", [](const Node &self) { return self.nReferences > 0; }, "Return True if this node is a leaf");
}

template <size_t DIM>
static void bind_bvh_node_soa(nb::module_ &m, const char *name) {
    using NodeSoA = diff_wost::BVHNodeSoA<DIM>;

    nb::class_<NodeSoA>(m, name)
        .def(nb::init<>())
        .def_rw("box", &NodeSoA::box)
        .def_rw("referenceOffset", &NodeSoA::referenceOffset)
        .def_rw("nReferences", &NodeSoA::nReferences);
}

template <size_t DIM>
static void bind_convert_bvh_nodes(nb::module_ &m, const char *name) {
    m.def(
        name, [](const std::vector<diff_wost::BVHNode<DIM>> &nodes) {
            diff_wost::BVHNodeSoA<DIM> soa;
            soa.resize(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                soa.box.pMin[i]        = nodes[i].box.pMin;
                soa.box.pMax[i]        = nodes[i].box.pMax;
                soa.referenceOffset[i] = nodes[i].referenceOffset;
                soa.nReferences[i]     = nodes[i].nReferences;
            }
            return soa;
        },
        nb::arg("nodes"), "Convert a list of BVH nodes to SoA format");
}

// Helper to bind BVH<DIM, Primitive> under a given Python name
template <size_t DIM, typename Primitive>
static void bind_bvh(nb::module_ &m, const char *name) {
    using BVHType = BVH<DIM, Primitive>;
    using Vector  = Eigen::Matrix<float, DIM, 1>;

    auto cls = nb::class_<BVHType>(m, name);

    // Main constructor with options
    cls.def(nb::init<const std::vector<Primitive> &,
                     CostHeuristic,
                     bool,
                     int,
                     int>(),
            nb::arg("primitives"),
            nb::arg("cost_heuristic") = CostHeuristic::SurfaceArea,
            nb::arg("pack_leaves")    = false,
            nb::arg("leaf_size")      = 4,
            nb::arg("n_buckets")      = 8,
            "Construct a BVH over a sequence of primitives")
        .def("build", &BVHType::build, "Rebuild the BVH")
        .def("primitives", &BVHType::primitives)
        .def("nodes", &BVHType::nodes);

    // Convenience constructors
    if constexpr (DIM == 2 && std::is_same_v<Primitive, LineSegment>) {
        cls.def(nb::init<const std::vector<Vector> &,
                         const std::vector<Eigen::Vector2i> &,
                         CostHeuristic, bool, int, int>(),
                nb::arg("vertices"), nb::arg("indices"),
                nb::arg("cost_heuristic") = CostHeuristic::SurfaceArea,
                nb::arg("pack_leaves")    = false,
                nb::arg("leaf_size")      = 4,
                nb::arg("n_buckets")      = 8);
    } else if constexpr (DIM == 3 && std::is_same_v<Primitive, Triangle>) {
        cls.def(nb::init<const std::vector<Vector> &,
                         const std::vector<Eigen::Vector3i> &,
                         CostHeuristic, bool, int, int>(),
                nb::arg("vertices"), nb::arg("indices"),
                nb::arg("cost_heuristic") = CostHeuristic::SurfaceArea,
                nb::arg("pack_leaves")    = false,
                nb::arg("leaf_size")      = 4,
                nb::arg("n_buckets")      = 8);
    } else if constexpr (DIM == 3 && std::is_same_v<Primitive, LineSegment3>) {
        cls.def(nb::init<const std::vector<Vector> &,
                         const std::vector<Eigen::Vector2i> &,
                         CostHeuristic, bool, int, int>(),
                nb::arg("vertices"), nb::arg("indices"),
                nb::arg("cost_heuristic") = CostHeuristic::SurfaceArea,
                nb::arg("pack_leaves")    = false,
                nb::arg("leaf_size")      = 4,
                nb::arg("n_buckets")      = 8);
    } else if constexpr ((DIM == 2 && std::is_same_v<Primitive, Point2>) ||
                         (DIM == 3 && std::is_same_v<Primitive, Point3>) ) {
        cls.def(nb::init<const std::vector<Vector> &,
                         CostHeuristic, bool, int, int>(),
                nb::arg("points"),
                nb::arg("cost_heuristic") = CostHeuristic::SurfaceArea,
                nb::arg("pack_leaves")    = false,
                nb::arg("leaf_size")      = 4,
                nb::arg("n_buckets")      = 8);
    }
}

// Exposed entry point called from `src/diff_wost.cpp`
void export_bvh(nb::module_ &m) {
    // --- Cost heuristic enum -------------------------------------------------
    nb::enum_<CostHeuristic>(m, "CostHeuristic", "Split cost heuristic for BVH construction")
        .value("LongestAxisCenter", CostHeuristic::LongestAxisCenter)
        .value("SurfaceArea", CostHeuristic::SurfaceArea)
        .value("OverlapSurfaceArea", CostHeuristic::OverlapSurfaceArea)
        .value("Volume", CostHeuristic::Volume)
        .value("OverlapVolume", CostHeuristic::OverlapVolume)
        .export_values();

    // --- BVH node types ------------------------------------------------------
    bind_bvh_node<2>(m, "BVHNode2");
    bind_bvh_node<3>(m, "BVHNode3");

    bind_bvh_node_soa<2>(m, "BvhNodeSoA2");
    bind_bvh_node_soa<3>(m, "BvhNodeSoA3");

    bind_convert_bvh_nodes<2>(m, "convert_bvh_nodes_2d");
    bind_convert_bvh_nodes<3>(m, "convert_bvh_nodes_3d");

    // --- Primitive types used by BVH ----------------------------------------
    // 2D line segment primitive
    nb::class_<LineSegment>(m, "LineSegmentPrimitive2")
        .def(nb::init<>())
        .def(nb::init<const Vector2 &, const Vector2 &>(),
             nb::arg("a"), nb::arg("b"),
             "Create a 2D line segment primitive from endpoints a and b")
        .def_rw("a", &LineSegment::a,
                "First endpoint")
        .def_rw("b", &LineSegment::b,
                "Second endpoint")
        .def_rw("index", &LineSegment::pIndex,
                "Application-defined primitive index");

    // 3D line segment primitive
    nb::class_<LineSegment3>(m, "LineSegmentPrimitive3")
        .def(nb::init<>())
        .def(nb::init<const Vector3 &, const Vector3 &>(),
             nb::arg("a"), nb::arg("b"),
             "Create a 3D line segment primitive from endpoints a and b")
        .def_rw("a", &LineSegment3::a,
                "First endpoint")
        .def_rw("b", &LineSegment3::b,
                "Second endpoint")
        .def_rw("index", &LineSegment3::pIndex,
                "Application-defined primitive index");

    // 3D triangle primitive
    nb::class_<Triangle>(m, "TrianglePrimitive")
        .def(nb::init<>())
        .def(nb::init<const Vector3 &, const Vector3 &, const Vector3 &>(),
             nb::arg("a"), nb::arg("b"), nb::arg("c"),
             "Create a 3D triangle primitive from vertices a, b, c")
        .def_rw("a", &Triangle::a,
                "First vertex")
        .def_rw("b", &Triangle::b,
                "Second vertex")
        .def_rw("c", &Triangle::c,
                "Third vertex")
        .def_rw("index", &Triangle::pIndex,
                "Application-defined primitive index");

    // 2D point primitive
    nb::class_<Point2>(m, "PointPrimitive2")
        .def(nb::init<>())
        .def(nb::init<const Vector2 &>(),
             nb::arg("p"),
             "Create a 2D point primitive from position p")
        .def(nb::init<const Vector2 &, float>(),
             nb::arg("p"), nb::arg("radius"),
             "Create a 2D point primitive with a small support radius")
        .def_rw("p", &Point2::p,
                "Point position")
        .def_rw("radius", &Point2::radius,
                "Support radius used for BVH construction")
        .def_rw("index", &Point2::pIndex,
                "Application-defined primitive index");

    // 3D point primitive
    nb::class_<Point3>(m, "PointPrimitive3")
        .def(nb::init<>())
        .def(nb::init<const Vector3 &>(),
             nb::arg("p"),
             "Create a 3D point primitive from position p")
        .def(nb::init<const Vector3 &, float>(),
             nb::arg("p"), nb::arg("radius"),
             "Create a 3D point primitive with a small support radius")
        .def_rw("p", &Point3::p,
                "Point position")
        .def_rw("radius", &Point3::radius,
                "Support radius used for BVH construction")
        .def_rw("index", &Point3::pIndex,
                "Application-defined primitive index");

    // --- Additional SoA Structs ---
    nb::class_<diff_wost::Point2SoA>(m, "Point2SoA")
        .def(nb::init<>())
        .def_rw("p", &diff_wost::Point2SoA::p)
        .def_rw("radius", &diff_wost::Point2SoA::radius)
        .def_rw("index", &diff_wost::Point2SoA::pIndex);

    nb::class_<diff_wost::Point3SoA>(m, "Point3SoA")
        .def(nb::init<>())
        .def_rw("p", &diff_wost::Point3SoA::p)
        .def_rw("radius", &diff_wost::Point3SoA::radius)
        .def_rw("index", &diff_wost::Point3SoA::pIndex);

    nb::class_<diff_wost::LineSegment3SoA>(m, "LineSegment3SoA")
        .def(nb::init<>())
        .def_rw("a", &diff_wost::LineSegment3SoA::a)
        .def_rw("b", &diff_wost::LineSegment3SoA::b)
        .def_rw("index", &diff_wost::LineSegment3SoA::pIndex);

    nb::class_<diff_wost::LineSegmentSoA>(m, "LineSegmentSoA")
        .def(nb::init<>())
        .def_rw("a", &diff_wost::LineSegmentSoA::a)
        .def_rw("b", &diff_wost::LineSegmentSoA::b)
        .def_rw("index", &diff_wost::LineSegmentSoA::pIndex);

    nb::class_<diff_wost::TriangleSoA>(m, "TriangleSoA")
        .def(nb::init<>())
        .def_rw("a", &diff_wost::TriangleSoA::a)
        .def_rw("b", &diff_wost::TriangleSoA::b)
        .def_rw("c", &diff_wost::TriangleSoA::c)
        .def_rw("index", &diff_wost::TriangleSoA::pIndex);

    // --- Conversion Functions ---
    m.def(
        "convert_line_segments_2d", [](const std::vector<LineSegment> &prims) {
            diff_wost::LineSegmentSoA soa;
            soa.resize(prims.size());
            for (size_t i = 0; i < prims.size(); ++i) {
                soa.a[i]      = prims[i].a;
                soa.b[i]      = prims[i].b;
                soa.pIndex[i] = prims[i].pIndex;
            }
            return soa;
        },
        nb::arg("primitives"), "Convert list of 2D line segments to SoA");

    m.def(
        "convert_line_segments_3d", [](const std::vector<LineSegment3> &prims) {
            diff_wost::LineSegment3SoA soa;
            soa.resize(prims.size());
            for (size_t i = 0; i < prims.size(); ++i) {
                soa.a[i]      = prims[i].a;
                soa.b[i]      = prims[i].b;
                soa.pIndex[i] = prims[i].pIndex;
            }
            return soa;
        },
        nb::arg("primitives"), "Convert list of 3D line segments to SoA");

    m.def(
        "convert_triangles", [](const std::vector<Triangle> &prims) {
            diff_wost::TriangleSoA soa;
            soa.resize(prims.size());
            for (size_t i = 0; i < prims.size(); ++i) {
                soa.a[i]      = prims[i].a;
                soa.b[i]      = prims[i].b;
                soa.c[i]      = prims[i].c;
                soa.pIndex[i] = prims[i].pIndex;
            }
            return soa;
        },
        nb::arg("primitives"), "Convert list of triangles to SoA");

    m.def(
        "convert_points_2d", [](const std::vector<Point2> &prims) {
            diff_wost::Point2SoA soa;
            soa.resize(prims.size());
            for (size_t i = 0; i < prims.size(); ++i) {
                soa.p[i]      = prims[i].p;
                soa.radius[i] = prims[i].radius;
                soa.pIndex[i] = prims[i].pIndex;
            }
            return soa;
        },
        nb::arg("primitives"), "Convert list of 2D points to SoA");

    m.def(
        "convert_points_3d", [](const std::vector<Point3> &prims) {
            diff_wost::Point3SoA soa;
            soa.resize(prims.size());
            for (size_t i = 0; i < prims.size(); ++i) {
                soa.p[i]      = prims[i].p;
                soa.radius[i] = prims[i].radius;
                soa.pIndex[i] = prims[i].pIndex;
            }
            return soa;
        },
        nb::arg("primitives"), "Convert list of 3D points to SoA");

    // --- Concrete BVH instantiations ----------------------------------------
    bind_bvh<2, LineSegment>(m, "BVH2LineSegment");
    bind_bvh<3, Triangle>(m, "BVH3Triangle");
    bind_bvh<2, Point2>(m, "BVH2Point");
    bind_bvh<3, Point3>(m, "BVH3Point");
    bind_bvh<3, LineSegment3>(m, "BVH3LineSegment");
}
