#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <snch.h>

namespace nb = nanobind;
using namespace diff_wost;

// Helper to bind SnchNode<DIM>
template <size_t DIM>
static void bind_snch_node(nb::module_ &m, const char *name) {
    using Node = SnchNode<DIM>;

    nb::class_<Node>(m, name)
        .def(nb::init<>())
        .def_rw("box", &Node::box, "Axis-aligned bounding box")
        .def_rw("cone", &Node::cone, "Bounding cone")
        .def_rw("reference_offset", &Node::referenceOffset,
                "Offset into the primitive index range for leaf nodes")
        .def_rw("second_child_offset", &Node::secondChildOffset,
                "Relative offset to the second child node for inner nodes")
        .def_rw("silhouette_reference_offset", &Node::silhouetteReferenceOffset,
                "Offset into the silhouette reference array")
        .def_rw("n_references", &Node::nReferences,
                "Number of primitives referenced by this node")
        .def_rw("n_silhouette_references", &Node::nSilhouetteReferences,
                "Number of silhouettes referenced by this node");
}

// Helper to bind SnchNodeSoA<DIM>
template <size_t DIM>
static void bind_snch_node_soa(nb::module_ &m, const char *name) {
    using NodeSoA = SnchNodeSoA<DIM>;

    nb::class_<NodeSoA>(m, name)
        .def(nb::init<>())
        .def_rw("box", &NodeSoA::box)
        .def_rw("cone", &NodeSoA::cone)
        .def_rw("reference_offset", &NodeSoA::referenceOffset)
        .def_rw("silhouette_reference_offset", &NodeSoA::silhouetteReferenceOffset)
        .def_rw("n_references", &NodeSoA::nReferences)
        .def_rw("n_silhouette_references", &NodeSoA::nSilhouetteReferences)
        .def("resize", &NodeSoA::resize)
        .def("clear", &NodeSoA::clear);
}

// Helper to bind conversion function
template <size_t DIM>
static void bind_convert_snch_nodes(nb::module_ &m, const char *name) {
    m.def(
        name,
        [](const std::vector<SnchNode<DIM>> &nodes) {
            SnchNodeSoA<DIM> soa;
            soa.resize(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                soa.box.pMin[i]                  = nodes[i].box.pMin;
                soa.box.pMax[i]                  = nodes[i].box.pMax;
                soa.cone.axis[i]                 = nodes[i].cone.axis;
                soa.cone.halfAngle[i]            = nodes[i].cone.halfAngle;
                soa.cone.radius[i]               = nodes[i].cone.radius;
                soa.referenceOffset[i]           = nodes[i].referenceOffset;
                soa.silhouetteReferenceOffset[i] = nodes[i].silhouetteReferenceOffset;
                soa.nReferences[i]               = nodes[i].nReferences;
                soa.nSilhouetteReferences[i]     = nodes[i].nSilhouetteReferences;
            }
            return soa;
        },
        nb::arg("nodes"), "Convert a list of SNCH nodes to SoA format");
}

// Helper to bind SNCH<DIM, Primitive, Silhouette>
template <size_t DIM, typename Primitive, typename Silhouette>
static void bind_snch(nb::module_ &m, const char *name) {
    using SNCHType     = SNCH<DIM, Primitive, Silhouette>;
    using BVHType      = BVH<DIM, Primitive>;
    using Connectivity = typename SNCHType::Connectivity;

    nb::class_<SNCHType>(m, name)
        .def(nb::init<const BVHType &,
                      const std::vector<Silhouette> &,
                      const std::vector<Connectivity> &>(),
             nb::arg("bvh"),
             nb::arg("silhouettes"),
             nb::arg("connectivity"),
             "Construct a SNCH from a BVH, silhouettes, and connectivity info")
        .def(
            "__init__", [](SNCHType *self, const std::vector<Eigen::Matrix<float, DIM, 1>> &vertices, const std::vector<Eigen::Matrix<int, DIM, 1>> &indices) { new (self) SNCHType(vertices, indices); }, nb::arg("vertices"), nb::arg("indices"), "Construct a SNCH directly from vertices and indices")
        .def("nodes", &SNCHType::nodes, "Return the flat list of SNCH nodes")
        .def("silhouette_refs", &SNCHType::silhouetteRefs, "Return the list of silhouette pointers referenced by nodes")
        .def("convert_to_soa", &SNCHType::convertToSoA, nb::arg("nodes_soa"), "Convert the SNCH nodes to SoA format")
        .def("primitives", &SNCHType::primitives, "Return the primitives owned by the SNCH")
        .def("silhouettes", &SNCHType::silhouettes, "Return the silhouettes owned by the SNCH");
}

// Specialized binding for 2D SNCH with silhouette_refs_soa method
static void bind_snch_2d(nb::module_ &m) {
    using SNCHType     = SNCH<2, LineSegment, SilhouetteVertex>;
    using BVHType      = BVH<2, LineSegment>;
    using Connectivity = typename SNCHType::Connectivity;

    nb::class_<SNCHType>(m, "Snch2")
        .def(nb::init<const BVHType &,
                      const std::vector<SilhouetteVertex> &,
                      const std::vector<Connectivity> &>(),
             nb::arg("bvh"),
             nb::arg("silhouettes"),
             nb::arg("connectivity"),
             "Construct a SNCH from a BVH, silhouettes, and connectivity info")
        .def(
            "__init__", [](SNCHType *self, const std::vector<Eigen::Matrix<float, 2, 1>> &vertices, const std::vector<Eigen::Matrix<int, 2, 1>> &indices) { new (self) SNCHType(vertices, indices); }, nb::arg("vertices"), nb::arg("indices"), "Construct a SNCH directly from vertices and indices")
        .def("nodes", &SNCHType::nodes, "Return the flat list of SNCH nodes")
        .def("silhouette_refs", &SNCHType::silhouetteRefs, "Return the list of silhouette pointers referenced by nodes")
        .def("convert_to_soa", &SNCHType::convertToSoA, nb::arg("nodes_soa"), "Convert the SNCH nodes to SoA format")
        .def("primitives", &SNCHType::primitives, "Return the primitives owned by the SNCH")
        .def("silhouettes", &SNCHType::silhouettes, "Return the silhouettes owned by the SNCH")
        .def(
            "silhouette_refs_soa", [](const SNCHType &snch) {
                const auto         &refs = snch.silhouetteRefs();
                SilhouetteVertexSoA soa;
                soa.resize(refs.size());
                for (size_t i = 0; i < refs.size(); ++i) {
                    const SilhouetteVertex *sv = refs[i];
                    soa.a[i]                   = sv->a;
                    soa.b[i]                   = sv->b;
                    soa.c[i]                   = sv->c;
                    soa.indices[0][i]          = sv->indices[0];
                    soa.indices[1][i]          = sv->indices[1];
                    soa.indices[2][i]          = sv->indices[2];
                    soa.pIndex[i]              = sv->pIndex;
                }
                return soa;
            },
            "Return silhouette_refs as SoA (for SNCH traversal)");
}

// Specialized binding for 3D SNCH with silhouette_refs_soa method
static void bind_snch_3d(nb::module_ &m) {
    using SNCHType     = SNCH<3, Triangle, SilhouetteEdge>;
    using BVHType      = BVH<3, Triangle>;
    using Connectivity = typename SNCHType::Connectivity;

    nb::class_<SNCHType>(m, "Snch3")
        .def(nb::init<const BVHType &,
                      const std::vector<SilhouetteEdge> &,
                      const std::vector<Connectivity> &>(),
             nb::arg("bvh"),
             nb::arg("silhouettes"),
             nb::arg("connectivity"),
             "Construct a SNCH from a BVH, silhouettes, and connectivity info")
        .def(
            "__init__", [](SNCHType *self, const std::vector<Eigen::Matrix<float, 3, 1>> &vertices, const std::vector<Eigen::Matrix<int, 3, 1>> &indices) { new (self) SNCHType(vertices, indices); }, nb::arg("vertices"), nb::arg("indices"), "Construct a SNCH directly from vertices and indices")
        .def("nodes", &SNCHType::nodes, "Return the flat list of SNCH nodes")
        .def("silhouette_refs", &SNCHType::silhouetteRefs, "Return the list of silhouette pointers referenced by nodes")
        .def("convert_to_soa", &SNCHType::convertToSoA, nb::arg("nodes_soa"), "Convert the SNCH nodes to SoA format")
        .def("primitives", &SNCHType::primitives, "Return the primitives owned by the SNCH")
        .def("silhouettes", &SNCHType::silhouettes, "Return the silhouettes owned by the SNCH")
        .def(
            "silhouette_refs_soa", [](const SNCHType &snch) {
                const auto       &refs = snch.silhouetteRefs();
                SilhouetteEdgeSoA soa;
                soa.resize(refs.size());
                for (size_t i = 0; i < refs.size(); ++i) {
                    const SilhouetteEdge *se = refs[i];
                    soa.a[i]                 = se->a;
                    soa.b[i]                 = se->b;
                    soa.c[i]                 = se->c;
                    soa.d[i]                 = se->d;
                    soa.indices[0][i]        = se->indices[0];
                    soa.indices[1][i]        = se->indices[1];
                    soa.indices[2][i]        = se->indices[2];
                    soa.indices[3][i]        = se->indices[3];
                    soa.pIndex[i]            = se->pIndex;
                }
                return soa;
            },
            "Return silhouette_refs as SoA (for SNCH traversal)");
}

static void bind_convert_silhouettes_2d(nb::module_ &m, const char *name) {
    m.def(
        name,
        [](const std::vector<SilhouetteVertex> &silhouettes) {
            SilhouetteVertexSoA soa;
            soa.resize(silhouettes.size());
            for (size_t i = 0; i < silhouettes.size(); ++i) {
                soa.a[i]          = silhouettes[i].a;
                soa.b[i]          = silhouettes[i].b;
                soa.c[i]          = silhouettes[i].c;
                soa.indices[0][i] = silhouettes[i].indices[0];
                soa.indices[1][i] = silhouettes[i].indices[1];
                soa.indices[2][i] = silhouettes[i].indices[2];
                soa.pIndex[i]     = silhouettes[i].pIndex;
            }
            return soa;
        },
        nb::arg("silhouettes"), "Convert a list of SilhouetteVertex to SoA format");
}

static void bind_convert_silhouettes_3d(nb::module_ &m, const char *name) {
    m.def(
        name,
        [](const std::vector<SilhouetteEdge> &silhouettes) {
            SilhouetteEdgeSoA soa;
            soa.resize(silhouettes.size());
            for (size_t i = 0; i < silhouettes.size(); ++i) {
                soa.a[i]          = silhouettes[i].a;
                soa.b[i]          = silhouettes[i].b;
                soa.c[i]          = silhouettes[i].c;
                soa.d[i]          = silhouettes[i].d;
                soa.indices[0][i] = silhouettes[i].indices[0];
                soa.indices[1][i] = silhouettes[i].indices[1];
                soa.indices[2][i] = silhouettes[i].indices[2];
                soa.indices[3][i] = silhouettes[i].indices[3];
                soa.pIndex[i]     = silhouettes[i].pIndex;
            }
            return soa;
        },
        nb::arg("silhouettes"), "Convert a list of SilhouetteEdge to SoA format");
}

void export_snch(nb::module_ &m) {
    // --- SNCH Node Types ---
    bind_snch_node<2>(m, "SnchNode2");
    bind_snch_node<3>(m, "SnchNode3");

    bind_snch_node_soa<2>(m, "SnchNodeSoA2");
    bind_snch_node_soa<3>(m, "SnchNodeSoA3");

    bind_convert_snch_nodes<2>(m, "convert_snch_nodes_2d");
    bind_convert_snch_nodes<3>(m, "convert_snch_nodes_3d");

    bind_convert_silhouettes_2d(m, "convert_silhouettes_2d");
    bind_convert_silhouettes_3d(m, "convert_silhouettes_3d");

    // --- SNCH Classes (with silhouette_refs_soa method) ---
    // 2D: LineSegment + SilhouetteVertex
    bind_snch_2d(m);

    // 3D: Triangle + SilhouetteEdge
    bind_snch_3d(m);
}
