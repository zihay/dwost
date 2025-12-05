#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <silhouette_edge.h>
#include <silhouette_vertex.h>

namespace nb = nanobind;

using diff_wost::SilhouetteEdge;
using diff_wost::SilhouetteVertex;

void export_silhouette(nb::module_ &m) {
    // --- 2D silhouette vertex primitive --------------------------------------
    nb::class_<SilhouetteVertex>(m, "SilhouetteVertex")
        .def(nb::init<>())
        .def(nb::init<const Vector2 &,
                      const Vector2 &,
                      const Vector2 &,
                      int,
                      int,
                      int,
                      int>(),
             nb::arg("a"),
             nb::arg("b"),
             nb::arg("c"),
             nb::arg("idx_prev"),
             nb::arg("idx_center"),
             nb::arg("idx_next"),
             nb::arg("index") = -1,
             "Create a 2D silhouette vertex from three consecutive polyline "
             "vertices and their indices.")
        .def_rw("a", &SilhouetteVertex::a,
                "Previous vertex position")
        .def_rw("b", &SilhouetteVertex::b,
                "Central vertex position")
        .def_rw("c", &SilhouetteVertex::c,
                "Next vertex position")
        .def("get_indices", [](const SilhouetteVertex &self) { return std::array<int, 3>{
                                                                   self.indices[0], self.indices[1], self.indices[2]
                                                               }; }, "Return (idx_prev, idx_center, idx_next) as a 3‑tuple.")
        .def("set_indices", [](SilhouetteVertex &self, const std::array<int, 3> &idx) {
                 self.indices[0] = idx[0];
                 self.indices[1] = idx[1];
                 self.indices[2] = idx[2]; }, nb::arg("indices"), "Set (idx_prev, idx_center, idx_next) from a 3‑tuple.")
        .def_prop_rw("index", &SilhouetteVertex::getIndex, &SilhouetteVertex::setIndex, "Application‑defined vertex index.")
        .def("bounding_box", &SilhouetteVertex::boundingBox, "Return a tight bounding box around the central vertex.")
        .def("centroid", &SilhouetteVertex::centroid, "Return the centroid (central vertex position).")
        .def("surface_area", &SilhouetteVertex::surfaceArea, "Return the effective surface area used in BVH heuristics "
                                                             "(always zero for vertices).")
        .def("has_face", &SilhouetteVertex::hasFace, nb::arg("face_index"), "Return True if the silhouette has the given adjacent face "
                                                                            "(0: forward segment, 1: backward segment).")
        .def("face_normal", [](const SilhouetteVertex &self, int face_index, bool normalize) { return self.normal(face_index, normalize); }, nb::arg("face_index"), nb::arg("normalize") = true, "Return outward normal of the adjacent segment.")
        .def("normal", [](const SilhouetteVertex &self) { return self.normal(); }, "Return averaged outward normal of the silhouette vertex.");

    // --- 3D silhouette edge primitive ----------------------------------------
    nb::class_<SilhouetteEdge>(m, "SilhouetteEdge")
        .def(nb::init<>())
        .def(nb::init<const Vector3 &,
                      const Vector3 &,
                      const Vector3 &,
                      const Vector3 &,
                      int,
                      int,
                      int,
                      int,
                      int>(),
             nb::arg("a"),
             nb::arg("b"),
             nb::arg("c"),
             nb::arg("d"),
             nb::arg("idx_c"),
             nb::arg("idx_a"),
             nb::arg("idx_b"),
             nb::arg("idx_d"),
             nb::arg("index") = -1,
             "Create a 3D silhouette edge for triangles (A,B,C) and (D,A,B).")
        .def_rw("a", &SilhouetteEdge::a,
                "First endpoint of the edge.")
        .def_rw("b", &SilhouetteEdge::b,
                "Second endpoint of the edge.")
        .def_rw("c", &SilhouetteEdge::c,
                "Third vertex of triangle (A,B,C).")
        .def_rw("d", &SilhouetteEdge::d,
                "Third vertex of triangle (D,A,B).")
        .def("get_indices", [](const SilhouetteEdge &self) { return std::array<int, 4>{ self.indices[0],
                                                                                        self.indices[1],
                                                                                        self.indices[2],
                                                                                        self.indices[3] }; }, "Return (idx_c, idx_a, idx_b, idx_d) as a 4‑tuple.")
        .def("set_indices", [](SilhouetteEdge &self, const std::array<int, 4> &idx) {
                 self.indices[0] = idx[0];
                 self.indices[1] = idx[1];
                 self.indices[2] = idx[2];
                 self.indices[3] = idx[3]; }, nb::arg("indices"), "Set (idx_c, idx_a, idx_b, idx_d) from a 4‑tuple.")
        .def_prop_rw("index", &SilhouetteEdge::getIndex, &SilhouetteEdge::setIndex, "Application‑defined edge index.")
        .def("bounding_box", &SilhouetteEdge::boundingBox, "Return a bounding box around the edge endpoints.")
        .def("centroid", &SilhouetteEdge::centroid, "Return midpoint of the edge.")
        .def("surface_area", &SilhouetteEdge::surfaceArea, "Return effective surface area (edge length) used in BVH heuristics.")
        .def("has_face", &SilhouetteEdge::hasFace, nb::arg("face_index"), "Return True if the silhouette edge has the given adjacent face "
                                                                          "(0 or 1).")
        .def("face_normal", [](const SilhouetteEdge &self, int face_index, bool normalize) { return self.normal(face_index, normalize); }, nb::arg("face_index"), nb::arg("normalize") = true, "Return the normal of the adjacent triangle.")
        .def("normal", [](const SilhouetteEdge &self) { return self.normal(); }, "Return the averaged outward normal of the silhouette edge.");

    // --- Free helpers mirroring Python is_silhouette logic -------------------
    m.def(
        "is_silhouette_vertex",
        &diff_wost::isSilhouetteVertex,
        nb::arg("n0"),
        nb::arg("n1"),
        nb::arg("view_dir"),
        nb::arg("distance"),
        nb::arg("flip_normal_orientation"),
        nb::arg("precision"),
        "Classify whether a vertex is a silhouette vertex given two adjacent "
        "segment normals, view direction, and distance.");

    m.def(
        "is_silhouette_edge",
        &diff_wost::isSilhouetteEdge,
        nb::arg("pa"),
        nb::arg("pb"),
        nb::arg("n0"),
        nb::arg("n1"),
        nb::arg("view_dir"),
        nb::arg("distance"),
        nb::arg("flip_normal_orientation"),
        nb::arg("precision"),
        "Classify whether an edge is a silhouette edge given two adjacent "
        "triangle normals, view direction, and distance.");
    // --- SoA Bindings --------------------------------------------------------
    nb::class_<diff_wost::SilhouetteVertexSoA>(m, "SilhouetteVertexSoA")
        .def(nb::init<>())
        .def_rw("a", &diff_wost::SilhouetteVertexSoA::a)
        .def_rw("b", &diff_wost::SilhouetteVertexSoA::b)
        .def_rw("c", &diff_wost::SilhouetteVertexSoA::c)
        .def_prop_rw(
            "indices",
            [](diff_wost::SilhouetteVertexSoA &s) {
                return std::make_tuple(std::ref(s.indices[0]), std::ref(s.indices[1]), std::ref(s.indices[2]));
            },
            [](diff_wost::SilhouetteVertexSoA &s, const std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> &v) {
                s.indices[0] = std::get<0>(v);
                s.indices[1] = std::get<1>(v);
                s.indices[2] = std::get<2>(v);
            })
        .def_rw("p_index", &diff_wost::SilhouetteVertexSoA::pIndex)
        .def("resize", &diff_wost::SilhouetteVertexSoA::resize)
        .def("clear", &diff_wost::SilhouetteVertexSoA::clear);

    nb::class_<diff_wost::SilhouetteEdgeSoA>(m, "SilhouetteEdgeSoA")
        .def(nb::init<>())
        .def_rw("a", &diff_wost::SilhouetteEdgeSoA::a)
        .def_rw("b", &diff_wost::SilhouetteEdgeSoA::b)
        .def_rw("c", &diff_wost::SilhouetteEdgeSoA::c)
        .def_rw("d", &diff_wost::SilhouetteEdgeSoA::d)
        .def_prop_rw(
            "indices",
            [](diff_wost::SilhouetteEdgeSoA &s) {
                return std::make_tuple(std::ref(s.indices[0]), std::ref(s.indices[1]), std::ref(s.indices[2]), std::ref(s.indices[3]));
            },
            [](diff_wost::SilhouetteEdgeSoA &s, const std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> &v) {
                s.indices[0] = std::get<0>(v);
                s.indices[1] = std::get<1>(v);
                s.indices[2] = std::get<2>(v);
                s.indices[3] = std::get<3>(v);
            })
        .def_rw("p_index", &diff_wost::SilhouetteEdgeSoA::pIndex)
        .def("resize", &diff_wost::SilhouetteEdgeSoA::resize)
        .def("clear", &diff_wost::SilhouetteEdgeSoA::clear);
}
