// Python bindings for diff_wost bounding volume utilities using nanobind.

#include <bounding_volume.h>
#include <fwd.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using diff_wost::BoundingBox;
using diff_wost::BoundingCone;
using diff_wost::BoundingSphere;

using BoundingBox2    = BoundingBox<2>;
using BoundingBox3    = BoundingBox<3>;
using BoundingSphere2 = BoundingSphere<2>;
using BoundingSphere3 = BoundingSphere<3>;
using BoundingCone2   = BoundingCone<2>;
using BoundingCone3   = BoundingCone<3>;

// Wrapper helpers for functions with out-parameters

// BoundingCone::overlap wrappers returning a nanobind tuple
nb::tuple bounding_cone_overlap_2d(const BoundingCone2 &cone,
                                   const Vector2       &origin,
                                   const BoundingBox2  &box,
                                   float                dist_to_box) {
    float min_angle = 0.0f, max_angle = 0.0f;
    bool  result = cone.overlap(origin, box, dist_to_box, min_angle, max_angle);
    return nb::make_tuple(result, min_angle, max_angle);
}

nb::tuple bounding_cone_overlap_3d(const BoundingCone3 &cone,
                                   const Vector3       &origin,
                                   const BoundingBox3  &box,
                                   float                dist_to_box) {
    float min_angle = 0.0f, max_angle = 0.0f;
    bool  result = cone.overlap(origin, box, dist_to_box, min_angle, max_angle);
    return nb::make_tuple(result, min_angle, max_angle);
}

// computeOrthonormalBasis wrapper returning a nanobind tuple
nb::tuple compute_orthonormal_basis_wrapper(const Vector3 &n) {
    Vector3 b1, b2;
    diff_wost::computeOrthonormalBasis(n, b1, b2);
    return nb::make_tuple(b1, b2);
}

// mergeBoundingCones wrappers for 2D and 3D
BoundingCone2 merge_bounding_cones_2d(const BoundingCone2 &cone_a,
                                      const BoundingCone2 &cone_b,
                                      const Vector2       &origin_a,
                                      const Vector2       &origin_b,
                                      const Vector2       &new_origin) {
    return diff_wost::mergeBoundingCones<2>(cone_a, cone_b,
                                            origin_a, origin_b, new_origin);
}

BoundingCone3 merge_bounding_cones_3d(const BoundingCone3 &cone_a,
                                      const BoundingCone3 &cone_b,
                                      const Vector3       &origin_a,
                                      const Vector3       &origin_b,
                                      const Vector3       &new_origin) {
    return diff_wost::mergeBoundingCones<3>(cone_a, cone_b,
                                            origin_a, origin_b, new_origin);
}

void export_bounding_volume(nb::module_ &m) {
    // --- BoundingSphere ---
    nb::class_<BoundingSphere2>(m, "BoundingSphere2")
        .def(nb::init<const diff_wost::Vector<2> &, float>(),
             nb::arg("center"), nb::arg("radius_squared"))
        .def("transform", &BoundingSphere2::transform, nb::arg("transform"),
             "Return a transformed copy of this bounding sphere")
        .def_rw("c", &BoundingSphere2::c,
                "Center of the bounding sphere")
        .def_rw("r2", &BoundingSphere2::r2,
                "Squared radius of the bounding sphere");

    nb::class_<BoundingSphere3>(m, "BoundingSphere3")
        .def(nb::init<const diff_wost::Vector<3> &, float>(),
             nb::arg("center"), nb::arg("radius_squared"))
        .def("transform", &BoundingSphere3::transform, nb::arg("transform"),
             "Return a transformed copy of this bounding sphere")
        .def_rw("c", &BoundingSphere3::c,
                "Center of the bounding sphere")
        .def_rw("r2", &BoundingSphere3::r2,
                "Squared radius of the bounding sphere");

    // --- BoundingBox ---
    nb::class_<BoundingBox2>(m, "BoundingBox2")
        .def(nb::init<>())
        .def(nb::init<const diff_wost::Vector<2> &>(), nb::arg("p"))
        .def("expand_to_include_point",
             (void(BoundingBox2::*)(const diff_wost::Vector<2> &)) & BoundingBox2::expandToInclude,
             nb::arg("p"),
             "Expand this box to include a point")
        .def("expand_to_include_box",
             (void(BoundingBox2::*)(const BoundingBox2 &)) & BoundingBox2::expandToInclude,
             nb::arg("box"),
             "Expand this box to include another box")
        .def("extent", &BoundingBox2::extent,
             "Return the box extent along each dimension")
        .def("contains", &BoundingBox2::contains, nb::arg("p"),
             "Return True if the point is inside the box")
        .def("overlap_box",
             static_cast<bool (BoundingBox2::*)(const BoundingBox2 &) const>(
                 &BoundingBox2::overlap),
             nb::arg("other"),
             "Return True if this box overlaps another box")
        .def("is_valid", &BoundingBox2::isValid,
             "Return True if this box is valid (min <= max in all dimensions)")
        .def("max_dimension", &BoundingBox2::maxDimension,
             "Return the index of the longest axis of the box")
        .def("centroid", &BoundingBox2::centroid,
             "Return the centroid of the box")
        .def("surface_area", &BoundingBox2::surfaceArea,
             "Return the surface area of the box")
        .def("volume", &BoundingBox2::volume,
             "Return the volume of the box")
        .def("bounding_sphere", &BoundingBox2::boundingSphere,
             "Return a bounding sphere of the box")
        .def("transform", &BoundingBox2::transform, nb::arg("transform"),
             "Return a transformed copy of this bounding box")
        .def("intersect", &BoundingBox2::intersect, nb::arg("other"),
             "Return the intersection of this box with another box")
        .def_rw("p_min", &BoundingBox2::pMin,
                "Minimum corner of the box")
        .def_rw("p_max", &BoundingBox2::pMax,
                "Maximum corner of the box");

    nb::class_<BoundingBox3>(m, "BoundingBox3")
        .def(nb::init<>())
        .def(nb::init<const diff_wost::Vector<3> &>(), nb::arg("p"))
        .def("expand_to_include_point",
             (void(BoundingBox3::*)(const diff_wost::Vector<3> &)) & BoundingBox3::expandToInclude,
             nb::arg("p"),
             "Expand this box to include a point")
        .def("expand_to_include_box",
             (void(BoundingBox3::*)(const BoundingBox3 &)) & BoundingBox3::expandToInclude,
             nb::arg("box"),
             "Expand this box to include another box")
        .def("extent", &BoundingBox3::extent,
             "Return the box extent along each dimension")
        .def("contains", &BoundingBox3::contains, nb::arg("p"),
             "Return True if the point is inside the box")
        .def("overlap_box",
             static_cast<bool (BoundingBox3::*)(const BoundingBox3 &) const>(
                 &BoundingBox3::overlap),
             nb::arg("other"),
             "Return True if this box overlaps another box")
        .def("is_valid", &BoundingBox3::isValid,
             "Return True if this box is valid (min <= max in all dimensions)")
        .def("max_dimension", &BoundingBox3::maxDimension,
             "Return the index of the longest axis of the box")
        .def("centroid", &BoundingBox3::centroid,
             "Return the centroid of the box")
        .def("surface_area", &BoundingBox3::surfaceArea,
             "Return the surface area of the box")
        .def("volume", &BoundingBox3::volume,
             "Return the volume of the box")
        .def("bounding_sphere", &BoundingBox3::boundingSphere,
             "Return a bounding sphere of the box")
        .def("transform", &BoundingBox3::transform, nb::arg("transform"),
             "Return a transformed copy of this bounding box")
        .def("intersect", &BoundingBox3::intersect, nb::arg("other"),
             "Return the intersection of this box with another box")
        .def_rw("p_min", &BoundingBox3::pMin,
                "Minimum corner of the box")
        .def_rw("p_max", &BoundingBox3::pMax,
                "Maximum corner of the box");

    // --- BoundingCone ---
    nb::class_<BoundingCone2>(m, "BoundingCone2")
        .def(nb::init<>())
        .def(nb::init<const diff_wost::Vector<2> &, float, float>(),
             nb::arg("axis"), nb::arg("half_angle"), nb::arg("radius"))
        .def("overlap",
             &bounding_cone_overlap_2d,
             nb::arg("origin"), nb::arg("box"), nb::arg("dist_to_box"),
             "Check for overlap with a view cone defined by origin and box. "
             "Returns (overlap: bool, min_angle_range: float, "
             "max_angle_range: float).")
        .def("is_valid", &BoundingCone2::isValid,
             "Return True if this cone is valid")
        .def_rw("axis", &BoundingCone2::axis,
                "Cone axis")
        .def_rw("half_angle", &BoundingCone2::halfAngle,
                "Cone half-angle in radians")
        .def_rw("radius", &BoundingCone2::radius,
                "Bounding sphere radius for the cone");

    nb::class_<BoundingCone3>(m, "BoundingCone3")
        .def(nb::init<>())
        .def(nb::init<const diff_wost::Vector<3> &, float, float>(),
             nb::arg("axis"), nb::arg("half_angle"), nb::arg("radius"))
        .def("overlap",
             &bounding_cone_overlap_3d,
             nb::arg("origin"), nb::arg("box"), nb::arg("dist_to_box"),
             "Check for overlap with a view cone defined by origin and box. "
             "Returns (overlap: bool, min_angle_range: float, "
             "max_angle_range: float).")
        .def("is_valid", &BoundingCone3::isValid,
             "Return True if this cone is valid")
        .def_rw("axis", &BoundingCone3::axis,
                "Cone axis")
        .def_rw("half_angle", &BoundingCone3::halfAngle,
                "Cone half-angle in radians")
        .def_rw("radius", &BoundingCone3::radius,
                "Bounding sphere radius for the cone");

    // --- Free functions ---
    m.def("compute_orthonormal_basis", &compute_orthonormal_basis_wrapper,
          nb::arg("n"),
          "Compute an orthonormal basis (b1, b2) around normal n");

    m.def("merge_bounding_cones_2d", &merge_bounding_cones_2d,
          nb::arg("cone_a"), nb::arg("cone_b"),
          nb::arg("origin_a"), nb::arg("origin_b"), nb::arg("new_origin"),
          "Merge two 2D bounding cones given their origins and a new origin");

    m.def("merge_bounding_cones_3d", &merge_bounding_cones_3d,
          nb::arg("cone_a"), nb::arg("cone_b"),
          nb::arg("origin_a"), nb::arg("origin_b"), nb::arg("new_origin"),
          "Merge two 3D bounding cones given their origins and a new origin");

    // --- SoA Structs ---
    nb::class_<diff_wost::BoundingBoxSoA<2>>(m, "BoundingBoxSoA2")
        .def(nb::init<>())
        .def_rw("pMin", &diff_wost::BoundingBoxSoA<2>::pMin)
        .def_rw("pMax", &diff_wost::BoundingBoxSoA<2>::pMax);

    nb::class_<diff_wost::BoundingBoxSoA<3>>(m, "BoundingBoxSoA3")
        .def(nb::init<>())
        .def_rw("pMin", &diff_wost::BoundingBoxSoA<3>::pMin)
        .def_rw("pMax", &diff_wost::BoundingBoxSoA<3>::pMax);

    nb::class_<diff_wost::BoundingConeSoA<2>>(m, "BoundingConeSoA2")
        .def(nb::init<>())
        .def_rw("axis", &diff_wost::BoundingConeSoA<2>::axis)
        .def_rw("halfAngle", &diff_wost::BoundingConeSoA<2>::halfAngle)
        .def_rw("radius", &diff_wost::BoundingConeSoA<2>::radius);

    nb::class_<diff_wost::BoundingConeSoA<3>>(m, "BoundingConeSoA3")
        .def(nb::init<>())
        .def_rw("axis", &diff_wost::BoundingConeSoA<3>::axis)
        .def_rw("halfAngle", &diff_wost::BoundingConeSoA<3>::halfAngle)
        .def_rw("radius", &diff_wost::BoundingConeSoA<3>::radius);
}
