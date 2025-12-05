#pragma once

#include <bounding_volume.h>
#include <fwd.h>

// Lightweight primitives that own their vertices directly (no soup reference).
// Designed to satisfy the minimal interface used by the local BVH scaffold:
// - boundingBox(), centroid(), surfaceArea(), signedVolume(), getIndex(), setIndex(int)

namespace diff_wost {

struct LineSegment {
    // members
    Vector2 a      = Vector2::Zero();
    Vector2 b      = Vector2::Zero();
    int     pIndex = -1;

    // constructors
    LineSegment() = default;
    LineSegment(const Vector2 &a_, const Vector2 &b_) : a(a_), b(b_) {}

    // interface
    BoundingBox<2> boundingBox() const {
        BoundingBox<2> box(a);
        box.expandToInclude(b);
        return box;
    }

    Vector<2> centroid() const {
        return (a + b) * 0.5f;
    }

    float surfaceArea() const {
        return (b - a).norm();
    }

    float signedVolume() const {
        // Oriented area contribution w.r.t. origin (consistent with LineSegment)
        return 0.5f * (a.x() * b.y() - a.y() * b.x());
    }

    int  getIndex() const { return pIndex; }
    void setIndex(int index) { pIndex = index; }
};

// 3D Line segment primitive
struct LineSegment3 {
    // members
    Vector3 a      = Vector3::Zero();
    Vector3 b      = Vector3::Zero();
    int     pIndex = -1;

    // constructors
    LineSegment3() = default;
    LineSegment3(const Vector3 &a_, const Vector3 &b_) : a(a_), b(b_) {}

    // interface
    BoundingBox<3> boundingBox() const {
        BoundingBox<3> box(a);
        box.expandToInclude(b);
        return box;
    }

    Vector<3> centroid() const {
        return (a + b) * 0.5f;
    }

    float surfaceArea() const {
        return (b - a).norm();
    }

    float signedVolume() const {
        return 0.0f;
    }

    int  getIndex() const { return pIndex; }
    void setIndex(int index) { pIndex = index; }
};

struct Triangle {
    // members
    Vector3 a      = Vector3::Zero();
    Vector3 b      = Vector3::Zero();
    Vector3 c      = Vector3::Zero();
    int     pIndex = -1;

    // constructors
    Triangle() = default;
    Triangle(const Vector3 &a_, const Vector3 &b_, const Vector3 &c_) : a(a_), b(b_), c(c_) {}

    // interface
    BoundingBox<3> boundingBox() const {
        BoundingBox<3> box(a);
        box.expandToInclude(b);
        box.expandToInclude(c);
        return box;
    }

    Vector<3> centroid() const {
        return (a + b + c) / 3.0f;
    }

    float surfaceArea() const {
        return 0.5f * ((b - a).cross(c - a)).norm();
    }

    float signedVolume() const {
        // Not used by the local BVH; return a robust proxy (0)
        return 0.0f;
    }

    int  getIndex() const { return pIndex; }
    void setIndex(int index) { pIndex = index; }
};

// 2D Point primitive
struct Point2 {
    // members
    Vector2 p      = Vector2::Zero();
    float   radius = 1e-4f; // tiny radius to give points finite extent
    int     pIndex = -1;

    // constructors
    Point2() = default;
    explicit Point2(const Vector2 &p_) : p(p_) {}
    Point2(const Vector2 &p_, float r_) : p(p_), radius(r_) {}

    // interface
    BoundingBox<2> boundingBox() const {
        BoundingBox<2> box(p);
        Vector<2>      r = Vector<2>::Constant(radius);
        box.pMin -= r;
        box.pMax += r;
        return box;
    }

    Vector<2> centroid() const {
        return p;
    }

    float surfaceArea() const {
        // Treat point as a tiny disk of radius r: use circumference in 2D context
        return 2.0f * static_cast<float>(M_PI) * radius;
    }

    float signedVolume() const {
        return 0.0f;
    }

    int  getIndex() const { return pIndex; }
    void setIndex(int index) { pIndex = index; }
};

// 3D Point primitive
struct Point3 {
    // members
    Vector3 p      = Vector3::Zero();
    float   radius = 1e-4f; // tiny radius to give points finite extent
    int     pIndex = -1;

    // constructors
    Point3() = default;
    explicit Point3(const Vector3 &p_) : p(p_) {}
    Point3(const Vector3 &p_, float r_) : p(p_), radius(r_) {}

    // interface
    BoundingBox<3> boundingBox() const {
        BoundingBox<3> box(p);
        Vector<3>      r = Vector<3>::Constant(radius);
        box.pMin -= r;
        box.pMax += r;
        return box;
    }

    Vector<3> centroid() const {
        return p;
    }

    float surfaceArea() const {
        // Treat point as a tiny sphere of radius r: surface area 4*pi*r^2
        return 4.0f * static_cast<float>(M_PI) * radius * radius;
    }

    float signedVolume() const {
        // Volume of a sphere
        return (4.0f / 3.0f) * static_cast<float>(M_PI) * radius * radius * radius;
    }

    int  getIndex() const { return pIndex; }
    void setIndex(int index) { pIndex = index; }
};

// SoA Structures

struct LineSegmentSoA {
    std::vector<Vector2> a;
    std::vector<Vector2> b;
    std::vector<int>     pIndex;

    void resize(size_t n) {
        a.resize(n);
        b.resize(n);
        pIndex.resize(n);
    }

    void clear() {
        a.clear();
        b.clear();
        pIndex.clear();
    }
};

struct LineSegment3SoA {
    std::vector<Vector3> a;
    std::vector<Vector3> b;
    std::vector<int>     pIndex;

    void resize(size_t n) {
        a.resize(n);
        b.resize(n);
        pIndex.resize(n);
    }

    void clear() {
        a.clear();
        b.clear();
        pIndex.clear();
    }
};

struct TriangleSoA {
    std::vector<Vector3> a;
    std::vector<Vector3> b;
    std::vector<Vector3> c;
    std::vector<int>     pIndex;

    void resize(size_t n) {
        a.resize(n);
        b.resize(n);
        c.resize(n);
        pIndex.resize(n);
    }

    void clear() {
        a.clear();
        b.clear();
        c.clear();
        pIndex.clear();
    }
};

struct Point2SoA {
    std::vector<Vector2> p;
    std::vector<float>   radius;
    std::vector<int>     pIndex;

    void resize(size_t n) {
        p.resize(n);
        radius.resize(n);
        pIndex.resize(n);
    }

    void clear() {
        p.clear();
        radius.clear();
        pIndex.clear();
    }
};

struct Point3SoA {
    std::vector<Vector3> p;
    std::vector<float>   radius;
    std::vector<int>     pIndex;

    void resize(size_t n) {
        p.resize(n);
        radius.resize(n);
        pIndex.resize(n);
    }

    void clear() {
        p.clear();
        radius.clear();
        pIndex.clear();
    }
};

} // namespace diff_wost