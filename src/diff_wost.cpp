// Main nanobind module entry point for diff_wost.

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Forward declarations of exporters implemented in src/python/*.cpp
void export_bounding_volume(nb::module_ &m);
void export_bvh(nb::module_ &m);
void export_silhouette(nb::module_ &m);
void export_snch(nb::module_ &m);

NB_MODULE(diff_wost_ext, m) {
    // Group bindings into logically separated exporters.
    export_bounding_volume(m);
    export_bvh(m);
    export_silhouette(m);
    export_snch(m);
}
