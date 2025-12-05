cmake_minimum_required(VERSION 3.15...3.27)
project(diff_wost_cpp) # Replace 'my_project' with the name of your project

# Set C++20 globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Keep C++ standard pure, avoid compiler-specific extensions
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Fix for GCC 13+ -Wchanges-meaning error in TBB headers
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wno-changes-meaning)
endif()

# import the nanobind package
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# import zombie
set(ZOMBIE_BUILD_DEMO OFF CACHE BOOL "Build zombie demo" FORCE)

# import Eigen (prefer system install, fall back to bundled copy if available)
find_package(Eigen3 QUIET)

if(TARGET Eigen3::Eigen)
  # Modern CMake module or config package that defines the imported target
elseif(Eigen3_FOUND AND EIGEN3_INCLUDE_DIR)
  # Older FindEigen3 that only sets EIGEN3_INCLUDE_DIR
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
else()
  # Fallback: look for a vendored Eigen checkout in ext/eigen
  set(EIGEN3_FALLBACK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen")
  if(EXISTS "${EIGEN3_FALLBACK_INCLUDE_DIR}/Eigen/Core")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_FALLBACK_INCLUDE_DIR}"
    )
  else()
    message(FATAL_ERROR "Eigen3 not found. Install Eigen3 (e.g. via your package manager) or place a copy in ext/eigen.")
  endif()
endif()

# import TBB
# find_package(TBB REQUIRED COMPONENTS tbb)

include_directories(include)

# add_library(diff_wost_cpp 
# src/scenes/circle.cpp
# src/solvers/wost.cpp
# src/main.cpp)
# target_link_libraries(diff_wost PUBLIC Eigen3::Eigen zombie tbb)

# add_executable(diff_wost_main src/main.cpp)
# target_link_libraries(diff_wost_main PRIVATE diff_wost)

nanobind_add_module(
  # Name of the extension
  diff_wost_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
#   src/core/log.cpp
#   src/diff_wost.cpp
#   src/scenes/circle.cpp
#   src/scenes/ring.cpp
#   src/scenes/polyline.cpp
#   src/scenes/mesh.cpp
#   src/solvers/wos.cpp
#   src/solvers/wost.cpp
#   src/solvers/wost_3d.cpp
#   src/solvers/wost_one_step.cpp
#   src/solvers/wost_original.cpp
#   src/solvers/wos_grad.cpp
#   src/solvers/wost_grad.cpp
#   src/solvers/wost_der.cpp
#   src/scenes/slab.cpp
#   src/test_scenes/polyline_no_source.cpp
#   src/test_scenes/polyline_with_source.cpp
#   src/test_scenes/mesh_with_source.cpp
#   src/test_scenes/mesh_no_source.cpp
#   src/utils/BVH.cpp
#   src/aggregates/bvh.cpp
#   src/diff_wost_bvh_bindings.cpp
# #   src/main.cpp
    src/diff_wost.cpp
    src/python/bind_bounding_volume.cpp
    src/python/bind_bvh.cpp
    src/python/bind_silhouette.cpp
    src/python/bind_snch.cpp
)

target_link_libraries(diff_wost_ext PRIVATE Eigen3::Eigen)

# Set RPATH for both Apple and Linux
if(APPLE)
  set_target_properties(diff_wost_ext PROPERTIES
      INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(diff_wost_ext PROPERTIES
      INSTALL_RPATH "$ORIGIN")
endif()

nanobind_add_stub(
    diff_wost_ext_stub
    MODULE diff_wost_ext
    OUTPUT diff_wost_ext.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:diff_wost_ext>
    MARKER_FILE py.typed
    DEPENDS diff_wost_ext
)

install(TARGETS diff_wost_ext LIBRARY DESTINATION .)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed 
              ${CMAKE_CURRENT_BINARY_DIR}/diff_wost_ext.pyi DESTINATION .)
