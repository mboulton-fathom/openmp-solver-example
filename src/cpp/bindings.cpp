#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <new>
#include <stdexcept>

#include "jacobi.hpp"

namespace nb = nanobind;

using InGrid = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OutGrid = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using NumpyGrid = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

namespace {

template <typename Array>
std::size_t square_size(const Array& grid) {
    if (grid.shape(0) != grid.shape(1)) {
        throw std::invalid_argument("grid must be square");
    }
    if (grid.shape(0) < 3) {
        throw std::invalid_argument("grid size must be at least 3");
    }
    return grid.shape(0);
}

// Allocates a new numpy grid and calls fill(data) to set its values.
template <typename Fill>
NumpyGrid new_grid(std::size_t n, Fill fill) {
    double* data = new double[n * n];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    fill(data);
    return NumpyGrid(data, {n, n}, owner);
}

}  // namespace

NB_MODULE(_jacobi, m) {
    m.doc() = "OpenMP Jacobi 2D diffusion solver";

    m.def(
        "jacobi_step",
        [](InGrid src) {
            const std::size_t n = square_size(src);
            return new_grid(n, [&](double* dst) {
                std::copy(src.data(), src.data() + n * n, dst);
                nb::gil_scoped_release release;
                jacobi_step(src.data(), dst, n);
            });
        },
        nb::arg("grid"),
        "Return one Jacobi step of a square float64 grid. Maps, runs and unmaps on each call.");

    nb::class_<JacobiSolver>(m, "Solver",
                             "Jacobi solver whose fields stay on the device between calls.")
        .def(
            "__init__",
            [](JacobiSolver* self, InGrid initial) {
                new (self) JacobiSolver(initial.data(), square_size(initial));
            },
            nb::arg("initial"))
        .def("step", &JacobiSolver::step, nb::arg("iterations") = 1,
             nb::call_guard<nb::gil_scoped_release>(), "Advance the field on the device.")
        .def(
            "sync_to_host",
            [](const JacobiSolver& self, OutGrid out) {
                if (out.shape(0) != self.size() || out.shape(1) != self.size()) {
                    throw std::invalid_argument("out has the wrong shape");
                }
                nb::gil_scoped_release release;
                self.sync_to_host(out.data());
            },
            nb::arg("out").noconvert(), "Copy the current field into an existing float64 array.")
        .def(
            "to_numpy",
            [](const JacobiSolver& self) {
                return new_grid(self.size(), [&](double* data) {
                    nb::gil_scoped_release release;
                    self.sync_to_host(data);
                });
            },
            "Copy the current field into a new numpy array.")
        .def_prop_ro("size", &JacobiSolver::size)
        .def_prop_ro("steps", &JacobiSolver::steps);
}
