#include "jacobi.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {

// src and dst must already be mapped to the device.
void step_kernel(const double* src, double* dst, std::size_t n) {
    #pragma omp target teams distribute parallel for collapse(2)
    for (std::size_t i = 1; i < n - 1; ++i) {
        for (std::size_t j = 1; j < n - 1; ++j) {
            dst[i * n + j] = 0.25 * (src[(i - 1) * n + j] + src[(i + 1) * n + j] +
                                     src[i * n + j - 1] + src[i * n + j + 1]);
        }
    }
}

}  // namespace

void jacobi_step(const double* src, double* dst, std::size_t n) {
    if (n < 3) {
        return;
    }
    const std::size_t size = n * n;
    #pragma omp target data map(to: src[0:size]) map(tofrom: dst[0:size])
    step_kernel(src, dst, n);
}

JacobiSolver::JacobiSolver(const double* initial, std::size_t n)
    : n_(n), a_(initial, initial + n * n), b_(a_), cur_(a_.data()), next_(b_.data()) {
    if (n < 3) {
        throw std::invalid_argument("grid size must be at least 3");
    }
    double* a = a_.data();
    double* b = b_.data();
    const std::size_t size = n_ * n_;
    #pragma omp target enter data map(to: a[0:size], b[0:size])
}

JacobiSolver::~JacobiSolver() {
    double* a = a_.data();
    double* b = b_.data();
    const std::size_t size = n_ * n_;
    #pragma omp target exit data map(delete: a[0:size], b[0:size])
}

void JacobiSolver::step(int iterations) {
    if (iterations < 0) {
        throw std::invalid_argument("iterations must not be negative");
    }
    for (int it = 0; it < iterations; ++it) {
        step_kernel(cur_, next_, n_);
        std::swap(cur_, next_);
    }
    steps_ += iterations;
}

void JacobiSolver::sync_to_host(double* out) const {
    double* cur = cur_;
    const std::size_t size = n_ * n_;
    #pragma omp target update from(cur[0:size])
    std::copy(cur, cur + size, out);
}
