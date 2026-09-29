#pragma once

#include <cstddef>
#include <vector>

// One Jacobi step on row-major n x n host arrays: maps both to the device, runs the kernel,
// and copies dst back. Writes only the interior of dst; its boundary cells stay as they are.
void jacobi_step(const double* src, double* dst, std::size_t n);

// Owns two n x n fields that stay mapped on the device for the life of the object.
// step() runs on the device only; sync_to_host() is the only copy back to the host.
class JacobiSolver {
public:
    JacobiSolver(const double* initial, std::size_t n);
    ~JacobiSolver();

    JacobiSolver(const JacobiSolver&) = delete;
    JacobiSolver& operator=(const JacobiSolver&) = delete;

    void step(int iterations);
    void sync_to_host(double* out) const;

    std::size_t size() const { return n_; }
    long steps() const { return steps_; }

private:
    std::size_t n_;
    long steps_ = 0;
    std::vector<double> a_;
    std::vector<double> b_;
    double* cur_;
    double* next_;
};
