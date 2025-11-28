#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vol::calib {

struct ParameterSpec {
    std::string name;
    double lower;
    double upper;
};

struct CalibratorConfig {
    int max_local_iters = 400;
    int global_restarts = 4;
    double grad_tol = 1e-2;
    double cond_tol = 1e8;
    double penalty_weight = 1.0;
    std::uint64_t seed = 42;
    std::vector<double> initial_guess;
    bool require_convergence = true;
};

struct CalibrationDiagnostics {
    int iterations = 0;
    int best_restart = 0;
    double grad_norm = 0.0;
    double cond_proxy = 0.0;
    double penalty_value = 0.0;
};

struct CalibrationResult {
    std::vector<double> params;
    double objective = 0.0;
    bool converged = false;
    CalibrationDiagnostics diagnostics;
};

struct CalibrationProblem {
    std::vector<ParameterSpec> parameters;
    std::function<void(const std::vector<double>&, double&, std::vector<double>&)> objective;
    std::function<void(const std::vector<double>&, double&, std::vector<double>&)> penalty;
};

CalibrationResult run_calibration(const CalibrationProblem& problem,
                                  const CalibratorConfig& cfg);

} // namespace vol::calib
