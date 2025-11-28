#include "libvol/calib/calibrator.hpp"
#include "libvol/calib/least_squares.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace vol::calib {

namespace {

double penalty_only(const CalibrationProblem& problem,
                    const std::vector<double>& x,
                    double weight) {
    if (!problem.penalty) {
        return 0.0;
    }
    double pen = 0.0;
    std::vector<double> grad(x.size(), 0.0);
    problem.penalty(x, pen, grad);
    return weight * pen;
}

} // namespace

CalibrationResult run_calibration(const CalibrationProblem& problem,
                                  const CalibratorConfig& cfg) {
    CalibrationResult best;
    best.objective = std::numeric_limits<double>::infinity();

    if (!problem.objective || problem.parameters.empty()) {
        return best;
    }

    const std::size_t n = problem.parameters.size();
    std::vector<double> lb(n), ub(n);
    for (std::size_t i = 0; i < n; ++i) {
        lb[i] = problem.parameters[i].lower;
        ub[i] = problem.parameters[i].upper;
        if (lb[i] > ub[i]) {
            std::swap(lb[i], ub[i]);
        }
    }

    auto clamp = [&](std::vector<double>& x) {
        if (x.size() != n) {
            x.resize(n, 0.0);
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isfinite(lb[i])) {
                x[i] = std::max(lb[i], x[i]);
            }
            if (std::isfinite(ub[i])) {
                x[i] = std::min(ub[i], x[i]);
            }
        }
    };

    std::mt19937_64 rng(cfg.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto sample = [&]() {
        std::vector<double> x(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double lo = lb[i];
            const double hi = ub[i];
            if (std::isfinite(lo) && std::isfinite(hi)) {
                x[i] = lo + (hi - lo) * uni(rng);
            } else if (std::isfinite(lo)) {
                const double span = std::max(1.0, std::abs(lo));
                x[i] = lo + span * uni(rng);
            } else if (std::isfinite(hi)) {
                const double span = std::max(1.0, std::abs(hi));
                x[i] = hi - span * uni(rng);
            } else {
                x[i] = 2.0 * uni(rng) - 1.0;
            }
        }
        return x;
    };

    const int restarts = std::max(1, cfg.global_restarts);
    for (int restart = 0; restart < restarts; ++restart) {
        std::vector<double> x0;
        if (restart == 0 && cfg.initial_guess.size() == n) {
            x0 = cfg.initial_guess;
        } else {
            x0 = sample();
        }
        clamp(x0);

        auto f_grad = [&](const std::vector<double>& x, double& obj, std::vector<double>& grad) {
            grad.assign(n, 0.0);
            problem.objective(x, obj, grad);
            if (problem.penalty) {
                std::vector<double> grad_pen(n, 0.0);
                double pen = 0.0;
                problem.penalty(x, pen, grad_pen);
                obj += cfg.penalty_weight * pen;
                for (std::size_t i = 0; i < n; ++i) {
                    grad[i] += cfg.penalty_weight * grad_pen[i];
                }
            }
        };

        auto res = projected_gradient_descent(x0, lb, ub, f_grad, cfg.max_local_iters, cfg.grad_tol);
        if (res.x.size() != n) {
            continue;
        }

        const double penalty_val = penalty_only(problem, res.x, cfg.penalty_weight);
        const bool candidate_converged =
            res.converged &&
            res.grad_norm < cfg.grad_tol &&
            res.cond_proxy < cfg.cond_tol;

        if (res.obj < best.objective) {
            best.params = res.x;
            best.objective = res.obj;
            best.converged = candidate_converged;
            best.diagnostics.iterations = res.iters;
            best.diagnostics.best_restart = restart;
            best.diagnostics.grad_norm = res.grad_norm;
            best.diagnostics.cond_proxy = res.cond_proxy;
            best.diagnostics.penalty_value = penalty_val;
        }
    }

    return best;
}

} // namespace vol::calib
