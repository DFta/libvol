#include "libvol/calib/heston.hpp"

#include "libvol/models/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>

namespace vol::calib {

namespace {

std::vector<ParameterSpec> default_bounds() {
    return {
        {"kappa", 0.1, 8.0},
        {"theta", 1e-4, 0.5},
        {"sigma", 0.05, 2.5},
        {"rho",   -0.999, 0.999},
        {"v0",    1e-4, 0.8}
    };
}

double normalization_weight(const vol::OptionSpec& opt) {
    const double F = opt.S * std::exp((opt.r - opt.q) * opt.T);
    const double k = std::log(std::max(1e-12, opt.K / F));
    return 1.0 / (1.0 + std::abs(k));
}

struct PreparedOption {
    double S;
    double K;
    double r;
    double q;
    double T;
    double mid;
    bool is_call;
    double weight;
};

} // namespace

HestonCalibrationResult calibrate_heston_smile(const HestonSlice& slice,
                                               const CalibratorConfig& user_cfg) {
    if (slice.options.empty() || slice.mids.size() != slice.options.size()) {
        throw std::invalid_argument("calibrate_heston_smile: mismatched option data");
    }

    auto bounds = slice.bounds;
    if (bounds.size() != 5) {
        bounds = default_bounds();
    }

    std::vector<double> lb(5), ub(5);
    for (std::size_t i = 0; i < 5; ++i) {
        lb[i] = bounds[i].lower;
        ub[i] = bounds[i].upper;
    }

    double sum_iv2 = 0.0;
    int iv_count = 0;
    for (std::size_t i = 0; i < slice.options.size(); ++i) {
        const auto& opt = slice.options[i];
        const double mid = slice.mids[i];
        auto iv = vol::bs::implied_vol(opt.S, opt.K, opt.r, opt.q, opt.T, mid, opt.is_call, 0.2, 1e-8);
        if (iv.converged && std::isfinite(iv.iv)) {
            sum_iv2 += iv.iv * iv.iv;
            ++iv_count;
        }
    }
    const double theta_guess = std::max(1e-4, iv_count > 0 ? sum_iv2 / iv_count : 0.04);
    const double v0_guess = theta_guess;
    const std::vector<double> initial = { 1.5, theta_guess, 0.5, -0.5, v0_guess };

    std::vector<PreparedOption> prepared;
    prepared.reserve(slice.options.size());
    for (std::size_t i = 0; i < slice.options.size(); ++i) {
        const auto& opt = slice.options[i];
        PreparedOption p{opt.S, opt.K, opt.r, opt.q, opt.T, slice.mids[i], opt.is_call, normalization_weight(opt)};
        prepared.push_back(p);
    }

    auto objective_only = [&](const std::vector<double>& x) {
        const vol::heston::Params params{x[0], x[1], x[2], x[3], x[4]};
        double sumw = 0.0;
        double err = 0.0;
        for (const auto& opt : prepared) {
            const double model = vol::heston::price_cf(opt.S, opt.K, opt.r, opt.q, opt.T, params, opt.is_call, slice.quad_order);
            const double resid = model - opt.mid;
            const double wt = opt.weight;
            sumw += wt;
            err  += wt * resid * resid;
        }
        return (sumw > 0.0) ? 0.5 * err / sumw : 0.0;
    };

    CalibrationProblem problem;
    problem.parameters = bounds;
    problem.objective = [objective_only, lb, ub](const std::vector<double>& x, double& f, std::vector<double>& g) {
        f = objective_only(x);
        const std::size_t n = x.size();
        g.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double span = 1e-4 * std::max(1.0, std::abs(x[i]));
            std::vector<double> xp = x;
            std::vector<double> xm = x;
            xp[i] = std::min(ub[i], x[i] + span);
            xm[i] = std::max(lb[i], x[i] - span);
            const double delta = xp[i] - xm[i];
            if (delta == 0.0) {
                continue;
            }
            const double fp = objective_only(xp);
            const double fm = objective_only(xm);
            g[i] = (fp - fm) / delta;
        }
    };

    problem.penalty = [](const std::vector<double>& x, double& pen, std::vector<double>& gp) {
        pen = 0.0;
        gp.assign(5, 0.0);
        const double kappa = x[0];
        const double theta = x[1];
        const double sigma = x[2];
        const double rho   = x[3];
        const double v0    = x[4];

        const auto enforce_lower = [&](double value, std::size_t idx) {
            const double violation = std::max(0.0, 1e-4 - value);
            if (violation > 0.0) {
                const double scale = 25.0;
                pen += 0.5 * scale * violation * violation;
                gp[idx] -= scale * violation;
            }
        };

        enforce_lower(kappa, 0);
        enforce_lower(theta, 1);
        enforce_lower(sigma, 2);
        enforce_lower(v0, 4);

        const double rho_violation = std::max(0.0, std::abs(rho) - 0.999);
        if (rho_violation > 0.0) {
            const double scale = 25.0;
            pen += 0.5 * scale * rho_violation * rho_violation;
            gp[3] += scale * rho_violation * (rho >= 0.0 ? 1.0 : -1.0);
        }

        const double feller = sigma * sigma - 2.0 * kappa * theta;
        if (feller > 0.0) {
            const double scale = 5.0;
            pen += 0.5 * scale * feller * feller;
            gp[0] += -scale * feller * 2.0 * theta;
            gp[1] += -scale * feller * 2.0 * kappa;
            gp[2] +=  scale * feller * 2.0 * sigma;
        }
    };

    CalibratorConfig solver_cfg = user_cfg;
    if (solver_cfg.initial_guess.empty()) {
        solver_cfg.initial_guess = initial;
    }
    solver_cfg.global_restarts = std::max(6, solver_cfg.global_restarts);
    solver_cfg.max_local_iters = std::max(400, solver_cfg.max_local_iters);

    auto result = run_calibration(problem, solver_cfg);
    if (result.params.size() != 5) {
        throw std::runtime_error("calibrate_heston_smile: optimizer returned invalid parameter vector");
    }
    if (solver_cfg.require_convergence && !result.converged) {
        std::ostringstream oss;
        oss << "calibrate_heston_smile: optimizer failed (objective=" << result.objective
            << ", grad_norm=" << result.diagnostics.grad_norm
            << ", cond_proxy=" << result.diagnostics.cond_proxy
            << ", restarts=" << result.diagnostics.best_restart + 1 << ")";
        throw std::runtime_error(oss.str());
    }

    vol::heston::Params params{ result.params[0], result.params[1], result.params[2],
                                result.params[3], result.params[4] };

    return { params, result };
}

} // namespace vol::calib
