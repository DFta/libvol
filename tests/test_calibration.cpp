#include <catch2/catch_all.hpp>
#include <cmath>

#include "libvol/calib/calibrator.hpp"
#include "libvol/calib/heston.hpp"
#include "libvol/models/heston.hpp"

using Catch::Approx;

TEST_CASE("Generic calibrator minimizes convex quadratic", "[calibration]") {
    vol::calib::CalibrationProblem problem;
    problem.parameters = {
        {"x", -2.0, 2.0},
        {"y", -2.0, 2.0}
    };
    problem.objective = [](const std::vector<double>& theta, double& f, std::vector<double>& g) {
        const double x = theta[0];
        const double y = theta[1];
        f = 0.5 * ((x - 0.5) * (x - 0.5) + 4.0 * (y + 0.25) * (y + 0.25));
        g.assign(2, 0.0);
        g[0] = (x - 0.5);
        g[1] = 4.0 * (y + 0.25);
    };

    vol::calib::CalibratorConfig cfg;
    cfg.global_restarts = 5;
    cfg.max_local_iters = 200;
    cfg.grad_tol = 1e-10;
    cfg.initial_guess = { -1.5, 1.2 };

    const auto res = vol::calib::run_calibration(problem, cfg);
    REQUIRE(res.converged);
    REQUIRE(res.params.size() == 2);
    CHECK(res.params[0] == Approx(0.5).margin(1e-6));
    CHECK(res.params[1] == Approx(-0.25).margin(1e-6));
    CHECK(res.diagnostics.grad_norm < 1e-6);
}

TEST_CASE("Heston smile calibration fits synthetic smile prices", "[calibration][heston]") {
    const vol::heston::Params truth{2.1, 0.05, 0.4, -0.6, 0.045};
    std::vector<vol::OptionSpec> opts;
    std::vector<double> mids;

    const double S = 100.0;
    const double r = 0.01;
    const double q = 0.0;
    const double T = 1.0;

    // Generate synthetic call prices from the true parameters
    for (double k = 70.0; k <= 130.0; k += 10.0) {
        vol::OptionSpec opt{S, k, r, q, T, true};
        opts.push_back(opt);
        mids.push_back(vol::heston::price_cf(S, k, r, q, T, truth, true, 96));
    }

    vol::calib::HestonSlice slice;
    slice.options = opts;
    slice.mids    = mids;
    slice.quad_order = 96;
    slice.bounds = {
        {"kappa", 1.0, 3.5},
        {"theta", 0.02, 0.08},
        {"sigma", 0.2,  0.7},
        {"rho",   -0.9, -0.2},
        {"v0",    0.02, 0.08}
    };

    vol::calib::CalibratorConfig cfg;
    cfg.global_restarts   = 10;
    cfg.max_local_iters   = 1200;
    cfg.grad_tol          = 1e-2;       // realistic for FD gradients
    cfg.cond_tol          = 5e9;
    cfg.penalty_weight    = 1e-6;
    cfg.seed              = 20241123u;
    cfg.initial_guess     = {2.0, 0.05, 0.45, -0.5, 0.05};
    cfg.require_convergence = false;    // return best solution; test checks quality

    vol::calib::HestonCalibrationResult output;
    REQUIRE_NOTHROW(output = vol::calib::calibrate_heston_smile(slice, cfg));

    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < opts.size(); ++i) {
        const auto& opt = opts[i];
        double model = vol::heston::price_cf(
            opt.S, opt.K, opt.r, opt.q, opt.T,
            output.params, opt.is_call, slice.quad_order
        );
        double err = std::abs(model - mids[i]);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }

    CHECK(max_abs_err < 5e-2);

    CHECK(output.params.kappa >= slice.bounds[0].lower);
    CHECK(output.params.kappa <= slice.bounds[0].upper);
    CHECK(output.params.theta >= slice.bounds[1].lower);
    CHECK(output.params.theta <= slice.bounds[1].upper);
    CHECK(output.params.sigma >= slice.bounds[2].lower);
    CHECK(output.params.sigma <= slice.bounds[2].upper);
    CHECK(output.params.rho   >= slice.bounds[3].lower);
    CHECK(output.params.rho   <= slice.bounds[3].upper);
    CHECK(output.params.v0    >= slice.bounds[4].lower);
    CHECK(output.params.v0    <= slice.bounds[4].upper);
}
