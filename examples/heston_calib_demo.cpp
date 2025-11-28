#include "libvol/calib/heston.hpp"
#include "libvol/models/heston.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        const double S = 100.0;
        const double r = 0.01;
        const double q = 0.0;
        const double T = 0.75;
        const bool is_call = true;

        const vol::heston::Params trueparams{1.7, 0.045, 0.6, -0.55, 0.05};

        std::vector<vol::OptionSpec> opts;
        std::vector<double> mids;

        // Generate synthetic prices from the "true" Heston params
        for (double k = 60.0; k <= 140.0; k += 10.0) {
            vol::OptionSpec opt{S, k, r, q, T, is_call};
            const double price = vol::heston::price_cf(S, k, r, q, T, trueparams, is_call, 96);
            opts.push_back(opt);
            mids.push_back(price);
        }

        vol::calib::HestonSlice slice;
        slice.options    = opts;
        slice.mids       = mids;
        slice.quad_order = 96;
        slice.bounds = {
            {"kappa", 0.5, 3.0},
            {"theta", 0.01, 0.10},
            {"sigma", 0.1,  1.5},
            {"rho",   -0.9, -0.1},
            {"v0",    0.01, 0.10}
        };

        vol::calib::CalibratorConfig cfg;
        cfg.global_restarts     = 8;
        cfg.max_local_iters     = 800;
        cfg.grad_tol            = 1e-2;       // realistic for FD gradients
        cfg.cond_tol            = 5e9;
        cfg.penalty_weight      = 1e-6;
        cfg.seed                = 20241123u;
        cfg.initial_guess       = {1.5, 0.04, 0.5, -0.5, 0.04};
        cfg.require_convergence = false;      // return best effort; we inspect diagnostics

        const auto calibrated = vol::calib::calibrate_heston_smile(slice, cfg);

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "Ground truth params:\n"
                  << "  kappa=" << trueparams.kappa
                  << " theta=" << trueparams.theta
                  << " sigma=" << trueparams.sigma
                  << " rho="   << trueparams.rho
                  << " v0="    << trueparams.v0 << "\n\n";

        std::cout << "Calibrated params:\n"
                  << "  kappa=" << calibrated.params.kappa
                  << " theta=" << calibrated.params.theta
                  << " sigma=" << calibrated.params.sigma
                  << " rho="   << calibrated.params.rho
                  << " v0="    << calibrated.params.v0 << "\n\n";

        std::cout << "Solver diagnostics:\n"
                  << "  objective   = " << calibrated.solver.objective << "\n"
                  << "  grad_norm   = " << calibrated.solver.diagnostics.grad_norm << "\n"
                  << "  penalty     = " << calibrated.solver.diagnostics.penalty_value << "\n"
                  << "  best_restart= " << calibrated.solver.diagnostics.best_restart + 1 << "\n";

        // Also print max pricing error to sanity-check the fit
        double max_abs_err = 0.0;
        for (std::size_t i = 0; i < opts.size(); ++i) {
            const auto& opt = opts[i];
            const double model = vol::heston::price_cf(
                opt.S, opt.K, opt.r, opt.q, opt.T,
                calibrated.params, opt.is_call, slice.quad_order
            );
            const double err = std::abs(model - mids[i]);
            if (err > max_abs_err) {
                max_abs_err = err;
            }
        }
        std::cout << "\nMax abs price error: " << max_abs_err << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
