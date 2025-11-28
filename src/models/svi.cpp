#include "libvol/models/svi.hpp"
#include "libvol/calib/calibrator.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace vol::svi {

    // p = {a, b, rho, m, sigma}
    double total_variance(double k, const Params& p) {
        const double a = p[0], b = p[1], rho = p[2], m = p[3], sigma = p[4];
        const double x = k - m;
        const double R = std::sqrt(x * x + sigma * sigma);
        return a + b * (rho * x + R);
    }

    bool basic_no_arb(const Params& p) {
        const double b = p[1], rho = p[2], sigma = p[4];
        if (!(std::isfinite(b) && std::isfinite(rho) && std::isfinite(sigma))) return false;
        if (b <= 0.0) return false;
        if (std::abs(rho) >= 1.0) return false;
        if (sigma <= 0.0) return false;
        return true;
    }

    static inline double clamp(double x, double lo, double hi) {
        return std::max(lo, std::min(hi, x));
    }

    static inline void linreg_slope(const std::vector<double>& x, const std::vector<double>& y, std::size_t i0, std::size_t i1, double& slope_out) {
        // fit y = s*x + c on [i0, i1] (inclusive)
        const std::size_t n = (i1 >= i0) ? (i1 - i0 + 1) : 0;
        if (n < 2) { slope_out = 0.0; return; }
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (std::size_t i = i0; i <= i1; ++i) {
            sx  += x[i];
            sy  += y[i];
            sxx += x[i] * x[i];
            sxy += x[i] * y[i];
        }
        const double n_d = static_cast<double>(n);
        const double den = (n_d * sxx - sx * sx);
        slope_out = (std::abs(den) > 0.0) ? (n_d * sxy - sx * sy) / den : 0.0;
    }

    static inline double local_quadratic_curvature(const std::vector<double>& k, const std::vector<double>& w, std::size_t idx_min) {
        const std::size_t n = k.size();
        if (n < 3) return 0.0;

        //5 closest points
        std::vector<std::pair<double, std::size_t>> pts;
        pts.reserve(n);
        const double km = k[idx_min];
        for (std::size_t i = 0; i < n; ++i) {
            pts.emplace_back(std::abs(k[i] - km), i);
        }
        
        //top 5
        std::size_t take = std::min<std::size_t>(5, n);
        std::partial_sort(pts.begin(), pts.begin() + take, pts.end());

        //Least Squares)
        double S0 = 0, S1 = 0, S2 = 0, S3 = 0, S4 = 0;
        double Sy = 0, Sxy = 0, Sx2y = 0;

        for (std::size_t t = 0; t < take; ++t) {
            const std::size_t original_idx = pts[t].second;
            const double x = k[original_idx] - km; 
            const double y = w[original_idx];
            const double x2 = x * x;
            
            S0 += 1.0; S1 += x; S2 += x2; S3 += x2 * x; S4 += x2 * x2;
            Sy += y;   Sxy += x * y; Sx2y += x2 * y;
        }

        //determinant of coefficient matrix
        const double det = S0 * (S2 * S4 - S3 * S3) - 
                        S1 * (S1 * S4 - S2 * S3) + 
                        S2 * (S1 * S3 - S2 * S2);

        if (std::abs(det) < 1e-12) return 0.0;

        //determinant replacing the 3rd column with Y vector
        const double det_A = S0 * (S2 * Sx2y - S3 * Sxy) - 
                            S1 * (S1 * Sx2y - S2 * Sxy) + 
                            Sy * (S1 * S3 - S2 * S2);

        //curvature is 2 * a
        return 2.0 * (det_A / det);
    }

// Per-slice
    Params fit_raw_svi(const std::vector<double>& k, const std::vector<double>& w_mkt, const std::vector<double>& wts_in)
    {
        const std::size_t n = std::min(k.size(), w_mkt.size());
        if (n == 0) {
            return Params{1e-10, 0.1, 0.0, 0.0, 0.2};
        }

        if (n < 5) {
            const double kmin = *std::min_element(k.begin(), k.begin() + n);
            const double kmax = *std::max_element(k.begin(), k.begin() + n);
            const double wmin = *std::min_element(w_mkt.begin(), w_mkt.begin() + n);
            const double b0 = 0.1;
            const double sigma0 = std::max(1e-3, 0.2 * (kmax - kmin));
            const double a0 = std::max(1e-10, wmin - b0 * sigma0);
            return Params{ a0, b0, 0.0, 0.5 * (kmin + kmax), sigma0 };
        }

        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](std::size_t i, std::size_t j){ return k[i] < k[j]; });

        std::vector<double> k_sorted(n), w_sorted(n), wt(n, 1.0);
        for (std::size_t t = 0; t < n; ++t) {
            const std::size_t orig = idx[t];
            k_sorted[t] = k[orig];
            w_sorted[t] = w_mkt[orig];
            if (!wts_in.empty()) wt[t] = wts_in[orig];
        }

        const double kmin = k_sorted.front();
        const double kmax = k_sorted.back();
        auto [min_it, max_it] = std::minmax_element(w_sorted.begin(), w_sorted.end());
        const double wrange = *max_it - *min_it;
        const double range_k = std::max(1e-6, kmax - kmin);

        std::size_t i_min = 0;
        for (std::size_t i = 1; i < n; ++i) if (w_sorted[i] < w_sorted[i_min]) i_min = i;
        const double m0 = k_sorted[i_min];
        const double w_at_min = w_sorted[i_min];

        const std::size_t wing = std::max<std::size_t>(2, n / 5);
        double sL = 0.05, sR = 0.05;
        linreg_slope(k_sorted, w_sorted, 0, std::min(wing, n-1), sL);
        linreg_slope(k_sorted, w_sorted, (n > wing ? n - wing : 0), n - 1, sR);
        sL = std::max(1e-4, std::abs(sL));
        sR = std::max(1e-4, std::abs(sR));

        double b0   = 0.5 * (sL + sR);
        double rho0 = (sR - sL) / std::max(1e-12, (sR + sL));
        b0   = clamp(b0,   1e-6, 10.0);
        rho0 = clamp(rho0, -0.95, 0.95);

        double c2 = local_quadratic_curvature(k_sorted, w_sorted, i_min);
        double sigma0 = (c2 > 1e-6) ? clamp(b0 / c2, 1e-4, 2.0) : clamp(0.2 * range_k, 1e-4, 2.0);

        double a0 = std::max(1e-10, w_at_min - b0 * sigma0);

        const double a_max = std::max(1.0, 5.0 * (wrange > 0.0 ? wrange : (w_at_min + b0 * sigma0 + 1.0)));
        const std::vector<double> x0 = { a0, b0, rho0, m0, sigma0 };
        const std::vector<double> lb = { 1e-12, 1e-8, -0.999, kmin - 1.0 * range_k, 1e-6 };
        const std::vector<double> ub = { a_max,  10.0,  0.999, kmax + 1.0 * range_k,  5.0  };

        calib::CalibrationProblem problem;
        problem.parameters = {
            {"a", lb[0], ub[0]},
            {"b", lb[1], ub[1]},
            {"rho", lb[2], ub[2]},
            {"m", lb[3], ub[3]},
            {"sigma", lb[4], ub[4]}
        };

        problem.objective = [=,&k_sorted,&w_sorted,&wt](const std::vector<double>& x,
                                                        double& f,
                                                        std::vector<double>& g) {
            const double a = x[0], b = x[1], rho = x[2], m = x[3], sigma = x[4];
            const double eps = 1e-12;
            const double rho_c = clamp(rho, -0.999, 0.999);
            const double b_pos = std::max(b, 1e-12);
            const double s_pos = std::max(sigma, 1e-12);

            double sumw = 0.0, obj = 0.0;
            double ga = 0.0, gb = 0.0, gr = 0.0, gm = 0.0, gs = 0.0;

            for (std::size_t i = 0; i < k_sorted.size(); ++i) {
                const double wi = (i < wt.size() ? std::max(0.0, wt[i]) : 1.0);
                if (wi <= 0.0) continue;

                const double xk = k_sorted[i] - m;
                const double R  = std::sqrt(xk * xk + s_pos * s_pos);
                const double w_model = a + b_pos * (rho_c * xk + R);
                const double r  = (w_model - w_sorted[i]);

                sumw += wi;
                obj  += wi * r * r;

                const double dw_da = 1.0;
                const double dw_db = (rho_c * xk + R);
                const double dw_dr = b_pos * xk;
                const double dw_dm = b_pos * (-rho_c - xk / std::max(R, eps));
                const double dw_ds = b_pos * (s_pos / std::max(R, eps));

                ga += wi * r * dw_da;
                gb += wi * r * dw_db;
                gr += wi * r * dw_dr;
                gm += wi * r * dw_dm;
                gs += wi * r * dw_ds;
            }

            if (sumw <= 0.0) {
                f = 0.0;
                g.assign(5, 0.0);
                return;
            }

            const double inv = 1.0 / sumw;
            f = 0.5 * obj * inv;

            g.assign(5, 0.0);
            g[0] = ga * inv;
            g[1] = gb * inv;
            g[2] = gr * inv;
            g[3] = gm * inv;
            g[4] = gs * inv;
        };

        problem.penalty = [](const std::vector<double>& x, double& pen, std::vector<double>& gp) {
            gp.assign(5, 0.0);
            pen = 0.0;
            const double b = x[1];
            const double rho = x[2];
            const double sigma = x[4];

            const auto add_quad = [&](double violation, std::size_t idx, double dir = 1.0) {
                if (violation > 0.0) {
                    const double scale = 10.0;
                    pen += 0.5 * scale * violation * violation;
                    gp[idx] += scale * violation * dir;
                }
            };

            add_quad(std::max(0.0, 1e-6 - b), 1, -1.0);
            add_quad(std::max(0.0, std::abs(rho) - 0.999), 2, (rho >= 0.0 ? 1.0 : -1.0));
            add_quad(std::max(0.0, 1e-6 - sigma), 4, -1.0);
        };

        calib::CalibratorConfig solver_cfg;
        solver_cfg.max_local_iters = 500;
        solver_cfg.global_restarts = 6;
        solver_cfg.grad_tol = 1e-8;
        solver_cfg.penalty_weight = 1e-4;
        solver_cfg.seed = 1729;
        solver_cfg.initial_guess = x0;

        const auto result = calib::run_calibration(problem, solver_cfg);
        Params candidate{};
        if (result.params.size() == 5) {
            candidate = Params{ result.params[0], result.params[1], result.params[2],
                                result.params[3], result.params[4] };
        } else {
            candidate = Params{ clamp(x0[0], lb[0], ub[0]),
                                clamp(x0[1], lb[1], ub[1]),
                                clamp(x0[2], lb[2], ub[2]),
                                clamp(x0[3], lb[3], ub[3]),
                                clamp(x0[4], lb[4], ub[4]) };
        }

        if (!basic_no_arb(candidate)) {
            candidate = Params{ clamp(x0[0], lb[0], ub[0]),
                                clamp(x0[1], lb[1], ub[1]),
                                clamp(x0[2], lb[2], ub[2]),
                                clamp(x0[3], lb[3], ub[3]),
                                clamp(x0[4], lb[4], ub[4]) };
        }

        return candidate;
    }

} // namespace vol::svi
