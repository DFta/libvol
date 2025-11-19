#include "libvol/models/binom.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace vol::binom {

namespace {
    inline double intrinsic(bool is_call, double S, double K) {
        return is_call ? std::max(0.0, S - K) : std::max(0.0, K - S);
    }

    struct CRRParams {
        double dt;
        double u, d, p, disc;
    };

    inline CRRParams make_crr(double r, double q, double T, double vol, int steps) {
        const double dt = T / static_cast<double>(steps);
        const double vs = vol * std::sqrt(dt);
        const double u  = std::exp(vs);
        const double d  = 1.0 / u;
        const double a  = std::exp((r - q) * dt);
        const double p  = (a - d) / (u - d);
        const double disc = std::exp(-r * dt);
        return {dt, u, d, p, disc};
    }

    //CRR engine
    double price_crr(double S0, double K, double r, double q, double T, double vol,
                    int steps, bool is_call, bool is_american, int* early_ex_step_out){
        if (steps <= 0) {
            // degenerate: fallback to intrinsic at expiry
            return std::exp(-r*T) * intrinsic(is_call, S0 * std::exp((r - q - 0.5*vol*vol)*T), K);
        }
        const auto p = make_crr(r, q, T, vol, steps);
        double prob = p.p;
        if (!(prob >= 0.0 && prob <= 1.0) || !std::isfinite(prob)) {
            prob = std::min(1.0, std::max(0.0, prob));
        }       

        // Precomputing terminal asset prices and option values
        std::vector<double> V(steps + 1);
        std::vector<double> S(steps + 1);

        double Sj = S0 * std::pow(p.d, steps);
        const double ud = p.u / p.d;
        for (int j = 0; j <= steps; ++j) {
            S[j] = Sj;
            V[j] = intrinsic(is_call, S[j], K);
            Sj *= ud;
        }

        int earliest_ex_step = -1;

        //backward induction
        for (int n = steps - 1; n >= 0; --n) {
            for (int j = 0; j <= n; ++j) {
                const double cont = p.disc * (prob * V[j + 1] + (1.0 - prob) * V[j]);
                if (is_american) {
                    const double S_nj = S[j] / p.d;
                    const double exer = intrinsic(is_call, S_nj, K);
                    const double vn = std::max(cont, exer);
                    if (earliest_ex_step == -1 && exer > cont + 1e-16) {
                        earliest_ex_step = n;
                    }
                    V[j] = vn;
                    S[j] = S_nj;
                } else {
                    V[j] = cont;
                    S[j] = S[j] / p.d;
                }
            }
        }

        if (early_ex_step_out) *early_ex_step_out = earliest_ex_step;
        return V[0];
    }

    inline double safe_dt_for_theta(double T) {
        const double min_dt = 1.0 / 3650.0;       // ~0.1 day
        const double frac_T = 0.05 * T;           // 5% of T
        const double max_abs = 5.0 / 365.0;        // cap at ~5 days
        return std::min(std::max(min_dt, frac_T), std::max(min_dt, max_abs));
    }
} // namespace

double price(double S, double K, double r, double q, double T, double vol, int steps, bool is_call, bool is_american){
    return price_crr(S, K, r, q, T, vol, steps, is_call, is_american, nullptr);
}

BinomialResult price_w_info(double S, double K, double r, double q, double T,
                            double vol, int steps, bool is_call, bool is_american)
{
    int earliest = -1;
    const double px = price_crr(S, K, r, q, T, vol, steps, is_call, is_american, &earliest);
    return {px, earliest};
}

PriceGreeks price_greeks(double S, double K, double r, double q, double T, double vol, int steps, bool is_call, bool is_american)
{
    const double base = price(S, K, r, q, T, vol, steps, is_call, is_american);
    //Step Sizes
    //Small step for Delta Rho & Vega (precision)
    const double hS   = std::max(1e-8, 1e-4 * std::max(1.0, S));
    const double hvol = std::max(1e-8, 1e-4 * std::max(1.0, vol));
    const double hr   = std::max(1e-8, 1e-5 * std::max(1.0, std::fabs(r)));
    
    // Large step specifically for Binomial Gamma (smoothing), 1% of spot
    const double hS_binom = std::max(1e-4, 0.01 * S); 
    
    double hT = safe_dt_for_theta(T); 

    //Delta (Standard Finite Difference with small hS)
    const double p_delta_up = price(S + hS, K, r, q, T, vol, steps, is_call, is_american);
    const double p_delta_dn = price(std::max(1e-12, S - hS), K, r, q, T, vol, steps, is_call, is_american);
    const double delta = (p_delta_up - p_delta_dn) / (2.0 * hS);

    //Gamma (Wide Finite Difference with hS_binom)
    const double p_gamma_up = price(S + hS_binom, K, r, q, T, vol, steps, is_call, is_american);
    const double p_gamma_dn = price(std::max(1e-12, S - hS_binom), K, r, q, T, vol, steps, is_call, is_american);
    const double gamma = (p_gamma_up - 2.0 * base + p_gamma_dn) / (hS_binom * hS_binom);

    //Vega 
    const double p_vp = price(S, K, r, q, T, vol + hvol, steps, is_call, is_american);
    const double p_vm = price(S, K, r, q, T, std::max(1e-12, vol - hvol), steps, is_call, is_american);
    const double vega = (p_vp - p_vm) / (2.0 * hvol);

    //Rho 
    const double p_rp = price(S, K, r + hr, q, T, vol, steps, is_call, is_american);
    const double p_rm = price(S, K, std::max(-0.999, r - hr), q, T, vol, steps, is_call, is_american);
    const double rho = (p_rp - p_rm) / (2.0 * hr);

    //Theta
    double theta;
    if (T > hT) {
        const double p_tp = price(S, K, r, q, T + hT, vol, steps, is_call, is_american);
        const double p_tm = price(S, K, r, q, T - hT, vol, steps, is_call, is_american);
        theta = (p_tm - p_tp) / (2.0 * hT); 
    } else {
        const double p_tp = price(S, K, r, q, T + hT, vol, steps, is_call, is_american);
        theta = -(p_tp - base) / hT; 
    }

    return {base, delta, gamma, vega, theta, rho};
}

} // namespace vol::binom
