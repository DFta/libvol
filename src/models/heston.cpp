#include "libvol/models/heston.hpp"

#include "libvol/core/constants.hpp"
#include "libvol/math/quadrature.hpp"
#include "libvol/models/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace vol::heston {

namespace {

using Complex = std::complex<double>;
constexpr Complex I(0.0, 1.0);

Complex characteristic(const Complex& u,
                        double logS,
                        double drift,
                        double T,
                        const Params& p) {
    const double sigma = p.sigma;

    if (sigma <= 0.0) {
        throw std::invalid_argument("Heston vol-of-vol sigma must be positive");
    }
    
    const double sigma2 = sigma * sigma;
    const Complex iu = I * u;
    const Complex beta = static_cast<double>(p.kappa) - p.rho * sigma * iu;
    const Complex d = std::sqrt(beta * beta + sigma2 * (iu + u * u));
    const Complex g = (beta - d) / (beta + d);
    const Complex exp_term = std::exp(-d * T);
    const Complex C = (p.kappa * p.theta / sigma2) *
        ((beta - d) * T - 2.0 * std::log((Complex(1.0) - g * exp_term) / (Complex(1.0) - g)));
    const Complex D = (beta - d) / sigma2 * ((Complex(1.0) - exp_term) / (Complex(1.0) - g * exp_term));
    return std::exp(C + D * p.v0 + iu * (logS + drift));
}

double intrinsic(double S, double K, bool is_call) {
    return is_call ? std::max(0.0, S - K) : std::max(0.0, K - S);
}

} // namespace

double price_cf(double S,
                double K,
                double r,
                double q,
                double T,
                const Params& params,
                bool is_call,
                int n_gl) {
    if (S <= 0.0 || K <= 0.0) {
        throw std::invalid_argument("Spot and strike must be positive");
    }
    if (T <= 0.0) {
        return intrinsic(S, K, is_call);
    }
    if (n_gl <= 0) {
        throw std::invalid_argument("Gauss-Laguerre order must be positive");
    }
    if (params.sigma <= 0.0) {
        throw std::invalid_argument("Heston vol-of-vol sigma must be positive");
    }

    const double logS = std::log(S);
    const double logK = std::log(K);
    const double drift = (r - q) * T;
    const double disc_r = std::exp(-r * T);
    const double disc_q = std::exp(-q * T);

    const auto& rule = vol::quad::gauss_laguerre_rule(n_gl);
    const Complex phi_minus_i = characteristic(-I, logS, drift, T, params);

    double integral_p1 = 0.0;
    double integral_p2 = 0.0;

    for (std::size_t idx = 0; idx < rule.nodes.size(); ++idx) {
        const double u = rule.nodes[idx];
        const double weight = rule.weights[idx];
        const double exp_scale = std::exp(u);
        const Complex u_c(u, 0.0);
        const Complex phase = std::exp(-I * u * logK);
        const Complex denom = I * u_c;

        const Complex phi_shift = characteristic(u_c - I, logS, drift, T, params);
        const Complex phi_val = characteristic(u_c, logS, drift, T, params);

        const Complex term1 = phase * phi_shift / (denom * phi_minus_i);
        const Complex term2 = phase * phi_val / denom;

        integral_p1 += weight * exp_scale * term1.real();
        integral_p2 += weight * exp_scale * term2.real();
    }

    double P1 = 0.5 + integral_p1 / vol::PI;
    double P2 = 0.5 + integral_p2 / vol::PI;
    P1 = std::clamp(P1, 0.0, 1.0);
    P2 = std::clamp(P2, 0.0, 1.0);

    const double call_price = S * disc_q * P1 - K * disc_r * P2;
    if (is_call) {
        return call_price;
    }
    return call_price - (S * disc_q - K * disc_r);
}

} // namespace vol::heston