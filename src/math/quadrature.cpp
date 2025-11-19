#include "libvol/math/quadrature.hpp"

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace vol::quad {

namespace {

constexpr double EPS = 1e-14;
constexpr int MAX_ITERS = 64;

struct PolyVals {
    double Ln;
    double Lnm1;
};

PolyVals eval_laguerre(int n, double x) {
    double Lnm1 = 0.0;
    double Ln = 1.0;
    for (int k = 1; k <= n; ++k) {
        const double coeff = (2.0 * k - 1.0 - x);
        const double Lnp1 = (coeff * Ln - (k - 1.0) * Lnm1) / static_cast<double>(k);
        Lnm1 = Ln;
        Ln = Lnp1;
    }
    return {Ln, Lnm1};
}

double compute_weight(int n, double x, double Ln, double Lnm1) {
    const double Lnp1 = ((2.0 * n + 1.0 - x) * Ln - n * Lnm1) / (n + 1.0);
    const double denom = (n + 1.0) * (n + 1.0) * Lnp1 * Lnp1;
    return x / denom;
}

GaussLaguerreRule build_rule(int n) {
    if (n <= 0) {
        throw std::invalid_argument("Gauss-Laguerre order must be positive");
    }
    GaussLaguerreRule rule;
    rule.nodes.resize(n);
    rule.weights.resize(n);

    double z = 0.0;
    for (int i = 1; i <= n; ++i) {
        if (i == 1) {
            z = 3.0 / (1.0 + 2.4 * n);
        } else if (i == 2) {
            z += 15.0 / (1.0 + 1.7 * n);
        } else {
            const double ai = static_cast<double>(i - 2);
            z += ((1.0 + 2.55 * ai) / (1.9 * ai)) * (z - rule.nodes[i - 3]);
        }

        for (int it = 0; it < MAX_ITERS; ++it) {
            const PolyVals vals = eval_laguerre(n, z);
            const double Ln = vals.Ln;
            const double Lnm1 = vals.Lnm1;
            const double deriv = (n * Ln - n * Lnm1) / z;
            const double delta = Ln / deriv;
            z -= delta;
            if (std::abs(delta) <= EPS) {
                break;
            }
        }

        const PolyVals vals = eval_laguerre(n, z);
        rule.nodes[i - 1] = z;
        rule.weights[i - 1] = compute_weight(n, z, vals.Ln, vals.Lnm1);
    }
    return rule;
}

} // namespace

const GaussLaguerreRule& gauss_laguerre_rule(int n) {
    static std::mutex mtx;
    static std::unordered_map<int, GaussLaguerreRule> cache;

    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(n);
    if (it != cache.end()) {
        return it->second;
    }
    auto [inserted_it, inserted] = cache.emplace(n, GaussLaguerreRule{});
    inserted_it->second = build_rule(n);
    return inserted_it->second;
}

} // namespace vol::quad
