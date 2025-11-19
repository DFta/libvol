#pragma once

#include <vector>

namespace vol::quad {

struct GaussLaguerreRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

// Returns Gauss-Laguerre nodes/weights (alpha=0) for order n (n >= 1).
// Results are cached per-order and reused.
const GaussLaguerreRule& gauss_laguerre_rule(int n);

} // namespace vol::quad
