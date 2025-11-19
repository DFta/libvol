#pragma once
#include <array>
#include <vector>


namespace vol::svi {
using Params = std::array<double,5>;


double total_variance(double k, const Params& p);

bool basic_no_arb(const Params& p);

Params fit_raw_svi(const std::vector<double>& k, const std::vector<double>& w_mkt, const std::vector<double>& wts);

} // namespace vol::svi