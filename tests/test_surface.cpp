#include <catch2/catch_all.hpp>

#include "libvol/models/vol_surface.hpp"
#include "libvol/models/black_scholes.hpp"

#include <cmath>

namespace {

vol::surface::MaturitySlice make_slice(double T, const std::vector<double>& strikes) {
    vol::surface::MaturitySlice slice;
    slice.options.reserve(strikes.size());
    slice.mids.reserve(strikes.size());
    const double S = 100.0;
    const double r = 0.01;
    const double q = 0.0;
    for (double K : strikes) {
        vol::OptionSpec opt{S, K, r, q, T, true};
        const double k = std::log(K / S);
        const double vol = 0.2 + 0.05 * k - 0.02 * k * k + 0.02 * T;
        const double price = vol::bs::price(S, K, r, q, T, std::max(0.05, vol), true);
        slice.options.push_back(opt);
        slice.mids.push_back(price);
    }
    return slice;
}

} // namespace

TEST_CASE("Calibrated surface preserves calendar monotonicity", "[surface]") {
    const auto slice1 = make_slice(0.5, {70, 80, 90, 100, 110, 120, 130});
    const auto slice2 = make_slice(1.0, {70, 80, 90, 100, 110, 120, 130});
    const auto slice3 = make_slice(2.0, {70, 80, 90, 100, 110, 120, 130});

    vol::svi::SliceConfig cfg;
    const auto surface = vol::surface::calibrate_surface({slice1, slice2, slice3}, cfg);
    const auto diag = surface.diagnostics();

    REQUIRE(diag.calendar_ok);
    REQUIRE(diag.wings_ok);
    REQUIRE(diag.atm_vols.size() == 3);

    const double price = surface.price_option(100.0, 100.0, 0.01, 0.0, 1.5, true);
    const double ref_vol = 0.2 + 0.02 * 1.5;
    const double ref_price = vol::bs::price(100.0, 100.0, 0.01, 0.0, 1.5, ref_vol, true);
    REQUIRE(price == Catch::Approx(ref_price).margin(0.25));
}

TEST_CASE("Breeden-Litzenberger density is positive", "[surface][density]") {
    const auto slice1 = make_slice(0.75, {80, 90, 100, 110, 120});
    vol::svi::SliceConfig cfg;
    const auto surface = vol::surface::calibrate_surface({slice1}, cfg);

    const double density = vol::surface::Surface::breeden_litzenberger_density(surface, 100.0, 105.0, 0.01, 0.0, 0.75);
    REQUIRE(density > 0.0);
    REQUIRE(std::isfinite(density));
}
