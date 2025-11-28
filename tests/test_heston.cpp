#include <catch2/catch_all.hpp>

#include "libvol/models/heston.hpp"

#include <cmath>

using Catch::Approx;

TEST_CASE("Heston CF price matches reference value", "[heston]") {
    const vol::heston::Params params{1.5, 0.04, 0.5, -0.7, 0.04};
    const double price = vol::heston::price_cf(100.0, 100.0, 0.01, 0.0, 1.0, params, true, 128);
    REQUIRE(price == Approx(7.601755381347978).margin(1e-7));
}

TEST_CASE("Heston put-call parity holds", "[heston]") {
    const double S = 120.0;
    const double K = 100.0;
    const double r = 0.015;
    const double q = 0.01;
    const double T = 1.25;
    const vol::heston::Params params{2.0, 0.07, 0.4, -0.5, 0.05};
    const double call = vol::heston::price_cf(S, K, r, q, T, params, true, 64);
    const double put = vol::heston::price_cf(S, K, r, q, T, params, false, 64);
    const double parity = call - put - (S * std::exp(-q * T) - K * std::exp(-r * T));
    REQUIRE(std::abs(parity) < 1e-8);
}

TEST_CASE("Heston Gauss-Laguerre order convergence", "[heston]") {
    const vol::heston::Params params{2.0, 0.09, 0.5, -0.7, 0.09};
    const double price32 = vol::heston::price_cf(100.0, 80.0, 0.03, 0.0, 0.75, params, true, 32);
    const double price96 = vol::heston::price_cf(100.0, 80.0, 0.03, 0.0, 0.75, params, true, 96);
    REQUIRE(price32 == Approx(price96).margin(5e-5));
}

TEST_CASE("Heston handles Feller boundary", "[heston][edge]") {
    const vol::heston::Params params{3.0, 0.04, std::sqrt(2.0 * 3.0 * 0.04) * 0.99, -0.3, 0.04};
    const double call = vol::heston::price_cf(100.0, 110.0, 0.01, 0.0, 2.0, params, true, 128);
    REQUIRE(call > 0.0);
    REQUIRE(std::isfinite(call));
}
