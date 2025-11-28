#include <catch2/catch_all.hpp>
#include "libvol/models/black_scholes.hpp"
#include <cmath>


TEST_CASE("BS price symmetry put-call parity", "[bs]"){
double S=100,K=100,r=0.02,q=0.01,T=1,vol=0.2;
double c = vol::bs::price(S,K,r,q,T,vol,true);
double p = vol::bs::price(S,K,r,q,T,vol,false);
double parity = c - p - (S*std::exp(-q*T) - K*std::exp(-r*T));
REQUIRE(std::abs(parity) < 1e-10);
}


TEST_CASE("BS greeks vs finite diff", "[bs]"){
    double S=100, K=100, r=0.02, q=0.01, T=1, vol=0.25;

    double h = std::max(1e-4 * S, 1e-6);

    auto g = vol::bs::price_greeks(S, K, r, q, T, vol, true);

    // Delta from price (central diff)
    double f_up = vol::bs::price(S+h, K, r, q, T, vol, true);
    double f_dn = vol::bs::price(S-h, K, r, q, T, vol, true);
    double num_delta = (f_up - f_dn) / (2*h);

    // Gamma as derivative of delta (central diff on delta)
    auto g_up = vol::bs::price_greeks(S+h, K, r, q, T, vol, true);
    auto g_dn = vol::bs::price_greeks(S-h, K, r, q, T, vol, true);
    double num_gamma = (g_up.delta - g_dn.delta) / (2*h);

    INFO("delta analytic=" << g.delta << " numeric=" << num_delta);
    INFO("gamma analytic=" << g.gamma << " numeric=" << num_gamma);

    REQUIRE(std::abs(g.delta - num_delta) < 1e-5);
    REQUIRE(std::abs(g.gamma - num_gamma) < 1e-4);
}

TEST_CASE("BS put theta vs finite diff", "[bs]") {
    double S=100, K=100, r=0.02, q=0.01, T=1.0, vol=0.25;
    bool is_call = false;

    auto g = vol::bs::price_greeks(S, K, r, q, T, vol, is_call);

    double hT = 1e-4;
    double p_up = vol::bs::price(S, K, r, q, T + hT, vol, is_call);
    double p_dn = vol::bs::price(S, K, r, q, T - hT, vol, is_call);
    double dV_dT = (p_up - p_dn) / (2.0 * hT);
    double num_theta = -1.0 * dV_dT;

    REQUIRE(std::abs(g.theta - num_theta) < 1e-4);
}

TEST_CASE("BS pricing handles deep ITM/OTM and short maturities", "[bs][edge]") {
    const double r = 1e-8;
    const double q = 0.0;
    const double tiny_T = 1e-4;
    const double low_vol = 1e-3;

    const double itm_call = vol::bs::price(100.0, 10.0, r, q, tiny_T, low_vol, true);
    REQUIRE(itm_call == Catch::Approx(90.0).margin(1e-6));

    const double otm_call = vol::bs::price(100.0, 180.0, 0.02, 0.0, 5e-4, 0.6, true);
    REQUIRE(otm_call < 1e-3);

    const auto greeks = vol::bs::price_greeks(100.0, 90.0, 1e-6, 1e-6, 0.2, 0.35, true);
    REQUIRE(std::isfinite(greeks.delta));
    REQUIRE(std::isfinite(greeks.gamma));
    REQUIRE(std::isfinite(greeks.vega));
}

