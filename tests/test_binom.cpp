#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "libvol/models/binom.hpp"
#include "libvol/models/black_scholes.hpp"
#include <cmath>

TEST_CASE("Binomial pricing: no-dividend American call equal to euro", "[binomial]"){
    double S = 100.0, K = 100.0, r = 0.05, q = 0.0, T = 1.0, vol = 0.25;
    int steps = 100;
    bool is_call = true;
    bool is_american = true;

    double american_call = vol::binom::price(S, K, r, q, T, vol, steps, is_call, is_american);
    double euro_call = vol::bs::price(S, K, r, q, T, vol, is_call);

    REQUIRE_THAT(american_call, Catch::Matchers::WithinRel(euro_call, 0.005)); // 0.5% tolerance
}

TEST_CASE("Binomial pricing: American put has early exercise premium", "[binomial]"){
    double S=100, K=100, r=0.02, q=0.01, T=1.0, vol=0.2;
    int steps = 100;
    bool is_call = false;
    bool is_american = true;

    double american_put = vol::binom::price(S, K, r, q, T, vol, steps, is_call, is_american);
    double euro_put = vol::bs::price(S, K, r, q, T, vol, is_call);

    REQUIRE(american_put > euro_put);
    REQUIRE(american_put < euro_put * 1.5); // sanity bound
}

TEST_CASE("Binomial pricing: Binomial euro call converges to Black_Scholes", "[binomial]"){
    double S=100,K=100,r=0.05,q=0.02,T=1.0,vol=0.25;
    bool is_call = true;
    bool is_american = false;

    double euro_call_binom_50  = vol::binom::price(S, K, r, q, T, vol,  50, is_call, is_american);
    double euro_call_binom_100 = vol::binom::price(S, K, r, q, T, vol, 100, is_call, is_american);
    double euro_call_binom_200 = vol::binom::price(S, K, r, q, T, vol, 200, is_call, is_american);
    double euro_call_bs = vol::bs::price  (S, K, r, q, T, vol, is_call);

    double err_50 = euro_call_binom_50 - euro_call_bs;
    double err_100 = euro_call_binom_100 - euro_call_bs;
    double err_200 = euro_call_binom_200 - euro_call_bs;

    REQUIRE(std::abs(err_100) <= std::abs(err_50));
    REQUIRE(std::abs(err_200) <= std::abs(err_100));
    REQUIRE(std::abs(err_200) < 0.1);
}

TEST_CASE("Binomial pricing: Euro put-call parity", "[binomial]"){
    double S=100,K=100,r=0.05,q=0.02,T=1.0,vol=0.25;
    int steps = 100; 
    bool is_american = false;
    
    double call = vol::binom::price(S, K, r, q, T, vol, steps, true, is_american);
    double put  = vol::binom::price(S, K, r, q, T, vol, steps, false, is_american);

    double lhs = call - put;
    //put-call parity with continuous dividend yield q
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);

    REQUIRE_THAT(lhs, Catch::Matchers::WithinRel(rhs, 0.01)); // 1% relative tolerance
}

TEST_CASE("Binomial convergence follows sqrt step scaling", "[binomial][convergence]") {
    const double S = 100.0, K = 120.0, r = 0.03, q = 0.0, T = 1.25, vol = 0.35;
    const bool is_call = false;
    const bool is_american = false;
    const double ref = vol::bs::price(S, K, r, q, T, vol, is_call);

    const double price64  = vol::binom::price(S, K, r, q, T, vol, 64, is_call, is_american);
    const double price256 = vol::binom::price(S, K, r, q, T, vol, 256, is_call, is_american);
    const double price1024 = vol::binom::price(S, K, r, q, T, vol, 1024, is_call, is_american);

    const double err64 = std::abs(price64 - ref);
    const double err256 = std::abs(price256 - ref);
    const double err1024 = std::abs(price1024 - ref);

    REQUIRE(err256 < 0.8 * err64);
    REQUIRE(err1024 < 0.8 * err256);
    REQUIRE(err1024 < 1e-2);
}
