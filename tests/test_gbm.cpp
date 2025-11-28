#include <catch2/catch_all.hpp>
#include "libvol/mc/gbm.hpp"
#include "libvol/models/black_scholes.hpp"
#include <cmath>

TEST_CASE("Monte-Carlo vs Black-Scholes call, 200k paths","[gbm]"){
    double S=100,K=100,r=0.02,q=0.01,T=1,vol=0.2;
    bool is_call = true;
    auto res_mc = vol::mc::european_vanilla_gbm(S,K,r,q,T,vol,is_call,2e5);
    double price_mc = res_mc.price;
    double se_mc = res_mc.std_err;
    double price_bs = vol::bs::price(S,K,r,q,T,vol,is_call);

    double diff = std::abs(price_mc - price_bs);

    REQUIRE(diff < 4.0 * se_mc); //diff within 4 stderr
}

TEST_CASE("Monte-Carlo vs Black-Scholes put, 200k paths","[gbm]"){
    double S=100,K=100,r=0.02,q=0.01,T=1,vol=0.2;
    bool is_call = false;
    auto res_mc = vol::mc::european_vanilla_gbm(S,K,r,q,T,vol,is_call,2e5);
    double price_mc = res_mc.price;
    double se_mc = res_mc.std_err;
    double price_bs = vol::bs::price(S,K,r,q,T,vol,is_call);

    double diff = std::abs(price_mc - price_bs);

    REQUIRE(diff < 4.0 * se_mc); //same as above
}

TEST_CASE("Monte-Carlo standard error obeys sqrt(N)", "[gbm][convergence]") {
    const double S = 100.0, K = 110.0, r = 0.01, q = 0.0, T = 0.5, vol = 0.3;
    const bool is_call = true;

    const auto res_small = vol::mc::european_vanilla_gbm(S, K, r, q, T, vol, is_call, 50000);
    const auto res_large = vol::mc::european_vanilla_gbm(S, K, r, q, T, vol, is_call, 200000);

    const double ratio = res_small.std_err / res_large.std_err;
    const double expected = std::sqrt(200000.0 / 50000.0);

    REQUIRE(ratio == Catch::Approx(expected).margin(0.25));
}
