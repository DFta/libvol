#pragma once
#include <tuple>

namespace vol::binom {

struct PriceGreeks {
    double price, delta, gamma, vega, theta, rho; 
};

// Removed unused 'TreeType' enum

struct BinomialResult {
    double price; 
    int early_exercise_step;
};

// Core functions
double price(double S, double K, double r, double q, double T, double vol, int steps, bool is_call, bool is_american);

BinomialResult price_w_info(double S, double K, double r, double q, double T, double vol, int steps, bool is_call, bool is_american);

PriceGreeks price_greeks(double S, double K, double r, double q, double T, double vol, int steps, bool is_call, bool is_american);

} // namespace vol::binom