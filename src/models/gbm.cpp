#include "libvol/mc/gbm.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

namespace vol::mc {

    MCResult european_vanilla_gbm(double S, double K, double r, double q, double T, double vol, bool is_call, 
                                  std::uint64_t n_paths, std::uint64_t seed) {
        
        if (n_paths % 2 != 0) ++n_paths;

        std::mt19937_64 rng(seed);
        std::normal_distribution<double> Z(0.0, 1.0);

        const double sqT = std::sqrt(T);
        const double mu_T = (r - q - 0.5 * vol * vol) * T;
        const double vol_sqT = vol * sqT;
        const double disc = std::exp(-r * T);
        
        // Expected value of the Control Variate
        //E[e^{-rT} S_T] = S_0 * e^{-qT}
        const double EX = S * std::exp(-q * T); 

        const std::uint64_t pairs = n_paths / 2;

        //E[Y], E[X], Var(Y), Var(X), Cov(Y, X)
        double sum_Y = 0.0;
        double sum_Y2 = 0.0;
        double sum_X = 0.0;
        double sum_X2 = 0.0;
        double sum_YX = 0.0;

        for(std::uint64_t i = 0; i < pairs; ++i) {
            const double z = Z(rng);
            const double ez = vol_sqT * z;
            
            // Antithetic Paths
            // S_T = S * exp(mu*T +/- vol*sqrt(T)*z)
            const double term_p = std::exp(mu_T + ez);
            const double term_m = std::exp(mu_T - ez);
            
            const double STp = S * term_p;
            const double STm = S * term_m;

            // Payoffs
            const double payp = is_call ? std::max(0.0, STp - K) : std::max(0.0, K - STp);
            const double paym = is_call ? std::max(0.0, STm - K) : std::max(0.0, K - STm);

            // Average of the pair (Antithetic Variate)
            // Y = Discounted Option Payoff
            // X = Discounted Stock Price (Control Variate)
            const double Y_val = disc * 0.5 * (payp + paym);
            const double X_val = disc * 0.5 * (STp + STm);

            sum_Y  += Y_val;
            sum_Y2 += Y_val * Y_val;
            sum_X  += X_val;
            sum_X2 += X_val * X_val;
            sum_YX += Y_val * X_val;
        }

        const double N = static_cast<double>(pairs);
        
        // Compute statistics on the pairs
        const double mean_Y = sum_Y / N;
        const double mean_X = sum_X / N;
        
        // Sample Variances and Covariance
        // Var = (SumSq - N * Mean^2) / (N - 1)
        const double var_Y = (sum_Y2 - N * mean_Y * mean_Y) / (N - 1.0);
        const double var_X = (sum_X2 - N * mean_X * mean_X) / (N - 1.0);
        const double cov_YX = (sum_YX - N * mean_Y * mean_X) / (N - 1.0);

        // Optimal Beta for Control Variate
        // beta = Cov(Y, X) / Var(X)
        const double beta = (var_X > 1e-14) ? cov_YX / var_X : 0.0;

        // Adjusted Mean (Control Variate Estimator)
        // E[Y_cv] = E[Y] - beta * (E[X] - Theoretical_EX)
        const double price = mean_Y - beta * (mean_X - EX);

        // Adjusted Variance (theoretical reduction)
        // Var(Y_cv) = Var(Y) - Cov(Y, X)^2 / Var(X)
        //           = Var(Y) * (1 - rho^2)
        const double var_reduction = (var_X > 1e-14) ? (cov_YX * cov_YX) / var_X : 0.0;
        const double var_cv = std::max(0.0, var_Y - var_reduction);

        // Standard Error of the mean
        const double std_err = std::sqrt(var_cv / N);

        // Note: We return 2 * pairs as total paths
        return { price, std_err, n_paths };
    }

} // namespace vol::mc