#include "libvol/models/vol_surface.hpp"

#include "libvol/models/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vol::surface {

namespace {

std::vector<double> default_grid() {
    return {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
}

void enforce_calendar(std::vector<Slice>& slices) {
    if (slices.size() < 2) {
        return;
    }
    const auto grid = default_grid();
    for (std::size_t i = 1; i < slices.size(); ++i) {
        double needed = 0.0;
        for (double k : grid) {
            const double prev = vol::svi::total_variance(k, slices[i - 1].params);
            const double curr = vol::svi::total_variance(k, slices[i].params);
            needed = std::max(needed, prev - curr + 1e-8);
        }
        if (needed > 0.0) {
            slices[i].params[0] += needed;
            if (!vol::svi::basic_no_arb(slices[i].params)) {
                slices[i].params[0] = std::max(1e-10, slices[i].params[0]);
            }
        }
    }
}

} // namespace

Surface::Surface(std::vector<Slice> slices) : slices_(std::move(slices)) {
    std::sort(slices_.begin(), slices_.end(), [](const Slice& a, const Slice& b) {
        return a.T < b.T;
    });
}

double Surface::total_variance(double T, double k) const {
    if (slices_.empty()) return 0.0;

    // Fast handling for extrapolation (flat extrapolation of variance)
    if (T <= slices_.front().T) {
        return vol::svi::total_variance(k, slices_.front().params);
    }
    if (T >= slices_.back().T) {
        return vol::svi::total_variance(k, slices_.back().params);
    }

    // Binary Search to find the interval [t0, t1] containing T
    // std::upper_bound returns the first element > T
    const auto it = std::upper_bound(slices_.begin(), slices_.end(), T, 
        [](double val, const Slice& s) { return val < s.T; });

    // 'it' points to the slice AFTER T (t1)
    // 'prev' points to the slice BEFORE T (t0)
    const auto& slice1 = *it;
    const auto& slice0 = *std::prev(it);

    const double t0 = slice0.T;
    const double t1 = slice1.T;
    const double w0 = vol::svi::total_variance(k, slice0.params);
    const double w1 = vol::svi::total_variance(k, slice1.params);

    // Linear interpolation in total variance
    const double w = (T - t0) / (t1 - t0);
    return w0 + (w1 - w0) * w;
}
double Surface::implied_vol(double S, double K, double r, double q, double T) const {
    if (T <= 0.0 || K <= 0.0 || S <= 0.0) {
        return 0.0;
    }
    const double F = S * std::exp((r - q) * T);
    const double k = std::log(K / F);
    const double w = std::max(1e-12, total_variance(T, k));
    return std::sqrt(w / T);
}

double Surface::price_option(double S, double K, double r, double q, double T, bool is_call) const {
    const double iv = implied_vol(S, K, r, q, T);
    return vol::bs::price(S, K, r, q, T, iv, is_call);
}

SurfaceDiagnostics Surface::diagnostics() const {
    SurfaceDiagnostics diag;
    diag.calendar_flags.reserve(slices_.size());
    diag.atm_vols.reserve(slices_.size());
    diag.atm_skews.reserve(slices_.size());

    if (slices_.empty()) {
        diag.calendar_ok = true;
        diag.wings_ok = true;
        return diag;
    }

    for (const auto& slice : slices_) {
        const double atm_w = vol::svi::total_variance(0.0, slice.params);
        const double atm_vol = (slice.T > 0.0) ? std::sqrt(std::max(1e-12, atm_w) / slice.T) : 0.0;
        diag.atm_vols.push_back(atm_vol);
        diag.atm_skews.push_back(slice.params[1] * slice.params[2]);

        const double left_slope = slice.params[1] * (1 + slice.params[2]);
        const double right_slope = slice.params[1] * (1 - slice.params[2]);
        if (left_slope < 0.0 || right_slope < 0.0) {
            diag.wings_ok = false;
        }
    }

    const auto grid = default_grid();
    diag.calendar_flags.resize(slices_.size(), 0.0);
    for (std::size_t i = 1; i < slices_.size(); ++i) {
        double worst = 0.0;
        for (double k : grid) {
            const double prev = vol::svi::total_variance(k, slices_[i - 1].params);
            const double curr = vol::svi::total_variance(k, slices_[i].params);
            worst = std::min(worst, curr - prev);
            if (curr < prev - 1e-10) {
                diag.calendar_ok = false;
            }
        }
        diag.calendar_flags[i] = worst;
    }

    return diag;
}

double Surface::breeden_litzenberger_density(const Surface& surface,
                                             double S,
                                             double K,
                                             double r,
                                             double q,
                                             double T) {
    if (T <= 0.0) {
        return 0.0;
    }
    const double bump = std::max(1e-3, 0.01 * K);
    const double kp = K + bump;
    const double km = std::max(1e-4, K - bump);
    const double cp = surface.price_option(S, kp, r, q, T, true);
    const double c = surface.price_option(S, K, r, q, T, true);
    const double cm = surface.price_option(S, km, r, q, T, true);
    const double second = (cp - 2.0 * c + cm) / (bump * bump);
    return std::exp(r * T) * second;
}

const Slice& Surface::slice_for_T(double T) const {
    if (slices_.empty()) {
        throw std::runtime_error("vol surface has no slices");
    }
    if (T <= slices_.front().T) {
        return slices_.front();
    }
    if (T >= slices_.back().T) {
        return slices_.back();
    }
    for (std::size_t i = 1; i < slices_.size(); ++i) {
        if (T <= slices_[i].T) {
            return slices_[i];
        }
    }
    return slices_.back();
}

Surface calibrate_surface(const std::vector<MaturitySlice>& data,
                          const vol::svi::SliceConfig& slice_cfg) {
    if (data.empty()) {
        throw std::invalid_argument("calibrate_surface: no maturity data provided");
    }

    std::vector<Slice> slices;
    slices.reserve(data.size());
    for (const auto& mat : data) {
        if (mat.options.empty() || mat.options.size() != mat.mids.size()) {
            throw std::invalid_argument("calibrate_surface: invalid slice data");
        }
        const double T = mat.options.front().T;
        for (const auto& opt : mat.options) {
            if (std::abs(opt.T - T) > 1e-12) {
                throw std::invalid_argument("calibrate_surface: slice must have uniform maturity");
            }
        }
        auto params = vol::svi::calibrate_slice_from_prices(mat.options, mat.mids, slice_cfg);
        slices.push_back({T, params});
    }

    std::sort(slices.begin(), slices.end(), [](const Slice& a, const Slice& b) {
        return a.T < b.T;
    });

    enforce_calendar(slices);
    return Surface{std::move(slices)};
}

} // namespace vol::surface
