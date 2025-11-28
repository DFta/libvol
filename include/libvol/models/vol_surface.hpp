#pragma once

#include "libvol/calib/svi_slice.hpp"
#include "libvol/core/types.hpp"
#include "libvol/models/svi.hpp"

#include <vector>

namespace vol::surface {

struct Slice {
    double T;
    vol::svi::Params params;
};

struct SurfaceDiagnostics {
    bool calendar_ok = true;
    bool wings_ok = true;
    std::vector<double> atm_vols;
    std::vector<double> atm_skews;
    std::vector<double> calendar_flags;
};

class Surface {
public:
    Surface() = default;
    explicit Surface(std::vector<Slice> slices);

    double total_variance(double T, double k) const;
    double implied_vol(double S, double K, double r, double q, double T) const;
    double price_option(double S, double K, double r, double q, double T, bool is_call) const;

    const std::vector<Slice>& slices() const { return slices_; }

    SurfaceDiagnostics diagnostics() const;

    static double breeden_litzenberger_density(const Surface& surface,
                                               double S,
                                               double K,
                                               double r,
                                               double q,
                                               double T);

private:
    std::vector<Slice> slices_;

    const Slice& slice_for_T(double T) const;
};

struct MaturitySlice {
    std::vector<vol::OptionSpec> options;
    std::vector<double> mids;
};

Surface calibrate_surface(const std::vector<MaturitySlice>& data,
                          const vol::svi::SliceConfig& slice_cfg);

} // namespace vol::surface
