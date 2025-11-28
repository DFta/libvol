#pragma once

#include "libvol/calib/calibrator.hpp"
#include "libvol/core/types.hpp"
#include "libvol/models/heston.hpp"

#include <vector>

namespace vol::calib {

struct HestonSlice {
    std::vector<vol::OptionSpec> options;
    std::vector<double> mids;
    int quad_order = 64;
    std::vector<ParameterSpec> bounds;
};

struct HestonCalibrationResult {
    vol::heston::Params params;
    CalibrationResult solver;
};

// Throws std::runtime_error if the optimizer cannot converge to a feasible solution.
HestonCalibrationResult calibrate_heston_smile(const HestonSlice& slice,
                                               const CalibratorConfig& cfg = {});

} // namespace vol::calib
