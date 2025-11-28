# Libvol

## Overview
A small C++20 volatility and option pricing library implementing:
- Black-Scholes pricing + Greeks + robust implied vol solver
- CRR binomial tree (American/European, price + Greeks + early exercise info)
- GBM Monte Carlo with antithetic and control variate
- SVI slice calibration on top of BS implied vols + reusable calibration framework
- Heston CF vanilla pricing + smile calibration (Carr-Madan/Attari + Gauss-Laguerre integration)
- Volatility surface object with cross-maturity checks, BS projection, and Breeden-Litzenberger density extraction
- Benchmarks (~40 ns per BS price on i7-12650H)
- C++ and Python (pybind11) APIs

## Building & Testing
**With ninja:**
```bash
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build`
```
**With MSVC**
```bash
cmake -S . -B build
`cmake --build build --config Release
```

**Run tests**
```bash
from repo root
Using CTest (recommended)
ctest --test-dir build --output-on-failure
```
Or directly:
```bash
Ninja / Linux / macOS
./build/vol_tests`

MSVC / Windows
.\build\Release\vol_tests.exe
```
Run demo:
```bash
.\build\Release\demo.exe
```

(use / if on linux or macos)


**Next sprints**
- Heston calibration (global + local search, bounds/penalties)
- Calibration framework (global + local), parameter bounds/penalties
- RND extraction (Breeden-Litzenberger) + diagnostics

## Theory Snapshot
- **Black-Scholes.** Closed-form European pricing with dividends. Greeks (`delta`, `gamma`, `vega`, `theta`, `rho`) follow the textbook formulas and are validated numerically in tests even in the short-maturity/near-zero rate regime.
- **Carr–Madan / Attari Heston pricing.** Characteristic function integration via Gauss-Laguerre quadrature. The CF is evaluated at `u-i`, the damping and drift terms follow Gatheral. You can tune the quadrature order; convergence tests are in `tests/test_heston.cpp`.
- **SVI.** Raw SVI parameterization `w(k) = a + b ( \rho (k-m) + \sqrt{(k-m)^2 + \sigma^2})`. Calibration enforces `b>0`, `|\rho|<1`, `\sigma>0`, and adds soft penalties for Feller-esque wing slopes.
- **Vol surface.** Slice-by-slice SVI fits are stitched across maturities, calendar monotonicity is enforced on a k-grid, and Breeden–Litzenberger second derivatives recover densities for diagnostics.

See `docs/theory.md` for a compact derivation crib sheet (BS, Carr–Madan/Attari, SVI references to Gatheral, and how the quadrature weights tie in).

## API Reference (high level)
- **`vol::bs`** – pricing + Greeks + implied vol. Inputs: `(S,K,r,q,T,vol,bool)`. Guarantees: handles edge cases (deep ITM/OTM, tiny maturities) and returns finite Greeks.
- **`vol::binom`** – CRR tree, European & American. Inputs: `(steps, is_call, is_american)`. Converges to BS as steps ↑ (validated tests).
- **`vol::mc`** – GBM Monte Carlo with control variate and antithetic paths. `MCResult` reports standard error obeying √N scaling.
- **`vol::svi`** – slice fit + `vol::svi::calibrate_slice_from_prices`. Under the hood uses the generic calibrator (`vol::calib::run_calibration`) with random restarts, L-BFGS-B style projected steps, bounds, and soft penalties.
- **`vol::calib`** – `CalibratorConfig`, `ParameterSpec`, diagnostics (iterations, gradient norm, condition proxy). Plug-ins: BS IV (trivial), SVI slice, Heston smile. Exposed in C++ so you can inject your own objective.
- **`vol::surface`** – `Surface` aggregates SVI slices, exposes `total_variance`, `implied_vol`, `price_option`, plus `Surface::breeden_litzenberger_density` and diagnostics (ATM term structure, skews, arbitrage flags).
- **`vol::heston`** – CF pricing + `vol::calib::calibrate_heston_smile` to back out parameters from a smile with calendar/penalty handling near the Feller boundary.

## Demos & Notebooks
- `examples/heston_calib_demo.cpp` – generates a synthetic smile and recovers Heston params with diagnostics.
- `examples/svi_slice_demo.cpp` – one-slice calibration.
- `Notebooks/01_bs_vs_binom.ipynb` – binomial vs BS convergence, Greeks comparison, runtime.
- `Notebooks/02_svi_fit.ipynb` – synthetic smile → SVI fit, residual plot.
- `Notebooks/03_heston_pricing.ipynb` – parameter sweeps and smile impact for Heston.

## SVI Slice-by-Slice Calibration (New)

**What it does**

- Takes a **single expiry** option strip same $T$, varying $K$  
- Computes **implied vols** via a robust BS IV solver (safeguarded Newton + Brent)  
- Converts to **log-moneyness** $k = \ln(K/F)$ and **total variance** $w = \sigma^2 T$ 
- Fits a **raw SVI** smile per expiry
- Uses **vega-weighted least squares** with gentle wing down-weighting for stability  
- Enforces basic no-arb sanity: $b > 0$, $|\rho| < 1$, $\sigma > 0$ (soft penalties + box constraints)

**Pipeline**

1. Market prices -> BS IV (`vol::bs::implied_vol`)  
2. IVs -> total variance grid `$(k_i, w_i)$`  
3. Optimize raw SVI params `{a, b, rho, m, sigma}` per slice  
4. Use `total_variance(k, params)` + $w/T$ to recover model IVs for pricing / plotting

See `benchmarks.md` for full micro-benchmarks (Black-Scholes, binomial, SVI).

### Key Micro-Benchmarks
```
-------------------------------------------------------------------------
Benchmark                               Time             CPU   Iterations
-------------------------------------------------------------------------
BM_SVI_Calibrate_Clean             218825 ns       219727 ns         3200
BM_Price_ATM                       38.6 ns         38.5 ns       19478261
BM_PriceGreeks_ATM                 112 ns          109 ns         5600000
BM_Binom_Price_Amer_Put/256        37624 ns        37667 ns         18667
```

Additional Heston CF timings (ATM/OTM/portfolio prices + Gauss-Laguerre order sweeps) are available via the `heston_bench` target.

**Binomial Convergence to Black-Scholes:**
- 50 steps: 0.048 error
- 256 steps: 0.009 error
- 1024 steps: 0.002 error

## C++ Usage Example

```cpp
#include "libvol/models/black_scholes.hpp"

int main() {
    double S = 100.0, K = 100.0, r = 0.02, q = 0.01, T = 1.0, vol = 0.25;
    bool is_call = true;

    double price = vol::bs::price(S, K, r, q, T, vol, is_call);
    auto g = vol::bs::price_greeks(S, K, r, q, T, vol, is_call);

    // g.price, g.delta, g.gamma, g.vega, g.theta, g.rho
    return 0;
}
```
