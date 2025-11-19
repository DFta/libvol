# Libvol

## Overview
A small C++20 volatility and option pricing library implementing:
- Black-Scholes pricing + Greeks + robust implied vol solver
- CRR binomial tree (American/European, price + Greeks + early exercise info)
- GBM Monte Carlo with antithetic and control variate
- SVI slice calibration on top of BS implied vols
- Heston CF vanilla pricing (Carr-Madan/Attari + Gauss-Laguerre integration)
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
