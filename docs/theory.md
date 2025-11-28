# Libvol Theory Notes

This document collects the derivations and references that underpin the implementations. It is intentionally terse so you can jump from formula → implementation quickly.

## Black–Scholes

For a stock paying a continuous dividend yield `q`, spot `S`, strike `K`, rate `r`, expiry `T`, volatility `σ`:

```
d1 = [ln(S/K) + (r - q + ½σ²)T] / (σ√T)
d2 = d1 - σ√T
C = S e^{-qT} Φ(d1) - K e^{-rT} Φ(d2)
P = K e^{-rT} Φ(-d2) - S e^{-qT} Φ(-d1)
```

Greeks follow by differentiation (delta, gamma, vega, theta, rho) and we keep a single evaluation of `φ(d1)` (`φ` is the standard normal pdf) for numerical stability. Tests in `tests/test_black_scholes.cpp` compare analytic Greeks to finite differences even in the short-maturity, near-zero rate regime.

### Implied Volatility

`vol::bs::implied_vol` uses a safeguarded Newton solver with Brent brackets. We start from `σ₀ = 0.2`, expand the bracket until prices straddle the target, then alternate Newton and Brent steps until the relative price error is below `1e-10`.

## Heston (Carr–Madan / Attari)

Characteristic function for log price `x = ln S` with variance process parameters `(κ, θ, σ, ρ, v₀)`:

```
β(u) = κ - ρσiu
d(u) = sqrt(β(u)² + σ²(u² + iu))
g(u) = (β(u) - d(u)) / (β(u) + d(u))
```

The CF is

```
φ(u) = exp{iu(x + (r - q)T) + (κθ/σ²)[(β - d)T - 2 ln((1 - g e^{-dT})/(1 - g))]
        + v₀ (β - d)/σ² * (1 - e^{-dT})/(1 - g e^{-dT})}
```

Carr–Madan and Attari show how to express call prices via two damped integrals. We evaluate them with Gauss–Laguerre quadrature (`libvol::quad::gauss_laguerre_rule`). The tests confirm that increasing the order (32 → 96) changes prices by less than `5e-5`.

The calibrator (`vol::calib::calibrate_heston_smile`) uses finite-difference gradients, random restarts, bounds, and a Feller penalty `max(0, σ² - 2κθ)²`.

## Stochastic Volatility Inspired (SVI)

Raw SVI parameterization:

```
w(k) = a + b ( ρ (k - m) + sqrt((k - m)² + σ²) )
```

where `w = σ_imp² T` is the total variance, `k = ln(K/F)` is log-moneyness, `b > 0`, `σ > 0`, and `|ρ| < 1`. Calibration minimizes vega-weighted least squares in total variance. We analytically differentiate `w` with respect to `(a, b, ρ, m, σ)` and feed those gradients into the generic calibrator with:

- Box constraints on each parameter.
- Soft penalties on `b`, `σ`, `|ρ|`.
- Random restarts (uniform seeds in each bounding box) + projected gradient steps (`lbfgsb`), reporting iterations, gradient norm, and a simple condition proxy (`max|g| / min|g|`).

References: Gatheral, *The Volatility Surface* (2006).

## Surface Construction & Density

1. Calibrate each maturity slice independently (per above).
2. Enforce calendar monotonicity: for a grid `k ∈ {-2, -1, ..., 2}` ensure `w_{Tᵢ}(k) ≥ w_{Tᵢ₋₁}(k)` by nudging `aᵢ` upward if necessary.
3. Interpolate total variance linearly in maturity when pricing at intermediate expiries.
4. Extrapolate using nearest slice for `T` outside the calibrated range (typical in SVI workflows).

Breeden–Litzenberger (risk-neutral density) uses the standard identity `f(K) = e^{rT} ∂²C/∂K²`. We compute it via a central finite difference around `K` with a 1% bump.

Diagnostics reported by `vol::surface::Surface::diagnostics()`:

- ATM term structure (`σ_ATM(Tᵢ)`).
- ATM skew (`bρ`).
- Calendar flags (worst violation per adjacent pair).
- Wing slopes `b(1±ρ)` to ensure Lee-like conditions.
