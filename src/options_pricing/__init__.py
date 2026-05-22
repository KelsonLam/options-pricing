"""Black-Scholes option pricing, from scratch.

The package prices European options, computes the Greeks analytically, backs
out implied volatility from a market price, and cross-checks everything against
a Monte Carlo simulation.

Modules:

    black_scholes  the pricing formula and the d1 / d2 terms
    greeks         delta, gamma, vega, theta, and rho
    implied_vol    solve for the volatility implied by a market price
    montecarlo     an independent price estimate, used to validate the formula
    plotting       price and Greek curves, and the payoff diagram

Conventions: rates and the dividend yield are continuously compounded and
annualized, time to expiry T is in years, and volatility sigma is annualized.
"""

__version__ = "0.1.0"
