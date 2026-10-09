"""Inden-Hillert-Jarl magnetic contribution to the Gibbs energy.

    G_mag = R T ln(beta + 1) f(tau),    tau = T / T_C

per mole of formula units, with the polynomial (Hillert and Jarl, CALPHAD 2
(1978) 227)

    tau < 1:   f = 1 - [79 / (140 p tau) + (474/497) (1/p - 1)
                        (tau^3/6 + tau^9/135 + tau^15/600)] / A
    tau >= 1:  f = -[tau^-5/10 + tau^-15/315 + tau^-25/1500] / A

    A = 518/1125 + (11692/15975) (1/p - 1)

where p is the structure factor: 0.28 for FCC and HCP, 0.40 for BCC. f and
df/dtau are continuous at tau = 1.

TDB convention for antiferromagnetism: a negative T_C (Neel temperature) or
beta is divided by the antiferromagnetic factor of the phase's MAGNETIC type
definition (-3 for FCC/HCP, -1 for BCC), so both become positive.
"""
from dataclasses import dataclass

import numpy as np

FCC_STRUCTURE_FACTOR = 0.28
BCC_STRUCTURE_FACTOR = 0.40
FCC_AFM_FACTOR = -3.0
BCC_AFM_FACTOR = -1.0


@dataclass(frozen=True)
class MagneticModel:
    afm_factor: float
    structure_factor: float


def _A(p: float) -> float:
    return 518.0 / 1125.0 + (11692.0 / 15975.0) * (1.0 / p - 1.0)


def ihj_function(tau, p: float):
    """(f(tau), df/dtau) of the Inden-Hillert-Jarl model for structure factor p."""
    tau = np.asarray(tau, dtype=float)
    A = _A(p)
    c = (474.0 / 497.0) * (1.0 / p - 1.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        below = tau < 1.0
        f_low = 1.0 - (79.0 / (140.0 * p) / tau + c * (tau**3 / 6.0 + tau**9 / 135.0 + tau**15 / 600.0)) / A
        df_low = -(-79.0 / (140.0 * p) / tau**2 + c * (tau**2 / 2.0 + tau**8 / 15.0 + tau**14 / 40.0)) / A
        f_high = -(tau**-5 / 10.0 + tau**-15 / 315.0 + tau**-25 / 1500.0) / A
        df_high = (tau**-6 / 2.0 + tau**-16 / 21.0 + tau**-26 / 60.0) / A
    return np.where(below, f_low, f_high), np.where(below, df_low, df_high)


def magnetic_gibbs(T, TC, beta, model: MagneticModel, gas_constant: float):
    """(G_mag, dG/dTC, dG/dbeta) per mole of formula units, for the raw
    (composition-weighted) T_C and beta of a phase."""
    T = np.asarray(T, dtype=float)
    TC = np.asarray(TC, dtype=float)
    beta = np.asarray(beta, dtype=float)
    TC_scale = np.where(TC < 0.0, 1.0 / model.afm_factor, 1.0)
    beta_scale = np.where(beta < 0.0, 1.0 / model.afm_factor, 1.0)
    TC_eff = TC * TC_scale
    beta_eff = beta * beta_scale

    active = TC_eff > 0.0
    safe_TC = np.where(active, TC_eff, 1.0)
    tau = T / safe_TC
    f, df = ihj_function(tau, model.structure_factor)
    log_term = np.log1p(beta_eff)
    RT = gas_constant * T

    G = np.where(active, RT * log_term * f, 0.0)
    dG_dTC = np.where(active, RT * log_term * df * (-tau / safe_TC) * TC_scale, 0.0)
    dG_dbeta = np.where(active, RT * f / (1.0 + beta_eff) * beta_scale, 0.0)
    return G, dG_dTC, dG_dbeta
