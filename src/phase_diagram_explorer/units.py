"""Unit conversions between internal values (temperature in K, composition as
mole fraction of the dependent element) and display units.

Every function accepts a scalar or an array and returns the same kind.
"""
import numpy as np

KELVIN = "K"
CELSIUS = "°C"
TEMPERATURE_UNITS = (KELVIN, CELSIUS)

ATOMIC_PERCENT = "at%"
WEIGHT_PERCENT = "wt%"
COMPOSITION_UNITS = (ATOMIC_PERCENT, WEIGHT_PERCENT)

CELSIUS_OFFSET = 273.15


def _same_kind(value, result):
    return float(result) if np.ndim(value) == 0 else np.asarray(result, dtype=float)


def kelvin_to_celsius(T):
    return _same_kind(T, np.asarray(T, dtype=float) - CELSIUS_OFFSET)


def celsius_to_kelvin(T):
    return _same_kind(T, np.asarray(T, dtype=float) + CELSIUS_OFFSET)


def mole_to_weight_fraction(x_b, mass_a: float, mass_b: float):
    """Weight fraction of B from its mole fraction x_b in a binary A-B."""
    x_b_arr = np.asarray(x_b, dtype=float)
    w_b = x_b_arr * mass_b / (x_b_arr * mass_b + (1.0 - x_b_arr) * mass_a)
    return _same_kind(x_b, w_b)


def weight_to_mole_fraction(w_b, mass_a: float, mass_b: float):
    """Mole fraction of B from its weight fraction w_b in a binary A-B."""
    w_b_arr = np.asarray(w_b, dtype=float)
    x_b = (w_b_arr / mass_b) / (w_b_arr / mass_b + (1.0 - w_b_arr) / mass_a)
    return _same_kind(w_b, x_b)


def temperature_to_display(T_kelvin, unit: str):
    if unit == KELVIN:
        return _same_kind(T_kelvin, T_kelvin)
    if unit == CELSIUS:
        return kelvin_to_celsius(T_kelvin)
    raise ValueError(f"unknown temperature unit {unit!r}")


def temperature_from_display(T, unit: str):
    if unit == KELVIN:
        return _same_kind(T, T)
    if unit == CELSIUS:
        return celsius_to_kelvin(T)
    raise ValueError(f"unknown temperature unit {unit!r}")


def composition_to_display(x_b, unit: str, mass_a: float | None = None, mass_b: float | None = None):
    """Mole fraction of B -> percent of B in `unit` (at% or wt%)."""
    if unit == ATOMIC_PERCENT:
        return _same_kind(x_b, 100.0 * np.asarray(x_b, dtype=float))
    if unit == WEIGHT_PERCENT:
        _require_masses(mass_a, mass_b)
        return _same_kind(x_b, 100.0 * np.asarray(mole_to_weight_fraction(x_b, mass_a, mass_b)))
    raise ValueError(f"unknown composition unit {unit!r}")


def composition_from_display(value, unit: str, mass_a: float | None = None, mass_b: float | None = None):
    """Percent of B in `unit` (at% or wt%) -> mole fraction of B."""
    fraction = _same_kind(value, np.asarray(value, dtype=float) / 100.0)
    if unit == ATOMIC_PERCENT:
        return fraction
    if unit == WEIGHT_PERCENT:
        _require_masses(mass_a, mass_b)
        return weight_to_mole_fraction(fraction, mass_a, mass_b)
    raise ValueError(f"unknown composition unit {unit!r}")


def temperature_label(unit: str) -> str:
    return f"Temperature ({unit})"


def composition_label(symbol: str, unit: str) -> str:
    """Axis label for the composition of `symbol`, e.g. "x(Cu) (at%)"."""
    quantity = "x" if unit == ATOMIC_PERCENT else "w"
    return f"{quantity}({symbol}) ({unit})"


def _require_masses(mass_a: float | None, mass_b: float | None) -> None:
    if mass_a is None or mass_b is None:
        raise ValueError("weight fraction conversion needs the atomic masses of both elements")
