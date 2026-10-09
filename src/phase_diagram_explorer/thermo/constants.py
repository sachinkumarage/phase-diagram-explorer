"""Physical constants and defaults shared by the thermodynamic models."""

# Gas constant (J/(mol K)) conventionally used by CALPHAD databases; the
# default for systems read from TDB files.
DATABASE_GAS_CONSTANT = 8.31451
# CODATA 2018 value, used by 0.2.0 and earlier for every system; still the
# default for JSON systems, whose results are unchanged.
CODATA_GAS_CONSTANT = 8.314462618
# Pressure (Pa) at which pressure-dependent TDB expressions are evaluated.
STANDARD_PRESSURE = 101325.0
# Site fractions are kept within (MIN_SITE_FRACTION, 1) when minimising over
# internal degrees of freedom.
MIN_SITE_FRACTION = 1e-12
