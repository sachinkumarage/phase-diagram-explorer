"""Assessed Fe-C (data/systems/fe_c.meta.json, pycalphad's cfe_broshe.tdb):
loading and modes, the suspended phases, this database's reference values,
literature values and the invariant reactions of both diagrams.

The database is fetched from the pycalphad repository at tag 0.11.2 and
verified by SHA256 (scripts/fetch_tdb.py)."""
import numpy as np
import pytest

from phase_diagram_explorer import fe_c
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium
from phase_diagram_explorer.labels import PhaseNames
from phase_diagram_explorer.units import CELSIUS_OFFSET
from phase_diagram_explorer.visualization import reaction_names

SUSPENDED = {"M7C3_D101", "DIAMOND_A4"}


@pytest.fixture(scope="module")
def definition():
    return fe_c.definition()


@pytest.fixture(scope="module")
def reactions():
    return {mode: fe_c.by_type(fe_c.invariants(mode)) for mode in fe_c.MODES}


def _celsius(reaction):
    return reaction.temperature - CELSIUS_OFFSET


def _wt(reaction, phase, definition):
    return float(fe_c.weight_percent(fe_c.composition(reaction, phase), definition))


# --- loading, modes, suspended phases ---------------------------------------------


def test_suspended_phases_are_left_out_with_a_reason(definition):
    names = {phase.name for phase in definition.phases}
    assert not names & SUSPENDED
    assert set(definition.suspended_phases) == SUSPENDED
    assert "This work" in definition.suspended_phases["M7C3_D101"]
    assert "high pressure" in definition.suspended_phases["DIAMOND_A4"]
    full = {phase.name for phase in fe_c.definition(include_suspended=True).phases}
    assert full - names == SUSPENDED


def test_modes(definition):
    assert definition.mode_keys == ["metastable", "stable"]
    assert definition.mode().label == "Metastable Fe-Fe3C (steel practice)"
    assert definition.mode("stable").label == "Stable Fe-graphite"
    metastable = {phase.name for phase in definition.in_mode("metastable").phases}
    stable = {phase.name for phase in definition.in_mode("stable").phases}
    assert "GRAPHITE" not in metastable and "CEMENTITE_D011" in metastable
    assert stable - metastable == {"GRAPHITE"}
    with pytest.raises(ValueError):
        definition.in_mode("unknown")


def test_display_settings(definition):
    assert definition.x_range == (0.0, 0.25)
    assert definition.t_range_k == (673.15, 1873.15)
    assert definition.display["temperature_unit"] == "°C" and definition.display["composition_unit"] == "wt%"
    assert fe_c.weight_percent(0.25, definition) == pytest.approx(6.69, abs=0.005)
    assert [element.symbol for element in definition.elements] == ["Fe", "C"]


def test_alpha_and_delta_are_named_by_temperature_region(definition):
    names = PhaseNames(definition.display_names)
    assert names.name("BCC_A2", 0.001, T=1000.0) == "alpha (ferrite)"
    assert names.name("BCC_A2", 0.001, T=1750.0) == "delta (ferrite)"
    # not by composition-set order
    assert names.names([("BCC_A2#2", 0.001), ("FCC_A1", 0.03)], T=1000.0)[0] == "alpha (ferrite)"
    assert [names.name(p) for p in ("LIQUID", "FCC_A1", "CEMENTITE_D011", "GRAPHITE")] == [
        "L", "gamma (austenite)", "Fe3C (cementite)", "graphite",
    ]


@pytest.mark.slow
def test_reaction_names(definition, reactions):
    names = PhaseNames(definition.display_names)
    metastable = reactions["metastable"]
    assert reaction_names(metastable["eutectoid"], names) == ["alpha (ferrite)", "gamma (austenite)", "Fe3C (cementite)"]
    assert reaction_names(metastable["peritectic"], names) == ["delta (ferrite)", "gamma (austenite)", "L"]


# Points with x(C) < 0.25 across both diagrams: ferrite, austenite, the
# two-phase fields, the eutectic and peritectic regions, liquid.
GRID_T = (700.0, 950.0, 1100.0, 1300.0, 1450.0, 1600.0, 1750.0, 1790.0, 1850.0)
GRID_X = (0.0005, 0.003, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.24)


@pytest.mark.slow
@pytest.mark.parametrize("mode", fe_c.MODES)
def test_leaving_out_m7c3_changes_nothing_below_cementite(mode):
    """With and without M7C3_D101, every equilibrium with x(C) < 0.25 and
    every invariant reaction below Fe3C are identical within 1e-8, so the
    suspension does not change the displayed diagram."""
    without, with_ = fe_c.system(mode), fe_c.system(mode, include=("M7C3_D101",))
    assert "M7C3_D101" in with_ and "M7C3_D101" not in without
    for T in GRID_T:
        for x in GRID_X:
            a, b = compute_equilibrium(without, T, x), compute_equilibrium(with_, T, x)
            assert a.stable_phases == b.stable_phases, (T, x)
            for phase in a.stable_phases:
                assert b.phase_compositions[phase] == pytest.approx(a.phase_compositions[phase], abs=1e-8)
                assert b.phase_fractions[phase] == pytest.approx(a.phase_fractions[phase], abs=1e-8)
            assert b.total_gibbs == pytest.approx(a.total_gibbs, abs=1e-8 * abs(a.total_gibbs))

    def below_cementite(reactions):
        """Reactions whose line reaches below x = 0.25 (in the stable diagram
        the lines run on to graphite at x = 1)."""
        return [r for r in reactions if min(r.composition.values()) < fe_c.X_CEMENTITE - 1e-9]

    a = below_cementite(fe_c.invariants(mode))
    b = below_cementite(fe_c.invariants(mode, include=("M7C3_D101",)))
    assert [(r.type, r.phases) for r in a] == [(r.type, r.phases) for r in b]
    for ra, rb in zip(a, b):
        assert rb.temperature == pytest.approx(ra.temperature, abs=1e-8)
        for phase in ra.phases:
            assert rb.composition[phase] == pytest.approx(ra.composition[phase], abs=1e-8)


@pytest.mark.slow
def test_m7c3_only_adds_a_reaction_beyond_cementite():
    (extra,) = [r for r in fe_c.invariants("metastable", include=("M7C3_D101",)) if "M7C3_D101" in r.phases]
    assert min(extra.composition.values()) == pytest.approx(fe_c.X_CEMENTITE)
    assert _celsius(extra) == pytest.approx(1210.7, abs=0.1)


def test_diamond_must_stay_suspended_in_the_metastable_diagram():
    """With graphite suspended, ferrite + diamond would undercut ferrite +
    Fe3C below about 589 °C; in the stable diagram diamond never appears."""
    metastable = fe_c.system("metastable", include=("DIAMOND_A4",))
    assert "DIAMOND_A4" in compute_equilibrium(metastable, 850.0, 0.1).stable_phases
    assert compute_equilibrium(metastable, 875.0, 0.1).stable_phases == ["BCC_A2", "CEMENTITE_D011"]
    stable_with, stable_without = fe_c.system("stable", include=("DIAMOND_A4",)), fe_c.system("stable")
    for T in (700.0, 862.0, 1100.0, 1500.0):
        assert compute_equilibrium(stable_with, T, 0.1).stable_phases == compute_equilibrium(
            stable_without, T, 0.1
        ).stable_phases


# --- invariant reactions ------------------------------------------------------------


@pytest.mark.slow
def test_metastable_has_exactly_three_invariants(reactions):
    """Over 400-1600 °C: peritectic delta + L -> gamma, eutectic
    L -> gamma + Fe3C and eutectoid gamma -> alpha + Fe3C."""
    found = {kind: tuple(base_phase_name(p) for p in r.phases) for kind, r in reactions["metastable"].items()}
    assert found == {
        "eutectoid": ("BCC_A2", "FCC_A1", "CEMENTITE_D011"),
        "eutectic": ("FCC_A1", "LIQUID", "CEMENTITE_D011"),
        "peritectic": ("BCC_A2", "FCC_A1", "LIQUID"),
    }
    assert len(fe_c.invariants("metastable")) == 3


@pytest.mark.slow
def test_stable_has_eutectic_eutectoid_and_the_peritectic(reactions):
    found = {kind: tuple(base_phase_name(p) for p in r.phases) for kind, r in reactions["stable"].items()}
    assert found == {
        "eutectoid": ("BCC_A2", "FCC_A1", "GRAPHITE"),
        "eutectic": ("FCC_A1", "LIQUID", "GRAPHITE"),
        "peritectic": ("BCC_A2", "FCC_A1", "LIQUID"),
    }
    # the peritectic involves no carbon-rich phase: the same in both modes
    assert reactions["stable"]["peritectic"].temperature == pytest.approx(
        reactions["metastable"]["peritectic"].temperature, abs=1e-9
    )


# --- this database's reference values -----------------------------------------------


def _database_cases():
    for mode, values in fe_c.DATABASE_VALUES.items():
        for reaction, quantities in values.items():
            for quantity, value in quantities.items():
                yield pytest.param(mode, reaction, quantity, value, id=f"{mode}-{reaction}-{quantity}")


@pytest.mark.slow
@pytest.mark.parametrize("mode, reaction, quantity, value", list(_database_cases()))
def test_database_reference_values(definition, reactions, mode, reaction, quantity, value):
    """Values of this database recorded when Fe-C was added (this engine and
    pycalphad agreed to 0.002 K and 4e-5 in x). Tight tolerances (0.05 K,
    0.0005 wt%) catch regressions in the engine or the data."""
    found = reactions[mode][reaction]
    if quantity == "T":
        assert _celsius(found) == pytest.approx(value, abs=fe_c.DATABASE_T_TOLERANCE_K)
    else:
        assert _wt(found, quantity, definition) == pytest.approx(value, abs=fe_c.DATABASE_W_TOLERANCE)


# --- literature ---------------------------------------------------------------------


def _check_literature(definition, reactions, mode, reaction, quantity):
    (value,) = [v for v in fe_c.LITERATURE if (v.mode, v.reaction, v.quantity) == (mode, reaction, quantity)]
    found = reactions[mode][reaction]
    shown = _celsius(found) if quantity == "T" else _wt(found, quantity, definition)
    assert shown == pytest.approx(value.value, abs=value.tolerance), value.label


@pytest.mark.slow
def test_literature_eutectoid(definition, reactions):
    """Eutectoid gamma -> alpha + Fe3C: 727 °C at 0.76 wt% C. Compilations
    give 723-727 °C and 0.76-0.80 wt%, and assessments differ by about 1 °C,
    hence ±2 °C and ±0.03 wt% (this database: 726.5 °C, 0.762 wt%)."""
    _check_literature(definition, reactions, "metastable", "eutectoid", "T")
    _check_literature(definition, reactions, "metastable", "eutectoid", "FCC_A1")


@pytest.mark.slow
def test_literature_max_carbon_in_ferrite(definition, reactions):
    """Ferrite at the eutectoid: 0.022 wt% C. Quoted values range from about
    0.02 to 0.025 wt%; ±0.004 wt% covers them (this database: 0.019 wt%)."""
    _check_literature(definition, reactions, "metastable", "eutectoid", "BCC_A2")


@pytest.mark.slow
def test_literature_eutectic(definition, reactions):
    """Eutectic L -> gamma + Fe3C: 1147 °C at 4.30 wt% C. Quoted values are
    1146-1148 °C and 4.26-4.3 wt%; assessments place the liquid up to about
    0.1 wt% higher, hence ±2 °C and ±0.10 wt% (this database: 1148.2 °C,
    4.38 wt%)."""
    _check_literature(definition, reactions, "metastable", "eutectic", "T")
    _check_literature(definition, reactions, "metastable", "eutectic", "LIQUID")


@pytest.mark.slow
def test_literature_max_carbon_in_austenite(definition, reactions):
    """Austenite at the eutectic: 2.14 wt% C (also quoted as 2.11 and 2.08);
    assessments range down to about 2.05 wt%, hence ±0.12 wt% (this
    database: 2.054 wt%)."""
    _check_literature(definition, reactions, "metastable", "eutectic", "FCC_A1")


@pytest.mark.slow
def test_literature_peritectic(definition, reactions):
    """Peritectic delta + L -> gamma: about 1495 °C (quoted 1493-1495 °C),
    hence ±2 °C (this database: 1494.6 °C)."""
    _check_literature(definition, reactions, "metastable", "peritectic", "T")


@pytest.mark.slow
def test_literature_stable_diagram(definition, reactions):
    """Stable eutectoid about 738 °C (quoted 738-740 °C) and eutectic about
    1153 °C (quoted 1153-1154 °C), hence ±2 °C (this database: 738.0 and
    1153.4 °C)."""
    _check_literature(definition, reactions, "stable", "eutectoid", "T")
    _check_literature(definition, reactions, "stable", "eutectic", "T")


def test_literature_table_matches_the_task():
    assert {(v.reaction, v.quantity): (v.value, v.tolerance) for v in fe_c.LITERATURE if v.mode == "metastable"} == {
        ("eutectoid", "T"): (727.0, 2.0), ("eutectoid", "FCC_A1"): (0.76, 0.03), ("eutectoid", "BCC_A2"): (0.022, 0.004),
        ("eutectic", "T"): (1147.0, 2.0), ("eutectic", "LIQUID"): (4.30, 0.10), ("eutectic", "FCC_A1"): (2.14, 0.12),
        ("peritectic", "T"): (1495.0, 2.0),
    }
    assert np.isclose(sum(v.tolerance for v in fe_c.LITERATURE if v.mode == "stable"), 4.0)
