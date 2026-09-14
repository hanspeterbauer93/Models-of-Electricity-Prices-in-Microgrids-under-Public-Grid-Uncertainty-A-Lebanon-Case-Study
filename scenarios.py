"""
Stochastic public-grid (EDL) availability scenarios for the lower level.

The village of Deir Kanoun al Naher (DKN) is supplied by the Lebanese national
utility EDL only intermittently: EDL announces a daily rationing schedule, but
the delivered supply routinely deviates from it. This module encodes that
uncertainty as six discrete scenarios over the hourly public-grid
availability.

A scenario is a :class:`GridScenario`: an availability WINDOW (which hours EDL
delivers) plus a fraction scaling the delivered capacity within that window.
The window is stored as a key rather than as literal hours so that the six
scenarios can be declared against named windows. The actual max supply is:

    QbarPG[t, s] = BASELINE_KW * avail_fraction   if t in window
                 = 0                              otherwise   [kWh]

SEASON INDEPENDENCE
-------------------
The windows and the capacity are the SAME in every season, deliberately. No
source gives an hourly rationing schedule, a window length or a grid capacity
for DKN (see ``EDL_availability_sources.docx``), so any seasonal split here
would be assumed rather than observed -- and it would be assumed on top of
demand and PV data that ARE seasonal and ARE sourced. Holding EDL fixed keeps
the two apart: whatever differs between seasons in the results comes from the
demand profile and the PV capacity factor, not from a guess about the utility.

The seasonality does not disappear, it stops being an INPUT. The same 1650 kWh
window covers 39% of winter demand but only 24% of summer demand, because
summer demand is 64% higher; and the same six clock hours sit against an
evening-peaked winter profile but a NOCTURNAL summer one -- summer net demand
runs 355-415 kW from 22:00 to 06:00 and only 181-236 kW at midday, since
250 x 4 kWp of rooftop PV flattens the middle of the day. Summer therefore
remains by far the tightest season, as a computed result rather than as an
assumption.

What this gives up is the DIRECTION of a seasonal tightening, which is
supported -- Abi Ghanem (2018) notes outages are "exacerbated at times when
there is additional demand, such as during the heat of summer" -- even though
its magnitude is not. That belongs in a sensitivity run rather than the base
case, where it could not be told apart from the demand seasonality above.
An earlier version of this module encoded it three times over (season-specific
capacity, window length AND window placement); the placement was the most
influential of the three and the least visible.

The six scenarios:

    announced    supply delivered exactly as EDL announced it
    partial      announced window, but at half capacity
    delayed      the announced supply windows arrive a few hours late
    bonus        announced windows, extended by an hour on each side
    full_supply  uninterrupted 24 h supply at full capacity
    none         EDL fully off for the whole day

The six probabilities sum to 1 and are shared across seasons.

Usage
-----
    from scenarios import grid_scenarios

    S, prob, QbarPG = grid_scenarios("winter", T=range(24))
    #   S            : list of scenario keys
    #   prob[s]      : scenario probability (sum to 1)
    #   QbarPG[t, s] : max EDL supply in hour t under scenario s [kWh]

``season`` is still taken (and validated) so callers read naturally and typos
are caught, but per the above the returned profiles no longer depend on it.

All numeric levels below are modelling assumptions for the DKN case study.
Sources are given with each one; the full discussion, including what is and is
not supported by the literature, is in ``EDL_availability_sources.docx``.
"""

from dataclasses import dataclass
from typing import Iterable

import config


@dataclass(frozen=True)
class GridScenario:
    key: str
    label: str
    prob: float
    window: str | None                 # key into WINDOW_HOURS; None -> no supply
    avail_fraction: float = 1.0        # scales the cap in the "partial" case


# ---------------------------------------------------------------------------
# The scenario set (probabilities must sum to 1).                 [ASSUMPTION]
# ---------------------------------------------------------------------------
SCENARIOS: tuple[GridScenario, ...] = (
    GridScenario("announced",   "As announced",   0.40, "announced"),
    GridScenario("partial",     "Reduced supply", 0.15, "announced", 0.5),
    GridScenario("delayed",     "Shifted window", 0.20, "delayed"),
    GridScenario("bonus",       "Extended window", 0.10, "bonus"),
    GridScenario("full_supply", "Full supply",    0.05, "full"),
    GridScenario("none",        "No supply",      0.10, None),
)

assert abs(sum(s.prob for s in SCENARIOS) - 1.0) < 1e-9

# Convenience views, so consumers need not walk the dataclass.
SCENARIO_PROB: dict[str, float] = {s.key: s.prob for s in SCENARIOS}
SCENARIO_LABEL: dict[str, str] = {s.key: s.label for s in SCENARIOS}

# ---------------------------------------------------------------------------
# Baseline EDL grid capacity while supplying [kW].                [ASSUMPTION]
# Season-independent; see SEASON INDEPENDENCE in the module docstring.
#
# No source gives a grid capacity for DKN. Bou Gebrael, E., Olleik, M., &
# Zwickl-Bernhard, S. (2026), "Market Power and Distributed Solar Integration
# in Microgrids under Limited Regulation", arXiv:2603.16893 -- the paper the
# rest of the physical setup (400 kW genset, 650 households, 4 kWp per PV
# owner) comes from -- says only that EDL is "highly intermittent", with
# regional outages "of up to 20 hours per day". 275 kW is the midpoint of the
# 250-300 kW range this model previously assumed season by season, so the
# collapse to one number is neutral with respect to the old assumption.
#
# The level matters less than it looks. Village net demand stays below 275 kW
# in every hour of the announced window except summer h06 (339 kW) and h20
# (328 kW), so under "announced" the cap is slack and it is DEMAND, not EDL
# capacity, that sets the delivered energy -- the window carries the
# availability story almost by itself. The capacity does real work only in the
# "partial" scenario, where 0.5 x 275 = 138 kW binds right across the window.
# ---------------------------------------------------------------------------
BASELINE_KW: float = 275.0

# ---------------------------------------------------------------------------
# Availability windows, as hours (hour-beginning, 0..23).         [ASSUMPTION]
# One set for the whole year; see SEASON INDEPENDENCE in the module docstring.
#
# Block STRUCTURE -- Abi Ghanem, D. (2018), "Energy, the city and everyday
#   life: Living with power outages in post-war Lebanon", Energy Research &
#   Social Science. Rationing runs in daily BLOCKS: 3 h in Beirut, 8 h or
#   longer outside it. Hence contiguous morning/evening blocks rather than
#   hours scattered over the day.
#
# LENGTH (6 h) -- L'Orient Today (2025), "EDL announces return to 'normal'
#   electricity supply". By late February 2025 nationwide supply had recovered
#   to roughly 6-8 h/day. That date falls INSIDE the August 2024 - April 2025
#   period the DKN load loggers ran in (Bou Gebrael et al. 2026), so the window
#   length is anchored to the same period as the demand data. For scale, Human
#   Rights Watch (2023), "Cut Off From Life Itself", surveyed a median of
#   2 h/day in 2021/22 and ~12 h/day before the 2019 collapse: 6 h is a
#   recovery-period figure, neither a crisis nor a pre-crisis one. HRW also
#   found South Lebanon the LEAST rationed region (7 h/day without power,
#   against 11 h in Beirut and Mount Lebanon), which argues that 6 h is
#   conservative for DKN rather than generous.
#
# PLACEMENT (morning + evening) -- EDL rations against NATIONAL load, which is
#   morning- and evening-peaked. The window is deliberately NOT fitted to DKN's
#   own net peak: that peak is a local artifact of 250 x 4 kWp of rooftop PV,
#   which EDL neither sees nor schedules around. Any misalignment between the
#   window and village net demand is therefore a finding, not a free parameter.
#
# "delayed" is the announced window arriving two hours late.
# ---------------------------------------------------------------------------
WINDOW_HOURS: dict[str, tuple[int, ...]] = {
    "announced": (6, 7, 8, 18, 19, 20),
    "delayed":   (8, 9, 10, 20, 21, 22),
}


def _check_season(season: str) -> None:
    """Raise if ``season`` is not one of the representative days in config."""
    if season not in config.SEASONS:
        raise ValueError(
            f"Unknown season {season!r}. Expected one of {tuple(config.SEASONS)}."
        )


def _extend(hours: tuple[int, ...], by: int = 1,
            horizon: int = 24) -> tuple[int, ...]:
    """Widen each contiguous block of ``hours`` by ``by`` hours on both sides."""
    out = {h + d for h in hours for d in range(-by, by + 1)}
    return tuple(sorted(h for h in out if 0 <= h < horizon))


# The "bonus" window is the announced one extended by an hour on each side, and
# "full" is the whole day; both are derived so they follow any change to the
# announced window above.
WINDOW_HOURS["bonus"] = _extend(WINDOW_HOURS["announced"])
WINDOW_HOURS["full"] = tuple(range(24))


def grid_scenarios(
    season: str, T: Iterable[int] = tuple(range(24))
) -> tuple[list[str], dict[str, float], dict[tuple[int, str], float]]:
    """
    Public-grid (EDL) availability scenarios for ``season``.

    The EDL assumptions are season-independent (see the module docstring), so
    the returned profiles are the same for every season. ``season`` is kept and
    validated so callers read naturally and typos are still caught; the
    seasonality of the instance enters through the demand and PV data instead.

    Parameters
    ----------
    season : str
        One of :data:`config.SEASONS` ("summer", "winter", "fall_spring").
    T : iterable of int, optional
        Hour indices to build the profiles for (default ``range(24)``).

    Returns
    -------
    S : list[str]
        Scenario keys, in the fixed order of :data:`SCENARIOS`.
    prob : dict[str, float]
        ``{scenario: probability}``, summing to 1.
    QbarPG : dict[tuple[int, str], float]
        ``{(hour, scenario): max EDL supply [kWh]}``.

    Raises
    ------
    ValueError
        If ``season`` is unknown.
    """
    _check_season(season)

    hours = tuple(T)
    S = [s.key for s in SCENARIOS]
    prob = dict(SCENARIO_PROB)

    QbarPG: dict[tuple[int, str], float] = {}
    for sc in SCENARIOS:
        avail = () if sc.window is None else WINDOW_HOURS[sc.window]
        for t in hours:
            QbarPG[(t, sc.key)] = (BASELINE_KW * sc.avail_fraction
                                   if t in avail else 0.0)

    return S, prob, QbarPG


# ---------------------------------------------------------------------------
# Parametric availability profiles (for the availability sweep)
#
# The six scenarios above are the case study: named, discrete outcomes with
# probabilities. To show how availability influences the leader GENERALLY, the
# analysis needs a continuous knob instead -- a profile whose total energy is a
# chosen share of daily demand, laid out over the day by a chosen pattern.
# ---------------------------------------------------------------------------
AVAILABILITY_PATTERNS: tuple[str, ...] = (
    "proportional",   # Qbar_t ~ D_t: every hour leaves the MGO's market at the
                      # same price, so this is the kink-free reference case the
                      # closed form describes exactly
    "uniform",        # the same energy in every hour of the day
    "peak",           # concentrated on the highest-demand hours
    "offpeak",        # concentrated on the lowest-demand hours
    "announced",      # the season's announced EDL window
    "delayed",        # the announced window, arriving late
)

# How many hours the concentrated patterns spread over.            [ASSUMPTION]
_CONCENTRATED_HOURS = 8


def availability_shape(season: str, pattern: str,
                       D: dict[int, float]) -> dict[int, float]:
    """
    Hourly weights of an availability pattern, summing to 1.

    ``D`` is the hourly demand, which the demand-dependent patterns
    ("proportional", "peak", "offpeak") are defined against. ``season`` no
    longer selects the window -- the EDL windows are season-independent -- but
    is still validated, and the demand-dependent patterns remain seasonal
    through ``D``.
    """
    hours = sorted(D)
    if pattern == "proportional":
        total = sum(D.values())
        return {t: D[t] / total for t in hours}

    if pattern == "uniform":
        active = tuple(hours)
    elif pattern in ("peak", "offpeak"):
        ranked = sorted(hours, key=lambda t: D[t], reverse=(pattern == "peak"))
        active = tuple(ranked[:_CONCENTRATED_HOURS])
    elif pattern in ("announced", "delayed"):
        _check_season(season)
        active = WINDOW_HOURS[pattern]
    else:
        raise ValueError(
            f"Unknown pattern {pattern!r}. Expected one of "
            f"{AVAILABILITY_PATTERNS}."
        )

    w = 1.0 / len(active)
    return {t: (w if t in active else 0.0) for t in hours}


def uncapped_availability(season: str, pattern: str,
                          D: dict[int, float]) -> dict[int, float]:
    """
    An hourly EDL availability profile that is UNLIMITED where it is on.

    The alternative to :func:`synthetic_availability`, and the one the timing
    analysis uses. There, a pattern is an energy spread over its hours, so EDL's
    cap binds inside the window and the hour is only partly covered. Here the
    cap is :data:`config.CONFIG` ``M_phys`` in every hour of the pattern and
    zero outside, so EDL covers those hours COMPLETELY -- whatever the demand in
    them, and whatever the village shifts into them -- and nothing elsewhere.

    Two consequences worth being explicit about:

      * The patterns no longer carry equal energy. ``peak`` takes over the
        highest-demand hours and ``offpeak`` the lowest, so peak delivers far
        more kWh; what separates the two is no longer timing at a fixed energy
        but WHICH hours EDL takes over entirely.
      * Load shifting gains a lever it did not have. With a binding cap there
        was nothing to gain by moving load into a covered hour once the cap was
        exhausted; with no cap, every kWh moved in is bought at ``P_EDL``
        instead of from the microgrid.

    Parameters
    ----------
    season : str
        Season key (validated for the window-based patterns).
    pattern : str
        One of :data:`AVAILABILITY_PATTERNS`.
    D : dict
        Hourly demand ``{hour: kWh}``, which the ranked patterns are defined
        against.

    Returns
    -------
    dict
        ``{hour: max EDL supply [kWh]}``, either ``M_phys`` or 0.
    """
    shape = availability_shape(season, pattern, D)
    cap = config.CONFIG["M_phys"]
    return {t: (cap if w > 0.0 else 0.0) for t, w in shape.items()}


def synthetic_availability(season: str, share: float, pattern: str,
                           D: dict[int, float]) -> dict[int, float]:
    """
    An hourly EDL availability profile carrying ``share`` of daily demand.

    Parameters
    ----------
    season : str
        Season key (only the window-based patterns depend on it).
    share : float
        Total availability as a fraction of total daily demand:
        ``sum_t Qbar_t = share * sum_t D_t``. This is the raw, policy-facing
        quantity -- how much energy EDL puts on offer, before asking WHEN.
    pattern : str
        One of :data:`AVAILABILITY_PATTERNS`.
    D : dict
        Hourly demand ``{hour: kWh}``.

    Returns
    -------
    dict
        ``{hour: max EDL supply [kWh]}``.
    """
    shape = availability_shape(season, pattern, D)
    energy = share * sum(D.values())
    return {t: energy * w for t, w in shape.items()}
