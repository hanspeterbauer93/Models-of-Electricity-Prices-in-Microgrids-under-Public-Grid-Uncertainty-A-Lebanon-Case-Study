"""
Loader for the three microgrid input YAMLs.

Each YAML in ``data/`` describes one variable per (season, hour) with the
hour-key convention h00..h23:

    cap_factors.yaml       PV capacity factor φ           (kWh/kWp/h)
    demand_non_pv.yaml     non-PV household demand        (kWh/hh/h)
    demand_pv_owner.yaml   PV-owner household demand      (kWh/hh/h)

Typical use:

    data = load_data()
    data["seasons"]                       # ('summer', 'fall_spring', 'winter')
    data["cap_factor"]["winter"][12]      # 0.2759...
"""

from pathlib import Path

import yaml

import config

_YAML_FILES: dict[str, str] = {
    "cap_factor":      "cap_factors.yaml",
    "demand_non_pv":   "demand_non_pv.yaml",
    "demand_pv_owner": "demand_pv_owner.yaml",
}


def _parse_hour(key: str) -> int:
    """'h00' -> 0, 'h23' -> 23."""
    return int(key[1:])


def load_data() -> dict:
    """
    Read the three YAMLs and return a nested-dict bundle keyed for direct
    consumption in the MILP.

    Returns
    -------
    dict
        seasons         : tuple of season names (e.g. ("summer", "fall_spring", "winter"))
        hours           : tuple(range(24))
        cap_factor      : {season: {hour: float}}   — PV capacity factor φ
        demand_non_pv   : {season: {hour: float}}   — kWh/hh/h, non-PV household
        demand_pv_owner : {season: {hour: float}}   — kWh/hh/h, PV-owner household
        metadata        : {variable: {description, source_file, source_sheet, unit, ...}}
    """
    out: dict = {
        "cap_factor":      {},
        "demand_non_pv":   {},
        "demand_pv_owner": {},
        "metadata":        {},
    }

    for key, fname in _YAML_FILES.items():
        path: Path = config.DATA_DIR / fname
        with open(path) as f:
            doc = yaml.safe_load(f)

        out["metadata"][key] = {k: v for k, v in doc.items() if k != "seasons"}
        for season, hours in doc["seasons"].items():
            out[key][season] = {_parse_hour(h): float(v) for h, v in hours.items()}

    out["seasons"] = tuple(out["cap_factor"].keys())
    out["hours"]   = tuple(range(24))
    return out

def get_demand(data: dict, cfg: dict, season: str) -> dict[int, float]:
    """
    Hourly village net demand for ``season`` [kWh].

        net_demand[t] = N_nonPV * dem_nonPV[t]
                      + N_PV    * max(0, dem_PV[t] - cap_factor[t] * cap_PV_hh)

    The PV-owner term is clamped at ≥ 0: surplus PV is self-consumed
    only (no feed-in to the village grid).

    Parameters
    ----------
    data : dict
        The bundle returned by :func:`load_data`.
    cfg : dict
        Config providing ``n_Households_nonPV``, ``n_Households_PV`` and
        ``cap_PV_per_HH``.
    season : str
        One of ``data["seasons"]``.

    Returns
    -------
    dict[int, float]
        {hour: net_demand} for hours 0..23.
    """
    if data is None:
        raise RuntimeError(
            "data is unset — call load_data() and pass it to get_demand()."
        )
    if season not in data["seasons"]:
        raise ValueError(
            f"Unknown season {season!r}. Expected one of {data['seasons']}."
        )

    n_nonPV   = cfg["n_Households_nonPV"]
    n_PV      = cfg["n_Households_PV"]
    cap_PV_hh = cfg["cap_PV_per_HH"]

    dem_nonPV  = data["demand_non_pv"][season]
    dem_PV     = data["demand_pv_owner"][season]
    cap_factor = data["cap_factor"][season]

    return {
        t: n_nonPV * dem_nonPV[t]
           + n_PV * max(0.0, dem_PV[t] - cap_factor[t] * cap_PV_hh)
        for t in range(24)
    }
