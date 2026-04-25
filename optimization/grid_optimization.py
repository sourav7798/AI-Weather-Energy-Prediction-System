"""
Simple grid optimization heuristics: balance demand vs supply, recommend
energy storage sizing and distribution across nodes.
"""
import numpy as np
import pandas as pd

def balance_supply_demand(supply_kw, demand_kw):
    """Return net surplus/deficit and suggested curtailment if surplus."""
    supply = np.sum(supply_kw)
    demand = np.sum(demand_kw)
    net = supply - demand
    suggestion = {}
    if net >= 0:
        suggestion['status'] = 'surplus'
        suggestion['surplus_kw'] = net
        # recommend storing up to 80% of surplus
        suggestion['recommend_storage_kw'] = net * 0.8
    else:
        suggestion['status'] = 'deficit'
        suggestion['deficit_kw'] = -net
        # recommend load shedding priority or discharge storage
        suggestion['recommend_discharge_kw'] = -net * 0.7
    return suggestion

def distribute_load(nodes_demand, total_supply_kw):
    """Distribute available supply proportionally to node demand.
    Returns allocation and unmet demand per node.
    """
    nodes = np.array(nodes_demand)
    total_demand = nodes.sum()
    if total_demand == 0:
        return np.zeros_like(nodes), np.zeros_like(nodes)
    alloc = nodes / total_demand * total_supply_kw
    unmet = np.maximum(nodes - alloc, 0)
    return alloc, unmet

def recommend_battery_size(hourly_deficit_series, safety_margin=1.2):
    """Suggest battery size (kWh) to cover peak deficit for a day.
    Takes a pandas Series of hourly deficits (kW).
    """
    peak = hourly_deficit_series.max()
    # assume 4-hour discharge duration
    capacity_kwh = peak * 4 * safety_margin
    return capacity_kwh
