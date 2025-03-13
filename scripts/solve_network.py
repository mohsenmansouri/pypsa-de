# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""

import importlib
import logging
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from _benchmark import memory_logger
from pypsa.descriptors import get_activity_mask
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.prepare_sector_network import get

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def add_land_use_constraint_perfect(n):
    """
    Add global constraints for tech capacity limit.
    """
    logger.info("Add land-use constraint for perfect foresight")

    def compress_series(s):
        def process_group(group):
            if group.nunique() == 1:
                return pd.Series(group.iloc[0], index=[None])
            else:
                return group

        return s.groupby(level=[0, 1]).apply(process_group)

    def new_index_name(t):
        # Convert all elements to string and filter out None values
        parts = [str(x) for x in t if x is not None]
        # Join with space, but use a dash for the last item if not None
        return " ".join(parts[:2]) + (f"-{parts[-1]}" if len(parts) > 2 else "")

    def check_p_min_p_max(p_nom_max):
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}"
            )

    grouper = [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ext_i = n.generators.p_nom_extendable
    # get technical limit per node and investment period
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max
    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]
    # carrier
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0
    # check minimum capacities
    check_p_min_p_max(p_nom_max)
    # drop multi entries in case p_nom_max stays constant in different periods
    # p_nom_max = compress_series(p_nom_max)
    # adjust name to fit syntax of nominal constraint per bus
    df = p_nom_max.reset_index()
    df["name"] = df.apply(
        lambda row: f"nom_max_{row['carrier']}"
        + (f"_{row['build_year']}" if row["build_year"] is not None else ""),
        axis=1,
    )

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values

    return n


def add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in [
        "solar",
        "solar rooftop",
        "solar-hsat",
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
    ]:
        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        existing = (
            n.generators.loc[ext_i, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    # check if existing capacities are larger than technical potential
    existing_large = n.generators[
        n.generators["p_nom_min"] > n.generators["p_nom_max"]
    ].index
    if len(existing_large):
        logger.warning(
            f"Existing capacities larger than technical potential for {existing_large},\
                        adjust technical potential to existing capacities"
        )
        n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
            existing_large, "p_nom_min"
        ]

    n.generators["p_nom_max"] = n.generators["p_nom_max"].clip(lower=0)


def add_solar_potential_constraints(n, config):
    """
    Add constraint to make sure the sum capacity of all solar technologies (fixed, tracking, ets. ) is below the region potential.
    Example:
    ES1 0: total solar potential is 10 GW, meaning:
           solar potential : 10 GW
           solar-hsat potential : 8 GW (solar with single axis tracking is assumed to have higher land use)
    The constraint ensures that:
           solar_p_nom + solar_hsat_p_nom * 1.13 <= 10 GW
    """
    land_use_factors = {
        "solar-hsat": config["renewable"]["solar"]["capacity_per_sqkm"]
        / config["renewable"]["solar-hsat"]["capacity_per_sqkm"],
    }
    rename = {"Generator-ext": "Generator"}

    solar_carriers = ["solar", "solar-hsat"]
    solar = n.generators[
        n.generators.carrier.isin(solar_carriers) & n.generators.p_nom_extendable
    ].index

    solar_today = n.generators[
        (n.generators.carrier == "solar") & (n.generators.p_nom_extendable)
    ].index
    solar_hsat = n.generators[(n.generators.carrier == "solar-hsat")].index

    if solar.empty:
        return

    land_use = pd.DataFrame(1, index=solar, columns=["land_use_factor"])
    for carrier, factor in land_use_factors.items():
        land_use = land_use.apply(
            lambda x: (x * factor) if carrier in x.name else x, axis=1
        )

    location = pd.Series(n.buses.index, index=n.buses.index)
    ggrouper = n.generators.loc[solar].bus
    rhs = (
        n.generators.loc[solar_today, "p_nom_max"]
        .groupby(n.generators.loc[solar_today].bus.map(location))
        .sum()
        - n.generators.loc[solar_hsat, "p_nom"]
        .groupby(n.generators.loc[solar_hsat].bus.map(location))
        .sum()
        * land_use_factors["solar-hsat"]
    ).clip(lower=0)

    lhs = (
        (n.model["Generator-p_nom"].rename(rename).loc[solar] * land_use.squeeze())
        .groupby(ggrouper)
        .sum()
    )

    logger.info("Adding solar potential constraint.")
    n.model.add_constraints(lhs <= rhs, name="solar_potential")


def add_co2_sequestration_limit(n, limit_dict):
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.
    """

    if not n.investment_periods.empty:
        periods = n.investment_periods
        limit = pd.Series(
            {
                f"co2_sequestration_limit-{period}": limit_dict.get(period, 200)
                for period in periods
            }
        )
        names = limit.index
    else:
        limit = get(limit_dict, int(snakemake.wildcards.planning_horizons))
        periods = [np.nan]
        names = pd.Index(["co2_sequestration_limit"])

    n.add(
        "GlobalConstraint",
        names,
        sense=">=",
        constant=-limit * 1e6,
        type="operational_limit",
        carrier_attribute="co2 sequestered",
        investment_period=periods,
    )


def add_carbon_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "co2_atmosphere"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            lhs = final_e.loc[time_i, :] - final_e.shift(snapshot=1).loc[time_i, :]

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_carbon_budget_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Budget"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            weighting = n.investment_period_weightings.loc[time_valid, "years"]
            lhs = final_e.loc[time_i, :] * weighting

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_max_growth(n):
    """
    Add maximum growth rates for different carriers.
    """

    opts = snakemake.params["sector"]["limit_max_growth"]
    # take maximum yearly difference between investment periods since historic growth is per year
    factor = n.investment_period_weightings.years.max() * opts["factor"]
    for carrier in opts["max_growth"].keys():
        max_per_period = opts["max_growth"][carrier] * factor
        logger.info(
            f"set maximum growth rate per investment period of {carrier} to {max_per_period} GW."
        )
        n.carriers.loc[carrier, "max_growth"] = max_per_period * 1e3

    for carrier in opts["max_relative_growth"].keys():
        max_r_per_period = opts["max_relative_growth"][carrier]
        logger.info(
            f"set maximum relative growth per investment period of {carrier} to {max_r_per_period}."
        )
        n.carriers.loc[carrier, "max_relative_growth"] = max_r_per_period

    return n


def add_retrofit_gas_boiler_constraint(n, snapshots):
    """
    Allow retrofitting of existing gas boilers to H2 boilers.
    """
    c = "Link"
    logger.info("Add constraint for retrofitting gas boilers to H2 boilers.")
    # existing gas boilers
    mask = n.links.carrier.str.contains("gas boiler") & ~n.links.p_nom_extendable
    gas_i = n.links[mask].index
    mask = n.links.carrier.str.contains("retrofitted H2 boiler")
    h2_i = n.links[mask].index

    n.links.loc[gas_i, "p_nom_extendable"] = True
    p_nom = n.links.loc[gas_i, "p_nom"]
    n.links.loc[gas_i, "p_nom"] = 0

    # heat profile
    cols = n.loads_t.p_set.columns[
        n.loads_t.p_set.columns.str.contains("heat")
        & ~n.loads_t.p_set.columns.str.contains("industry")
        & ~n.loads_t.p_set.columns.str.contains("agriculture")
    ]
    profile = n.loads_t.p_set[cols].div(
        n.loads_t.p_set[cols].groupby(level=0).max(), level=0
    )
    # to deal if max value is zero
    profile.fillna(0, inplace=True)
    profile.rename(columns=n.loads.bus.to_dict(), inplace=True)
    profile = profile.reindex(columns=n.links.loc[gas_i, "bus1"])
    profile.columns = gas_i

    rhs = profile.mul(p_nom)

    dispatch = n.model["Link-p"]
    active = get_activity_mask(n, c, snapshots, gas_i)
    rhs = rhs[active]
    p_gas = dispatch.sel(Link=gas_i)
    p_h2 = dispatch.sel(Link=h2_i)

    lhs = p_gas + p_h2

    n.model.add_constraints(lhs == rhs, name="gas_retrofit")


def prepare_network(
    n,
    solve_opts=None,
    config=None,
    foresight=None,
    planning_horizons=None,
    co2_sequestration_potential=None,
):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.links_t.p_max_pu,
            n.links_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if load_shedding := solve_opts.get("load_shedding"):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.add(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("curtailment_mode"):
        n.add("Carrier", "curtailment", color="#fedfed", nice_name="Curtailment")
        n.generators_t.p_min_pu = n.generators_t.p_max_pu
        buses_i = n.buses.query("carrier == 'AC'").index
        n.add(
            "Generator",
            buses_i,
            suffix=" curtailment",
            bus=buses_i,
            p_min_pu=-1,
            p_max_pu=0,
            marginal_cost=-0.1,
            carrier="curtailment",
            p_nom=1e6,
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if foresight == "myopic":
        add_land_use_constraint(n)

    if foresight == "perfect":
        n = add_land_use_constraint_perfect(n)
        if snakemake.params["sector"]["limit_max_growth"]["enable"]:
            n = add_max_growth(n)

    if n.stores.carrier.eq("co2 sequestered").any():
        limit_dict = co2_sequestration_potential
        add_co2_sequestration_limit(n, limit_dict=limit_dict)

        # Add DSM if enabled
    if config["sector"].get("dsm", {}).get("enable", False):
        logging.info("[Setup] Adding demand-side management")
        n = add_dsm(n, config)  # Add DSM components

    return n


def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network and set fixed capacities.
    """
    logger.info("=== Starting CCL Constraint Addition ===")

    try:
        # Read limits with multi-level columns
        agg_p_nom_minmax = pd.read_csv(
            config["solving"]["agg_p_nom_limits"]["file"],
            index_col=[0, 1],
            header=[0, 1]
        )[snakemake.wildcards.planning_horizons]

        # List of technologies to fix
        fixed_techs = ['onwind', 'offwind-ac', 'offwind-dc', 'solar', 'solar rooftop']

        for idx, row in agg_p_nom_minmax.iterrows():
            country, carrier = idx
            min_val = row['min']
            max_val = row['max']

            # Get all generators for this country and carrier
            gens_mask = (
                    (n.generators.bus.map(n.buses.country) == country) &
                    (n.generators.carrier == carrier)
            )
            relevant_gens = n.generators[gens_mask]

            if relevant_gens.empty:
                logger.warning(f"No generators found for {carrier} in {country}")
                continue

            if carrier in fixed_techs:
                # For fixed technologies, set exact capacities
                if not np.isclose(min_val, max_val):
                    logger.warning(
                        f"Min and max values differ for fixed technology {carrier}. Using min value: {min_val}")

                target_capacity = min_val
                current_capacity = relevant_gens.p_nom.sum()

                logger.info(f"\nSetting fixed capacity for {carrier} in {country}:")
                logger.info(f"Target capacity: {target_capacity:.2f} MW")
                logger.info(f"Current capacity: {current_capacity:.2f} MW")

                # Make all generators non-extendable
                n.generators.loc[relevant_gens.index, 'p_nom_extendable'] = False

                if len(relevant_gens) > 1:
                    # Distribute capacity proportionally among generators
                    scaling_factor = target_capacity / current_capacity
                    for idx in relevant_gens.index:
                        old_capacity = n.generators.loc[idx, 'p_nom']
                        new_capacity = old_capacity * scaling_factor
                        n.generators.loc[idx, 'p_nom'] = new_capacity
                        logger.info(f"Generator {idx}: {old_capacity:.2f} MW → {new_capacity:.2f} MW")
                else:
                    # Set capacity directly for single generator
                    idx = relevant_gens.index[0]
                    old_capacity = n.generators.loc[idx, 'p_nom']
                    n.generators.loc[idx, 'p_nom'] = target_capacity
                    logger.info(f"Generator {idx}: {old_capacity:.2f} MW → {target_capacity:.2f} MW")

                # Verify the total capacity
                new_total = n.generators.loc[relevant_gens.index, 'p_nom'].sum()
                logger.info(f"New total capacity: {new_total:.2f} MW")

                if not np.isclose(new_total, target_capacity, rtol=1e-3):
                    logger.warning(
                        f"Warning: New capacity {new_total:.2f} MW differs from target {target_capacity:.2f} MW")

            else:
                # For other technologies, add min/max constraints as before
                extendable_gens = relevant_gens[relevant_gens.p_nom_extendable]

                if extendable_gens.empty:
                    logger.warning(f"No extendable generators found for {carrier} in {country}")
                    continue

                # Get model components
                p_nom = n.model["Generator-p_nom"]
                lhs = p_nom.loc[extendable_gens.index].sum()

                # Add minimum constraint if applicable
                if min_val > 0:
                    constraint_name = f"agg_p_nom_min_{country}_{carrier}"
                    n.model.add_constraints(
                        lhs >= min_val,
                        name=constraint_name
                    )
                    logger.info(f"Added minimum constraint: {constraint_name} >= {min_val}")

                # Add maximum constraint if applicable
                if max_val < float('inf'):
                    constraint_name = f"agg_p_nom_max_{country}_{carrier}"
                    n.model.add_constraints(
                        lhs <= max_val,
                        name=constraint_name
                    )
                    logger.info(f"Added maximum constraint: {constraint_name} <= {max_val}")

        logger.info("\nAll CCL constraints and fixed capacities set successfully")

    except Exception as e:
        logger.error(f"Error in CCL constraint addition: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

    return n


import logging

import logging


import logging


def add_CLL_constraints(n, config):
    """
    Add CLL constraints for coal, lignite, CCGT, and OCGT (implemented as links)
    using planning_year from snakemake.wildcards.planning_horizons.

    For each year, a dictionary defines min/max capacities.
    The function zeroes out existing capacities in DE and then adds or updates one link per carrier.

    For gas-based carriers:
      - For CCGT we now fix the capacity (p_nom_extendable=False) so that the solver is forced
        to invest at least the minimum capacity.
      - For OCGT the link remains extendable.

    Make sure that the buses 'EU gas', 'EU coal', 'EU lignite', and 'DE0 0' exist in n.buses.
    """
    logging.info("====== STARTING CLL CONSTRAINTS FOR LINKS AND GENERATORS ======")

    try:
        # 1) Read planning year from snakemake wildcards
        planning_year = int(snakemake.wildcards.planning_horizons)
        de_buses = n.buses[n.buses.country == 'DE'].index

        # 2) Capacity targets for each year
        capacities = {
            '2020': {
                'coal': {'min': 20000, 'max': 20000},
                'lignite': {'min': 20000, 'max': 20000},
                'CCGT': {'min': 10000, 'max': 80000},
                'OCGT': {'min': 50, 'max': 5000}
            },

            '2025': {
                'coal': {'min': 17000, 'max': 17000},
                'lignite': {'min': 17000, 'max': 17000},
                'CCGT': {'min': 10000, 'max': 80000},
                'OCGT': {'min': 50, 'max': 5000}
            },

            '2030': {
                'coal': {'min': 8000, 'max': 8000},
                'lignite': {'min': 8000, 'max': 8000},
                'CCGT': {'min': 39000, 'max': 80000},
                'OCGT': {'min': 50, 'max': 5000}
            },

            '2035': {
                'coal': {'min': 2000, 'max': 2000},
                'lignite': {'min': 2000, 'max': 2000},
                'CCGT': {'min': 50000, 'max': 80000},
                'OCGT': {'min': 50, 'max': 5000}
            },

            '2040': {
                'coal': {'min': 0, 'max': 0},
                'lignite': {'min': 0, 'max': 0},
                'CCGT': {'min': 52000, 'max': 80000},
                'OCGT': {'min': 50, 'max': 5000}
            },
            '2045': {
                'coal': {'min': 0, 'max': 0},
                'lignite': {'min': 0, 'max': 0},
                'CCGT': {'min': 51000, 'max': 80000},
                'OCGT': {'min': 50, 'max': 5000}
            }
        }

        year_str = str(planning_year)
        if year_str not in capacities:
            raise ValueError(f"No capacity targets defined for year {year_str}")

        year_capacities = capacities[year_str]

        # 3) Zero out existing capacity in DE for the four carriers
        for carrier in ['coal', 'lignite', 'CCGT', 'OCGT']:
            if carrier not in n.carriers.index:
                logging.info(f"Defining missing carrier '{carrier}' in n.carriers.")
                n.carriers.loc[carrier, "nice_name"] = carrier

            existing_units = n.links.loc[
                (n.links.carrier == carrier) &
                (n.links.bus1.isin(de_buses))
                ]
            if not existing_units.empty:
                logging.info(f"Zeroing capacity for {len(existing_units)} existing {carrier} link(s) in DE.")
                n.links.loc[existing_units.index, "p_nom"] = 0
                n.links.loc[existing_units.index, "p_nom_extendable"] = False
                n.links.loc[existing_units.index, "p_nom_min"] = 0
                n.links.loc[existing_units.index, "p_nom_max"] = 0

        # 4) Add or update link for each carrier from the capacity dictionary
        for carrier, caps in year_capacities.items():
            min_cap = caps['min']
            max_cap = caps['max']

            if min_cap <= 0 and max_cap <= 0:
                logging.info(f"{carrier.capitalize()} capacity in {year_str} is 0 MW - skipping link creation.")
                continue

            new_unit = f"DE0 0 {carrier}-{planning_year}"
            if carrier in ['CCGT', 'OCGT']:
                bus0 = 'EU gas'
                # For CCGT, we now force a fixed capacity; for OCGT, use extendable as usual.
                if carrier == 'CCGT':
                    is_extendable = False
                else:
                    is_extendable = (min_cap != max_cap)
                efficiency = 0.6 if carrier == 'CCGT' else 0.4
            else:
                bus0 = f"EU {carrier}"
                is_extendable = (min_cap != max_cap)
                efficiency = 1.0

            if "DE0 0" not in n.buses.index:
                logging.warning("Bus 'DE0 0' not found in n.buses; link might not connect properly.")

            if new_unit in n.links.index:
                n.links.at[new_unit, 'p_nom'] = min_cap
                n.links.at[new_unit, 'p_nom_min'] = min_cap
                n.links.at[new_unit, 'p_nom_max'] = max_cap
                n.links.at[new_unit, 'p_nom_extendable'] = is_extendable
                n.links.at[new_unit, 'efficiency'] = efficiency
                msg = "Updated"
            else:
                n.add(
                    "Link",
                    new_unit,
                    bus0=bus0,
                    bus1="DE0 0",
                    carrier=carrier,
                    p_nom=min_cap,
                    p_nom_min=min_cap,
                    p_nom_max=max_cap,
                    p_nom_extendable=is_extendable,
                    efficiency=efficiency
                )
                msg = "Added"

            if min_cap == max_cap:
                logging.info(f"{msg} {carrier} unit with fixed capacity: {min_cap} MW for year {year_str}")
            else:
                logging.info(f"{msg} {carrier} unit with capacity range: {min_cap}-{max_cap} MW for year {year_str}")

        # 5) Log final capacities for each carrier
        logging.info("\n====== FINAL CONFIGURATIONS ======")
        total_capacities = {c: 0 for c in ['coal', 'lignite', 'CCGT', 'OCGT']}

        for carrier in ['coal', 'lignite', 'CCGT', 'OCGT']:
            units = n.links.loc[
                (n.links.carrier == carrier) &
                (n.links.bus1.isin(de_buses))
                ]
            total_cap = units["p_nom"].sum()
            total_min = units["p_nom_min"].sum()
            total_max = units["p_nom_max"].sum()
            total_capacities[carrier] = total_min

            logging.info(f"\nDE {carrier}:")
            logging.info(f"  - Current capacity: {total_cap:.1f} MW")
            logging.info(f"  - Min capacity: {total_min:.1f} MW")
            logging.info(f"  - Max capacity: {total_max:.1f} MW")

        logging.info("\nTotal Capacities:")
        for c, cap in total_capacities.items():
            logging.info(f"{c}: {cap:.1f} MW")
        logging.info(f"Total Gas: {total_capacities['CCGT'] + total_capacities['OCGT']:.1f} MW")

    except Exception as e:
        logging.error(f"Error while processing limits: {e}", exc_info=True)
        raise

    return n


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    # TODO: Generalize to cover myopic and other sectors?
    float_regex = r"[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (
        (p * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling))
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24h]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]  # noqa: F841
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = summed_reserve + (
            p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))
        ).sum("Generator")

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_variable = n.model["Generator-p_nom"].rename(
        {"Generator-ext": "Generator"}
    )
    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve - capacity_variable * xr.DataArray(p_max_pu[ext_i])

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_lossy_bidirectional_link_constraints(n):
    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841

    forward_i = n.links.query(
        "carrier in @carriers and ~reversed and p_nom_extendable"
    ).index

    def get_backward_i(forward_i):
        return pd.Index(
            [
                (
                    re.sub(r"-(\d{4})$", r"-reversed-\1", s)
                    if re.search(r"-\d{4}$", s)
                    else s + "-reversed"
                )
                for s in forward_i
            ]
        )

    backward_i = get_backward_i(forward_i)

    lhs = n.model["Link-p_nom"].loc[backward_i]
    rhs = n.model["Link-p_nom"].loc[forward_i]

    n.model.add_constraints(lhs == rhs, name="Link-bidirectional_sync")


def add_chp_constraints(n):
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    if "reversed" not in n.links.columns:
        n.links["reversed"] = False
    gas_pipes_i = n.links.query(
        "carrier == 'gas pipeline' and p_nom_extendable and ~reversed"
    ).index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable and ~reversed"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i].rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")


def add_flexible_egs_constraint(n):
    """
    Upper bounds the charging capacity of the geothermal reservoir according to
    the well capacity.
    """
    well_index = n.links.loc[n.links.carrier == "geothermal heat"].index
    storage_index = n.storage_units.loc[
        n.storage_units.carrier == "geothermal heat"
    ].index

    p_nom_rhs = n.model["Link-p_nom"].loc[well_index]
    p_nom_lhs = n.model["StorageUnit-p_nom"].loc[storage_index]

    n.model.add_constraints(
        p_nom_lhs <= p_nom_rhs,
        name="upper_bound_charging_capacity_of_geothermal_reservoir",
    )


def add_co2_atmosphere_constraint(n, snapshots):
    glcs = n.global_constraints[n.global_constraints.type == "co2_atmosphere"]

    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last_i = snapshots[-1]
            lhs = n.model["Store-e"].loc[last_i, stores.index]
            rhs = glc.constant

            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


import logging


def add_heat_pump_constraints(n):
    """
    Imposes min/max capacity constraints on heat pumps in Germany.
    """
    import logging
    import re


    logging.info("[HP constraints] Starting heat pump constraint application...")
    print("DEBUG: Heat pump constraints function is being called")

    # Get planning year more aggressively with multiple fallbacks
    planning_year = None

    # Method 1: Try to get from snakemake wildcards
    try:
        planning_year = int(snakemake.wildcards.planning_horizons)
        logging.info(f"[HP constraints] Got planning year {planning_year} from snakemake wildcards")
    except (NameError, AttributeError, KeyError, ValueError) as e:
        logging.info(f"[HP constraints] Could not get year from snakemake wildcards: {e}")

    # Method 2: Try to extract from network filename or attributes
    if planning_year is None:
        # Try from network name
        if hasattr(n, 'name'):
            match = re.search(r'(\d{4})', n.name)
            if match:
                planning_year = int(match.group(1))
                logging.info(f"[HP constraints] Extracted planning year {planning_year} from network name: {n.name}")

        # Try from filenames in metadata
        if planning_year is None and hasattr(n, 'meta'):
            for key, value in n.meta.items():
                if isinstance(value, str) and 'filename' in key.lower():
                    match = re.search(r'(\d{4})', value)
                    if match:
                        planning_year = int(match.group(1))
                        logging.info(f"[HP constraints] Extracted planning year {planning_year} from metadata: {value}")
                        break

    # Method 3: Look at the links in the network to find the planning year
    if planning_year is None:
        # Look at link names for year patterns
        years = []
        for link_name in n.links.index:
            if isinstance(link_name, str):
                match = re.search(r'-(\d{4})$', link_name)
                if match:
                    years.append(int(match.group(1)))

        if years:
            from collections import Counter
            most_common_year = Counter(years).most_common(1)[0][0]
            planning_year = most_common_year
            logging.info(f"[HP constraints] Using most common year {planning_year} from link names")

    # Method 4: Extract from the network filename if passed via n
    if planning_year is None and hasattr(n, 'filename'):
        match = re.search(r'(\d{4})', n.filename)
        if match:
            planning_year = int(match.group(1))
            logging.info(f"[HP constraints] Extracted planning year {planning_year} from n.filename")

    # Method 5: Fallback to snapshots as last resort
    if planning_year is None:
        try:
            if hasattr(n, 'snapshots') and len(n.snapshots) > 0:
                from collections import Counter
                years = [ts.year for ts in n.snapshots]
                most_common_year = Counter(years).most_common(1)[0][0]

                # If snapshot year is 2019, map it to the actual planning year
                # based on common patterns
                if most_common_year == 2019:
                    # Check if we're running with a specific config pattern that might indicate year
                    if "config" in dir(snakemake) and "scenario" in dir(snakemake.config):
                        scenario = snakemake.config.scenario
                        if "2030" in scenario:
                            planning_year = 2030
                        elif "2040" in scenario:
                            planning_year = 2040
                        elif "2045" in scenario:
                            planning_year = 2045
                        elif "2035" in scenario:
                            planning_year = 2035
                        elif "2025" in scenario:
                            planning_year = 2025
                        else:
                            # Fallback to using a fixed year if snapshot year is 2019
                            print(f"DEBUG: Using hardcoded year instead of {most_common_year}")
                            planning_year = 2030  # Default to 2030 if snapshot is 2019
                    else:
                        # Hardcode 2030 if snapshots are 2019
                        planning_year = 2030  # Default to 2030 if snapshot is 2019
                else:
                    planning_year = most_common_year

                logging.info(
                    f"[HP constraints] Using year {planning_year} (mapped from snapshots year {most_common_year})")
        except Exception as e:
            logging.error(f"[HP constraints] Failed to determine planning year from snapshots: {e}")

    # If all methods fail, use a hardcoded year as absolute last resort
    if planning_year is None:
        planning_year = 2030  # Default to 2030 if all else fails
        logging.warning(f"[HP constraints] Using hardcoded fallback year {planning_year}")

    year_str = str(planning_year)
    logging.info(f"[HP constraints] Final planning year: {year_str}")

    # 2) Predefine min/max capacity for relevant heat pump carriers, by year
    hp_capacities = {
        "2020": {
            "rural air heat pump": {"min": 0, "max": 1500},
            "rural ground heat pump": {"min": 0, "max": 1500},
            "urban central air heat pump": {"min": 0, "max": 2000},
            "urban decentral air heat pump": {"min": 0, "max": 2000},
        },
        "2025": {
            "rural air heat pump": {"min": 500, "max": 3500},
            "rural ground heat pump": {"min": 300, "max": 3500},
            "urban central air heat pump": {"min": 1000, "max": 5000},
            "urban decentral air heat pump": {"min": 1000, "max": 5000},
        },
        "2030": {
            "rural air heat pump": {"min": 100, "max": 100},
            "rural ground heat pump": {"min": 200, "max": 200},
            "urban central air heat pump": {"min": 300, "max": 300},
            "urban decentral air heat pump": {"min": 400, "max": 400},
        },
        "2035": {
            "rural air heat pump": {"min": 2000, "max": 9000},
            "rural ground heat pump": {"min": 1000, "max": 8000},
            "urban central air heat pump": {"min": 3000, "max": 12000},
            "urban decentral air heat pump": {"min": 3000, "max": 12000},
        },
        "2040": {
            "rural air heat pump": {"min": 3000, "max": 3000},
            "rural ground heat pump": {"min": 3000, "max": 3000},
            "urban central air heat pump": {"min": 6000, "max": 6000},
            "urban decentral air heat pump": {"min": 6000, "max": 6000},
        },
        "2045": {
            "rural air heat pump": {"min": 3000, "max": 3000},
            "rural ground heat pump": {"min": 3000, "max": 3000},
            "urban central air heat pump": {"min": 6000, "max": 6000},
            "urban decentral air heat pump": {"min": 6000, "max": 6000},
        },
    }

    if year_str not in hp_capacities:
        logging.warning(f"[HP constraints] No HP capacity targets defined for {year_str}; skipping.")
        return

    # Log all available carriers to help with debugging
    all_carriers = n.links.carrier.unique()
    logging.info(f"[HP constraints] All link carriers in network: {all_carriers}")

    # 3) Gather the bus indices for Germany
    de_buses = n.buses.index[n.buses.country == "DE"]
    if de_buses.empty:
        logging.warning("[HP constraints] Found no DE buses in the network. Skipping.")
        return

    # 4) Load the capacity targets for this year
    year_dict = hp_capacities[year_str]

    # Check for missing carriers
    missing_carriers = set(year_dict.keys()) - set(all_carriers)
    if missing_carriers:
        logging.warning(f"[HP constraints] Some heat pump carriers not found in network: {missing_carriers}")

    # 5) Loop over each HP carrier, apply min/max constraints
    logging.info(
        f"[HP constraints] Applying {'fixed' if year_str in ['2030', '2040', '2045'] else 'min/max'} capacities for heat pumps in {year_str}")

    # Track total values for summary
    total_min = 0
    total_max = 0
    total_links = 0
    applied_constraints = False

    # Check if we're in a year with fixed capacities
    fixed_capacity_years = ['2030', '2040', '2045']
    is_fixed_year = year_str in fixed_capacity_years

    for hp_carrier, caps in year_dict.items():
        min_cap = caps["min"]
        max_cap = caps["max"]

        # Find all links whose carrier matches hp_carrier AND whose bus1 is in DE
        relevant_links = n.links.index[
            (n.links.carrier == hp_carrier) &
            (n.links.bus1.isin(de_buses))
            ]

        if len(relevant_links) == 0:
            logging.info(f"[HP constraints] No links found with carrier '{hp_carrier}' in DE")
            continue

        # Calculate the current total capacity for this carrier
        current_capacity = n.links.loc[relevant_links, "p_nom"].sum()
        logging.info(
            f"[HP constraints] Found {len(relevant_links)} links with carrier '{hp_carrier}' - current capacity: {current_capacity} MW")

        # Apply constraints to each link
        for link_name in relevant_links:
            per_link_capacity = min_cap / len(relevant_links)

            if is_fixed_year or min_cap == max_cap:
                # For fixed capacity years, set p_nom directly and make non-extendable
                n.links.at[link_name, "p_nom"] = per_link_capacity
                n.links.at[link_name, "p_nom_extendable"] = False

                logging.info(f"[HP constraints] Fixed '{link_name}' ({hp_carrier}): "
                             f"p_nom={per_link_capacity:.2f} MW (non-extendable)")
            else:
                # For flexible years, set min/max and make extendable
                n.links.at[link_name, "p_nom_extendable"] = True
                n.links.at[link_name, "p_nom_min"] = per_link_capacity
                n.links.at[link_name, "p_nom_max"] = max_cap / len(relevant_links)

                logging.info(f"[HP constraints] Set '{link_name}' ({hp_carrier}): "
                             f"p_nom_min={per_link_capacity:.2f} MW, "
                             f"p_nom_max={(max_cap / len(relevant_links)):.2f} MW")

            applied_constraints = True
            total_links += 1

        total_min += min_cap
        total_max += max_cap

    # Log summary
    if applied_constraints:
        if is_fixed_year:
            logging.info(f"[HP constraints] Successfully applied fixed capacities to {total_links} heat pump links")
            logging.info(f"[HP constraints] Total heat pump capacity for {year_str}: {total_min} MW")
        else:
            logging.info(f"[HP constraints] Successfully applied constraints to {total_links} heat pump links")
            logging.info(f"[HP constraints] Total heat pump capacity range for {year_str}: {total_min}-{total_max} MW")
    else:
        logging.warning(f"[HP constraints] No heat pump constraints were applied for {year_str}")

    logging.info("[HP constraints] Completed heat pump constraint application")


def add_dsm(n, config):
    """
    Add demand-side management (DSM) components to the network using the approach
    from the paper 'Impact of Flexible Demand-Side Management in Open Energy System
    Models'.
    """
    import logging
    import pandas as pd
    import numpy as np

    logger = logging.getLogger(__name__)
    logger.info("\n=== Starting Enhanced DSM Addition ===")

    try:
        # Verify config structure
        if 'sector' not in config:
            raise KeyError("'sector' not found in config")
        if 'dsm' not in config['sector']:
            raise KeyError("'dsm' not found in sector config")

        dsm_config = config['sector']['dsm']
        logger.info(f"Working with DSM config: {dsm_config}")

        if not dsm_config.get('enable', False):
            logger.info("DSM is disabled in config. Skipping.")
            return n

        # Get planning year
        try:
            year = snakemake.wildcards.planning_horizons
            logger.info(f"Processing year: {year}")
        except (NameError, AttributeError):
            if hasattr(n, 'planning_year'):
                year = n.planning_year
            else:
                year = 2030  # Default
            logger.info(f"Using planning year: {year}")

        # Define DSM parameters for different demand types
        dsm_params = {
            'electricity': {
                'sflex': dsm_config.get("tech_potential", 0.1),  # Flexible share of load
                'sutil': 0.67,  # Utilization factor from paper
                'sinc': 1.0,  # Maximum increase factor
                'sdec': 0.5,  # Minimum decrease factor
                'delta_t': dsm_config.get("time_shifting", 6)  # Shifting window in hours
            },
            'industry electricity': {
                'sflex': min(dsm_config.get("tech_potential", 0.1) * 1.2, 0.6),  # Higher flexibility for industry
                'sutil': 0.8,  # Higher utilization for industry
                'sinc': 0.95,  # From paper for industrial processes
                'sdec': 0.0,  # From paper for industrial processes
                'delta_t': max(dsm_config.get("time_shifting", 6) - 2, 4)  # Shorter window for industry
            },
            'agriculture electricity': {
                'sflex': min(dsm_config.get("tech_potential", 0.1) * 0.8, 0.4),  # Lower flexibility
                'sutil': 0.7,  # Standard utilization
                'sinc': 0.9,  # Can increase load significantly
                'sdec': 0.5,  # Can decrease load by half
                'delta_t': dsm_config.get("time_shifting", 6) * 2  # Longer shifting window
            }
        }

        # Add general electric transport
        if len(n.loads[n.loads.carrier == 'land transport EV']) > 0:
            dsm_params['land transport EV'] = {
                'sflex': min(dsm_config.get("tech_potential", 0.1) * 1.5, 0.75),  # Higher flexibility for EVs
                'sutil': 0.6,  # Lower utilization for EV charging
                'sinc': 1.0,  # Can fully increase charging
                'sdec': 0.0,  # Can fully decrease charging
                'delta_t': dsm_config.get("time_shifting", 6) * 2  # Longer shifting window for EVs
            }

        # Get DSM costs
        dsm_costs = dsm_config.get("costs", {})
        if str(year) in dsm_costs:
            costs = dsm_costs[str(year)]
        elif int(year) in dsm_costs:
            costs = dsm_costs[int(year)]
        else:
            logger.warning(f"No cost data found for year {year} in config; using default cost values.")
            costs = {"capital_cost": 80, "storage_cost": 8}

        # Handle missing storage_cost value for 2045
        if 'storage_cost' not in costs or costs['storage_cost'] is None:
            costs['storage_cost'] = 4  # Default value

        logger.info(f"DSM cost data for year {year} used: {costs}")

        # Add DSM carrier if it doesn't exist
        dsm_carrier = "dsm"
        if dsm_carrier not in n.carriers.index:
            n.add("Carrier", dsm_carrier)
            logger.info(f"Added DSM carrier: {dsm_carrier}")

        # Define function to process DSM for each load
        def process_dsm_for_load(load_name, load, params):
            """Create DSM components for a specific load using paper's methodology"""
            try:
                logger.info(f"Creating DSM for load: {load_name} (carrier: {load.carrier})")

                # Get load time series
                if hasattr(n, 'loads_t') and hasattr(n.loads_t, 'p_set'):
                    load_ts = n.loads_t.p_set[load_name]
                else:
                    logger.warning(f"No time series for load {load_name}. Using static value.")
                    # Create synthetic time series if none exists
                    load_ts = pd.Series(
                        data=load.p_set,
                        index=n.snapshots
                    )

                # Extract DSM parameters for this load type
                sflex = params['sflex']
                sutil = params['sutil']
                sinc = params['sinc']
                sdec = params['sdec']
                delta_t = params['delta_t']

                # Step 1: Calculate shiftable load L(t)
                scheduled_load = load_ts * sflex

                # Step 2: Calculate maximum capacity Lambda
                energy_annual = load_ts.sum()
                hours_year = len(load_ts) if len(load_ts) > 0 else 8760
                lam = (energy_annual * sflex) / (hours_year * sutil)

                # Skip if lambda is too small
                if lam < 1e-3:  # Very small value - not worth modeling
                    logger.info(f"Lambda for {load_name} is too small ({lam:.5f} MW). Skipping.")
                    return {'power': 0, 'energy': 0}

                # CRITICAL FIX: Set reasonable bounds for time-series constraints
                # We'll avoid strict limits that could lead to infeasibilities

                # Create DSM components
                bus_name = load.bus
                dsm_bus_name = f"{bus_name} DSM"

                # Create DSM bus if it doesn't exist
                if dsm_bus_name not in n.buses.index:
                    n.add("Bus",
                          dsm_bus_name,
                          carrier=dsm_carrier,
                          location=bus_name)

                # Create DSM link - using reasonable fixed values
                link_name = f"{load_name} DSM"
                max_power = lam * sinc * 2  # Maximum power capacity

                n.add("Link",
                      link_name,
                      bus0=dsm_bus_name,
                      bus1=bus_name,
                      carrier=dsm_carrier,
                      p_nom=max_power,  # Set fixed capacity based on calculations
                      p_nom_extendable=False,
                      efficiency=1.0,
                      marginal_cost=costs.get("marginal_cost", 1.0))  # Small cost

                # Create DSM store with reasonable capacity
                store_name = f"{bus_name} DSM store"
                max_energy = lam * delta_t * 2  # Maximum energy capacity

                if store_name not in n.stores.index:  # Create store per bus, not per load
                    n.add("Store",
                          store_name,
                          bus=dsm_bus_name,
                          carrier=dsm_carrier,
                          e_nom=max_energy,  # Set energy capacity based on time window
                          e_nom_extendable=False,
                          e_cyclic=True,
                          standing_loss=0.0)

                # CRITICAL FIX: Instead of using precise p_max_pu/p_min_pu values that might
                # lead to infeasibilities, we'll use simplified time series constraints

                # Create simpler, more relaxed time-series constraints
                if 'p_max_pu' not in n.links_t:
                    n.links_t.p_max_pu = pd.DataFrame(index=n.snapshots)
                if 'p_min_pu' not in n.links_t:
                    n.links_t.p_min_pu = pd.DataFrame(index=n.snapshots)

                # Use constant values (fully available) instead of calculated ones
                n.links_t.p_max_pu[link_name] = 1.0  # Full upward flexibility
                n.links_t.p_min_pu[link_name] = -1.0  # Full downward flexibility

                # DO NOT set store e_max_pu and e_min_pu - they cause the infeasibility
                # Let the solver determine these values within the physical capacity

                logger.info(f"Created DSM components for {load_name}:")
                logger.info(f"  - Lambda: {lam:.2f} MW")
                logger.info(f"  - DSM power capacity: {max_power:.2f} MW")
                logger.info(f"  - DSM energy capacity: {max_energy:.2f} MWh")
                logger.info(f"  - Time window: {delta_t} hours")

                return {
                    'power': max_power,
                    'energy': max_energy
                }

            except Exception as e:
                logger.error(f"Error creating DSM for load {load_name}: {str(e)}")
                logger.error("Traceback:", exc_info=True)
                return {'power': 0, 'energy': 0}

        # Process each electricity demand type with its specific parameters
        dsm_totals = {'power': 0, 'energy': 0, 'components': 0}

        # Process loads by carrier type
        for carrier, params in dsm_params.items():
            # Find loads of this carrier type
            carrier_loads = n.loads[n.loads.carrier == carrier]
            if len(carrier_loads) == 0:
                logger.info(f"No loads found for carrier: {carrier}")
                continue

            logger.info(f"\nProcessing {len(carrier_loads)} loads for carrier: {carrier}")
            logger.info(f"Parameters: sflex={params['sflex']}, delta_t={params['delta_t']} hours")

            # Process each load of this type
            for load_name, load in carrier_loads.iterrows():
                result = process_dsm_for_load(load_name, load, params)
                dsm_totals['power'] += result['power']
                dsm_totals['energy'] += result['energy']
                if result['power'] > 0:
                    dsm_totals['components'] += 1

        # Final DSM summary
        logger.info("\n=== DSM Addition Summary ===")
        logger.info(f"Total DSM components added: {dsm_totals['components']}")
        logger.info(f"Total DSM power capacity: {dsm_totals['power']:.2f} MW")
        logger.info(f"Total DSM energy capacity: {dsm_totals['energy']:.2f} MWh")

        # Verify components were added
        dsm_buses = n.buses[n.buses.carrier == dsm_carrier]
        dsm_links = n.links[n.links.carrier == dsm_carrier]
        dsm_stores = n.stores[n.stores.carrier == dsm_carrier]

        logger.info("\nDSM Components Added:")
        logger.info(f"  - DSM Buses: {len(dsm_buses)}")
        logger.info(f"  - DSM Links: {len(dsm_links)}")
        logger.info(f"  - DSM Stores: {len(dsm_stores)}")

    except Exception as e:
        logger.error(f"Error in add_dsm: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

    return n


def add_dsm_constraints(n, snapshots):
    """
    Add DSM-specific constraints to the optimization problem if needed.
    With e_cyclic=True, PyPSA should already ensure energy neutrality.
    This function is kept for compatibility or additional constraints.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Checking if additional DSM constraints are needed")

    # With e_cyclic=True, the energy neutrality constraint is already handled
    # by PyPSA. This function is kept for potential future extensions.

    # Example: You could add additional constraints here if needed
    # For example, forcing DSM links to be inactive during certain hours

    dsm_links = n.links[n.links.carrier == "dsm"]
    if len(dsm_links) > 0:
        logger.info(f"Found {len(dsm_links)} DSM links")
        # Add custom constraints here if needed

    dsm_stores = n.stores[n.stores.carrier == "dsm"]
    if len(dsm_stores) > 0:
        logger.info(f"Found {len(dsm_stores)} DSM stores")
        # Add custom constraints here if needed

    logger.info("No additional DSM constraints needed at this time")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    config = n.config
    constraints = config["solving"].get("constraints", {})

    if constraints["BAU"] and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)

    if constraints["SAFE"] and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)

    if constraints["CCL"] and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)

    # Add CLL constraints if enabled
    if constraints.get("CLL", False):
        add_CLL_constraints(n, config)

    # Add DSM-specific constraints if needed
    # (note: we're not adding DSM components here, just additional constraints if necessary)
    if config["sector"].get("dsm", {}).get("enable", False):
        add_dsm_constraints(n, snapshots)

    # Always add heat pump constraints (regardless of config setting)
    logging.info("[Setup] Adding heat pump constraints")
    add_heat_pump_constraints(n)

    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)

    if EQ_o := constraints["EQ"]:
        add_EQ_constraints(n, EQ_o.replace("EQ", ""))

    if {"solar-hsat", "solar"}.issubset(
        config["electricity"]["renewable_carriers"]
    ) and {"solar-hsat", "solar"}.issubset(
        config["electricity"]["extendable_carriers"]["Generator"]
    ):
        add_solar_potential_constraints(n, config)

    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)
    add_pipe_retrofit_constraint(n)
    if n._multi_invest:
        add_carbon_constraint(n, snapshots)
        add_carbon_budget_constraint(n, snapshots)
        add_retrofit_gas_boiler_constraint(n, snapshots)
    else:
        add_co2_atmosphere_constraint(n, snapshots)

    if config["sector"]["enhanced_geothermal"]["enable"]:
        add_flexible_egs_constraint(n)

    if n.params.custom_extra_functionality:
        source_path = pathlib.Path(n.params.custom_extra_functionality).resolve()
        assert source_path.exists(), f"{source_path} does not exist"
        sys.path.append(os.path.dirname(source_path))
        module_name = os.path.splitext(os.path.basename(source_path))[0]
        module = importlib.import_module(module_name)
        custom_extra_functionality = getattr(module, module_name)
        custom_extra_functionality(n, snapshots, snakemake)




def check_constraint_feasibility(n, config):
    """
    Check and adjust CLL constraints before solving.
    """
    logger.info("=== Checking Constraint Feasibility ===")

    try:
        # Use direct path to the limits file
        constraints_df = pd.read_csv("data/technology_limits.csv")

        planning_horizon = str(snakemake.wildcards.planning_horizons)
        year_constraints = constraints_df[constraints_df['year'] == int(planning_horizon)]

        if year_constraints.empty:
            raise KeyError(f"No constraints found for year {planning_horizon}")

        logger.info(f"\nProcessing constraints for year {planning_horizon}")

        for _, row in year_constraints.iterrows():
            country = row['country']
            technology = row['technology']
            min_cap = row['capacity_min_mw']
            max_cap = row['capacity_max_mw']

            if technology in ['coal', 'lignite']:
                # Get relevant links for coal and lignite
                relevant_units = n.links[
                    (n.links.carrier == technology) &
                    (n.links.bus0 == f'EU {technology}') &
                    (n.links.bus1.map(n.buses.country) == country)
                ]

                if relevant_units.empty:
                    logger.warning(f"No links found for {technology} in {country}")
                    continue

                logger.info(f"\nProcessing {country} {technology} (links):")
                logger.info(f"Required capacity: {min_cap} MW")

                # Check feasibility for fixed capacity
                total_possible = relevant_units.p_nom_max.sum()
                if total_possible < min_cap:
                    logger.warning(
                        f"Required capacity {min_cap} MW exceeds maximum possible "
                        f"capacity {total_possible} MW for {technology} in {country}"
                    )

            else:  # For CCGT and OCGT
                # Get relevant generators
                relevant_units = n.generators[
                    (n.generators.carrier == technology) &
                    (n.generators.bus.map(n.buses.country) == country)
                ]

                if relevant_units.empty:
                    logger.warning(f"No generators found for {technology} in {country}")
                    continue

                logger.info(f"\nProcessing {country} {technology} (generators):")
                logger.info(f"Required capacity range: {min_cap} MW - {max_cap} MW")

                # Check feasibility for extendable capacity
                existing_cap = relevant_units.p_nom.sum()
                max_possible = relevant_units.p_nom_max.sum()

                logger.info(f"Current capacity: {existing_cap:.2f} MW")
                logger.info(f"Maximum possible: {max_possible:.2f} MW")

                if max_possible < min_cap:
                    logger.warning(
                        f"Minimum required capacity {min_cap} MW exceeds maximum possible "
                        f"capacity {max_possible} MW for {technology} in {country}"
                    )

            logger.info(f"Processed {technology} in {country}")

        return True

    except Exception as e:
        logger.error(f"Error in constraint feasibility check: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        if 'constraints_df' in locals():
            logger.error("\nDataFrame information:")
            logger.error(f"Shape: {constraints_df.shape}")
            logger.error("Columns:")
            logger.error(constraints_df.columns.tolist())
            logger.error("\nFirst few rows:")
            logger.error(constraints_df.head())
        raise


def solve_network(n, config, params, solving, **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["multi_investment_periods"] = config["foresight"] == "perfect"
    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment", False
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    kwargs["io_api"] = cf_solving.get("io_api", None)

    if kwargs["solver_name"] == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.params = params

    if rolling_horizon and snakemake.rule == "solve_operations_network":
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = cf_solving["track_iterations"]
        kwargs["min_iterations"] = cf_solving["min_iterations"]
        kwargs["max_iterations"] = cf_solving["max_iterations"]
        if cf_solving["post_discretization"].pop("enable"):
            logger.info("Add post-discretization parameters.")
            kwargs.update(cf_solving["post_discretization"])
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs
        )

    if status != "ok" and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )

    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    if status == "warning":
        raise RuntimeError(
            "Solving status 'warning'. Results may not be reliable. Aborting."
        )

    return n


# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network_perfect",
            configfiles="../config/test/config.perfect.yaml",
            opts="",
            clusters="5",
            ll="v1.0",
            sector_opts="",
            # planning_horizons="2030",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
        config=snakemake.config,
        foresight=snakemake.params.foresight,
        planning_horizons=snakemake.params.planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
    )

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=30.0
    ) as mem:
        n = solve_network(
            n,
            config=snakemake.config,
            params=snakemake.params,
            solving=snakemake.params.solving,
            log_fn=snakemake.log.solver,
        )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)

    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
