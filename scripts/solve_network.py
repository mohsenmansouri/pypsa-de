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
    # if config["sector"].get("dsm", {}).get("enable", False):
    #     logging.info("[Setup] Adding demand-side management")
    #     n = add_dsm(n, config)  # Add DSM components

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
    Add CLL constraints for conventional generators and CHP units (implemented as links)
    using planning_year from snakemake.wildcards.planning_horizons.

    Reads capacity targets from config.solving.constraints.CLL.generation_target_ranges.
    The function zeroes out existing capacities in DE and then adds or updates one link per carrier.

    Make sure that the required buses exist in n.buses.
    """
    logging.info("====== STARTING CLL CONSTRAINTS FOR LINKS AND GENERATORS ======")

    try:
        # Check if CLL constraints are enabled in config
        cll_config = config.get("solving", {}).get("constraints", {}).get("CLL", {})

        # If CLL config exists but apply_constraints is False, skip
        if not cll_config.get("apply_constraints", False):
            logging.info("CLL constraints are disabled in config. Skipping.")
            return n

        # 1) Read planning year from snakemake wildcards
        planning_year = int(snakemake.wildcards.planning_horizons)
        de_buses = n.buses[n.buses.country == 'DE'].index

        # 2) Read capacity targets from config
        if 'generation_target_ranges' not in cll_config:
            raise ValueError("Missing 'generation_target_ranges' in CLL config")

        generation_targets = cll_config['generation_target_ranges']

        if str(planning_year) not in generation_targets:
            raise ValueError(f"No capacity targets defined for year {planning_year} in CLL.generation_target_ranges")

        year_capacities = generation_targets[str(planning_year)]

        # Define carriers that are likely implemented as AC links
        ac_carriers = [
            'coal', 'lignite', 'CCGT', 'OCGT',
            'urban central gas CHP', 'urban central gas CHP CC',
            'urban central solid biomass CHP', 'urban central solid biomass CHP CC'
        ]

        # Get all target carriers from the config
        target_carriers = list(year_capacities.keys())

        # Check if we have AC links in the network
        has_ac_links = 'AC' in n.links.carrier.unique()

        # Look for carrier properties if available in config
        carrier_properties = {}
        if ('solving' in config and
                'constraints' in config['solving'] and
                'CLL' in config['solving']['constraints'] and
                'carrier_properties' in config['solving']['constraints']['CLL']):
            carrier_properties = config['solving']['constraints']['CLL']['carrier_properties']

        for carrier in target_carriers:
            if carrier not in n.carriers.index:
                logging.info(f"Defining missing carrier '{carrier}' in n.carriers.")
                n.carriers.loc[carrier, "nice_name"] = carrier

            # Check for generators with this carrier
            if carrier in ['biomass', 'solid biomass']:
                # Check if there are generators with this carrier
                existing_generators = n.generators.loc[
                    (n.generators.carrier.isin(['biomass', 'solid biomass'])) &
                    (n.generators.bus.isin(de_buses))
                    ]
                if not existing_generators.empty:
                    logging.info(
                        f"Zeroing capacity for {len(existing_generators)} existing {carrier} generator(s) in DE.")
                    n.generators.loc[existing_generators.index, "p_nom"] = 0
                    n.generators.loc[existing_generators.index, "p_nom_extendable"] = False
                    n.generators.loc[existing_generators.index, "p_nom_min"] = 0
                    n.generators.loc[existing_generators.index, "p_nom_max"] = 0

            # Check for direct links with this carrier
            existing_links = n.links.loc[
                (n.links.carrier == carrier) &
                (n.links.bus1.isin(de_buses))
                ]
            if not existing_links.empty:
                logging.info(f"Zeroing capacity for {len(existing_links)} existing {carrier} link(s) in DE.")
                n.links.loc[existing_links.index, "p_nom"] = 0
                n.links.loc[existing_links.index, "p_nom_extendable"] = False
                n.links.loc[existing_links.index, "p_nom_min"] = 0
                n.links.loc[existing_links.index, "p_nom_max"] = 0

            # Special handling for carriers that might be implemented as AC links
            if has_ac_links and carrier in ac_carriers:
                # Look for AC links where bus0 contains carrier name
                ac_links_for_carrier = n.links.loc[
                    (n.links.carrier == 'AC') &
                    (n.links.bus1.isin(de_buses)) &
                    (n.links.bus0.str.contains(carrier, case=False))
                    ]

                if not ac_links_for_carrier.empty:
                    logging.info(f"Zeroing capacity for {len(ac_links_for_carrier)} AC links for {carrier} in DE.")
                    n.links.loc[ac_links_for_carrier.index, "p_nom"] = 0
                    n.links.loc[ac_links_for_carrier.index, "p_nom_extendable"] = False
                    n.links.loc[ac_links_for_carrier.index, "p_nom_min"] = 0
                    n.links.loc[ac_links_for_carrier.index, "p_nom_max"] = 0

        # 4) Add or update for each carrier from the capacity dictionary
        for carrier, caps in year_capacities.items():
            min_cap = caps.get('min', 0)
            max_cap = caps.get('max', 0)

            if min_cap <= 0 and max_cap <= 0:
                logging.info(f"{carrier.capitalize()} capacity in {planning_year} is 0 MW - skipping creation.")
                continue

            # For biomass, we could use either generators or links, check network structure
            if carrier in ['biomass', 'solid biomass']:
                # Check if biomass is modeled as generators or links in this network
                biomass_generators = n.generators[
                    (n.generators.carrier.isin(['biomass', 'solid biomass'])) &
                    (n.generators.bus.isin(de_buses))
                    ]
                biomass_links = n.links[
                    (n.links.carrier.isin(['biomass', 'solid biomass'])) &
                    (n.links.bus1.isin(de_buses))
                    ]

                # If there are existing biomass generators, add a new generator
                if len(biomass_generators) > 0 or len(biomass_links) == 0:
                    # Biomass is modeled as a generator
                    new_unit = f"DE0 biomass-{planning_year}"

                    # Find an appropriate bus to connect to
                    elec_buses = [bus for bus in de_buses if 'elec' in n.buses.at[bus, 'carrier']]
                    target_bus = elec_buses[0] if elec_buses else "DE0 0"

                    if target_bus not in n.buses.index:
                        logging.warning(
                            f"Bus '{target_bus}' not found in n.buses; generator might not connect properly.")

                    # Add or update generator
                    if new_unit in n.generators.index:
                        n.generators.at[new_unit, 'p_nom'] = min_cap
                        n.generators.at[new_unit, 'p_nom_min'] = min_cap
                        n.generators.at[new_unit, 'p_nom_max'] = max_cap
                        n.generators.at[new_unit, 'p_nom_extendable'] = (min_cap != max_cap)
                        msg = "Updated"
                    else:
                        n.add(
                            "Generator",
                            new_unit,
                            bus=target_bus,
                            carrier='biomass',
                            p_nom=min_cap,
                            p_nom_min=min_cap,
                            p_nom_max=max_cap,
                            p_nom_extendable=(min_cap != max_cap),
                            marginal_cost=80  # Example value, adjust as needed
                        )
                        msg = "Added"

                    if min_cap == max_cap:
                        logging.info(
                            f"{msg} biomass generator with fixed capacity: {min_cap} MW for year {planning_year}")
                    else:
                        logging.info(
                            f"{msg} biomass generator with capacity range: {min_cap}-{max_cap} MW for year {planning_year}")

                    continue  # Skip the link creation for biomass

                # If we reach here, continue with biomass as a link

            # Determine if this carrier should be implemented as an AC link or direct link
            should_use_ac = has_ac_links and carrier in ac_carriers

            # Create a unit name based on whether we're using AC or direct link
            if should_use_ac:
                # For AC links, create a source and a link
                source_name = f"DE0 {carrier}-{planning_year}"
                link_name = f"DE0 AC-{carrier}-{planning_year}"
            else:
                # For direct links, just need one name
                link_name = f"DE0 0 {carrier.replace(' ', '-')}-{planning_year}"

            # Configure link parameters based on carrier type
            # Get default efficiency from config if available
            efficiency = carrier_properties.get(carrier, {}).get('efficiency', None)

            if carrier in ['CCGT', 'OCGT']:
                bus0 = 'EU gas'
                bus1 = "DE0 0"
                # For CCGT, we now force a fixed capacity; for OCGT, use extendable as usual.
                is_extendable = (carrier != 'CCGT') and (min_cap != max_cap)
                if efficiency is None:
                    efficiency = 0.6 if carrier == 'CCGT' else 0.4

            elif carrier in ['coal', 'lignite']:
                bus0 = f"EU {carrier}"
                bus1 = "DE0 0"
                is_extendable = (min_cap != max_cap)
                if efficiency is None:
                    efficiency = 0.45 if carrier == 'coal' else 0.42

            elif carrier in ['biomass', 'solid biomass']:
                bus0 = "EU biomass"
                bus1 = "DE0 0"
                is_extendable = (min_cap != max_cap)
                if efficiency is None:
                    efficiency = 0.4

            elif 'gas CHP' in carrier:
                bus0 = "EU gas"
                bus1 = "DE0 0"
                is_extendable = (min_cap != max_cap)
                # Lower electrical efficiency for CHP units
                if efficiency is None:
                    efficiency = 0.40 if 'CC' in carrier else 0.35

            elif 'biomass CHP' in carrier:
                bus0 = "EU biomass"
                bus1 = "DE0 0"
                is_extendable = (min_cap != max_cap)
                # Lower electrical efficiency for biomass CHP units
                if efficiency is None:
                    efficiency = 0.28 if 'CC' in carrier else 0.25

            else:
                logging.warning(f"Unknown carrier type: {carrier}")
                continue

            # Check if required buses exist
            if bus0 not in n.buses.index:
                logging.warning(f"Bus '{bus0}' not found in n.buses. Creating it.")
                n.add("Bus", bus0, carrier="gas" if "gas" in bus0 else "biomass")

            if bus1 not in n.buses.index:
                logging.warning(f"Bus '{bus1}' not found in n.buses; link might not connect properly.")

            # For AC links, we need to create an intermediate bus and two links
            if should_use_ac:
                # 1. Create source bus if it doesn't exist
                if source_name not in n.buses.index:
                    n.add(
                        "Bus",
                        source_name,
                        carrier=carrier
                    )
                    logging.info(f"Created source bus {source_name} for {carrier}")

                # 2. Create source link that feeds fuel to the source bus
                source_link_name = f"DE0 source-{carrier}-{planning_year}"
                if source_link_name in n.links.index:
                    n.links.at[source_link_name, 'p_nom'] = min_cap / efficiency  # Adjust for efficiency
                    n.links.at[source_link_name, 'p_nom_min'] = min_cap / efficiency
                    n.links.at[source_link_name, 'p_nom_max'] = max_cap / efficiency
                    n.links.at[source_link_name, 'p_nom_extendable'] = is_extendable
                    n.links.at[source_link_name, 'efficiency'] = 1.0  # Just fuel transfer
                else:
                    n.add(
                        "Link",
                        source_link_name,
                        bus0=bus0,
                        bus1=source_name,
                        carrier=carrier,
                        p_nom=min_cap / efficiency,
                        p_nom_min=min_cap / efficiency,
                        p_nom_max=max_cap / efficiency,
                        p_nom_extendable=is_extendable,
                        efficiency=1.0  # Just fuel transfer
                    )

                # 3. Add or update the AC link between source bus and target bus
                if link_name in n.links.index:
                    n.links.at[link_name, 'p_nom'] = min_cap
                    n.links.at[link_name, 'p_nom_min'] = min_cap
                    n.links.at[link_name, 'p_nom_max'] = max_cap
                    n.links.at[link_name, 'p_nom_extendable'] = is_extendable
                    n.links.at[link_name, 'efficiency'] = efficiency
                    msg = "Updated"
                else:
                    n.add(
                        "Link",
                        link_name,
                        bus0=source_name,
                        bus1=bus1,
                        carrier='AC',  # AC link
                        p_nom=min_cap,
                        p_nom_min=min_cap,
                        p_nom_max=max_cap,
                        p_nom_extendable=is_extendable,
                        efficiency=efficiency
                    )
                    msg = "Added"
            else:
                # Regular link implementation (not using AC)
                if link_name in n.links.index:
                    n.links.at[link_name, 'p_nom'] = min_cap
                    n.links.at[link_name, 'p_nom_min'] = min_cap
                    n.links.at[link_name, 'p_nom_max'] = max_cap
                    n.links.at[link_name, 'p_nom_extendable'] = is_extendable
                    n.links.at[link_name, 'efficiency'] = efficiency
                    msg = "Updated"
                else:
                    n.add(
                        "Link",
                        link_name,
                        bus0=bus0,
                        bus1=bus1,
                        carrier=carrier,
                        p_nom=min_cap,
                        p_nom_min=min_cap,
                        p_nom_max=max_cap,
                        p_nom_extendable=is_extendable,
                        efficiency=efficiency
                    )
                    msg = "Added"

            if min_cap == max_cap:
                logging.info(
                    f"{msg} {carrier} {'AC ' if should_use_ac else ''}link with fixed capacity: {min_cap} MW for year {planning_year}")
            else:
                logging.info(
                    f"{msg} {carrier} {'AC ' if should_use_ac else ''}link with capacity range: {min_cap}-{max_cap} MW for year {planning_year}")

        # 5) Log final capacities for each carrier
        logging.info("\n====== FINAL CONFIGURATIONS ======")
        total_capacities = {c: 0 for c in target_carriers}

        for carrier in target_carriers:
            # Check for generators first (relevant for biomass)
            if carrier in ['biomass', 'solid biomass']:
                gen_units = n.generators.loc[
                    (n.generators.carrier.isin(['biomass', 'solid biomass'])) &
                    (n.generators.bus.isin(de_buses))
                    ]
                if not gen_units.empty:
                    gen_cap = gen_units["p_nom"].sum()
                    gen_min = gen_units["p_nom_min"].sum()
                    gen_max = gen_units["p_nom_max"].sum()
                    total_capacities[carrier] = gen_min

                    logging.info(f"\nDE {carrier} (generators):")
                    logging.info(f"  - Current capacity: {gen_cap:.1f} MW")
                    logging.info(f"  - Min capacity: {gen_min:.1f} MW")
                    logging.info(f"  - Max capacity: {gen_max:.1f} MW")

            # Check for links with direct carrier
            link_units = n.links.loc[
                (n.links.carrier == carrier) &
                (n.links.bus1.isin(de_buses))
                ]

            # Also check for AC links if this carrier might use them
            ac_link_units = pd.DataFrame()
            if has_ac_links and carrier in ac_carriers:
                # Look for AC links where bus0 contains the carrier name
                source_pattern = f"DE0 {carrier}"
                ac_link_units = n.links.loc[
                    (n.links.carrier == 'AC') &
                    (n.links.bus1.isin(de_buses)) &
                    (n.links.bus0.str.startswith(source_pattern))
                    ]

            # Combine direct and AC links
            all_links = pd.concat([link_units, ac_link_units])

            if not all_links.empty:
                link_cap = all_links["p_nom"].sum()
                link_min = all_links["p_nom_min"].sum()
                link_max = all_links["p_nom_max"].sum()

                # Add to total capacities (may add to existing biomass from generators)
                if carrier in total_capacities:
                    total_capacities[carrier] += link_min
                else:
                    total_capacities[carrier] = link_min

                # Show separate metrics for direct and AC links
                if not link_units.empty and not ac_link_units.empty:
                    logging.info(f"\nDE {carrier} (direct links):")
                    logging.info(f"  - Current capacity: {link_units['p_nom'].sum():.1f} MW")
                    logging.info(f"  - Min capacity: {link_units['p_nom_min'].sum():.1f} MW")
                    logging.info(f"  - Max capacity: {link_units['p_nom_max'].sum():.1f} MW")

                    logging.info(f"\nDE {carrier} (AC links):")
                    logging.info(f"  - Current capacity: {ac_link_units['p_nom'].sum():.1f} MW")
                    logging.info(f"  - Min capacity: {ac_link_units['p_nom_min'].sum():.1f} MW")
                    logging.info(f"  - Max capacity: {ac_link_units['p_nom_max'].sum():.1f} MW")

                    logging.info(f"\nDE {carrier} (total):")
                else:
                    logging.info(f"\nDE {carrier}:")

                logging.info(f"  - Current capacity: {link_cap:.1f} MW")
                logging.info(f"  - Min capacity: {link_min:.1f} MW")
                logging.info(f"  - Max capacity: {link_max:.1f} MW")

        logging.info("\nTotal Capacities:")
        for c, cap in total_capacities.items():
            logging.info(f"{c}: {cap:.1f} MW")

        # Calculate some useful aggregates
        total_gas = total_capacities.get('CCGT', 0) + total_capacities.get('OCGT', 0)
        total_gas_chp = total_capacities.get('urban central gas CHP', 0) + total_capacities.get(
            'urban central gas CHP CC', 0)

        # Use either 'biomass' or 'solid biomass', whichever is non-zero
        total_biomass = max(total_capacities.get('biomass', 0), total_capacities.get('solid biomass', 0))

        total_biomass_chp = total_capacities.get('urban central solid biomass CHP', 0) + total_capacities.get(
            'urban central solid biomass CHP CC', 0)

        logging.info(f"Total Gas (CCGT+OCGT): {total_gas:.1f} MW")
        logging.info(f"Total Gas CHP: {total_gas_chp:.1f} MW")
        logging.info(f"Total Biomass: {total_biomass:.1f} MW")
        logging.info(f"Total Biomass CHP: {total_biomass_chp:.1f} MW")

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

    # Get config and check if constraints are enabled
    if not hasattr(n, 'config'):
        try:
            n.config = snakemake.config
            logging.info("[HP constraints] Got config from snakemake")
        except (NameError, AttributeError):
            logging.warning("[HP constraints] No config found, skipping heat pump constraints")
            return

    # Check the path in the config following PyPSA-DE structure
    # Changed from hp_config = n.config.get('solving', {}).get('constraints', {}).get('heat_pumps', {})
    hp_config = n.config["solving"].get("constraints", {}).get("heat_pumps", {})

    # DEBUG: Log the structure of the config
    try:
        # First, try to log high-level keys
        logging.info("[HP constraints] Config contains these top-level keys: " +
                     str(list(n.config.keys())))

        # Check if 'solving' exists
        if 'solving' in n.config:
            logging.info("[HP constraints] 'solving' section contains: " +
                         str(list(n.config['solving'].keys())))

            # Check if 'constraints' exists under solving
            if 'constraints' in n.config['solving']:
                logging.info("[HP constraints] 'solving.constraints' section contains: " +
                             str(list(n.config['solving']['constraints'].keys())))

                # Check if heat_pumps exists
                if 'heat_pumps' in n.config['solving']['constraints']:
                    logging.info("[HP constraints] Found heat_pumps configuration!")
                    heat_pump_config = n.config['solving']['constraints']['heat_pumps']
                    logging.info("[HP constraints] Heat pump config has keys: " +
                                 str(list(heat_pump_config.keys())))
                    logging.info("[HP constraints] apply_constraints is set to: " +
                                 str(heat_pump_config.get('apply_constraints')))

        # Also check if heat_pumps might be at top level or under constraints directly
        if 'constraints' in n.config:
            logging.info("[HP constraints] Top-level 'constraints' section contains: " +
                         str(list(n.config['constraints'].keys())))

            if 'heat_pumps' in n.config['constraints']:
                logging.info("[HP constraints] Found heat_pumps in top-level constraints!")

        if 'heat_pumps' in n.config:
            logging.info("[HP constraints] Found heat_pumps at top level!")

    except Exception as e:
        logging.error(f"[HP constraints] Error inspecting config: {str(e)}")

    # Try multiple possible paths for heat pump config
    hp_config = None
    possible_paths = [
        # Path 1: As in your function
        n.config.get('solving', {}).get('constraints', {}).get('heat_pumps', {}),
        # Path 2: Directly under constraints
        n.config.get('constraints', {}).get('heat_pumps', {}),
        # Path 3: At top level
        n.config.get('heat_pumps', {})
    ]

    # Try each path and use the first one that has apply_constraints=True
    for i, path_config in enumerate(possible_paths):
        if path_config.get('apply_constraints', False):
            hp_config = path_config
            logging.info(f"[HP constraints] Found enabled heat pump config at path {i + 1}")
            break

    # If none of the paths had apply_constraints=True
    if hp_config is None or not hp_config.get('apply_constraints', False):
        logging.info("[HP constraints] Heat pump constraints disabled in config, skipping")
        return

    # Get planning year with multiple fallbacks
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

    # Get the active scenario from config
    active_scenario = hp_config.get('active_scenario', 'medium')
    logging.info(f"[HP constraints] Using {active_scenario} heat pump scenario from config")

    # Get countries to apply constraints to
    countries_to_process = hp_config.get('countries', [])
    if not countries_to_process:
        # If empty, apply to all countries in the network
        countries_to_process = n.buses.country.unique().tolist()
        logging.info(f"[HP constraints] Applying to all countries in network: {countries_to_process}")
    else:
        logging.info(f"[HP constraints] Applying to specified countries: {countries_to_process}")

    # Get capacity targets by scenario
    scenario_config = hp_config.get('scenarios', {}).get(active_scenario, {})

    # Add debug logging to see what's actually in the config
    logging.info(f"[HP constraints] Available years in {active_scenario} scenario: {list(scenario_config.keys())}")
    logging.info(f"[HP constraints] Types of year keys: {[type(k).__name__ for k in scenario_config.keys()]}")

    # Track totals for summary
    countries_processed = 0
    total_links_constrained = 0

    # Process each country
    for country in countries_to_process:
        # In your config, the scenario data isn't nested under countries
        # So we use the scenario_config directly
        country_config = scenario_config  # Use scenario_config directly without country nesting

        # Try to find the year in all possible formats
        year_dict = None

        # Method 1: Direct lookup using string
        if year_str in country_config:
            year_dict = country_config[year_str]
            logging.info(f"[HP constraints] Found year {year_str} as string key")

        # Method 2: Direct lookup using integer
        elif planning_year in country_config:
            year_dict = country_config[planning_year]
            logging.info(f"[HP constraints] Found year {planning_year} as integer key")

        # Method 3: Compare string representations of keys
        else:
            for key in country_config.keys():
                if str(key) == year_str:
                    year_dict = country_config[key]
                    logging.info(
                        f"[HP constraints] Found year as key {key} (type: {type(key).__name__}) matching {year_str}")
                    break

        # If we still couldn't find the year
        if year_dict is None:
            logging.warning(
                f"[HP constraints] No HP capacity targets defined for {year_str} in {active_scenario}. Skipping {country}.")
            # Debug what years are available
            logging.info(f"[HP constraints] Available years: {list(country_config.keys())}")
            continue

        # Get buses for this country
        country_buses = n.buses.index[n.buses.country == country]
        if country_buses.empty:
            logging.warning(f"[HP constraints] Found no buses for {country}. Skipping.")
            continue  # Skip this country and continue with the next one

        logging.info(f"[HP constraints] Processing country: {country} with {len(country_buses)} buses")

        # Check for missing carriers
        all_carriers = n.links.carrier.unique()
        missing_carriers = set(year_dict.keys()) - set(all_carriers)
        if missing_carriers:
            logging.warning(f"[HP constraints] Some heat pump carriers not found for {country}: {missing_carriers}")

        # Check if we have fixed capacities (min == max)
        fixed_capacities = all(
            caps.get("min", 0) == caps.get("max", 0)
            for caps in year_dict.values()
            if "min" in caps and "max" in caps
        )

        # Track country-specific totals
        country_min = 0
        country_max = 0
        country_links = 0

        # Apply constraints to each carrier
        for hp_carrier, caps in year_dict.items():
            min_cap = caps.get("min", 0)
            max_cap = caps.get("max", float('inf'))

            # Find relevant links in this country
            relevant_links = n.links.index[
                (n.links.carrier == hp_carrier) &
                (n.links.bus1.isin(country_buses))
                ]

            if len(relevant_links) == 0:
                logging.info(f"[HP constraints] No {hp_carrier} found in {country}")
                continue  # Skip this carrier and continue with the next one

            # Apply constraints to each link
            for link_name in relevant_links:
                per_link_capacity = min_cap / len(relevant_links)

                if fixed_capacities or min_cap == max_cap:
                    # Fixed capacity
                    n.links.at[link_name, "p_nom"] = per_link_capacity
                    n.links.at[link_name, "p_nom_extendable"] = False
                else:
                    # Min/max capacity
                    n.links.at[link_name, "p_nom_extendable"] = True
                    n.links.at[link_name, "p_nom_min"] = per_link_capacity
                    n.links.at[link_name, "p_nom_max"] = max_cap / len(relevant_links)

                country_links += 1
                total_links_constrained += 1

            country_min += min_cap
            country_max += max_cap

        # Log country summary
        if country_links > 0:
            if fixed_capacities:
                logging.info(
                    f"[HP constraints] {country}: Applied fixed capacity to {country_links} links, total: {country_min} MW")
            else:
                logging.info(
                    f"[HP constraints] {country}: Applied constraints to {country_links} links, range: {country_min}-{country_max} MW")
            countries_processed += 1

    # Log overall summary
    if total_links_constrained > 0:
        logging.info(
            f"[HP constraints] Successfully applied constraints to {total_links_constrained} heat pump links across {countries_processed} countries")
    else:
        logging.warning(f"[HP constraints] No heat pump constraints were applied")

    logging.info("[HP constraints] Completed heat pump constraint application")


def add_dsm(n, config=None):
    """
    Add demand-side management (DSM) components to the network.
    Enhanced version with improved parameters and flexibility options.
    """
    import logging
    import pandas as pd
    import numpy as np

    logger = logging.getLogger(__name__)
    logger.info("\n=== Starting Enhanced DSM Addition ===")

    try:
        # Handle the config parameter
        if config is None:
            # Try to get config from network or snakemake
            if hasattr(n, 'config'):
                config = n.config
                logger.info("Using config from network object")
            else:
                try:
                    # Try to get config from snakemake
                    config = snakemake.config
                    logger.info("Got config from snakemake")
                except (NameError, AttributeError):
                    logger.warning("No config found, using default DSM parameters")
                    config = {}

        # Store the config in the network for reference
        n.config = config

        # Check if DSM is enabled in config - look for it under sector.dsm
        dsm_config = config.get('sector', {}).get('dsm', {})
        if not dsm_config.get('enable', False):
            logger.info("DSM is disabled in config. Skipping.")
            return n

        # Get active scenario configuration
        active_scenario = dsm_config.get('active_scenario', 'medium')
        logger.info(f"Using {active_scenario} DSM scenario from config")

        # Get scenario configuration
        scenario_config = dsm_config.get('scenarios', {}).get(active_scenario, {})
        if not scenario_config:
            logger.warning(f"No configuration found for {active_scenario} scenario. Using default parameters.")
            scenario_config = {
                "default": {
                    "tech_potential": 0.1,
                    "time_shifting": 6,
                    "costs": {
                        "2030": {"capital_cost": 30, "storage_cost": 3, "marginal_cost": 0.1}
                    },
                    "demand_types": {
                        "electricity": {"sflex_modifier": 1.0, "time_window_modifier": 1.0},
                        "industry_electricity": {"sflex_modifier": 1.2, "time_window_modifier": 0.8}
                    }
                }
            }

        # Get countries to apply DSM to
        countries_to_process = dsm_config.get('countries', [])
        if not countries_to_process:
            # If empty, apply to all countries in the network
            countries_to_process = n.buses.country.unique().tolist()
            logger.info(f"Applying DSM to all countries in network: {countries_to_process}")
        else:
            logger.info(f"Applying DSM to specified countries: {countries_to_process}")

        # Get planning year
        try:
            year = snakemake.wildcards.planning_horizons
            logger.info(f"Processing year: {year}")
        except (NameError, AttributeError):
            if hasattr(n, 'planning_year'):
                year = n.planning_year
            else:
                # Try to extract year from network name or attributes
                year = "2030"  # Default
                if hasattr(n, 'name'):
                    import re
                    match = re.search(r'(\d{4})', n.name)
                    if match:
                        year = match.group(1)
            logger.info(f"Using planning year: {year}")

        # Store planning year for later use
        n.planning_year = year

        # Add DSM carrier if it doesn't exist
        dsm_carrier = "dsm"
        if dsm_carrier not in n.carriers.index:
            n.add("Carrier", dsm_carrier)
            logger.info(f"Added DSM carrier: {dsm_carrier}")

        # Track totals for summary
        total_dsm = {'power': 0, 'energy': 0, 'components': 0}
        countries_processed = 0

        # Prepare storage for initial values if they don't exist
        if 'e_initial' not in n.stores_t:
            n.stores_t['e_initial'] = pd.Series(index=n.stores.index)

        # Process each country
        for country in countries_to_process:
            # Get country-specific config or fall back to default
            country_config = scenario_config.get(country, scenario_config.get('default', {}))
            if not country_config:
                logger.warning(f"No DSM configuration found for {country}. Skipping.")
                continue

            # Get country-specific parameters
            tech_potential = country_config.get('tech_potential', 0.1)
            time_shifting = country_config.get('time_shifting', 6)
            demand_type_modifiers = country_config.get('demand_types', {})

            # Get country-specific costs for this year
            costs_config = country_config.get('costs', {})
            if str(year) in costs_config:
                costs = costs_config[str(year)]
            elif int(year) in costs_config:
                costs = costs_config[int(year)]
            else:
                logger.warning(f"No cost data found for {country} in year {year}; using default cost values.")
                # Improved default cost values
                costs = {"capital_cost": 20, "storage_cost": 1, "marginal_cost": 0.05}

            logger.info(f"\n--- Processing DSM for {country} ---")
            logger.info(f"Base tech_potential: {tech_potential}, time_shifting: {time_shifting}")
            logger.info(f"Cost data for year {year}: {costs}")

            # Get buses for this country
            country_buses = n.buses.index[n.buses.country == country]
            if country_buses.empty:
                logger.warning(f"Found no buses for {country}. Skipping.")
                continue

            # Define DSM parameters for different demand types in this country
            dsm_params = {}

            # Add electricity as base type
            dsm_params['electricity'] = {
                'sflex': tech_potential *
                         demand_type_modifiers.get('electricity', {}).get('sflex_modifier', 1.0),
                'sutil': 0.5,  # Lower utilization factor to allow more capacity
                'sinc': 1.2,  # Increased maximum increase factor
                'sdec': 0.7,  # Improved minimum decrease factor
                'delta_t': int(time_shifting *
                               demand_type_modifiers.get('electricity', {}).get('time_window_modifier', 1.0))
            }

            # Add industry electricity if present
            if 'industry_electricity' in demand_type_modifiers:
                dsm_params['industry electricity'] = {
                    'sflex': min(tech_potential *
                                 demand_type_modifiers.get('industry_electricity', {}).get('sflex_modifier', 1.2), 0.8),
                    # Higher limit
                    'sutil': 0.6,  # Improved utilization for industry
                    'sinc': 1.1,  # More balanced for industrial processes
                    'sdec': 0.3,  # More balanced for industrial processes
                    'delta_t': int(max(time_shifting *
                                       demand_type_modifiers.get('industry_electricity', {}).get('time_window_modifier',
                                                                                                 0.8), 6))
                    # Minimum 6 hours
                }

            # Add other demand types that have modifiers defined
            for type_key, modifiers in demand_type_modifiers.items():
                carrier = type_key.replace('_', ' ')
                if carrier != 'electricity' and carrier != 'industry electricity' and carrier not in dsm_params:
                    # Default parameters with modifiers applied
                    dsm_params[carrier] = {
                        'sflex': min(tech_potential * modifiers.get('sflex_modifier', 1.0), 0.7),  # Higher limit
                        'sutil': 0.55,  # Improved utilization
                        'sinc': 1.1,  # Improved increase factor
                        'sdec': 0.4,  # Improved decrease factor
                        'delta_t': int(max(time_shifting * modifiers.get('time_window_modifier', 1.0), 4))
                        # Minimum 4 hours
                    }

            # Check if EV DSM is already handled in the model
            ev_dsm_already_handled = False
            if 'transport' in config and 'bev_dsm' in config['transport']:
                try:
                    bev_dsm_year = int(config['transport']['bev_dsm'])
                    current_year = int(year)
                    if current_year >= bev_dsm_year:
                        ev_dsm_already_handled = True
                        logger.info(f"EV DSM is already handled by the transport model since {bev_dsm_year}")
                except (ValueError, TypeError):
                    logger.warning("Could not parse bev_dsm year from config")

            # Remove EV transport if already handled
            if ev_dsm_already_handled and 'land transport EV' in dsm_params:
                del dsm_params['land transport EV']
                logger.info("Skipping DSM for EV transport as it's already handled by the transport model")

            # Log the country-specific DSM parameters
            for carrier, params in dsm_params.items():
                logger.info(f"DSM parameters for {country} - {carrier}:")
                logger.info(f"  - sflex: {params['sflex']:.2f}")
                logger.info(f"  - delta_t: {params['delta_t']} hours")
                logger.info(f"  - sutil: {params['sutil']:.2f}")
                logger.info(f"  - sinc: {params['sinc']:.2f}")
                logger.info(f"  - sdec: {params['sdec']:.2f}")

            # Define function to process DSM for each load
            def process_dsm_for_load(load_name, load, params):
                """Create improved DSM components for a specific load"""
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

                    # Step 2: Calculate maximum capacity Lambda - IMPROVED CALCULATION
                    energy_annual = load_ts.sum()
                    hours_year = len(load_ts) if len(load_ts) > 0 else 8760

                    # More generous lambda calculation
                    lam = (energy_annual * sflex) / (hours_year * sutil * 0.5)

                    # Skip if lambda is too small
                    if lam < 1e-3:  # Very small value - not worth modeling
                        logger.info(f"Lambda for {load_name} is too small ({lam:.5f} MW). Skipping.")
                        return {'power': 0, 'energy': 0}

                    # Create DSM components
                    bus_name = load.bus
                    dsm_bus_name = f"{bus_name} DSM"

                    # Create DSM bus if it doesn't exist
                    if dsm_bus_name not in n.buses.index:
                        n.add("Bus",
                              dsm_bus_name,
                              carrier=dsm_carrier,
                              location=bus_name)

                    # Create DSM link with improved parameters
                    link_name = f"{load_name} DSM"
                    max_power = lam * sinc * 2  # More generous maximum power capacity

                    n.add("Link",
                          link_name,
                          bus0=dsm_bus_name,
                          bus1=bus_name,
                          carrier=dsm_carrier,
                          p_nom=max_power,
                          p_nom_extendable=False,
                          efficiency=1.0,
                          marginal_cost=costs.get("marginal_cost", 0.1),  # Lower marginal cost
                          capital_cost=costs.get("capital_cost", 20))  # Include capital cost

                    # Create DSM store with improved parameters
                    store_name = f"{bus_name} DSM store"
                    max_energy = lam * delta_t * 2  # Base energy capacity calculation

                    if store_name not in n.stores.index:  # Create store per bus, not per load
                        n.add("Store",
                              store_name,
                              bus=dsm_bus_name,
                              carrier=dsm_carrier,
                              e_nom=max_energy,  # Base capacity
                              e_nom_extendable=True,  # Allow optimization
                              e_nom_min=max_energy * 0.5,  # Minimum capacity
                              e_nom_max=max_energy * 4,  # Maximum capacity
                              e_cyclic=False,  # Don't force cyclicity over full period
                              standing_loss=0.001,  # Small standing loss
                              capital_cost=costs.get("storage_cost", 1))  # Storage cost

                        # Set initial state of charge to 50%
                        n.stores_t.e_initial[store_name] = max_energy * 0.5

                    # Create time-series constraints with load-based variation
                    if 'p_max_pu' not in n.links_t:
                        n.links_t.p_max_pu = pd.DataFrame(index=n.snapshots)
                    if 'p_min_pu' not in n.links_t:
                        n.links_t.p_min_pu = pd.DataFrame(index=n.snapshots)

                    # Create time-varying flexibility that matches load patterns
                    normalized_load = load_ts / load_ts.max()
                    n.links_t.p_max_pu[link_name] = 1.0 + 0.5 * normalized_load  # Higher during peak demand
                    n.links_t.p_min_pu[link_name] = -1.0 - 0.5 * normalized_load  # Higher during peak demand

                    logger.info(f"Created DSM components for {load_name}:")
                    logger.info(f"  - Lambda: {lam:.2f} MW")
                    logger.info(f"  - DSM power capacity: {max_power:.2f} MW")
                    logger.info(f"  - DSM base energy capacity: {max_energy:.2f} MWh")
                    logger.info(f"  - DSM capacity range: {max_energy * 0.5:.2f} - {max_energy * 4:.2f} MWh")

                    return {
                        'power': max_power,
                        'energy': max_energy
                    }

                except Exception as e:
                    logger.error(f"Error creating DSM for load {load_name}: {str(e)}")
                    logger.error("Traceback:", exc_info=True)
                    return {'power': 0, 'energy': 0}

            # Country-specific totals
            country_dsm = {'power': 0, 'energy': 0, 'components': 0}

            # Process loads by carrier type for this country
            for carrier, params in dsm_params.items():
                # Find loads of this carrier type in this country
                carrier_loads = n.loads[
                    (n.loads.carrier == carrier) &
                    (n.loads.bus.isin(country_buses))
                    ]

                if len(carrier_loads) == 0:
                    logger.info(f"No loads found for carrier: {carrier} in {country}")
                    continue

                logger.info(f"\nProcessing {len(carrier_loads)} loads for {carrier} in {country}")

                # Process each load of this type
                for load_name, load in carrier_loads.iterrows():
                    result = process_dsm_for_load(load_name, load, params)
                    country_dsm['power'] += result['power']
                    country_dsm['energy'] += result['energy']
                    if result['power'] > 0:
                        country_dsm['components'] += 1

            # Country summary
            if country_dsm['components'] > 0:
                logger.info(f"\n--- DSM Summary for {country} ---")
                logger.info(f"DSM components added: {country_dsm['components']}")
                logger.info(f"Total DSM power capacity: {country_dsm['power']:.2f} MW")
                logger.info(f"Total DSM energy capacity: {country_dsm['energy']:.2f} MWh")
                countries_processed += 1

                # Add to overall totals
                total_dsm['power'] += country_dsm['power']
                total_dsm['energy'] += country_dsm['energy']
                total_dsm['components'] += country_dsm['components']

        # Final overall DSM summary
        logger.info("\n=== Overall Enhanced DSM Addition Summary ===")
        logger.info(f"Countries processed: {countries_processed}")
        logger.info(f"Total DSM components added: {total_dsm['components']}")
        logger.info(f"Total DSM power capacity: {total_dsm['power']:.2f} MW")
        logger.info(f"Total DSM energy capacity: {total_dsm['energy']:.2f} MWh")

        # Store minimum utilization in network for constraints
        n.dsm_minimum_utilization = dsm_config.get('minimum_utilization', 0.0)
        if n.dsm_minimum_utilization > 0:
            logger.info(f"Set minimum DSM utilization to {n.dsm_minimum_utilization * 100:.1f}% of demand")

    except Exception as e:
        logger.error(f"Error in add_dsm: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        import traceback
        traceback.print_exc()

    return n


def add_dsm_constraints(n, snapshots):
    """
    Add enhanced DSM-specific constraints to the optimization problem.
    - Adds daily energy neutrality constraints
    - Supports minimum utilization
    - Adds optional ramping limits
    """
    import logging
    import pandas as pd
    import numpy as np
    from datetime import timedelta

    logger = logging.getLogger(__name__)
    logger.info("Adding enhanced DSM constraints")

    # Check if DSM is enabled in the model
    dsm_links = n.links[n.links.carrier == "dsm"]
    dsm_stores = n.stores[n.stores.carrier == "dsm"]

    if len(dsm_links) == 0 and len(dsm_stores) == 0:
        logger.info("No DSM components found in the network. Skipping constraints.")
        return

    logger.info(f"Found {len(dsm_links)} DSM links and {len(dsm_stores)} DSM stores")

    # 1. Add daily energy neutrality constraints (if not using e_cyclic)
    links_by_bus = {}
    for link_name, link in dsm_links.iterrows():
        if link.bus1 not in links_by_bus:
            links_by_bus[link.bus1] = []
        links_by_bus[link.bus1].append(link_name)

    # Group snapshots by day
    daily_groups = {}
    if isinstance(snapshots[0], pd.Timestamp):
        # Group timestamps by day
        for i, s in enumerate(snapshots):
            day = s.date()
            if day not in daily_groups:
                daily_groups[day] = []
            daily_groups[day].append(i)
    else:
        # If not timestamps, create artificial days of 24 periods each
        period_length = 24  # Assuming 24 hours per day
        for i in range(0, len(snapshots), period_length):
            day_index = i // period_length
            end_idx = min(i + period_length, len(snapshots))
            daily_groups[f"day_{day_index}"] = list(range(i, end_idx))

    # Add daily neutrality constraints for each store
    for store_name, store in dsm_stores.iterrows():
        bus = store.bus
        links = []
        # Find links connected to this DSM bus
        for link_name, link in dsm_links.iterrows():
            if link.bus0 == bus:
                links.append({"name": link_name, "direction": "outflow"})
            elif link.bus1 == bus:
                links.append({"name": link_name, "direction": "inflow"})

        if not links:
            continue

        # Add daily energy neutrality constraints
        for day, day_indices in daily_groups.items():
            if len(day_indices) < 2:
                continue  # Skip if less than 2 periods in the day

            # Create expression for net energy flow in this day
            day_expr = 0
            for link in links:
                for t in day_indices:
                    if link["direction"] == "outflow":
                        # Energy leaving the store (negative impact)
                        day_expr -= n.model.variables[f"Link-p-{link['name']}"][t]
                    else:
                        # Energy entering the store (positive impact)
                        day_expr += n.model.variables[f"Link-p-{link['name']}"][t]

            # Allow small tolerance (0.5% of store capacity)
            tolerance = store.e_nom_opt if hasattr(store, 'e_nom_opt') else store.e_nom
            tolerance *= 0.005  # 0.5% tolerance

            # Add neutrality constraints
            n.model.addConstr(day_expr >= -tolerance, name=f"dsm_daily_min_{store_name}_{day}")
            n.model.addConstr(day_expr <= tolerance, name=f"dsm_daily_max_{store_name}_{day}")

    # 2. Add minimum utilization constraint if configured
    min_utilization = getattr(n, 'dsm_minimum_utilization', 0.0)

    if min_utilization > 0:
        logger.info(f"Adding minimum DSM utilization constraint: {min_utilization}")

        # Create a dictionary to track DSM activity by bus
        dsm_by_bus = {}
        for link_name, link in dsm_links.iterrows():
            bus_name = link.bus1  # The main grid bus
            if bus_name not in dsm_by_bus:
                dsm_by_bus[bus_name] = []
            dsm_by_bus[bus_name].append(link_name)

        # Add constraint for each bus with DSM
        for bus_name, link_names in dsm_by_bus.items():
            # Get total demand at this bus
            bus_loads = n.loads[n.loads.bus == bus_name]
            total_demand = 0

            for load_name, load in bus_loads.iterrows():
                # Get load time series
                if hasattr(n, 'loads_t') and hasattr(n.loads_t, 'p_set'):
                    load_ts = n.loads_t.p_set[load_name].sum()
                else:
                    load_ts = load.p_set * len(snapshots)
                total_demand += load_ts

            if total_demand > 0:
                # Calculate required minimum DSM activity
                min_dsm_activity = total_demand * min_utilization

                # Create auxiliary variables for absolute values
                abs_vars = []
                for link_name in link_names:
                    for t in range(len(snapshots)):
                        var_name = f"dsm_abs_{link_name}_{t}"
                        up_var = n.model.addVar(lb=0, name=f"{var_name}_up")
                        down_var = n.model.addVar(lb=0, name=f"{var_name}_down")

                        # Link to the DSM variable
                        dsm_var = n.model.variables[f"Link-p-{link_name}"][t]

                        # Add constraints to compute absolute value
                        n.model.addConstr(up_var >= dsm_var)
                        n.model.addConstr(down_var >= -dsm_var)

                        # Add both directions to track absolute value
                        abs_vars.append(up_var + down_var)

                # Add the minimum utilization constraint - sum of all absolute values
                if abs_vars:
                    abs_sum = sum(abs_vars)
                    n.model.addConstr(abs_sum >= min_dsm_activity, name=f"min_dsm_{bus_name}")
                    logger.info(
                        f"Added minimum DSM utilization constraint for bus {bus_name}: {min_dsm_activity:.2f} MWh")

    # 3. Add ramping constraints if configured
    if hasattr(n, 'config'):
        dsm_constraint_config = n.config.get('sector', {}).get('dsm', {}).get('constraints', {})

        # Add DSM ramping restrictions if configured
        if dsm_constraint_config.get('ramping_limit', {}).get('enable', False):
            logger.info("Adding ramping restrictions for DSM")
            ramp_limit = dsm_constraint_config.get('ramping_limit', {}).get('value', 0.5)

            for link_name in dsm_links.index:
                capacity = dsm_links.at[link_name, 'p_nom']
                max_ramp = capacity * ramp_limit

                for t in range(1, len(snapshots)):
                    # Add constraint limiting ramp rate between consecutive time steps
                    prev_var = n.model.variables[f"Link-p-{link_name}"][t - 1]
                    current_var = n.model.variables[f"Link-p-{link_name}"][t]

                    # Add constraints: |current - prev| <= max_ramp
                    n.model.addConstr(current_var - prev_var <= max_ramp)
                    n.model.addConstr(prev_var - current_var <= max_ramp)

            logger.info(f"Added ramping constraints with limit of {ramp_limit * 100}% of capacity")

    logger.info("Finished adding enhanced DSM constraints")


def analyze_dsm_usage(n):
    """
    Analyze and log DSM usage after optimization to diagnose any issues.
    This should be called after the model has been solved.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("\n=== DSM USAGE ANALYSIS ===")

    # Check if network has been solved
    if not hasattr(n, 'results') and not hasattr(n.links_t, 'p'):
        logger.warning("Network hasn't been solved yet. No DSM usage to analyze.")
        return

    # Get DSM links
    dsm_links = n.links[n.links.carrier == "dsm"]

    if len(dsm_links) == 0:
        logger.info("No DSM links found in network.")
        return

    logger.info(f"Analyzing DSM usage for {len(dsm_links)} links...")

    # Analyze DSM activity
    total_dsm_activity = 0
    total_dsm_capacity = 0
    active_links = 0

    for link_name in dsm_links.index:
        # Get the link's power timeseries
        if link_name in n.links_t.p:
            link_p = n.links_t.p[link_name]
            capacity = dsm_links.at[link_name, 'p_nom']

            # Calculate metrics
            abs_activity = link_p.abs().sum()
            max_activity = capacity * len(n.snapshots)
            utilization_rate = abs_activity / max_activity if max_activity > 0 else 0

            if abs_activity > 1e-6:  # If there's any significant activity
                active_links += 1
                logger.info(f"DSM Link {link_name}:")
                logger.info(f"  - Absolute activity: {abs_activity:.2f} MWh")
                logger.info(f"  - Capacity: {capacity:.2f} MW")
                logger.info(f"  - Utilization rate: {utilization_rate * 100:.2f}%")

                # Count positive and negative shifts
                pos_shifts = link_p[link_p > 1e-6].sum()
                neg_shifts = link_p[link_p < -1e-6].sum()
                logger.info(f"  - Load increases: {pos_shifts:.2f} MWh")
                logger.info(f"  - Load decreases: {neg_shifts:.2f} MWh")

                total_dsm_activity += abs_activity
                total_dsm_capacity += capacity

    if active_links > 0:
        logger.info(f"\nTotal DSM Summary:")
        logger.info(f"  - Active DSM links: {active_links} out of {len(dsm_links)}")
        logger.info(f"  - Total absolute DSM activity: {total_dsm_activity:.2f} MWh")
        logger.info(f"  - Total DSM capacity: {total_dsm_capacity:.2f} MW")
        logger.info(
            f"  - Overall utilization rate: {(total_dsm_activity / (total_dsm_capacity * len(n.snapshots))) * 100:.2f}%")
    else:
        logger.warning("No DSM activity detected in any link!")

        # Try to diagnose why
        logger.info("Potential reasons for zero DSM usage:")

        # Check costs vs. alternatives
        battery_links = n.links[n.links.carrier.str.contains('battery', case=False)]
        if len(battery_links) > 0:
            avg_battery_cost = battery_links.capital_cost.mean() if 'capital_cost' in battery_links.columns else "N/A"
            logger.info(f"  - Battery capital cost comparison: {avg_battery_cost}")

        # Check if there's storage utilization in the model
        if hasattr(n.stores_t, 'e'):
            dsm_stores = n.stores[n.stores.carrier == "dsm"]
            for store in dsm_stores.index:
                if store in n.stores_t.e:
                    store_ts = n.stores_t.e[store]
                    if store_ts.max() - store_ts.min() > 1e-6:
                        logger.info(f"  - DSM store {store} shows energy level changes but no link activity")

        # Check for binding constraints
        if hasattr(n, 'model') and hasattr(n.model, 'getConstrs'):
            dsm_constraints = [c for c in n.model.getConstrs() if 'dsm' in c.ConstrName.lower()]
            binding_constraints = []

            for constr in dsm_constraints:
                slack = n.model.getConstrByName(constr.ConstrName).getAttr("slack")
                if abs(slack) < 1e-6:  # Binding constraint
                    binding_constraints.append(constr.ConstrName)

            if binding_constraints:
                logger.info(f"  - Found {len(binding_constraints)} binding DSM constraints:")
                for constr in binding_constraints[:5]:  # Show first 5
                    logger.info(f"    * {constr}")

        # Check if costs are configured correctly
        dsm_config = n.config.get('sector', {}).get('dsm', {})
        active_scenario = dsm_config.get('active_scenario', 'medium')
        logger.info(f"  - Using DSM scenario: {active_scenario}")
        scenario_costs = dsm_config.get('scenarios', {}).get(active_scenario, {}).get('DE', {}).get('costs', {})
        year = n.planning_year if hasattr(n, 'planning_year') else "2030"
        year_costs = scenario_costs.get(str(year), {})
        logger.info(f"  - DSM costs for year {year}: {year_costs}")

        # Check system characteristics
        logger.info(f"  - System generation/load balance may not require DSM")
        logger.info(f"  - Cheaper flexibility options like interconnection might be preferred")


def add_resistive_heater_constraints(n):
    """
    Imposes max capacity constraints on resistive heaters in Germany.
    Uses the same approach as the working heat_pump_constraints function.
    """
    import logging
    import pandas as pd

    logging.info("[Resistive heater constraints] Starting resistive heater constraint application...")

    # Get config and check if constraints are enabled
    if not hasattr(n, 'config'):
        try:
            n.config = snakemake.config
            logging.info("[Resistive heater constraints] Got config from snakemake")
        except (NameError, AttributeError):
            logging.warning("[Resistive heater constraints] No config found, skipping resistive heater constraints")
            return

    # Try multiple possible paths for resistive heater config
    resistive_config = None
    possible_paths = [
        # Path 1: Under solving.constraints
        n.config.get('solving', {}).get('constraints', {}).get('resistive_heaters', {}),
        # Path 2: Directly under constraints
        n.config.get('constraints', {}).get('resistive_heaters', {}),
        # Path 3: At top level
        n.config.get('resistive_heaters', {})
    ]

    # Log the config structure for debugging
    logging.info("[Resistive heater constraints] Config contains these top-level keys: " +
                 str(list(n.config.keys())))
    if 'solving' in n.config:
        logging.info("[Resistive heater constraints] 'solving' section contains: " +
                     str(list(n.config['solving'].keys())))
        if 'constraints' in n.config['solving']:
            logging.info("[Resistive heater constraints] 'solving.constraints' section contains: " +
                         str(list(n.config['solving']['constraints'].keys())))

    # Try each path and use the first one that has apply_constraints=True
    for i, path_config in enumerate(possible_paths):
        if path_config and path_config.get('apply_constraints', False):
            resistive_config = path_config
            logging.info(f"[Resistive heater constraints] Found enabled resistive heater config at path {i + 1}")
            break

    # If none of the paths had apply_constraints=True
    if resistive_config is None or not resistive_config.get('apply_constraints', False):
        logging.info("[Resistive heater constraints] Resistive heater constraints disabled in config, skipping")
        return

    # Get planning year with multiple fallbacks
    try:
        planning_year = int(snakemake.wildcards.planning_horizons)
        logging.info(f"[Resistive heater constraints] Got planning year {planning_year} from snakemake wildcards")
    except:
        # Try other methods to determine planning year
        planning_year = 2045  # Default fallback
        logging.info(f"[Resistive heater constraints] Using default planning year {planning_year}")

    year_str = str(planning_year)
    logging.info(f"[Resistive heater constraints] Final planning year: {year_str}")

    # Get active scenario
    active_scenario = resistive_config.get('active_scenario', 'restrictive')
    logging.info(f"[Resistive heater constraints] Using {active_scenario} resistive heater scenario from config")

    # Get countries to apply constraints to
    countries_to_process = resistive_config.get('countries', [])
    if not countries_to_process:
        # If empty, apply to all countries in the network
        countries_to_process = n.buses.country.unique().tolist()
        logging.info(f"[Resistive heater constraints] Applying to all countries in network: {countries_to_process}")
    else:
        logging.info(f"[Resistive heater constraints] Applying to specified countries: {countries_to_process}")

    # Get scenario config
    scenario_config = resistive_config.get('scenarios', {}).get(active_scenario, {})

    # Debug log the scenario config
    logging.info(
        f"[Resistive heater constraints] Available years in {active_scenario} scenario: {list(scenario_config.keys())}")
    logging.info(
        f"[Resistive heater constraints] Types of year keys: {[type(k).__name__ for k in scenario_config.keys()]}")

    # Find all resistive heater carriers in the network
    all_carriers = n.links.carrier.unique()
    resistive_heater_carriers = [c for c in all_carriers if 'resistive' in c.lower()]
    logging.info(f"[Resistive heater constraints] Detected resistive heater carriers: {resistive_heater_carriers}")

    # Ensure all carriers are in the carriers component
    if "carriers" not in n.components:
        n.add("Carrier", resistive_heater_carriers)
        logging.info(f"[Resistive heater constraints] Added resistive heater carriers to network")
    elif not all(c in n.carriers.index for c in resistive_heater_carriers):
        # Add any missing carriers
        missing_carriers = [c for c in resistive_heater_carriers if c not in n.carriers.index]
        for c in missing_carriers:
            n.add("Carrier", c)
        logging.info(f"[Resistive heater constraints] Added missing carriers: {missing_carriers}")

    # Log the initial capacities
    logging.info("[Resistive heater constraints] Initial capacities before constraints:")
    for rh_carrier in resistive_heater_carriers:
        total_cap = n.links.loc[n.links.carrier == rh_carrier, 'p_nom'].sum()
        num_links = len(n.links[n.links.carrier == rh_carrier])
        logging.info(f"  {rh_carrier}: {total_cap:.2f} MW across {num_links} links")

    # Process each country
    countries_processed = 0
    total_links_constrained = 0

    for country in countries_to_process:
        # Get year config
        year_dict = None
        for key in scenario_config:
            if str(key) == year_str:
                year_dict = scenario_config[key]
                logging.info(f"[Resistive heater constraints] Found year {key} matching {year_str}")
                break

        if year_dict is None:
            logging.warning(f"[Resistive heater constraints] No data for year {year_str} in {active_scenario} scenario")
            continue

        # Get buses for this country
        country_buses = n.buses.index[n.buses.country == country]
        if len(country_buses) == 0:
            logging.warning(f"[Resistive heater constraints] No buses found for {country}")
            continue

        logging.info(f"[Resistive heater constraints] Processing country: {country} with {len(country_buses)} buses")

        # Track country totals
        country_max = 0
        country_links = 0

        # Find all resistive heater links in this country
        for rh_carrier, caps in year_dict.items():
            if rh_carrier in resistive_heater_carriers:
                # Get the maximum capacity constraint
                max_cap = caps.get("max", float('inf'))

                # Look for links with bus0 (input) in this country
                relevant_links = n.links.index[
                    (n.links.carrier == rh_carrier) &
                    (n.links.bus0.isin(country_buses))
                    ]

                if len(relevant_links) == 0:
                    logging.warning(f"[Resistive heater constraints] No {rh_carrier} links found in {country}")
                    continue

                # Calculate current total capacity
                current_capacity = n.links.loc[relevant_links, 'p_nom'].sum()

                # Log what we found
                logging.info(
                    f"[Resistive heater constraints] Found {len(relevant_links)} {rh_carrier} links in {country} "
                    f"with initial capacity {current_capacity:.2f} MW, target: {max_cap} MW")

                # Set capacity equally across all links
                capacity_per_link = max_cap / len(relevant_links)

                # Apply the constraint to each link
                for link in relevant_links:
                    # CRITICAL: Following the same pattern as the heat_pump_constraints function
                    # Set fixed capacity and make non-extendable
                    original_capacity = n.links.at[link, 'p_nom']
                    n.links.at[link, 'p_nom'] = capacity_per_link
                    n.links.at[link, 'p_nom_extendable'] = False

                    logging.info(
                        f"[Resistive heater constraints] RESET: {link} capacity from {original_capacity:.2f} MW to {capacity_per_link:.2f} MW")

                logging.info(f"[Resistive heater constraints] Limited {rh_carrier} capacity to {max_cap} MW "
                             f"({capacity_per_link:.2f} MW per link across {len(relevant_links)} links)")

                country_max += max_cap
                country_links += len(relevant_links)
                total_links_constrained += len(relevant_links)

            elif rh_carrier not in resistive_heater_carriers and rh_carrier != 'total':
                logging.warning(f"[Resistive heater constraints] Carrier {rh_carrier} not found in network")

        # Process 'total' constraint if present
        if 'total' in year_dict and resistive_heater_carriers:
            total_max = year_dict['total'].get('max', float('inf'))

            # Get all resistive heater links in this country
            all_resistive_links = []
            for carrier in resistive_heater_carriers:
                carrier_links = n.links.index[
                    (n.links.carrier == carrier) &
                    (n.links.bus0.isin(country_buses))
                    ]
                all_resistive_links.extend(carrier_links)

            # Remove duplicates
            all_resistive_links = list(set(all_resistive_links))

            if all_resistive_links:
                # Calculate current total capacity
                current_total = n.links.loc[all_resistive_links, 'p_nom'].sum()

                if current_total > total_max:
                    # Scale down all links proportionally
                    scaling_factor = total_max / current_total if current_total > 0 else 0

                    for link in all_resistive_links:
                        original_capacity = n.links.at[link, 'p_nom']
                        new_capacity = original_capacity * scaling_factor
                        n.links.at[link, 'p_nom'] = new_capacity
                        n.links.at[link, 'p_nom_extendable'] = False

                        logging.info(
                            f"[Resistive heater constraints] TOTAL ADJUST: {link} capacity from {original_capacity:.2f} MW to {new_capacity:.2f} MW")

                logging.info(
                    f"[Resistive heater constraints] Applied total resistive heater constraint: {total_max} MW")

        # Log country summary
        if country_links > 0:
            logging.info(
                f"[Resistive heater constraints] {country}: Applied capacity constraints to {country_links} links, max total: {country_max} MW")
            countries_processed += 1

    # Log overall summary
    if total_links_constrained > 0:
        logging.info(
            f"[Resistive heater constraints] Successfully applied constraints to {total_links_constrained} resistive heater links across {countries_processed} countries")
    else:
        logging.warning(f"[Resistive heater constraints] No resistive heater constraints were applied")

    # Add extra log to show the post-constraint capacities
    try:
        resistive_capacities = []
        for idx, row in n.links.iterrows():
            if 'resistive' in row['carrier'].lower():
                if row['bus0'].startswith(tuple(countries_to_process)):
                    resistive_capacities.append({
                        'Link': idx,
                        'Carrier': row['carrier'],
                        'Country': row['bus0'][:2],
                        'Max Capacity': row.get('p_nom_max', 'Not set'),
                        'Current Capacity': row.get('p_nom', 0),
                        'Extendable': row.get('p_nom_extendable', False)
                    })

        if resistive_capacities:
            capacities_df = pd.DataFrame(resistive_capacities)
            # Group by carrier and country
            capacities_summary = capacities_df.groupby(['Country', 'Carrier']).agg({
                'Current Capacity': 'sum',
                'Link': 'count',
                'Max Capacity': lambda x: sum([y for y in x if y != 'Not set'])
            }).reset_index()
            capacities_summary.columns = ['Country', 'Carrier', 'Total Capacity (MW)', 'Number of Links',
                                          'Max Capacity (MW)']
            logging.info(
                f"[Resistive heater constraints] Current resistive heater capacities by carrier:\n{capacities_summary.to_string()}")

            # Add overall summary by country
            country_summary = capacities_df.groupby('Country').agg({
                'Current Capacity': 'sum',
                'Link': 'count',
                'Max Capacity': lambda x: sum([y for y in x if y != 'Not set'])
            }).reset_index()
            country_summary.columns = ['Country', 'Total Capacity (MW)', 'Number of Links', 'Max Capacity (MW)']
            logging.info(
                f"[Resistive heater constraints] Current resistive heater capacities by country:\n{country_summary.to_string()}")

            # Verify final capacities
            total_capacity = capacities_df['Current Capacity'].sum()
            if total_capacity > country_max:
                logging.error(f"[Resistive heater constraints] WARNING: Final capacity ({total_capacity:.2f} MW) "
                              f"still exceeds limit ({country_max} MW)!")
            else:
                logging.info(f"[Resistive heater constraints] SUCCESS: Final capacity ({total_capacity:.2f} MW) "
                             f"now respects limit ({country_max} MW)")
        else:
            logging.info("[Resistive heater constraints] No resistive heater links with installed capacity found")
    except Exception as e:
        logging.error(f"[Resistive heater constraints] Error generating capacity summary: {e}")

    logging.info("[Resistive heater constraints] Completed resistive heater constraint application")


def add_renewable_share_constraints(n):
    """
    Adds constraints to enforce minimum/maximum renewable generation share for Germany (DE).
    Uses direct constraint creation approach as in add_power_limits function.

    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network to which constraints will be applied
    """
    import logging
    import json

    logging.info("[Renewable share constraints] Starting renewable share constraint application...")

    # Get config and check if constraints are enabled
    config = n.config
    renewable_config = config.get('solving', {}).get('constraints', {}).get('renewable_share', {})

    if not renewable_config:
        # Try to find renewable_share at root level
        renewable_config = config.get('renewable_share', {})

    if not renewable_config:
        # Try to find it in constraints at root level
        renewable_config = config.get('constraints', {}).get('renewable_share', {})

    # Log the full config structure for debugging
    try:
        logging.info(f"[Renewable share constraints] Full renewable config: {json.dumps(renewable_config, indent=2)}")
    except:
        logging.info(f"[Renewable share constraints] Renewable config: {renewable_config}")

    if not renewable_config.get('apply_constraints', False):
        logging.info("[Renewable share constraints] Renewable share constraints disabled in config")
        return

    # Get planning year
    try:
        planning_year = int(snakemake.wildcards.planning_horizons)
        logging.info(f"[Renewable share constraints] Using planning year {planning_year} from wildcards")
    except (NameError, AttributeError, ValueError):
        # Fallback to default
        planning_year = 2045
        logging.warning(
            f"[Renewable share constraints] Could not get planning year from wildcards, using default: {planning_year}")

    year_str = str(planning_year)

    # Get active scenario from config
    active_scenario = renewable_config.get('active_scenario', 'target')
    logging.info(f"[Renewable share constraints] Using {active_scenario} scenario")

    # Get the constraint configuration for this year and scenario
    scenario_data = renewable_config.get('scenarios', {}).get(active_scenario, {})

    # Log the scenario data
    try:
        logging.info(f"[Renewable share constraints] Scenario data: {json.dumps(scenario_data, indent=2)}")
    except:
        logging.info(f"[Renewable share constraints] Scenario data: {scenario_data}")

    # Check if we have data for this year
    if year_str not in scenario_data:
        # Log all available years for debugging
        available_years = list(scenario_data.keys())
        logging.warning(f"[Renewable share constraints] Available years in scenario: {available_years}")
        logging.warning(
            f"[Renewable share constraints] No constraints defined for year {year_str} in {active_scenario} scenario")

        # Try with integer key in case the config has numeric keys
        if planning_year in scenario_data:
            year_data = scenario_data[planning_year]
            logging.info(f"[Renewable share constraints] Found data using integer key {planning_year}")
        else:
            # Try with nearest year if no exact match
            nearest_year = min(available_years, key=lambda x: abs(int(x) - planning_year)) if available_years else None
            if nearest_year:
                year_data = scenario_data[nearest_year]
                logging.info(
                    f"[Renewable share constraints] Using nearest year {nearest_year} instead of {planning_year}")
            else:
                return
    else:
        year_data = scenario_data[year_str]

    logging.info(f"[Renewable share constraints] Applying constraints for year {year_str}")

    # Use specified renewable carriers from config or default to the provided list
    default_renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'hydro']
    renewable_carriers = renewable_config.get('renewable_carriers', default_renewable_carriers)

    # Get non-renewable carriers from config or use a default list
    default_non_renewable_carriers = ['gas', 'oil', 'lignite', 'coal', 'nuclear', 'OCGT', 'CCGT']
    non_renewable_carriers = renewable_config.get('non_renewable_carriers', default_non_renewable_carriers)

    # Get ambiguous carriers (those that need special handling)
    default_ambiguous_carriers = {'H2': False, 'battery': True,
                                  'PHS': True}  # Default classification (True = renewable)
    ambiguous_carriers_map = renewable_config.get('ambiguous_carriers', default_ambiguous_carriers)

    logging.info(f"[Renewable share constraints] Base renewable carriers: {renewable_carriers}")
    logging.info(f"[Renewable share constraints] Non-renewable carriers: {non_renewable_carriers}")
    logging.info(f"[Renewable share constraints] Ambiguous carriers: {ambiguous_carriers_map}")

    # Expand each carrier to include variations in naming
    def expand_carrier_list(carrier_list):
        expanded = []
        for carrier in carrier_list:
            # Find generators with this carrier (exact match or containing the carrier name)
            matching_gen_carriers = [c for c in n.generators.carrier.unique()
                                     if carrier == c or carrier in c.lower().split('-')]
            expanded.extend(matching_gen_carriers)
        return list(set(expanded))  # Remove duplicates

    expanded_renewable_carriers = expand_carrier_list(renewable_carriers)
    expanded_non_renewable_carriers = expand_carrier_list(non_renewable_carriers)

    # For ambiguous carriers, check each generator carrier name
    ambiguous_renewable = []
    ambiguous_non_renewable = []

    for gen_carrier in n.generators.carrier.unique():
        for amb_carrier, is_renewable in ambiguous_carriers_map.items():
            if amb_carrier in gen_carrier.lower():
                if is_renewable:
                    ambiguous_renewable.append(gen_carrier)
                else:
                    ambiguous_non_renewable.append(gen_carrier)

    # Add ambiguous carriers to their respective lists
    expanded_renewable_carriers.extend(ambiguous_renewable)
    expanded_non_renewable_carriers.extend(ambiguous_non_renewable)

    # Remove any duplicates (ensure no carrier is in both lists)
    expanded_renewable_carriers = list(set(expanded_renewable_carriers) - set(expanded_non_renewable_carriers))

    logging.info(f"[Renewable share constraints] Expanded renewable carriers: {expanded_renewable_carriers}")
    logging.info(f"[Renewable share constraints] Expanded non-renewable carriers: {expanded_non_renewable_carriers}")

    # Focus on Germany (DE)
    country = 'DE'
    if country not in year_data:
        logging.warning(f"[Renewable share constraints] No constraint defined for {country}")
        return

    country_data = year_data[country]
    min_share = country_data.get('min', None)
    max_share = country_data.get('max', None)

    logging.info(f"[Renewable share constraints] Constraints for {country}: min={min_share}, max={max_share}")

    if min_share is None and max_share is None:
        logging.warning(f"[Renewable share constraints] No min/max share defined for {country}")
        return

    try:
        # Find buses in this country
        country_buses = n.buses.index[n.buses.country == country]
        if len(country_buses) == 0:
            logging.warning(f"[Renewable share constraints] No buses found for {country}, skipping")
            return

        # Find generators connected to these buses
        country_gens = n.generators.index[n.generators.bus.isin(country_buses)]
        renewable_country_gens = []
        non_renewable_country_gens = []
        unclassified_gens = []

        for gen in country_gens:
            carrier = n.generators.at[gen, 'carrier']
            if carrier in expanded_renewable_carriers:
                renewable_country_gens.append(gen)
            elif carrier in expanded_non_renewable_carriers:
                non_renewable_country_gens.append(gen)
            else:
                # If not explicitly classified, log it for review and treat as non-renewable
                unclassified_gens.append(gen)
                non_renewable_country_gens.append(gen)

        if unclassified_gens:
            unclassified_carriers = [n.generators.at[gen, 'carrier'] for gen in unclassified_gens]
            logging.warning(
                f"[Renewable share constraints] Unclassified generator carriers: {set(unclassified_carriers)}")

        logging.info(
            f"[Renewable share constraints] {country}: found {len(renewable_country_gens)} renewable generators, "
            f"{len(non_renewable_country_gens)} non-renewable generators "
            f"out of {len(country_gens)} total generators")

        # Ensure we have both renewable and total generators
        if len(renewable_country_gens) == 0:
            logging.warning(f"[Renewable share constraints] No renewable generators found for {country}, skipping")
            return

        # Key approach: iterate over snapshots as in the add_power_limits function
        constraint_counter = 0
        for t in n.snapshots:
            try:
                # Get generation variables for this snapshot
                renewable_gen_vars = n.model["Generator-p"].loc[t, renewable_country_gens]
                total_gen_vars = n.model["Generator-p"].loc[t, country_gens]

                # For minimum share: renewable_gen >= min_share * total_gen
                if min_share is not None:
                    # Create expressions for minimum share constraint
                    renewable_sum = renewable_gen_vars.sum()
                    total_sum = total_gen_vars.sum()

                    # Create the constraint: renewable_sum >= min_share * total_sum
                    # Rearranged as: renewable_sum - min_share * total_sum >= 0
                    cname_min = f"renewable-share-min-{country}-{t}"
                    n.model.add_constraints(renewable_sum - min_share * total_sum >= 0, name=cname_min)
                    constraint_counter += 1

                # For maximum share: renewable_gen <= max_share * total_gen
                if max_share is not None:
                    # Create expressions for maximum share constraint
                    renewable_sum = renewable_gen_vars.sum()
                    total_sum = total_gen_vars.sum()

                    # Create the constraint: renewable_sum <= max_share * total_sum
                    # Rearranged as: renewable_sum - max_share * total_sum <= 0
                    cname_max = f"renewable-share-max-{country}-{t}"
                    n.model.add_constraints(renewable_sum - max_share * total_sum <= 0, name=cname_max)
                    constraint_counter += 1

                # Log progress for every 100 constraints
                if constraint_counter % 100 == 0:
                    logging.info(f"[Renewable share constraints] Added {constraint_counter} constraints so far...")

            except Exception as e:
                logging.error(f"[Renewable share constraints] Error creating constraints for snapshot {t}: {e}")
                import traceback
                logging.error(traceback.format_exc())

        logging.info(
            f"[Renewable share constraints] Successfully added {constraint_counter} renewable share constraints for {country}")

        # Add a callback to check the final values after solving
        def log_final_renewable_share():
            try:
                if not hasattr(n, 'generators_t') or not hasattr(n.generators_t, 'p'):
                    logging.info(f"[Renewable share constraints] No generator outputs in results")
                    return

                # Get the actual generation data
                gen_data = n.generators_t.p

                # Calculate total generation by country
                country_gen_total = gen_data[country_gens].sum().sum()
                country_renewable_gen = gen_data[renewable_country_gens].sum().sum()

                # Calculate the share
                if country_gen_total > 0:
                    actual_share = country_renewable_gen / country_gen_total
                    logging.info(
                        f"[Renewable share constraints] Final renewable share for {country}: {actual_share:.2%}")

                    # Compare with constraints
                    if min_share is not None and actual_share < min_share:
                        logging.warning(
                            f"[Renewable share constraints] Final share ({actual_share:.2%}) is less than minimum ({min_share:.0%})")

                    if max_share is not None and actual_share > max_share:
                        logging.warning(
                            f"[Renewable share constraints] Final share ({actual_share:.2%}) exceeds maximum ({max_share:.0%})")

                    # Detailed generation breakdown
                    logging.info(f"[Renewable share constraints] Total generation: {country_gen_total:.2f} MWh")
                    logging.info(f"[Renewable share constraints] Renewable generation: {country_renewable_gen:.2f} MWh")

                    # Show largest contributors
                    top_gens = gen_data[country_gens].sum().sort_values(ascending=False).head(10)
                    logging.info(f"[Renewable share constraints] Top generators:")
                    for gen, value in top_gens.items():
                        carrier = n.generators.at[gen, 'carrier']
                        is_renewable = "Renewable" if gen in renewable_country_gens else "Non-renewable"
                        logging.info(f"  - {gen} ({carrier}, {is_renewable}): {value:.2f} MWh")
                else:
                    logging.warning(f"[Renewable share constraints] No generation found for {country}")
            except Exception as e:
                logging.error(f"[Renewable share constraints] Error calculating final share: {e}")
                import traceback
                logging.error(traceback.format_exc())

        # Store the callback function to be called after solving
        if not hasattr(n, 'post_solve_callbacks'):
            n.post_solve_callbacks = []
        n.post_solve_callbacks.append(log_final_renewable_share)

    except Exception as e:
        logging.error(f"[Renewable share constraints] Error processing {country}: {e}")
        import traceback
        logging.error(traceback.format_exc())

    logging.info("[Renewable share constraints] Completed renewable share constraint application")


def add_storage_capacity_limits(n):
    """
    Adds constraints to enforce minimum/maximum battery storage capacity for Germany (DE).

    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network to which constraints will be applied
    """
    import logging
    import json

    logging.info("[Storage capacity limits] Starting storage capacity limit constraint application...")

    # Get config
    config = n.config
    storage_config = config.get('solving', {}).get('constraints', {}).get('storage_capacity_limits', {})

    if not storage_config:
        # Try to find storage_capacity_limits at root level
        storage_config = config.get('storage_capacity_limits', {})

    if not storage_config:
        # Try to find it in constraints at root level
        storage_config = config.get('constraints', {}).get('storage_capacity_limits', {})

    # Log the full config structure for debugging
    try:
        logging.info(f"[Storage capacity limits] Full storage config: {json.dumps(storage_config, indent=2)}")
    except:
        logging.info(f"[Storage capacity limits] Storage config: {storage_config}")

    if not storage_config.get('apply_constraints', False):
        logging.info("[Storage capacity limits] Storage capacity limit constraints disabled in config")
        return

    # Get planning year
    try:
        planning_year = int(snakemake.wildcards.planning_horizons)
        logging.info(f"[Storage capacity limits] Using planning year {planning_year} from wildcards")
    except (NameError, AttributeError, ValueError):
        # Fallback to default
        planning_year = 2045
        logging.warning(
            f"[Storage capacity limits] Could not get planning year from wildcards, using default: {planning_year}")

    year_str = str(planning_year)

    # Get the constraint limits for this year
    year_limits = storage_config.get('limits', {}).get(year_str, {})

    # If not found as string, try with integer
    if not year_limits and planning_year in storage_config.get('limits', {}):
        year_limits = storage_config.get('limits', {}).get(planning_year, {})
        logging.info(f"[Storage capacity limits] Found limits using integer key {planning_year}")

    # If still not found, try nearest year
    if not year_limits:
        available_years = [int(y) for y in storage_config.get('limits', {}).keys() if str(y).isdigit()]
        if available_years:
            nearest_year = min(available_years, key=lambda x: abs(x - planning_year))
            year_limits = storage_config.get('limits', {}).get(str(nearest_year), {})
            logging.info(f"[Storage capacity limits] Using nearest year {nearest_year} instead of {planning_year}")

    if not year_limits:
        logging.warning(f"[Storage capacity limits] No limits defined for year {planning_year}")
        return

    logging.info(f"[Storage capacity limits] Applying limits for year {planning_year}")

    # Focus on Germany (DE)
    country = 'DE'
    if country not in year_limits:
        logging.warning(f"[Storage capacity limits] No limits defined for {country}")
        return

    country_limits = year_limits[country]
    min_capacity = country_limits.get('min', None)  # In GW
    max_capacity = country_limits.get('max', None)  # In GW

    if min_capacity is not None:
        min_capacity = min_capacity * 1000  # Convert GW to MW
    if max_capacity is not None:
        max_capacity = max_capacity * 1000  # Convert GW to MW

    logging.info(f"[Storage capacity limits] Limits for {country}: min={min_capacity} MW, max={max_capacity} MW")

    if min_capacity is None and max_capacity is None:
        logging.warning(f"[Storage capacity limits] No min/max capacity defined for {country}")
        return

    # Get carriers to consider as battery storage
    default_battery_carriers = ['battery', 'home battery']
    battery_carriers = storage_config.get('battery_carriers', default_battery_carriers)

    # Expand to include variations (like 'battery charger', 'battery discharger', etc.)
    expanded_battery_carriers = []
    for carrier in battery_carriers:
        expanded_battery_carriers.append(carrier)
        expanded_battery_carriers.append(f"{carrier} charger")
        expanded_battery_carriers.append(f"{carrier} discharger")

    logging.info(f"[Storage capacity limits] Battery carriers (expanded): {expanded_battery_carriers}")

    try:
        # Find all links in Germany that are battery-related
        # First find buses in Germany
        country_buses = n.buses.index[n.buses.country == country]
        if len(country_buses) == 0:
            logging.warning(f"[Storage capacity limits] No buses found for {country}, skipping")
            return

        # Find battery links connected to these buses
        battery_links = []
        for link in n.links.index:
            link_carrier = n.links.at[link, 'carrier']
            bus0 = n.links.at[link, 'bus0']
            bus1 = n.links.at[link, 'bus1']

            # Check if it's a battery link (more flexible matching)
            is_battery = any(battery in link_carrier.lower() for battery in battery_carriers)
            is_in_germany = (bus0 in country_buses) or (bus1 in country_buses)

            if is_battery and is_in_germany:
                battery_links.append(link)

        # Log all found battery links for debugging
        logging.info(f"[Storage capacity limits] Found {len(battery_links)} battery links in {country}")
        for i, link in enumerate(battery_links):
            if i < 10:  # Limit to first 10 to avoid too much logging
                logging.info(
                    f"[Storage capacity limits] Battery link {i + 1}: {link}, carrier: {n.links.at[link, 'carrier']}")

        if len(battery_links) == 0:
            logging.warning(f"[Storage capacity limits] No battery links found for {country}, skipping")
            return

        # Group battery links by type
        chargers = []
        dischargers = []
        for link in battery_links:
            carrier = n.links.at[link, 'carrier'].lower()
            if 'charger' in carrier:
                chargers.append(link)
            elif 'discharger' in carrier:
                dischargers.append(link)
            else:
                # For other battery links without explicit charger/discharger in name
                # Check efficiency or bus connection pattern
                efficiency = n.links.at[link, 'efficiency']
                bus0 = n.links.at[link, 'bus0']
                bus1 = n.links.at[link, 'bus1']

                if 'battery' in bus1.lower():
                    # If the destination bus has 'battery' in the name, it's likely a charger
                    chargers.append(link)
                elif 'battery' in bus0.lower():
                    # If the source bus has 'battery' in the name, it's likely a discharger
                    dischargers.append(link)
                elif efficiency > 0:
                    # Fallback - based on efficiency
                    chargers.append(link)
                else:
                    dischargers.append(link)

        logging.info(f"[Storage capacity limits] Found {len(chargers)} chargers and {len(dischargers)} dischargers")

        # Check if we have any battery links
        if len(chargers) + len(dischargers) == 0:
            logging.warning(f"[Storage capacity limits] No battery chargers/dischargers found for {country}, skipping")
            return

        # Since we're constraining capacity, we need to be careful about double-counting
        # We'll focus on dischargers since they represent the usable capacity
        target_links = dischargers if dischargers else chargers

        # To avoid double counting when both chargers and dischargers exist, we'll use just one type
        if len(dischargers) > 0 and len(chargers) > 0:
            target_links = dischargers
            logging.info(f"[Storage capacity limits] Using only dischargers to avoid double-counting")

        # For extendable links, we need to constrain the p_nom_opt variable
        extendable_links = [link for link in target_links if n.links.at[link, 'p_nom_extendable']]

        # Also check for already fixed capacity in non-extendable links
        fixed_links = [link for link in target_links if not n.links.at[link, 'p_nom_extendable']]
        fixed_capacity = sum(n.links.at[link, 'p_nom'] for link in fixed_links)

        logging.info(
            f"[Storage capacity limits] Found {len(extendable_links)} extendable and {len(fixed_links)} fixed battery links")
        logging.info(f"[Storage capacity limits] Fixed capacity: {fixed_capacity} MW")

        if not extendable_links and fixed_capacity == 0:
            logging.warning(
                f"[Storage capacity limits] No extendable battery links found for {country} and no fixed capacity, skipping")
            return

        # Now add constraints
        try:
            # Adjust limits to account for fixed capacity
            adjusted_min = min_capacity - fixed_capacity if min_capacity is not None else None
            adjusted_max = max_capacity - fixed_capacity if max_capacity is not None else None

            logging.info(f"[Storage capacity limits] Adjusted limits: min={adjusted_min} MW, max={adjusted_max} MW")

            # Skip if no extendable links but fixed capacity satisfies constraints
            if not extendable_links:
                if (adjusted_min is None or adjusted_min <= 0) and (
                        adjusted_max is None or fixed_capacity <= max_capacity):
                    logging.info(
                        f"[Storage capacity limits] Fixed capacity ({fixed_capacity} MW) satisfies constraints")
                    return
                else:
                    logging.warning(
                        f"[Storage capacity limits] Fixed capacity ({fixed_capacity} MW) doesn't satisfy constraints, but no extendable links found")
                    return

            # If min constraint is already satisfied by fixed capacity, set to 0
            if adjusted_min is not None and adjusted_min <= 0:
                logging.info(f"[Storage capacity limits] Minimum constraint already satisfied by fixed capacity")
                adjusted_min = 0

            # If max constraint is smaller than fixed capacity, issue warning
            if adjusted_max is not None and adjusted_max < 0:
                logging.warning(
                    f"[Storage capacity limits] Maximum constraint ({max_capacity} MW) is smaller than fixed capacity ({fixed_capacity} MW)")
                # We'll still apply the constraint, but it might be infeasible

            # For minimum capacity: sum of p_nom_opt >= adjusted_min
            if adjusted_min is not None and adjusted_min > 0:
                min_cname = f"storage-capacity-min-{country}"
                n.model.add_constraints(
                    n.model["Link-p_nom"].loc[extendable_links].sum() >= adjusted_min,
                    name=min_cname
                )
                logging.info(f"[Storage capacity limits] Added minimum capacity constraint: {adjusted_min} MW")

            # For maximum capacity: sum of p_nom_opt <= adjusted_max
            if adjusted_max is not None and adjusted_max >= 0:
                max_cname = f"storage-capacity-max-{country}"
                n.model.add_constraints(
                    n.model["Link-p_nom"].loc[extendable_links].sum() <= adjusted_max,
                    name=max_cname
                )
                logging.info(f"[Storage capacity limits] Added maximum capacity constraint: {adjusted_max} MW")

            logging.info(f"[Storage capacity limits] Successfully added capacity constraints for {country}")

            # Add a callback to check the final values after solving
            def log_final_battery_capacity():
                if n.links.p_nom_opt.empty:
                    logging.info(f"[Storage capacity limits] No p_nom_opt attribute in results")
                    return

                # Calculate total battery capacity
                total_extendable = 0
                for link in extendable_links:
                    if link in n.links.index and hasattr(n.links, 'p_nom_opt'):
                        total_extendable += n.links.p_nom_opt.get(link, 0)

                total_capacity = total_extendable + fixed_capacity
                logging.info(
                    f"[Storage capacity limits] Final battery capacity: {total_capacity} MW (extendable: {total_extendable} MW, fixed: {fixed_capacity} MW)")

                if min_capacity is not None and total_capacity < min_capacity:
                    logging.warning(
                        f"[Storage capacity limits] Final capacity ({total_capacity} MW) is less than minimum ({min_capacity} MW)")

                if max_capacity is not None and total_capacity > max_capacity:
                    logging.warning(
                        f"[Storage capacity limits] Final capacity ({total_capacity} MW) exceeds maximum ({max_capacity} MW)")

            # Store the callback function to be called after solving
            if not hasattr(n, 'post_solve_callbacks'):
                n.post_solve_callbacks = []
            n.post_solve_callbacks.append(log_final_battery_capacity)

        except Exception as e:
            logging.error(f"[Storage capacity limits] Error adding constraints: {e}")
            import traceback
            logging.error(traceback.format_exc())

            # Try alternative approach using individual variables
            try:
                logging.info("[Storage capacity limits] Trying alternative approach...")

                # Get p_nom variables for extendable links
                p_nom_vars = n.model["Link-p_nom"]

                # Create sum expression manually
                sum_expr = sum(p_nom_vars.loc[link] for link in extendable_links)

                # Add constraints
                if adjusted_min is not None and adjusted_min > 0:
                    n.model.add_constraints(sum_expr >= adjusted_min, name=f"storage-capacity-min-{country}-alt")
                    logging.info(
                        f"[Storage capacity limits] Added minimum capacity constraint (alt): {adjusted_min} MW")

                if adjusted_max is not None and adjusted_max >= 0:
                    n.model.add_constraints(sum_expr <= adjusted_max, name=f"storage-capacity-max-{country}-alt")
                    logging.info(
                        f"[Storage capacity limits] Added maximum capacity constraint (alt): {adjusted_max} MW")

                logging.info(
                    f"[Storage capacity limits] Successfully added capacity constraints using alternative approach")

            except Exception as e2:
                logging.error(f"[Storage capacity limits] Alternative approach also failed: {e2}")
                import traceback
                logging.error(traceback.format_exc())

    except Exception as e:
        logging.error(f"[Storage capacity limits] Error: {e}")
        import traceback
        logging.error(traceback.format_exc())

    logging.info("[Storage capacity limits] Completed storage capacity limit constraint application")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.
    """
    config = n.config
    constraints = config["solving"].get("constraints", {})

    # Add renewable share constraints
    add_renewable_share_constraints(n)

    # Add storage capacity limits
    add_storage_capacity_limits(n)

    # Check if generators are extendable before applying certain constraints
    has_extendable_generators = n.generators.p_nom_extendable.any()

    if constraints.get("BAU", False) and has_extendable_generators:
        add_BAU_constraints(n, config)

    if constraints.get("SAFE", False) and has_extendable_generators:
        add_SAFE_constraints(n, config)

    if constraints.get("CCL", False) and has_extendable_generators:
        add_CCL_constraints(n, config)

    # Add CLL constraints if enabled
    try:
        cll_config = constraints.get("CLL", {})
        if cll_config.get("apply_constraints", False):
            logging.info("Applying CLL constraints for conventional generation...")
            add_CLL_constraints(n, config)
        else:
            logging.info("CLL constraints disabled in config, skipping")
    except Exception as e:
        logging.error(f"Error applying CLL constraints: {e}")
        # Decide whether to re-raise or continue
        # raise

    # DSM constraints
    # if config["sector"].get("dsm", {}).get("enable", False):
    #     add_dsm_constraints(n, snapshots)

    # Heat pump constraints
    hp_enabled = False
    possible_paths = [
        config.get('solving', {}).get('constraints', {}).get('heat_pumps', {}),
        config.get('constraints', {}).get('heat_pumps', {}),
        config.get('heat_pumps', {})
    ]

    for path_config in possible_paths:
        if path_config.get('apply_constraints', False):
            hp_enabled = True
            break

    if hp_enabled:
        logging.info("[Setup] Adding heat pump constraints")
        add_heat_pump_constraints(n)
    else:
        logging.info("[Setup] Heat pump constraints disabled in config, skipping")

    # Add DSM-specific constraints if needed
    # (note: we're not adding DSM components here, just additional constraints if necessary)
    # if config["sector"].get("dsm", {}).get("enable", False):
    #     add_dsm_constraints(n, snapshots)

        # Only add heat pump constraints if enabled in config
    # Try multiple paths for heat pump configuration
    hp_enabled = False
    possible_paths = [
        config.get('solving', {}).get('constraints', {}).get('heat_pumps', {}),
        config.get('constraints', {}).get('heat_pumps', {}),
        config.get('heat_pumps', {})
    ]

    for path_config in possible_paths:
        if path_config.get('apply_constraints', False):
            hp_enabled = True
            break

    if hp_enabled:
        logging.info("[Setup] Adding heat pump constraints")
        add_heat_pump_constraints(n)
    else:
        logging.info("[Setup] Heat pump constraints disabled in config, skipping")

    # Add resistive heater constraints if enabled in config
    # Try multiple paths for resistive heater configuration
    rh_enabled = False
    possible_paths = [
        config.get('solving', {}).get('constraints', {}).get('resistive_heaters', {}),
        config.get('constraints', {}).get('resistive_heaters', {}),
        config.get('resistive_heaters', {})
    ]

    for path_config in possible_paths:
        if path_config.get('apply_constraints', False):
            rh_enabled = True
            break

    if rh_enabled:
        logging.info("[Setup] Adding resistive heater constraints")
        add_resistive_heater_constraints(n)
    else:
        logging.info("[Setup] Resistive heater constraints disabled in config, skipping")

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

    if config['run'].get('quick_test', False):
        start_date = '2019-01-15 00:00:00'
        end_date = '2019-01-22 00:00:00'
        selected_snapshots = n.snapshots[(n.snapshots >= start_date) & (n.snapshots <= end_date)]
        kwargs['snapshots'] = selected_snapshots
        n.snapshots = selected_snapshots

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
