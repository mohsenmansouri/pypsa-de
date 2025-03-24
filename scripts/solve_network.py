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

        # Add this debug statement
        logging.info("Calling add_dsm_storage_units function...")

        # Call the DSM function and capture its return value
        dsm_added = add_dsm_storage_units(n, config)

    # Move DSM function call outside the if-condition
    logging.info("Calling add_dsm_storage_units function...")
    dsm_added = add_dsm_storage_units(n, config)
    logging.info(f"DSM storage units added: {dsm_added}")

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
    import logging
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
            'urban central solid biomass CHP', 'urban central solid biomass CHP CC',
            'H2 OCGT', 'urban central H2 CHP'
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

            # Zero H2 OCGT generators if they exist
            if carrier == 'H2 OCGT':
                h2_generators = n.generators.loc[
                    (n.generators.carrier == 'H2 OCGT') &
                    (n.generators.bus.isin(de_buses))
                    ]
                if not h2_generators.empty:
                    logging.info(
                        f"Zeroing capacity for {len(h2_generators)} existing H2 OCGT generator(s) in DE.")
                    n.generators.loc[h2_generators.index, "p_nom"] = 0
                    n.generators.loc[h2_generators.index, "p_nom_extendable"] = False
                    n.generators.loc[h2_generators.index, "p_nom_min"] = 0
                    n.generators.loc[h2_generators.index, "p_nom_max"] = 0

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

            elif carrier == 'H2 OCGT':
                bus0 = "EU H2"
                bus1 = "DE0 0"
                is_extendable = (min_cap != max_cap)
                if efficiency is None:
                    efficiency = 0.55  # Higher efficiency for H2 OCGT

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

            elif carrier == 'urban central H2 CHP':
                bus0 = "EU H2"
                bus1 = "DE0 0"
                is_extendable = (min_cap != max_cap)
                # Efficiency for H2 CHP
                if efficiency is None:
                    efficiency = 0.45  # Higher efficiency for H2 CHP

            else:
                logging.warning(f"Unknown carrier type: {carrier}")
                continue

            # Check if required buses exist
            if bus0 not in n.buses.index:
                logging.warning(f"Bus '{bus0}' not found in n.buses. Creating it.")
                # Create an appropriate carrier type based on the bus name
                bus_carrier = None
                if "gas" in bus0:
                    bus_carrier = "gas"
                elif "biomass" in bus0:
                    bus_carrier = "biomass"
                elif "H2" in bus0:
                    bus_carrier = "H2"
                else:
                    bus_carrier = bus0.split()[-1]  # Use the last part of the name

                n.add("Bus", bus0, carrier=bus_carrier)

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
                # For H2 OCGT, ensure we're using a direct link implementation
                if carrier == 'H2 OCGT':
                    # Create a direct link from H2 source to electricity bus
                    h2_ocgt_name = f"DE0 0 H2-OCGT-{planning_year}"

                    if h2_ocgt_name in n.links.index:
                        n.links.at[h2_ocgt_name, 'p_nom'] = min_cap
                        n.links.at[h2_ocgt_name, 'p_nom_min'] = min_cap
                        n.links.at[h2_ocgt_name, 'p_nom_max'] = max_cap
                        n.links.at[h2_ocgt_name, 'p_nom_extendable'] = is_extendable
                        n.links.at[h2_ocgt_name, 'efficiency'] = efficiency
                        msg = "Updated"
                    else:
                        n.add(
                            "Link",
                            h2_ocgt_name,
                            bus0=bus0,  # EU H2
                            bus1=bus1,  # DE0 0
                            carrier="H2 OCGT",
                            p_nom=min_cap,
                            p_nom_min=min_cap,
                            p_nom_max=max_cap,
                            p_nom_extendable=is_extendable,
                            efficiency=efficiency
                        )
                        msg = "Added"

                    if min_cap == max_cap:
                        logging.info(f"{msg} H2 OCGT link with fixed capacity: {min_cap} MW for year {planning_year}")
                    else:
                        logging.info(
                            f"{msg} H2 OCGT link with capacity range: {min_cap}-{max_cap} MW for year {planning_year}")

                    # Continue to next carrier
                    continue

                # Regular link implementation (not using AC) for other technologies
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

            # Check for H2 OCGT as a special case
            if carrier == 'H2 OCGT':
                h2_ocgt_link_pattern = "DE0 0 H2-OCGT"
                h2_ocgt_links = n.links.loc[
                    (n.links.carrier == "H2 OCGT") &
                    (n.links.index.str.startswith(h2_ocgt_link_pattern))
                    ]

                if not h2_ocgt_links.empty:
                    link_cap = h2_ocgt_links["p_nom"].sum()
                    link_min = h2_ocgt_links["p_nom_min"].sum()
                    link_max = h2_ocgt_links["p_nom_max"].sum()
                    total_capacities[carrier] = link_min

                    logging.info(f"\nDE {carrier} (special implementation):")
                    logging.info(f"  - Current capacity: {link_cap:.1f} MW")
                    logging.info(f"  - Min capacity: {link_min:.1f} MW")
                    logging.info(f"  - Max capacity: {link_max:.1f} MW")

                    # Continue to next carrier, we've already handled H2 OCGT
                    continue

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

        # Add hydrogen technologies
        total_h2 = total_capacities.get('H2 OCGT', 0)
        total_h2_chp = total_capacities.get('urban central H2 CHP', 0)

        logging.info(f"Total Gas (CCGT+OCGT): {total_gas:.1f} MW")
        logging.info(f"Total Gas CHP: {total_gas_chp:.1f} MW")
        logging.info(f"Total Biomass: {total_biomass:.1f} MW")
        logging.info(f"Total Biomass CHP: {total_biomass_chp:.1f} MW")
        logging.info(f"Total H2 OCGT: {total_h2:.1f} MW")
        logging.info(f"Total H2 CHP: {total_h2_chp:.1f} MW")

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
    Adds constraints to enforce minimum/maximum battery storage capacity for Germany (DE),
    with support for different battery types.

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

    # Focus on Germany (DE)
    country = 'DE'

    # Check if we're using the new battery_types structure
    battery_types = storage_config.get('battery_types', {})

    if battery_types:
        # Using new structure with multiple battery types
        for battery_type, type_config in battery_types.items():
            logging.info(f"[Storage capacity limits] Processing battery type: {battery_type}")

            # Get carriers for this battery type
            battery_carriers = type_config.get('carriers', [])
            if not battery_carriers:
                logging.warning(f"[Storage capacity limits] No carriers defined for battery type: {battery_type}")
                continue

            # Get limits for this year and battery type
            year_limits = type_config.get('limits', {}).get(year_str, {})

            # If not found as string, try with integer
            if not year_limits and planning_year in type_config.get('limits', {}):
                year_limits = type_config.get('limits', {}).get(planning_year, {})
                logging.info(f"[Storage capacity limits] Found limits using integer key {planning_year}")

            # If still not found, try nearest year
            if not year_limits:
                available_years = [int(y) for y in type_config.get('limits', {}).keys() if str(y).isdigit()]
                if available_years:
                    nearest_year = min(available_years, key=lambda x: abs(x - planning_year))
                    year_limits = type_config.get('limits', {}).get(str(nearest_year), {})
                    logging.info(
                        f"[Storage capacity limits] Using nearest year {nearest_year} instead of {planning_year}")

            if not year_limits:
                logging.warning(
                    f"[Storage capacity limits] No limits defined for year {planning_year} and battery type {battery_type}")
                continue

            if country not in year_limits:
                logging.warning(
                    f"[Storage capacity limits] No limits defined for {country} in battery type {battery_type}")
                continue

            country_limits = year_limits[country]
            min_capacity = country_limits.get('min', None)  # In GW
            max_capacity = country_limits.get('max', None)  # In GW

            if min_capacity is not None:
                min_capacity = min_capacity * 1000  # Convert GW to MW
            if max_capacity is not None:
                max_capacity = max_capacity * 1000  # Convert GW to MW

            logging.info(
                f"[Storage capacity limits] Limits for {country}, {battery_type}: min={min_capacity} MW, max={max_capacity} MW")

            if min_capacity is None and max_capacity is None:
                logging.warning(f"[Storage capacity limits] No min/max capacity defined for {country}, {battery_type}")
                continue

            # Apply constraints for this battery type
            apply_battery_type_constraints(n, country, battery_carriers, min_capacity, max_capacity, battery_type)

    else:
        # Fallback to old structure with single battery type
        logging.info("[Storage capacity limits] Using legacy config structure with single battery type")

        # Get limits for this year
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

        # Get list of battery carriers from config or use defaults
        default_battery_carriers = ['battery', 'home battery']
        battery_carriers = storage_config.get('battery_carriers', default_battery_carriers)

        # Apply constraints for the legacy single battery type
        apply_battery_type_constraints(n, country, battery_carriers, min_capacity, max_capacity, "all_batteries")

    logging.info("[Storage capacity limits] Completed storage capacity limit constraint application")


def apply_battery_type_constraints(n, country, battery_carriers, min_capacity, max_capacity, battery_type_name):
    """
    Helper function to apply constraints for a specific battery type
    """
    import logging

    logging.info(f"[Storage capacity limits] Applying constraints for battery type: {battery_type_name}")
    logging.info(f"[Storage capacity limits] Battery carriers: {battery_carriers}")

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
        logging.info(f"[Storage capacity limits] Found {len(battery_links)} {battery_type_name} links in {country}")
        for i, link in enumerate(battery_links):
            if i < 10:  # Limit to first the 10 to avoid too much logging
                logging.info(
                    f"[Storage capacity limits] Battery link {i + 1}: {link}, carrier: {n.links.at[link, 'carrier']}")

        if len(battery_links) == 0:
            logging.warning(f"[Storage capacity limits] No {battery_type_name} links found for {country}, skipping")
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

        logging.info(
            f"[Storage capacity limits] Found {len(chargers)} chargers and {len(dischargers)} dischargers for {battery_type_name}")

        # Check if we have any battery links
        if len(chargers) + len(dischargers) == 0:
            logging.warning(
                f"[Storage capacity limits] No {battery_type_name} chargers/dischargers found for {country}, skipping")
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
            f"[Storage capacity limits] Found {len(extendable_links)} extendable and {len(fixed_links)} fixed {battery_type_name} links")
        logging.info(f"[Storage capacity limits] Fixed capacity: {fixed_capacity} MW")

        if not extendable_links and fixed_capacity == 0:
            logging.warning(
                f"[Storage capacity limits] No extendable {battery_type_name} links found for {country} and no fixed capacity, skipping")
            return

        # Now add constraints
        try:
            # Adjust limits to account for fixed capacity
            adjusted_min = min_capacity - fixed_capacity if min_capacity is not None else None
            adjusted_max = max_capacity - fixed_capacity if max_capacity is not None else None

            logging.info(
                f"[Storage capacity limits] Adjusted limits for {battery_type_name}: min={adjusted_min} MW, max={adjusted_max} MW")

            # Skip if no extendable links but fixed capacity satisfies constraints
            if not extendable_links:
                if (adjusted_min is None or adjusted_min <= 0) and (
                        adjusted_max is None or fixed_capacity <= max_capacity):
                    logging.info(
                        f"[Storage capacity limits] Fixed capacity ({fixed_capacity} MW) satisfies constraints for {battery_type_name}")
                    return
                else:
                    logging.warning(
                        f"[Storage capacity limits] Fixed capacity ({fixed_capacity} MW) doesn't satisfy constraints for {battery_type_name}, but no extendable links found")
                    return

            # If min constraint is already satisfied by fixed capacity, set to 0
            if adjusted_min is not None and adjusted_min <= 0:
                logging.info(
                    f"[Storage capacity limits] Minimum constraint already satisfied by fixed capacity for {battery_type_name}")
                adjusted_min = 0

            # If max constraint is smaller than fixed capacity, issue warning
            if adjusted_max is not None and adjusted_max < 0:
                logging.warning(
                    f"[Storage capacity limits] Maximum constraint ({max_capacity} MW) is smaller than fixed capacity ({fixed_capacity} MW) for {battery_type_name}")
                # We'll still apply the constraint, but it might be infeasible

            # For minimum capacity: sum of p_nom_opt >= adjusted_min
            if adjusted_min is not None and adjusted_min > 0:
                min_cname = f"storage-capacity-min-{country}-{battery_type_name}"
                n.model.add_constraints(
                    n.model["Link-p_nom"].loc[extendable_links].sum() >= adjusted_min,
                    name=min_cname
                )
                logging.info(
                    f"[Storage capacity limits] Added minimum capacity constraint for {battery_type_name}: {adjusted_min} MW")

            # For maximum capacity: sum of p_nom_opt <= adjusted_max
            if adjusted_max is not None and adjusted_max >= 0:
                max_cname = f"storage-capacity-max-{country}-{battery_type_name}"
                n.model.add_constraints(
                    n.model["Link-p_nom"].loc[extendable_links].sum() <= adjusted_max,
                    name=max_cname
                )
                logging.info(
                    f"[Storage capacity limits] Added maximum capacity constraint for {battery_type_name}: {adjusted_max} MW")

            logging.info(
                f"[Storage capacity limits] Successfully added capacity constraints for {country}, {battery_type_name}")

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
                    f"[Storage capacity limits] Final {battery_type_name} capacity: {total_capacity} MW (extendable: {total_extendable} MW, fixed: {fixed_capacity} MW)")

                if min_capacity is not None and total_capacity < min_capacity:
                    logging.warning(
                        f"[Storage capacity limits] Final {battery_type_name} capacity ({total_capacity} MW) is less than minimum ({min_capacity} MW)")

                if max_capacity is not None and total_capacity > max_capacity:
                    logging.warning(
                        f"[Storage capacity limits] Final {battery_type_name} capacity ({total_capacity} MW) exceeds maximum ({max_capacity} MW)")

            # Store the callback function to be called after solving
            if not hasattr(n, 'post_solve_callbacks'):
                n.post_solve_callbacks = []
            n.post_solve_callbacks.append(log_final_battery_capacity)

        except Exception as e:
            logging.error(f"[Storage capacity limits] Error adding constraints for {battery_type_name}: {e}")
            import traceback
            logging.error(traceback.format_exc())

            # Try alternative approach using individual variables
            try:
                logging.info(f"[Storage capacity limits] Trying alternative approach for {battery_type_name}...")

                # Get p_nom variables for extendable links
                p_nom_vars = n.model["Link-p_nom"]

                # Create sum expression manually
                sum_expr = sum(p_nom_vars.loc[link] for link in extendable_links)

                # Add constraints
                if adjusted_min is not None and adjusted_min > 0:
                    n.model.add_constraints(sum_expr >= adjusted_min,
                                            name=f"storage-capacity-min-{country}-{battery_type_name}-alt")
                    logging.info(
                        f"[Storage capacity limits] Added minimum capacity constraint (alt) for {battery_type_name}: {adjusted_min} MW")

                if adjusted_max is not None and adjusted_max >= 0:
                    n.model.add_constraints(sum_expr <= adjusted_max,
                                            name=f"storage-capacity-max-{country}-{battery_type_name}-alt")
                    logging.info(
                        f"[Storage capacity limits] Added maximum capacity constraint (alt) for {battery_type_name}: {adjusted_max} MW")

                logging.info(
                    f"[Storage capacity limits] Successfully added capacity constraints using alternative approach for {battery_type_name}")

            except Exception as e2:
                logging.error(
                    f"[Storage capacity limits] Alternative approach also failed for {battery_type_name}: {e2}")
                import traceback
                logging.error(traceback.format_exc())

    except Exception as e:
        logging.error(f"[Storage capacity limits] Error processing {battery_type_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())


def add_zero_electricity_imports_for_germany(n, snapshots):
    """
    Custom function to force zero electricity imports for Germany.
    This adds a hard constraint regardless of other settings in the config.
    """
    import logging

    # Target country
    country = "DE"

    logging.info(f"ENFORCING ZERO ELECTRICITY IMPORTS FOR {country} (CUSTOM OVERRIDE)")

    # Find all cross-border links and lines that bring electricity to Germany
    incoming_links = n.links.index[
        (n.links.bus0.str[:2] != country) &
        (n.links.bus1.str[:2] == country) &
        (n.links.carrier == "DC")
        ]

    incoming_lines = n.lines.index[
        (n.lines.bus0.str[:2] != country) &
        (n.lines.bus1.str[:2] == country) &
        (n.lines.carrier == "AC")
        ]

    # For each snapshot, create a constraint that sets the sum of imports to zero
    for t in snapshots:
        lhs = 0  # Left-hand side of the constraint equation

        # Add link imports (if any)
        if len(incoming_links) > 0:
            lhs += n.model["Link-p"].loc[t, incoming_links].sum()

        # Add line imports (if any)
        if len(incoming_lines) > 0:
            lhs += n.model["Line-s"].loc[t, incoming_lines].sum()

        # Add the constraint: imports must equal zero at all times
        constraint_name = f"zero-electricity-import-{country}-{t}"
        n.model.add_constraints(lhs == 0, name=constraint_name)

    # Also add a constraint for the total annual imports
    annual_lhs = 0
    for t in snapshots:
        # Sum import flows for this timestep and weight by snapshot duration
        if len(incoming_links) > 0:
            annual_lhs += n.snapshot_weightings.at[t, "objective"] * n.model["Link-p"].loc[t, incoming_links].sum()

        if len(incoming_lines) > 0:
            annual_lhs += n.snapshot_weightings.at[t, "objective"] * n.model["Line-s"].loc[t, incoming_lines].sum()

    # Add the constraint: total annual imports must equal zero
    constraint_name = f"zero-annual-electricity-import-{country}"
    n.model.add_constraints(annual_lhs == 0, name=constraint_name)


def determine_planning_year(n):
    """
    Determine the planning year from the network using various methods.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network

    Returns
    -------
    int
        The identified planning year
    """
    import logging

    # Approach 1: Try to get year from wildcards
    planning_year = None
    try:
        planning_year = int(snakemake.wildcards.planning_horizons)
        logging.info(f"Using planning year {planning_year} from wildcards")
        return planning_year
    except (NameError, AttributeError, ValueError):
        logging.info("Could not get planning year from wildcards")

    # Approach 2: Check if network has investment_period attribute
    planning_year = getattr(n, 'investment_period', None)
    if planning_year is not None:
        logging.info(f"Using planning year {planning_year} from network investment_period attribute")
        return planning_year

    # Approach 3: Try to extract from snapshots
    if not n.snapshots.empty:
        try:
            first_snapshot = n.snapshots[0]
            if hasattr(first_snapshot, 'year'):
                planning_year = first_snapshot.year
                logging.info(f"Using planning year {planning_year} from first snapshot year attribute")
                return planning_year
            else:
                # Try to parse year from snapshot string
                try:
                    planning_year = int(str(first_snapshot)[:4])
                    logging.info(f"Using planning year {planning_year} from first snapshot string")
                    return planning_year
                except:
                    pass
        except:
            pass

    # Approach 4: Try to extract from filename if network has a 'filename' attribute
    filename = getattr(n, 'filename', '')
    if filename:
        import re
        year_match = re.search(r'_(\d{4})\.', filename)
        if year_match:
            planning_year = int(year_match.group(1))
            logging.info(f"Using planning year {planning_year} from filename")
            return planning_year

    # Approach 5: Default to 2030 if all else fails
    planning_year = 2030
    logging.warning(f"Could not determine planning year, using default: {planning_year}")
    return planning_year


def add_dsm_storage_units(n, config=None):
    """
    Add DSM components as distributed storage units based on load profiles.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network
    config : dict, optional
        Configuration dictionary

    Returns
    -------
    bool
        True if DSM storage units were successfully added, False otherwise
    """
    import logging
    import pandas as pd
    import numpy as np
    import os

    log_file = "dsm_implementation.log"

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info("Starting DSM implementation...")

    # Get config if not provided
    if config is None:
        config = getattr(n, 'config', {})

    # Get DSM constraints configuration
    dsm_config = config.get("solving", {}).get("constraints", {}).get("dsm", {})

    # Skip if DSM constraints are not enabled
    if not dsm_config.get("apply_constraints", False):
        logging.info("DSM constraints disabled in config")
        return False

    # Get parameters from config
    s_util = dsm_config.get("s_util", 0.85)
    s_flex = dsm_config.get("s_flex", 0.75)
    s_inc = dsm_config.get("s_inc", 0.7)
    s_dec = dsm_config.get("s_dec", 0.7)
    delta_t = dsm_config.get("delta_t", 12)
    resample = dsm_config.get("resample", 3)
    marginal_cost = dsm_config.get("marginal_cost", -2.0)

    # Determine planning year (from snakemake wildcards or filename)
    planning_year = None
    try:
        # Try to get from snakemake wildcards
        planning_year = int(snakemake.wildcards.planning_horizons)
        logging.info(f"Using planning year {planning_year} from wildcards")
    except (NameError, AttributeError, ValueError):
        # Try to extract from filename
        if hasattr(n, 'name'):
            import re
            year_match = re.search(r'(\d{4})', n.name)
            if year_match:
                planning_year = int(year_match.group(1))
                logging.info(f"Using planning year {planning_year} from network name")

    # If still not found, try to extract from snapshots
    if planning_year is None and not n.snapshots.empty:
        try:
            # Try to get year from first snapshot
            first_snapshot = n.snapshots[0]
            if hasattr(first_snapshot, 'year'):
                planning_year = first_snapshot.year
            else:
                # Try to parse from string
                planning_year = int(str(first_snapshot)[:4])
            logging.info(f"Using planning year {planning_year} from snapshots")
        except:
            pass

    # Default to 2045 if we still don't have a year
    if planning_year is None:
        planning_year = 2045
        logging.info(f"Using default planning year {planning_year}")

    year_str = str(planning_year)

    # Get year-specific parameters
    year_parameters = dsm_config.get("year_parameters", {}).get(year_str, {})
    if not year_parameters and planning_year in dsm_config.get("year_parameters", {}):
        year_parameters = dsm_config.get("year_parameters", {}).get(planning_year, {})

    # If still not found, try nearest year
    if not year_parameters:
        available_years = [int(y) for y in dsm_config.get("year_parameters", {}).keys() if str(y).isdigit()]
        if available_years:
            nearest_year = min(available_years, key=lambda x: abs(x - planning_year))
            year_parameters = dsm_config.get("year_parameters", {}).get(str(nearest_year), {})
            logging.info(f"Using DSM parameters from nearest year {nearest_year} instead of {planning_year}")

    # Get target capacity from year parameters (in GW)
    target_capacity = year_parameters.get("target_capacity", 21.0)
    logging.info(f"Using DSM target capacity of {target_capacity} GW for year {planning_year}")

    # Override parameters with year-specific values if available
    for param in ["s_util", "s_flex", "s_inc", "s_dec", "delta_t", "resample", "marginal_cost"]:
        if param in year_parameters:
            locals()[param] = year_parameters[param]
            logging.info(f"Using year-specific {param}={year_parameters[param]}")

    # Define sector allocation (can be specified in config)
    sector_allocation = year_parameters.get("sector_allocation", {
        "general_electricity": 0.4,  # 40% to general electricity
        "industry_electricity": 0.4,  # 40% to industry electricity
        "agriculture_electricity": 0.05,  # 5% to agriculture
        "heat": 0.1,  # 10% to heat sector
        "ev": 0.05  # 5% to EV sector
    })

    # Ensure the allocation sums to 1.0
    total_allocation = sum(sector_allocation.values())
    if abs(total_allocation - 1.0) > 0.0001:
        for sector in sector_allocation:
            sector_allocation[sector] /= total_allocation

    # Target capacity per sector (in MW)
    target_capacity_mw = target_capacity * 1000  # Convert GW to MW
    sector_capacities = {
        sector: target_capacity_mw * allocation
        for sector, allocation in sector_allocation.items()
    }

    # Find all German low voltage buses
    de_low_voltage_buses = [bus for bus in n.buses.index if bus.startswith('DE') and 'low voltage' in bus]
    logging.info(f"Found {len(de_low_voltage_buses)} German low voltage buses for DSM allocation")

    if not de_low_voltage_buses:
        logging.error("No German low voltage buses found for DSM allocation")
        return False

    # Find regions (DE0 0, DE0 1, etc.)
    regions = []
    for bus in de_low_voltage_buses:
        region = " ".join(bus.split()[:2])  # Get first two parts (e.g., "DE0 0")
        if region not in regions:
            regions.append(region)

    logging.info(f"Found {len(regions)} German regions: {', '.join(regions)}")

    # For each region, find its loads
    region_loads = {}
    for region in regions:
        region_loads[region] = {
            'general': [],
            'industry': [],
            'agriculture': []
        }

    # Categorize loads by region and type
    for idx, load in n.loads.iterrows():
        # Skip non-German loads
        if not load.bus.startswith('DE'):
            continue

        # Extract region
        parts = load.bus.split()
        if len(parts) >= 2:
            region = f"{parts[0]} {parts[1]}"
        else:
            continue

        if region not in region_loads:
            continue

        # Categorize by type
        if 'industry' in idx.lower() or 'industry' in str(load.get('carrier', '')).lower():
            region_loads[region]['industry'].append(idx)
        elif 'agriculture' in idx.lower() or 'agriculture' in str(load.get('carrier', '')).lower():
            region_loads[region]['agriculture'].append(idx)
        elif 'electricity' in idx.lower() or 'electricity' in str(load.get('carrier', '')).lower() or (
                len(idx.split()) == 1):
            # General electricity loads often have simple names like "DE0 0"
            if idx in n.loads_t.p_set.columns:  # Only include if it has time series
                region_loads[region]['general'].append(idx)

    # Count the loads we found
    for sector in ['general', 'industry', 'agriculture']:
        count = sum(len(region_loads[region][sector]) for region in region_loads)
        logging.info(f"Found {count} {sector} electricity loads across all regions")

    # Get time series for general electricity (the only ones with time series)
    general_ts_by_region = {}
    for region in regions:
        general_loads = region_loads[region]['general']
        if general_loads:
            valid_loads = [load for load in general_loads if load in n.loads_t.p_set.columns]
            if valid_loads:
                general_ts_by_region[region] = n.loads_t.p_set[valid_loads].sum(axis=1)
                logging.info(f"Region {region}: Found time series for {len(valid_loads)} general electricity loads")
            else:
                logging.warning(f"Region {region}: No valid time series found for general electricity loads")

    # Get static values for industry and agriculture
    industry_load_by_region = {}
    agriculture_load_by_region = {}
    for region in regions:
        # Sum industry loads
        industry_loads = region_loads[region]['industry']
        if industry_loads:
            industry_load_by_region[region] = sum(abs(n.loads.at[load, 'p_set']) for load in industry_loads)
            logging.info(f"Region {region}: Industry electricity load: {industry_load_by_region[region]:.2f} MW")

        # Sum agriculture loads
        agriculture_loads = region_loads[region]['agriculture']
        if agriculture_loads:
            agriculture_load_by_region[region] = sum(abs(n.loads.at[load, 'p_set']) for load in agriculture_loads)
            logging.info(f"Region {region}: Agriculture electricity load: {agriculture_load_by_region[region]:.2f} MW")

    # Remove existing DSM units if any
    existing_dsm = [unit for unit in n.storage_units.index if "DSM" in unit]
    if existing_dsm:
        logging.info(f"Removing {len(existing_dsm)} existing DSM units")
        n.storage_units.drop(existing_dsm, errors='ignore', inplace=True)

    # Make sure the DSM carrier exists
    if "DSM" not in n.carriers.index:
        n.add("Carrier", "DSM")

    # Function to add DSM units for general electricity (with time series)
    def add_general_electricity_dsm(allocated_capacity):
        units_added = 0
        capacity_added = 0

        # Skip if no time series available
        if not general_ts_by_region:
            logging.warning("No time series available for general electricity DSM")
            return 0, 0

        # Distribute capacity proportional to load
        total_load = sum(ts.abs().sum() for ts in general_ts_by_region.values())
        if total_load < 1:  # Avoid division by zero
            return 0, 0

        for region, ts in general_ts_by_region.items():
            region_load = ts.abs().sum()
            region_capacity = allocated_capacity * (region_load / total_load)

            if region_capacity < 1:  # Skip tiny capacities
                continue

            # Create DSM unit
            bus_name = f"{region} low voltage"
            dsm_name = f"{region} general electricity DSM"

            # Calculate DSM parameters based on time series
            l_t = s_flex * ts
            p_max_t = region_capacity * s_inc - l_t
            p_min_t = -(l_t - region_capacity * s_dec)

            p_max_pu = p_max_t.clip(lower=0) / region_capacity
            p_min_pu = p_min_t.clip(upper=0) / region_capacity

            try:
                # Add the storage unit
                n.add(
                    "StorageUnit",
                    dsm_name,
                    bus=bus_name,
                    carrier="DSM",
                    max_hours=delta_t,
                    p_nom=region_capacity,
                    cyclic_state_of_charge=True,
                    efficiency_store=0.95,
                    efficiency_dispatch=0.95,
                    marginal_cost=marginal_cost
                )

                # Add time series data
                if not hasattr(n.storage_units_t, 'p_max_pu'):
                    n.storage_units_t['p_max_pu'] = pd.DataFrame(index=n.snapshots)
                if not hasattr(n.storage_units_t, 'p_min_pu'):
                    n.storage_units_t['p_min_pu'] = pd.DataFrame(index=n.snapshots)

                n.storage_units_t.p_max_pu[dsm_name] = p_max_pu
                n.storage_units_t.p_min_pu[dsm_name] = p_min_pu

                units_added += 1
                capacity_added += region_capacity
                logging.info(f"Added DSM unit: {dsm_name} with capacity {region_capacity:.2f} MW")

            except Exception as e:
                logging.error(f"Error adding DSM unit {dsm_name}: {e}")

        return units_added, capacity_added

    # Function to add DSM units for industry or agriculture (static load values)
    def add_static_load_dsm(load_by_region, sector_name, allocated_capacity):
        units_added = 0
        capacity_added = 0

        # Skip if no load data available
        if not load_by_region:
            logging.warning(f"No load data available for {sector_name} DSM")
            return 0, 0

        # Distribute capacity proportional to load
        total_load = sum(load for load in load_by_region.values())
        if total_load < 1:  # Avoid division by zero
            return 0, 0

        for region, load in load_by_region.items():
            region_capacity = allocated_capacity * (load / total_load)

            if region_capacity < 1:  # Skip tiny capacities
                continue

            # Create DSM unit
            bus_name = f"{region} low voltage"
            dsm_name = f"{region} {sector_name} DSM"

            # For static loads, create synthetic time series based on general pattern
            # If we have general electricity time series for this region, use its pattern
            if region in general_ts_by_region:
                pattern = general_ts_by_region[region] / general_ts_by_region[region].mean()

                # Industry has flatter load, agriculture more variable
                if sector_name == 'industry':
                    pattern = 0.7 + 0.3 * pattern  # Flatter (70% constant, 30% variable)
                else:  # agriculture
                    pattern = 0.3 + 0.7 * pattern  # More variable (30% constant, 70% variable)

                # Calculate DSM parameters
                synthetic_load = pattern * load
                l_t = s_flex * synthetic_load
                p_max_t = region_capacity * s_inc - l_t
                p_min_t = -(l_t - region_capacity * s_dec)

                p_max_pu = p_max_t.clip(lower=0) / region_capacity
                p_min_pu = p_min_t.clip(upper=0) / region_capacity
            else:
                # No pattern available, use constant values
                p_max_pu = pd.Series(s_inc, index=n.snapshots)
                p_min_pu = pd.Series(-s_dec, index=n.snapshots)

            try:
                # Add the storage unit
                n.add(
                    "StorageUnit",
                    dsm_name,
                    bus=bus_name,
                    carrier="DSM",
                    max_hours=delta_t,
                    p_nom=region_capacity,
                    cyclic_state_of_charge=True,
                    efficiency_store=0.95,
                    efficiency_dispatch=0.95,
                    marginal_cost=marginal_cost
                )

                # Add time series data
                if not hasattr(n.storage_units_t, 'p_max_pu'):
                    n.storage_units_t['p_max_pu'] = pd.DataFrame(index=n.snapshots)
                if not hasattr(n.storage_units_t, 'p_min_pu'):
                    n.storage_units_t['p_min_pu'] = pd.DataFrame(index=n.snapshots)

                n.storage_units_t.p_max_pu[dsm_name] = p_max_pu
                n.storage_units_t.p_min_pu[dsm_name] = p_min_pu

                units_added += 1
                capacity_added += region_capacity
                logging.info(f"Added DSM unit: {dsm_name} with capacity {region_capacity:.2f} MW")

            except Exception as e:
                logging.error(f"Error adding DSM unit {dsm_name}: {e}")

        return units_added, capacity_added

    # Add DSM units for the three electricity sectors
    general_units, general_capacity = add_general_electricity_dsm(
        sector_capacities.get('general_electricity', 0))

    industry_units, industry_capacity = add_static_load_dsm(
        industry_load_by_region, 'industry',
        sector_capacities.get('industry_electricity', 0))

    agriculture_units, agriculture_capacity = add_static_load_dsm(
        agriculture_load_by_region, 'agriculture',
        sector_capacities.get('agriculture_electricity', 0))

    # Find existing heat and EV loads for potential extension to these sectors
    heat_load_by_region = {}
    ev_load_by_region = {}

    # Look for heat and EV buses and loads
    heat_buses = [bus for bus in n.buses.index if any(term in bus.lower() for term in ['heat', 'thermal'])]
    ev_buses = [bus for bus in n.buses.index if any(term in bus.lower() for term in ['ev', 'battery'])]

    logging.info(f"Found {len(heat_buses)} heat buses and {len(ev_buses)} EV buses")

    # Process heat loads
    for bus in heat_buses:
        if not bus.startswith('DE'):
            continue

        # Extract region
        parts = bus.split()
        if len(parts) >= 2:
            region = f"{parts[0]} {parts[1]}"
        else:
            continue

        if region not in regions:
            continue

        # Find loads on this bus
        bus_loads = n.loads[n.loads.bus == bus]
        if not bus_loads.empty:
            total_load = abs(bus_loads['p_set'].sum())
            if total_load > 0:
                if region not in heat_load_by_region:
                    heat_load_by_region[region] = 0
                heat_load_by_region[region] += total_load

    # Process EV loads
    for bus in ev_buses:
        if not bus.startswith('DE'):
            continue

        # Extract region
        parts = bus.split()
        if len(parts) >= 2:
            region = f"{parts[0]} {parts[1]}"
        else:
            continue

        if region not in regions:
            continue

        # Find loads on this bus
        bus_loads = n.loads[n.loads.bus == bus]
        if not bus_loads.empty:
            total_load = abs(bus_loads['p_set'].sum())
            if total_load > 0:
                if region not in ev_load_by_region:
                    ev_load_by_region[region] = 0
                ev_load_by_region[region] += total_load

    # Add DSM for heat and EV if data available
    heat_units, heat_capacity = add_static_load_dsm(
        heat_load_by_region, 'heat',
        sector_capacities.get('heat', 0))

    ev_units, ev_capacity = add_static_load_dsm(
        ev_load_by_region, 'ev',
        sector_capacities.get('ev', 0))

    # Summary of added DSM units
    total_units = general_units + industry_units + agriculture_units + heat_units + ev_units
    total_capacity = general_capacity + industry_capacity + agriculture_capacity + heat_capacity + ev_capacity

    logging.info(f"\n=== DSM IMPLEMENTATION SUMMARY ===")
    logging.info(f"Total DSM units added: {total_units}")
    logging.info(f"Total DSM capacity: {total_capacity:.2f} MW ({total_capacity / 1000:.2f} GW)")
    logging.info(f"  - General Electricity: {general_units} units, {general_capacity:.2f} MW")
    logging.info(f"  - Industry Electricity: {industry_units} units, {industry_capacity:.2f} MW")
    logging.info(f"  - Agriculture Electricity: {agriculture_units} units, {agriculture_capacity:.2f} MW")
    logging.info(f"  - Heat: {heat_units} units, {heat_capacity:.2f} MW")
    logging.info(f"  - EV: {ev_units} units, {ev_capacity:.2f} MW")

    # Store DSM data for analysis
    n.dsm_storage_units = [unit for unit in n.storage_units.index if "DSM" in unit]
    n.dsm_parameters = {
        'planning_year': planning_year,
        'target_capacity_gw': target_capacity,
        's_util': s_util,
        's_flex': s_flex,
        's_inc': s_inc,
        's_dec': s_dec,
        'delta_t': delta_t,
        'resample': resample,
        'marginal_cost': marginal_cost,
        'sector_allocation': sector_allocation,
    }

    logging.info(f"DSM implementation completed successfully")

    return total_units > 0

def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.
    """
    import logging
    config = getattr(n, 'config', {})  # Get config from n if available

    # If n doesn't have config attribute, try to get it from a global/parent scope
    if not config:
        try:
            config = snakemake.config  # Try to get from snakemake if available
        except (NameError, AttributeError):
            import logging
            logging.warning("Cannot find configuration in network or snakemake objects")
            config = {}

    constraints = config.get("solving", {}).get("constraints", {})

    # DSM handling - check for storage-based DSM first
    if hasattr(n, 'dsm_storage_units') and n.dsm_storage_units:
        dsm_units_exist = any(unit in n.storage_units.index for unit in n.dsm_storage_units)

        if dsm_units_exist:
            logging.info("Found storage-based DSM units in the network")
            logging.info("Using storage-based DSM model without explicit constraints")
            # No constraints needed for storage-based DSM since it uses standard storage units
        else:
            logging.warning("DSM storage units were defined but not found in the network")
    # Then check for legacy DSM components
    elif hasattr(n, 'dsm') and n.dsm.get('components'):
        try:
            logging.info("Adding advanced DSM constraints based on configuration")
            add_dsm_storage_units(n, snapshots)
        except Exception as e:
            logging.error(f"Error adding DSM constraints: {e}")
            logging.warning("Continuing without DSM constraints")
    else:
        logging.info("No DSM components found, skipping DSM constraints")

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

    try:
        cll_config = constraints.get("CLL", {})
        if cll_config.get("apply_constraints", False):
            logging.info("Applying CLL constraints for conventional generation...")
            add_CLL_constraints(n, config)
        else:
            logging.info("CLL constraints disabled in config, skipping")
    except Exception as e:
        logging.error(f"Error applying CLL constraints: {e}")

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

        # ADD OUR CUSTOM CONSTRAINT FOR ZERO ELECTRICITY IMPORTS IN GERMANY
    add_zero_electricity_imports_for_germany(n, snapshots)

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
    # Check DSM units before solving
    if hasattr(n, 'dsm_storage_units'):
        dsm_units = [unit for unit in n.dsm_storage_units if unit in n.storage_units.index]
        logging.info(f"Before solving: Found {len(dsm_units)} DSM storage units")

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
    # Check DSM units after solving
    if hasattr(n, 'dsm_storage_units'):
        dsm_units = [unit for unit in n.dsm_storage_units if unit in n.storage_units.index]
        logging.info(f"After solving: Found {len(dsm_units)} DSM storage units")

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
