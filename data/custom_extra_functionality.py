# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import logging

logger = logging.getLogger(__name__)


def add_power_limits(n):
    """
    " Restricts the maximum inflow/outflow of electricity from/to a country.
    """
    ct = 'DE'

    limit = 0
    # limit = 1e3 * limits_power_max[ct][investment_year] / 10

    logger.info(
        f"Adding constraint on electricity import/export from/to {ct} to be < {limit} MW"
    )
    incoming_line = n.lines.index[
        (n.lines.carrier == "AC")
        & (n.lines.bus0.str[:2] != ct)
        & (n.lines.bus1.str[:2] == ct)
        ]
    outgoing_line = n.lines.index[
        (n.lines.carrier == "AC")
        & (n.lines.bus0.str[:2] == ct)
        & (n.lines.bus1.str[:2] != ct)
        ]

    incoming_link = n.links.index[
        (n.links.carrier == "DC")
        & (n.links.bus0.str[:2] != ct)
        & (n.links.bus1.str[:2] == ct)
        ]
    outgoing_link = n.links.index[
        (n.links.carrier == "DC")
        & (n.links.bus0.str[:2] == ct)
        & (n.links.bus1.str[:2] != ct)
        ]

    # iterate over snapshots - otherwise exporting of postnetwork fails since
    # the constraints are time dependent
    for t in n.snapshots:
        incoming_line_p = n.model["Line-s"].loc[t, incoming_line]
        outgoing_line_p = n.model["Line-s"].loc[t, outgoing_line]
        incoming_link_p = n.model["Link-p"].loc[t, incoming_link]
        outgoing_link_p = n.model["Link-p"].loc[t, outgoing_link]

        lhs = (
                      incoming_link_p.sum()
                      - outgoing_link_p.sum()
                      + incoming_line_p.sum()
                      - outgoing_line_p.sum()
              ) / 10
        # divide by 10 to avoid numerical issues

        cname_upper = f"Power-import-limit-{ct}-{t}"
        cname_lower = f"Power-export-limit-{ct}-{t}"

        n.model.add_constraints(lhs <= limit, name=cname_upper)
        n.model.add_constraints(lhs >= -limit, name=cname_lower)

        # not adding to network as the shadow prices are not needed

def custom_extra_functionality(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints.
    """
    pass
