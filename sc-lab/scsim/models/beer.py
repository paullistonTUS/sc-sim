"""Beer Game: a four-echelon serial chain showing the bullwhip effect.

Each week, in order from retailer to factory, every echelon receives the
shipments and orders that have arrived, ships what it can against backlog plus
new demand, updates an exponentially smoothed demand forecast and places an
order using an anchoring-and-adjustment rule of the kind Sterman (1989) fitted
to human players:

    order = forecast + alpha_s * (desired inventory - net inventory)
                     + alpha_s * beta * (desired supply line - supply line)

beta < 1 means the supply line (orders placed but not yet received) is
under-weighted, the behaviour that drives over-ordering. Orders and shipments
travel as timed SimPy events, landing mid-week before the next decision.
"""
from __future__ import annotations

import numpy as np
import simpy

from ..common import TraceBuilder

ECH = [("retailer", "Retailer", "retailer"), ("wholesaler", "Wholesaler", "warehouse"),
       ("distributor", "Distributor", "warehouse"), ("factory", "Factory", "factory")]

NOTES = [
    "Customer demand steps up once, and never changes again. Watch how far each echelon's orders swing.",
    "Orders grow larger and more erratic moving upstream: the bullwhip effect.",
    "Compare backlog and inventory: the chain is starved first, then flooded.",
    "Try the variants: which change calms the chain most, and why?",
]


def run(p, rep, seed, variant, description, record=True):
    env = simpy.Environment()
    rng = np.random.default_rng([seed, rep, 11])
    E, W = 4, p.weeks
    Lo, Ls, Lp = p.order_delay, p.ship_delay, p.production_delay
    lead = [Lo + Ls] * 3 + [Lp]

    tb = TraceBuilder("beer", variant, "Beer Game", description, "week", W + 1, (3.6, 1.6), 2) if record else None
    ids = [e[0] for e in ECH]
    if tb:
        xs = [0.95, 0.3, -0.35, -1.0]
        tb.node("customers", "Customers", "customers", 1.6, 0.0)
        for (id_, name, kind), x in zip(ECH, xs):
            tb.node(id_, name, kind, x, 0.0, inv=f"{id_}.inv", back=f"{id_}.back")
        tb.node("raw", "Raw materials", "supplier", -1.6, 0.0)
        for a, b in [("raw", "factory"), ("factory", "distributor"), ("distributor", "wholesaler"),
                     ("wholesaler", "retailer"), ("retailer", "customers")]:
            tb.link(a, b)

    inv = [float(p.initial_inventory)] * E
    back = [0.0] * E
    fc = [float(p.initial_flow)] * E
    sl = [p.initial_flow * lead[i] for i in range(E)]
    recv = [0.0] * E
    orders_in = [0.0] * E
    H = {k: [[] for _ in range(E)] for k in ("inv", "back", "net", "order", "demand")}
    for i in range(E):
        for k, v in (("inv", inv[i]), ("back", 0), ("net", inv[i]), ("order", p.initial_flow), ("demand", p.initial_flow)):
            H[k][i].append(v)
    cust_hist = [p.demand_before]
    cost = [0.0] * E

    def land_ship(i, q, t):
        yield env.timeout(t - env.now)
        recv[i] += q

    def land_order(i, q, t):
        yield env.timeout(t - env.now)
        orders_in[i] += q

    def upstream(i):
        return ids[i + 1] if i < 3 else "raw"

    # pipeline in steady state at the start
    for i in range(E):
        d = Ls if i < 3 else Lp
        for j in range(d):
            t = j + 0.5
            env.process(land_ship(i, p.initial_flow, t))
            if tb:
                tb.ship(upstream(i), ids[i], p.initial_flow, t - d, t)
        if i > 0:
            for j in range(Lo):
                t = j + 0.5
                env.process(land_order(i, p.initial_flow, t))
                if tb:
                    tb.order(ids[i - 1], ids[i], p.initial_flow, t - Lo, t)

    def week_loop():
        yield env.timeout(1)
        for k in range(1, W + 1):
            base = p.demand_before if k < p.step_week else p.demand_after
            cust = max(0, int(round(base + rng.normal(0, p.demand_noise_sd)))) if p.demand_noise_sd > 0 else int(base)
            cust_hist.append(cust)
            for i in range(E):
                dem = cust if i == 0 else orders_in[i]
                orders_in[i] = 0.0
                r = recv[i]
                recv[i] = 0.0
                inv[i] += r
                sl[i] -= r
                ship = min(inv[i], back[i] + dem)
                inv[i] -= ship
                back[i] += dem - ship
                if tb:
                    if i == 0:
                        tb.ship("retailer", "customers", ship, k, k + 0.5)
                    else:
                        tb.ship(ids[i], ids[i - 1], ship, k, k + Ls - 0.5)
                if i > 0:
                    env.process(land_ship(i - 1, ship, k + Ls - 0.5))
                signal = cust if p.info_sharing else dem
                fc[i] = p.theta * signal + (1 - p.theta) * fc[i]
                o = (fc[i] + p.alpha_s * (p.desired_inventory - (inv[i] - back[i]))
                     + p.alpha_s * p.beta * (fc[i] * lead[i] - sl[i]))
                o = max(0.0, float(round(o)))
                sl[i] += o
                if i < 3:
                    env.process(land_order(i + 1, o, k + Lo - 0.5))
                    if tb:
                        tb.order(ids[i], ids[i + 1], o, k, k + Lo - 0.5)
                else:
                    env.process(land_ship(3, o, k + Lp - 0.5))
                    if tb:
                        tb.ship("raw", "factory", o, k, k + Lp - 0.5)
                cost[i] += p.holding_cost * inv[i] + p.backlog_cost * back[i]
                for key, v in (("inv", inv[i]), ("back", back[i]), ("net", inv[i] - back[i]), ("order", o), ("demand", dem)):
                    H[key][i].append(v)
            yield env.timeout(1)

    env.process(week_loop())
    env.run(until=W + 1)

    cd = np.asarray(cust_hist[1:], dtype=float)
    vd = cd.var(ddof=1) if len(cd) > 1 else 0.0
    k = {"total_cost": sum(cost)}
    for i, (id_, name, _) in enumerate(ECH):
        orders = np.asarray(H["order"][i][1:], dtype=float)
        k[f"{id_}_cost"] = cost[i]
        k[f"{id_}_order_var_ratio"] = orders.var(ddof=1) / vd if vd > 0 else float("nan")
        k[f"{id_}_max_backlog"] = max(H["back"][i])
        k[f"{id_}_peak_order"] = float(orders.max())

    if tb:
        for i, id_ in enumerate(ids):
            for key in ("inv", "back", "net", "order", "demand"):
                tb.series(f"{id_}.{key}", H[key][i])
        tb.series("customers.demand", cust_hist)
        tb.chart("Orders placed each week", ["customers.demand"] + [f"{i}.order" for i in ids],
                 ["Customer demand", "Retailer", "Wholesaler", "Distributor", "Factory"], "cases")
        tb.chart("Net inventory (stock minus backlog)", [f"{i}.net" for i in ids],
                 ["Retailer", "Wholesaler", "Distributor", "Factory"], "cases")
        tb.notes(NOTES)
    return k, (tb.result() if tb else None), {"week": list(range(W + 1)), "customer_demand": cust_hist,
                                             **{f"{ids[i]}_{key}": H[key][i] for i in range(E) for key in H}}


KPI_LABELS = [
    ("total_cost", "Total chain cost", "", 0),
    ("retailer_order_var_ratio", "Order variance ratio: retailer", "x", 1),
    ("wholesaler_order_var_ratio", "Order variance ratio: wholesaler", "x", 1),
    ("distributor_order_var_ratio", "Order variance ratio: distributor", "x", 1),
    ("factory_order_var_ratio", "Order variance ratio: factory", "x", 1),
    ("factory_peak_order", "Peak factory order", "cases", 0),
    ("retailer_max_backlog", "Peak retailer backlog", "cases", 0),
]
