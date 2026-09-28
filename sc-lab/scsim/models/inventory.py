"""Inventory policy: one warehouse facing random daily demand from a supplier
with a (possibly variable) lead time, under a continuous-review (s,Q) policy
checked daily, or a periodic-review (R,S) order-up-to policy."""
from __future__ import annotations

import numpy as np
import simpy

from ..common import StockPoint, TraceBuilder, lead_sampler, rounded_normal

NOTES = [
    "The dashed line is the reorder point or order-up-to level. Orders are triggered by the inventory position (stock plus on order minus backorders), not by stock on the shelf.",
    "Stock falls during the lead time. Safety stock is what is left, on average, when the delivery lands.",
    "Two service measures: cycles without a stockout (what the z formula targets) and fill rate (share of units served at once). Why is fill rate so much higher?",
    "With variable lead times, the same reorder point gives worse service. Why?",
]


def run(p, rep, seed, variant, description, record=True):
    env = simpy.Environment()
    rng_d = np.random.default_rng([seed, rep, 21])
    rng_l = np.random.default_rng([seed, rep, 22])
    rng_t = np.random.default_rng([seed, rep, 23])
    tb = TraceBuilder("inventory", variant, "Inventory policy", description, "day", p.days + 1, (2.2, 1.0), 5) if record else None
    if tb:
        tb.node("supplier", "Supplier", "supplier", -0.85, 0.0)
        tb.node("warehouse", "Warehouse", "warehouse", 0.0, 0.0, inv="warehouse.inv", back="warehouse.back")
        tb.node("customers", "Customers", "customers", 0.85, 0.0)
        tb.link("supplier", "warehouse")
        tb.link("warehouse", "customers")

    start = p.reorder_point + p.order_qty if p.policy == "sQ" else p.order_up_to
    wh = StockPoint(env, "warehouse", "supplier", p.policy, lead_sampler(rng_l, p.lead_time_mean, p.lead_time_sd),
                    tb, p.warmup_days, s=p.reorder_point, Q=p.order_qty, R=p.review_period, S=p.order_up_to,
                    start_stock=start, record=record)
    demand_hist = [0]
    cost = {"holding": 0.0, "ordering": 0.0, "backorder": 0.0}

    def days():
        for d in range(1, p.days + 1):
            q = rounded_normal(rng_d, p.demand_mean, p.demand_sd)
            demand_hist.append(q)
            orders_before = wh.st["orders"]
            for t in np.sort(rng_t.uniform(d - 1, d, q)):
                yield env.timeout(t - env.now)
                wh.demand(1, "customers", 0.35)
            yield env.timeout(d - env.now)
            wh.end_of_day(d)
            wh.periodic_review(d)
            if d > p.warmup_days:
                cost["holding"] += p.holding_cost * wh.on_hand
                cost["backorder"] += p.backorder_cost * wh.back_qty
                cost["ordering"] += p.order_cost * (wh.st["orders"] - orders_before)

    env.process(days())
    env.run(until=p.days + 1)

    s = wh.summary()
    n = p.days - p.warmup_days
    k = {**s, "cost_per_day": sum(cost.values()) / n, "holding_cost_per_day": cost["holding"] / n,
         "ordering_cost_per_day": cost["ordering"] / n, "backorder_cost_per_day": cost["backorder"] / n,
         "orders_per_year": s["orders"] * 365 / n}
    if tb:
        tb.series("warehouse.inv", wh.hist_on)
        tb.series("warehouse.back", wh.hist_back)
        tb.series("warehouse.pos", wh.hist_pos)
        tb.series("customers.demand", demand_hist)
        level = p.reorder_point if p.policy == "sQ" else p.order_up_to
        tb.series("policy.level", [level] * len(wh.hist_on))
        tb.chart("Warehouse inventory", ["warehouse.inv", "warehouse.pos", "warehouse.back", "policy.level"],
                 ["On hand", "Inventory position", "Backorders",
                  "Reorder point" if p.policy == "sQ" else "Order-up-to level"], "units", window=60)
        tb.chart("Daily customer demand", ["customers.demand"], ["Demand"], "units", window=60)
        tb.notes(NOTES)
    period = {"day": list(range(p.days + 1)), "demand": demand_hist, "on_hand": wh.hist_on,
              "backorders": wh.hist_back, "position": wh.hist_pos}
    return k, (tb.result() if tb else None), period


KPI_LABELS = [
    ("cycle_service_pct", "Cycles without a stockout", "%", 1),
    ("fill_rate_pct", "Fill rate from stock", "%", 1),
    ("avg_on_hand", "Average stock on hand", "units", 0),
    ("stockout_days_pct", "Days with backorders", "%", 1),
    ("orders_per_year", "Orders per year", "", 0),
    ("cost_per_day", "Total cost per day", "", 1),
]
