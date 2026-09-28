"""Risk pooling: n regional warehouses against one central warehouse, run side
by side on identical daily demand (common random numbers).

Both systems use (s,Q) with reorder points set by the textbook formula for the
same cycle service target:
    decentralised, each region:  s = mu*L + z*sigma*sqrt(L)
    centralised:                 s = n*mu*L + z*sigma_c*sqrt(L)
where sigma_c = sigma*sqrt(n + n(n-1)rho) is the standard deviation of total
demand. With independent demand sigma_c = sigma*sqrt(n), so central safety
stock is sqrt(n) times smaller in total. Positive correlation erodes the gain.
"""
from __future__ import annotations

import math

import numpy as np
import simpy

from ..common import StockPoint, TraceBuilder, lead_sampler

NOTES = [
    "Both systems see exactly the same customer demand every day.",
    "Both systems target the same service level. Compare safety stock and average stock: only the safety stock shrinks, because the central warehouse orders the same total quantity.",
    "Pooling works because regional highs and lows cancel out in the total. Square-root law: n sites need about sqrt(n) times the safety stock of one.",
    "Try the correlated-demand variant: when regions rise and fall together, what happens to the saving?",
]


def run(p, rep, seed, variant, description, record=True):
    env = simpy.Environment()
    rng = np.random.default_rng([seed, rep, 31])
    n, mu, sd, rho, L, z = p.regions, p.demand_mean, p.demand_sd, p.correlation, p.lead_time, p.service_z
    cov = sd * sd * ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))
    s_dec = mu * L + z * sd * math.sqrt(L)
    sd_c = sd * math.sqrt(n + n * (n - 1) * rho)
    s_cen = n * mu * L + z * sd_c * math.sqrt(L)
    Qd, Qc = p.order_qty, p.order_qty * n

    tb = TraceBuilder("pooling", variant, "Risk pooling", description, "day", p.days + 1, (3.6, 2.2), 10) if record else None
    xs = np.linspace(-0.6, 0.6, n) if n > 1 else np.array([0.0])
    if tb:
        tb.label("Decentralised", -0.95, 0.95)
        tb.label("Centralised", 0.95, 0.95)
        tb.node("dsup", "Supplier", "supplier", -0.95, -0.72)
        tb.node("csup", "Supplier", "supplier", 0.95, -0.72)
        tb.node("hub", "Central warehouse", "warehouse", 0.95, -0.12, inv="hub.inv", back="hub.back")
        tb.link("csup", "hub")
        for i in range(n):
            tb.node(f"dw{i}", f"Warehouse {i + 1}", "warehouse", -0.95 + xs[i], -0.12, inv=f"dw{i}.inv", back=f"dw{i}.back")
            tb.node(f"dr{i}", f"Region {i + 1}", "customers", -0.95 + xs[i], 0.62)
            tb.node(f"cr{i}", f"Region {i + 1}", "customers", 0.95 + xs[i], 0.62)
            tb.link("dsup", f"dw{i}")
            tb.link(f"dw{i}", f"dr{i}")
            tb.link("hub", f"cr{i}")

    fixed = lead_sampler(None, L, 0)
    dec = [StockPoint(env, f"dw{i}", "dsup", "sQ", fixed, tb, p.warmup_days, s=s_dec, Q=Qd,
                      start_stock=s_dec + Qd / 2, record=record) for i in range(n)]
    cen = StockPoint(env, "hub", "csup", "sQ", fixed, tb, p.warmup_days, s=s_cen, Q=Qc,
                     start_stock=s_cen + Qc / 2, record=record)
    dem_hist = [[0] * n]

    def days():
        for d in range(1, p.days + 1):
            q = np.maximum(0, np.rint(rng.multivariate_normal(np.full(n, mu), cov))).astype(int)
            dem_hist.append(q.tolist())
            region = np.repeat(np.arange(n), q)
            times = rng.uniform(d - 1, d, len(region))
            for k in np.argsort(times):
                yield env.timeout(times[k] - env.now)
                i = int(region[k])
                dec[i].demand(1, f"dr{i}", 0.35)
                cen.demand(1, f"cr{i}", p.central_transit_days)
            yield env.timeout(d - env.now)
            for w in dec + [cen]:
                w.end_of_day(d)

    env.process(days())
    env.run(until=p.days + 1)

    ds = [w.summary() for w in dec]
    cs = cen.summary()
    tot_dem = sum(w.st["demand"] for w in dec)
    dec_on = sum(x["avg_on_hand"] for x in ds)
    nd = p.days - p.warmup_days
    k = {
        "dec_avg_stock": dec_on, "cen_avg_stock": cs["avg_on_hand"],
        "stock_saving_pct": 100 * (1 - cs["avg_on_hand"] / dec_on) if dec_on else float("nan"),
        "dec_fill_rate_pct": 100 * sum(w.st["served_now"] for w in dec) / tot_dem if tot_dem else float("nan"),
        "dec_cycle_service_pct": 100 * (1 - sum(w.st["stockout_cycles"] for w in dec) / max(1, sum(w.st["cycles"] for w in dec))),
        "cen_cycle_service_pct": cs["cycle_service_pct"],
        "cen_fill_rate_pct": cs["fill_rate_pct"],
        "dec_safety_stock": n * z * sd * math.sqrt(L), "cen_safety_stock": z * sd_c * math.sqrt(L),
        "dec_holding_cost_per_day": p.holding_cost * dec_on, "cen_holding_cost_per_day": p.holding_cost * cs["avg_on_hand"],
        "dec_orders_per_year": sum(x["orders"] for x in ds) * 365 / nd, "cen_orders_per_year": cs["orders"] * 365 / nd,
    }
    if tb:
        tot_on = np.sum([w.hist_on for w in dec], axis=0)
        tot_back = np.sum([w.hist_back for w in dec], axis=0)
        for i, w in enumerate(dec):
            tb.series(f"dw{i}.inv", w.hist_on)
            tb.series(f"dw{i}.back", w.hist_back)
        tb.series("hub.inv", cen.hist_on)
        tb.series("hub.back", cen.hist_back)
        tb.series("dec.inv", tot_on)
        tb.series("dec.back", tot_back)
        tb.chart("Total stock on hand", ["dec.inv", "hub.inv"], ["Decentralised (all warehouses)", "Centralised"], "units", window=60)
        tb.chart("Total backorders", ["dec.back", "hub.back"], ["Decentralised", "Centralised"], "units", window=60)
        tb.notes(NOTES)
    period = {"day": list(range(p.days + 1)),
              **{f"region{i + 1}_demand": [row[i] for row in dem_hist] for i in range(n)},
              **{f"wh{i + 1}_on_hand": w.hist_on for i, w in enumerate(dec)},
              "central_on_hand": cen.hist_on, "central_backorders": cen.hist_back}
    return k, (tb.result() if tb else None), period


KPI_LABELS = [
    ("dec_avg_stock", "Average stock: decentralised", "units", 0),
    ("cen_avg_stock", "Average stock: centralised", "units", 0),
    ("stock_saving_pct", "Stock saving from pooling", "%", 1),
    ("dec_cycle_service_pct", "Cycles without a stockout: decentralised", "%", 1),
    ("cen_cycle_service_pct", "Cycles without a stockout: centralised", "%", 1),
    ("dec_fill_rate_pct", "Fill rate: decentralised", "%", 1),
    ("cen_fill_rate_pct", "Fill rate: centralised", "%", 1),
    ("dec_safety_stock", "Safety stock: decentralised (total)", "units", 0),
    ("cen_safety_stock", "Safety stock: centralised", "units", 0),
]
