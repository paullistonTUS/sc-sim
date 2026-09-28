"""Pieces shared by the models: a trace builder for the viewer and a stock point
with (s,Q) or (R,S) replenishment used by the inventory and risk pooling models."""
from __future__ import annotations

import math

import numpy as np


class TraceBuilder:
    """Collects everything the viewer needs. Series are sampled once per period;
    index 0 is the starting state and index k the state at the end of period k."""

    def __init__(self, model, variant, title, description, time_unit, horizon, table, units_per_crate):
        self.d = {
            "format": "sc-trace-v1", "model": model, "variant": variant, "title": title,
            "description": description, "time_unit": time_unit, "horizon": horizon,
            "table": {"w": table[0], "d": table[1]}, "units_per_crate": units_per_crate,
            "nodes": [], "links": [], "labels": [], "series": {}, "shipments": [], "orders": [],
            "charts": [], "notes": [], "kpis": [],
        }

    def node(self, id, name, kind, x, z, inv=None, back=None):
        self.d["nodes"].append({"id": id, "name": name, "kind": kind, "x": x, "z": z, "inv": inv, "back": back})

    def link(self, a, b):
        self.d["links"].append([a, b])

    def label(self, text, x, z):
        self.d["labels"].append({"text": text, "x": x, "z": z})

    def series(self, key, values):
        self.d["series"][key] = [round(float(v), 2) for v in values]

    def ship(self, a, b, qty, t0, t1):
        if qty > 0:
            self.d["shipments"].append([a, b, round(float(qty), 2), round(t0, 3), round(t1, 3)])

    def order(self, a, b, qty, t0, t1):
        self.d["orders"].append([a, b, round(float(qty), 2), round(t0, 3), round(t1, 3)])

    def chart(self, title, keys, names, unit="", window=None):
        """window: number of periods shown around the playhead (None = whole run)."""
        self.d["charts"].append({"title": title, "keys": keys, "names": names, "unit": unit, "window": window})

    def notes(self, lines):
        self.d["notes"] = list(lines)

    def result(self):
        self.d["shipments"].sort(key=lambda s: s[3])
        self.d["orders"].sort(key=lambda s: s[3])
        return self.d


class StockPoint:
    """A stocking location with backorders.

    Day d runs from time d-1 to d. Customers arrive one unit at a time at
    random moments within the day. Under (s,Q) the inventory position is
    checked after every sale, so orders are triggered exactly at s. Under
    (R,S) the review happens at the start of every R-th day. An order placed
    at time t with lead time L arrives at t + L; backorders are filled first.
    Statistics are collected after the warm-up only."""

    def __init__(self, env, node, supplier, policy, lead_sampler, tb, warmup, s=0, Q=0, R=1, S=0,
                 start_stock=0.0, record=True):
        self.env, self.node, self.supplier = env, node, supplier
        self.policy, self.lead, self.tb, self.warmup = policy, lead_sampler, tb, warmup
        self.s, self.Q, self.R, self.S = s, Q, R, S
        self.on_hand, self.on_order = float(start_stock), 0.0
        self.backlog = []  # [destination, qty, transit]
        self.record = record
        self.day_ship = {}
        self.hist_on, self.hist_back, self.hist_pos = [self.on_hand], [0.0], [self.on_hand]
        self.st = dict(demand=0.0, served_now=0.0, on_sum=0.0, back_sum=0.0, days=0, stockout_days=0, orders=0,
                       cycles=0, stockout_cycles=0)
        self._short_this_cycle = False

    @property
    def back_qty(self):
        return sum(b[1] for b in self.backlog)

    @property
    def position(self):
        return self.on_hand - self.back_qty + self.on_order

    def demand(self, qty, dest, transit):
        t = self.env.now
        now = min(self.on_hand, qty)
        self.on_hand -= now
        if now > 0 and self.tb:
            q, tr = self.day_ship.get(dest, (0.0, transit))
            self.day_ship[dest] = (q + now, tr)
        if qty - now > 1e-9:
            self.backlog.append([dest, qty - now, transit])
            self._short_this_cycle = True
        if t > self.warmup:
            self.st["demand"] += qty
            self.st["served_now"] += now
        if self.policy == "sQ" and self.position <= self.s:
            n = math.floor((self.s - self.position) / self.Q) + 1
            self._place(n * self.Q)

    def periodic_review(self, day):
        """Called at time `day`, the start of day `day + 1`."""
        if self.policy == "RS" and day % self.R == 0:
            q = self.S - self.position
            if q > 0:
                self._place(q)

    def _receive(self, qty, t):
        yield self.env.timeout(t - self.env.now)
        # a replenishment cycle ends at each receipt; count whether it saw any backorders
        if self.env.now > self.warmup:
            self.st["cycles"] += 1
            self.st["stockout_cycles"] += 1 if self._short_this_cycle else 0
        self._short_this_cycle = False
        self.on_hand += qty
        self.on_order -= qty
        while self.backlog and self.on_hand > 1e-9:
            dest, q, tr = self.backlog[0]
            x = min(q, self.on_hand)
            self.on_hand -= x
            if self.tb:
                self.tb.ship(self.node, dest, x, self.env.now, self.env.now + max(0.35, tr))
            if x >= q - 1e-9:
                self.backlog.pop(0)
            else:
                self.backlog[0][1] -= x

    def _place(self, qty):
        t, lt = self.env.now, self.lead()
        self.on_order += qty
        self.env.process(self._receive(qty, t + lt))
        if self.tb:
            self.tb.order(self.node, self.supplier, qty, t, t + 0.3)
            self.tb.ship(self.supplier, self.node, qty, t + 0.3, t + lt)
        if t > self.warmup:
            self.st["orders"] += 1

    def end_of_day(self, day):
        b = self.back_qty
        if self.tb:
            for dest, (q, tr) in self.day_ship.items():
                self.tb.ship(self.node, dest, q, day - 0.5, day - 0.5 + max(0.35, tr))
        self.day_ship = {}
        if day > self.warmup:
            self.st["on_sum"] += self.on_hand
            self.st["back_sum"] += b
            self.st["days"] += 1
            self.st["stockout_days"] += 1 if b > 1e-9 else 0
        if self.record:
            self.hist_on.append(self.on_hand)
            self.hist_back.append(b)
            self.hist_pos.append(self.position)

    def summary(self):
        d = max(1, self.st["days"])
        return {
            "fill_rate_pct": 100 * self.st["served_now"] / self.st["demand"] if self.st["demand"] else float("nan"),
            "avg_on_hand": self.st["on_sum"] / d,
            "avg_backorders": self.st["back_sum"] / d,
            "stockout_days_pct": 100 * self.st["stockout_days"] / d,
            "orders": self.st["orders"],
            "cycle_service_pct": 100 * (1 - self.st["stockout_cycles"] / self.st["cycles"]) if self.st["cycles"] else float("nan"),
        }


def rounded_normal(rng, mean, sd):
    return max(0, int(round(rng.normal(mean, sd)))) if sd > 0 else int(round(mean))


def lead_sampler(rng, mean, sd):
    if sd <= 0:
        return lambda: int(mean)
    return lambda: max(1, int(round(rng.normal(mean, sd))))
