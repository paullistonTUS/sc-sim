"""Load and validate the supply chain lab workbook.

Each model has its own parameter sheet (parameter, value, unit, description,
source). Variants override parameters of one model. Every problem found is
reported together, naming the sheet and spreadsheet row.
"""
from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

# name: (kind, min, max)   kind: int | num | flag | choice:<a>/<b>
BEER = {
    "weeks": ("int", 10, 200), "initial_inventory": ("num", 0, None), "initial_flow": ("num", 0, None),
    "desired_inventory": ("num", 0, None), "demand_before": ("num", 0, None), "demand_after": ("num", 0, None),
    "step_week": ("int", 1, None), "demand_noise_sd": ("num", 0, None), "order_delay": ("int", 1, 10),
    "ship_delay": ("int", 1, 10), "production_delay": ("int", 1, 10), "theta": ("num", 0, 1),
    "alpha_s": ("num", 0, 1), "beta": ("num", 0, 1), "info_sharing": ("flag", 0, 1),
    "holding_cost": ("num", 0, None), "backlog_cost": ("num", 0, None), "replications": ("int", 1, 1000),
}
INVENTORY = {
    "days": ("int", 30, 3650), "warmup_days": ("int", 0, None), "demand_mean": ("num", 0, None),
    "demand_sd": ("num", 0, None), "lead_time_mean": ("int", 1, 60), "lead_time_sd": ("num", 0, None),
    "policy": ("choice:sQ/RS", None, None), "reorder_point": ("num", 0, None), "order_qty": ("num", 1, None),
    "review_period": ("int", 1, 60), "order_up_to": ("num", 0, None), "holding_cost": ("num", 0, None),
    "order_cost": ("num", 0, None), "backorder_cost": ("num", 0, None), "replications": ("int", 1, 1000),
}
POOLING = {
    "days": ("int", 30, 3650), "warmup_days": ("int", 0, None), "regions": ("int", 2, 8),
    "demand_mean": ("num", 0, None), "demand_sd": ("num", 0, None), "correlation": ("num", 0, 0.95),
    "lead_time": ("int", 1, 60), "service_z": ("num", 0, 4), "order_qty": ("num", 1, None),
    "central_transit_days": ("num", 0, 10), "holding_cost": ("num", 0, None), "replications": ("int", 1, 1000),
}
MODELS = {"beer": ("Beer game", BEER), "inventory": ("Inventory policy", INVENTORY), "pooling": ("Risk pooling", POOLING)}


class InputError(Exception):
    def __init__(self, problems):
        self.problems = problems
        super().__init__("\n".join(f"  - {p}" for p in problems))


@dataclass
class Variant:
    name: str
    model: str
    description: str
    overrides: list


@dataclass
class Config:
    params: dict        # model -> {param: value}
    provenance: dict    # model -> {param: source}
    variants: dict      # name -> Variant
    base_seed: int
    trace_replication: int
    source_file: str
    sha256: str


def _blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def _coerce(name, raw, spec, where, problems):
    kind, lo, hi = spec
    if _blank(raw):
        problems.append(f"{where}: value for {name} is missing")
        return None
    if kind.startswith("choice:"):
        opts = kind.split(":", 1)[1].split("/")
        v = str(raw).strip()
        if v not in opts:
            problems.append(f"{where}: {name} must be one of {', '.join(opts)} (got '{raw}')")
            return None
        return v
    try:
        x = float(raw)
    except (TypeError, ValueError):
        problems.append(f"{where}: {name} value '{raw}' is not a number")
        return None
    if kind in ("int", "flag") and not x.is_integer():
        problems.append(f"{where}: {name} must be a whole number")
        return None
    if lo is not None and x < lo or hi is not None and x > hi:
        rng = f"between {lo} and {hi}" if hi is not None else f"at least {lo}"
        problems.append(f"{where}: {name} = {raw} should be {rng}")
        return None
    return int(x) if kind in ("int", "flag") else x


def _cross_checks(model, p, label, problems):
    if model in ("inventory", "pooling") and p["warmup_days"] >= p["days"]:
        problems.append(f"{label}: warmup_days must be less than days")
    if model == "beer" and p["step_week"] > p["weeks"]:
        problems.append(f"{label}: step_week is after the last week")


def load_config(path) -> Config:
    path = Path(path)
    if not path.exists():
        raise InputError([f"Input file not found: {path}"])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    problems = []
    need = [MODELS[m][0] for m in MODELS] + ["Variants", "Run settings"]
    missing = [s for s in need if s not in sheets]
    if missing:
        raise InputError([f"Missing sheet(s): {', '.join(missing)}"])

    params, prov = {}, {}
    for model, (sheet, schema) in MODELS.items():
        df = sheets[sheet]
        got, src = {}, {}
        for i, r in df.iterrows():
            if _blank(r.get("parameter")):
                continue
            name = str(r["parameter"]).strip()
            where = f"{sheet} row {i + 2}"
            if name not in schema:
                problems.append(f"{where}: unknown parameter '{name}'")
                continue
            got[name] = _coerce(name, r.get("value"), schema[name], where, problems)
            src[name] = None if _blank(r.get("source")) else str(r["source"])
        for name in schema:
            if name not in got:
                problems.append(f"{sheet}: parameter '{name}' is missing")
        params[model], prov[model] = got, src

    rs = {}
    for i, r in sheets["Run settings"].iterrows():
        if not _blank(r.get("parameter")):
            rs[str(r["parameter"]).strip()] = (r.get("value"), f"Run settings row {i + 2}")
    def setting(name, spec):
        raw, where = rs.get(name, (None, "Run settings"))
        return _coerce(name, raw, spec, where, problems)
    seed = setting("base_seed", ("int", 0, None))
    trep = setting("trace_replication", ("int", 1, None))

    variants = {}
    for model, (sheet, _) in MODELS.items():
        variants[f"{model}_base"] = Variant(f"{model}_base", model, "Base inputs", [])
    for i, r in sheets["Variants"].iterrows():
        if _blank(r.get("variant")):
            continue
        where = f"Variants row {i + 2}"
        name, model = str(r["variant"]).strip(), str(r.get("model", "")).strip()
        if model not in MODELS:
            problems.append(f"{where}: model '{r.get('model')}' must be one of {', '.join(MODELS)}")
            continue
        if name.endswith("_base"):
            problems.append(f"{where}: names ending in _base are reserved")
            continue
        par = "" if _blank(r.get("parameter")) else str(r["parameter"]).strip()
        schema = MODELS[model][1]
        desc = "" if _blank(r.get("description")) else str(r["description"])
        v = variants.setdefault(name, Variant(name, model, desc, []))
        if v.model != model:
            problems.append(f"{where}: variant {name} is already used for model {v.model}")
            continue
        if desc and not v.description:
            v.description = desc
        if not par:
            continue  # a row with no parameter just declares the variant
        if par not in schema:
            problems.append(f"{where}: '{par}' is not a parameter of the {MODELS[model][0]} model")
            continue
        val = _coerce(par, r.get("value"), schema[par], where, problems)
        if val is not None:
            v.overrides.append((par, val))

    if problems:
        raise InputError(problems)
    cfg = Config(params, prov, variants, seed, trep, str(path), sha)
    for name in variants:
        p = variant_params(cfg, name)
        _cross_checks(variants[name].model, vars(p), f"Variant {name}", problems)
        if cfg.trace_replication > p.replications:
            problems.append(f"Variant {name}: trace_replication is larger than its replications")
    if problems:
        raise InputError(problems)
    return cfg


def variant_params(cfg: Config, name: str) -> SimpleNamespace:
    v = cfg.variants[name]
    p = copy.deepcopy(cfg.params[v.model])
    for par, val in v.overrides:
        p[par] = val
    return SimpleNamespace(**p)
