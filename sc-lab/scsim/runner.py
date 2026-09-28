"""Run variants and replications; write KPIs, period data, viewer traces and a manifest."""
from __future__ import annotations

import datetime as dt
import json
import math
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from . import __version__
from .config import MODELS, variant_params
from .models import REGISTRY

_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]


def ci(x):
    x = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    if len(x) == 1:
        return float(x[0]), 0.0, 1
    t = _T95[len(x) - 2] if len(x) - 1 <= 30 else 1.96
    return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(len(x))), len(x)


def _clean(v):
    return None if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v


def run_all(cfg, variants=None, out_root="outputs", viewer_dir="viewer/traces", input_path=None, log=print):
    names = variants or list(cfg.variants)
    bad = [v for v in names if v not in cfg.variants]
    if bad:
        raise SystemExit(f"Unknown variant(s): {', '.join(bad)}. Available: {', '.join(cfg.variants)}")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    vdir = Path(viewer_dir) if viewer_dir else None
    if vdir:
        vdir.mkdir(parents=True, exist_ok=True)

    rows, summary, index = [], [], []
    for name in names:
        v = cfg.variants[name]
        mod = REGISTRY[v.model]
        p = variant_params(cfg, name)
        trace, reps = None, []
        for r in range(1, p.replications + 1):
            rec = (r == cfg.trace_replication)
            k, tr, period = mod.run(p, r, cfg.base_seed, name, v.description, record=rec)
            reps.append(k)
            rows.append({"variant": name, "model": v.model, "replication": r, **k})
            if rec:
                trace = tr
                pd.DataFrame(period).to_csv(out / f"period_data_{name}.csv", index=False)
        kp = []
        for key, label, unit, dec in mod.KPI_LABELS:
            m, hw, n = ci([x[key] for x in reps])
            kp.append({"key": key, "label": label, "unit": unit, "decimals": dec, "mean": _clean(m),
                       "half_width": _clean(hw), "n": n})
        for key in reps[0]:
            m, hw, n = ci([x[key] for x in reps])
            summary.append({"variant": name, "model": v.model, "metric": key, "n": n, "mean": m,
                            "ci95_low": m - hw, "ci95_high": m + hw})
        if vdir and trace:
            trace["kpis"] = kp
            trace["replications"] = p.replications
            trace["replay_replication"] = cfg.trace_replication
            trace["input_file"] = Path(cfg.source_file).name
            trace["input_sha256"] = cfg.sha256
            trace["overrides"] = [list(o) for o in v.overrides]
            (vdir / f"{name}.json").write_text(json.dumps(trace, separators=(",", ":")))
            index.append({"file": f"{name}.json", "variant": name, "model": v.model,
                          "model_title": MODELS[v.model][0], "description": v.description})
        log(f"  {name:<24} {p.replications:>3} reps")

    pd.DataFrame(rows).to_csv(out / "kpi_replications.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "kpi_summary.csv", index=False)
    if vdir:
        (vdir / "index.json").write_text(json.dumps({"generated": stamp, "traces": index}, indent=1))
    if input_path:
        shutil.copy2(input_path, out / Path(input_path).name)
    manifest = {
        "run_id": out.name, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "input_file": Path(cfg.source_file).name, "input_sha256": cfg.sha256, "base_seed": cfg.base_seed,
        "variants": {n: {"model": cfg.variants[n].model, "description": cfg.variants[n].description,
                         "overrides": cfg.variants[n].overrides} for n in names},
        "parameter_sources": cfg.provenance,
        "software": {"scsim": __version__, "python": platform.python_version(), "simpy": simpy.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "files": sorted(p.name for p in out.iterdir()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return out
