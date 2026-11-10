"""Average trajectories within each seed, then report across-seed sample SD."""
from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from ..utils import read_json, reserve_directory, sha256_file, write_csv, write_json


def summarize(directories, output, *, allow_diagnostic=False):
    tables, identity, seen, coverage = [], {}, set(), {}
    if not directories:
        raise ValueError("No evaluation directories supplied.")
    for directory in directories:
        directory = Path(directory)
        manifest = read_json(directory / "evaluation_manifest.json")
        if not manifest["complete"] or (manifest["diagnostic"] and not allow_diagnostic):
            raise ValueError("Only complete formal evaluations can enter a paper summary.")
        path = directory / "per_trajectory.csv"
        if sha256_file(path) != manifest["table_sha256"]:
            raise ValueError("Result table changed after evaluation.")
        context = (manifest["scenario"], manifest["split"])
        previous = identity.setdefault(context, (manifest["array_sha256"], manifest["normalization"]))
        if previous != (manifest["array_sha256"], manifest["normalization"]):
            raise ValueError("Dataset/normalization mismatch across methods.")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or {int(r["horizon"]) for r in rows} != set(manifest["horizons"]):
            raise ValueError("Missing horizon(s) or empty evaluation table.")
        groups = defaultdict(list)
        for row in rows:
            expected = {"method":manifest["method"],"seed":str(manifest["seed"]) if manifest["seed"] is not None else "",
                        "scenario":manifest["scenario"],"split":manifest["split"]}
            if any(row[k] != value for k,value in expected.items()):
                raise ValueError("Table identity does not match its evaluation manifest.")
            key = tuple(row[k] for k in ("method", "seed", "scenario", "split", "horizon"))
            groups[key].append(row)
        for key, values in groups.items():
            if key in seen:
                raise ValueError(f"Duplicate method/seed/horizon evaluation: {key}")
            seen.add(key)
            if len(values) != manifest["trajectory_count"] or len({v["trajectory_index"] for v in values}) != len(values):
                raise ValueError("Missing or duplicate trajectories.")
            if {int(v["trajectory_index"]) for v in values} != set(range(manifest["trajectory_count"])):
                raise ValueError("Wrong trajectory coverage.")
            mapping = {v["trajectory_index"]:(v["initial_id"],v["control_id"]) for v in values}
            reference = coverage.setdefault((context,manifest["trajectory_count"]),mapping)
            if mapping != reference:
                raise ValueError("Initial-field/control grouping differs between compared evaluations.")
            if not allow_diagnostic and (len(values) != 180 or len({v["initial_id"] for v in values}) != 18):
                raise ValueError("Formal summaries require 180 trajectories grouped by 18 initial fields.")
            if not allow_diagnostic:
                from collections import Counter
                if set(Counter(v["initial_id"] for v in values).values()) != {10}:
                    raise ValueError("Every initial field must have exactly ten controls.")
            row = dict(zip(("method","seed","scenario","split","horizon"), key))
            row.update(trajectory_count=len(values), initial_field_count=len({v["initial_id"] for v in values}))
            for metric in values[0]:
                if metric not in {"method","seed","scenario","split","horizon","trajectory_index","initial_id","control_id"}:
                    if all(v[metric] != "" for v in values):
                        data = np.asarray([float(v[metric]) for v in values])
                        if not np.isfinite(data).all():
                            raise ValueError(f"Nonfinite metric: {metric}")
                        row[metric] = float(data.mean())
            tables.append(row)
    combined = defaultdict(list)
    for row in tables:
        combined[tuple(row[k] for k in ("method","scenario","split","horizon"))].append(row)
    summary = []
    for key, values in sorted(combined.items()):
        row = dict(zip(("method","scenario","split","horizon"), key))
        row["seed_count"] = len(values)
        for metric in values[0]:
            if metric not in {"method","seed","scenario","split","horizon","trajectory_count","initial_field_count"}:
                data = [v[metric] for v in values]
                row[metric + "_mean"] = float(np.mean(data))
                row[metric + "_seed_std"] = float(np.std(data, ddof=1)) if len(data) > 1 else None
        summary.append(row)
    output = Path(output)
    reserve_directory(output)
    write_csv(output / "per_seed.csv", tables)
    write_csv(output / "across_seeds.csv", summary)
    write_json(output / "summary_manifest.json", dict(inputs=[str(Path(d).resolve()) for d in directories],
        diagnostic=allow_diagnostic, convention="mean over trajectories per seed, then mean and sample SD over seeds; no CI over 180 correlated trajectories"))
    return str(output / "across_seeds.csv")
