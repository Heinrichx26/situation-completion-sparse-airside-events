"""Five-layer situation completion on FAA SDR 2024. Smoke first, then full.

Layers: artifact=JASC/ATA component, event=discrepancy, norm=AMM clause,
action=corrective type. Same slot statuses as AWHSO. Analog CQs CQ1--CQ5.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strengthen5_ocrag import SDR_DIR, SDR_LEX, parse_sdr_ca, sdr_gold_from_ca  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "experiments" / "aei_situation"
AMM = re.compile(r"\b(?:IAW|IN ACCORDANCE WITH)\s+([A-Z0-9][A-Z0-9 ./_-]{2,40})", re.I)


def load_year(year: str, limit: int | None = None) -> list[dict]:
    path = SDR_DIR / f"SDR-{year}.csv"
    rows = []
    with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            disc = row.get("Discrepancy") or ""
            head, ca = parse_sdr_ca(disc)
            jasc = (row.get("JASCCode") or "").strip()
            if not jasc:
                continue
            ata = jasc[:2]
            nature = (row.get("NatureOfConditionA") or "").strip() or "UNK"
            op = (row.get("OperatorDesignator") or "").strip() or "UNK"
            gold = sdr_gold_from_ca(ca) if ca else set()
            amm = ""
            m = AMM.search(ca or disc)
            if m:
                amm = m.group(1).strip()[:48]
            rows.append({
                "id": f"{year}-{i}",
                "year": year,
                "jasc": jasc,
                "ata": ata,
                "nature": nature,
                "op": op,
                "has_ca": bool(ca),
                "gold": gold,
                "amm": amm,
            })
            if limit and len(rows) >= limit:
                break
    return rows


def mine_cells(train: list[dict], min_c: int = 2) -> dict[tuple[str, str], set[str]]:
    counts = defaultdict(lambda: defaultdict(int))
    for rec in train:
        if not rec["gold"]:
            continue
        for a in rec["gold"]:
            counts[(rec["jasc"], rec["nature"])][a] += 1
    cells = {}
    for key, c in counts.items():
        top = max(c.values()) if c else 0
        pred = {a for a, n in c.items() if n == top and n >= min_c}
        if pred:
            cells[key] = pred
    return cells


def complete(rec: dict, cells: dict) -> dict:
    key = (rec["jasc"], rec["nature"])
    inferred = set(cells.get(key, ()))
    observed = set(rec["gold"])
    action_status = "observed" if observed else ("inferred" if inferred else "unknown")
    return {
        **rec,
        "inferred": inferred,
        "union": observed | inferred,
        "action_status": action_status,
        "norm_status": "observed" if rec["amm"] else "unknown",
    }


def cq_counts(completed: list[dict], cells: dict) -> dict:
    """Analog CQs. Nonempty = at least one typed row."""
    cq1 = {(r["ata"], r["nature"]) for r in completed}
    cq2 = {(k, a) for k, acts in cells.items() for a in acts}
    # CQ3: per operator, action in union with local observed vs fleet inferred
    op_obs = defaultdict(set)
    op_inf = defaultdict(set)
    fleet = set()
    for r in completed:
        op_obs[r["op"]] |= r["gold"]
        op_inf[r["op"]] |= r["inferred"]
        fleet |= r["union"]
    cq3 = []
    for op, obs in op_obs.items():
        inf = op_inf[op]
        for a in sorted(obs | inf | fleet):
            cq3.append((op, a, a in obs, a in inf, a in fleet))
    # CQ4: observed C/A type at operator with no inferred cell support
    cq4 = [(op, a) for op, obs in op_obs.items() for a in obs if a not in op_inf[op]]
    # CQ5: retract ATA '32' (landing gear) — actions whose cells all have ata 32
    ata_actions = defaultdict(set)
    for (jasc, nature), acts in cells.items():
        ata_actions[jasc[:2]] |= acts
    all_act = set().union(*ata_actions.values()) if ata_actions else set()
    lost = {a for a in all_act if a in ata_actions.get("32", set()) and all(
        a not in acts for ata, acts in ata_actions.items() if ata != "32"
    )}
    # if exclusive-to-32 is empty, report actions licensed by ATA 32 (precondition present)
    cq5 = lost or ata_actions.get("32", set())
    return {
        "cq1": len(cq1),
        "cq2": len(cq2),
        "cq3": len(cq3),
        "cq4": len(cq4),
        "cq5": len(cq5),
        "n_ops_cq3": len(op_obs),
        "cq5_actions": sorted(cq5),
    }


def ablation(completed: list[dict], cells: dict) -> dict:
    n = len(completed)
    n_ca = sum(1 for r in completed if r["has_ca"])
    n_inf = sum(1 for r in completed if r["inferred"])
    n_fill = sum(1 for r in completed if r["union"])
    n_amm = sum(1 for r in completed if r["amm"])
    b0 = {"cq1": 0, "cq2": 0, "cq3": 0, "cq4": 0, "cq5": 0}
    b2 = {"cq1": 0, "cq2": 0, "cq3": 0, "cq4": 0, "cq5": 0}
    bdoc = {"cq1": 0, "cq2": 0, "cq3": 0, "cq4": n_ca, "cq5": 0}
    b3 = cq_counts(completed, cells)
    return {
        "n": n,
        "observed_ca": n_ca,
        "observed_ca_rate": round(n_ca / n, 4) if n else 0,
        "inferred_action": n_inf,
        "inferred_rate": round(n_inf / n, 4) if n else 0,
        "filled_union": n_fill,
        "filled_rate": round(n_fill / n, 4) if n else 0,
        "observed_amm": n_amm,
        "amm_rate": round(n_amm / n, 4) if n else 0,
        "n_cells": len(cells),
        "B0": b0,
        "B2": b2,
        "B-doc": bdoc,
        "B3": {k: b3[k] for k in ("cq1", "cq2", "cq3", "cq4", "cq5")},
        "cq5_actions": b3["cq5_actions"],
        "n_ops": b3["n_ops_cq3"],
    }


def operator_buckets(completed: list[dict], op: str) -> list[dict]:
    obs = set()
    inf = set()
    for r in completed:
        if r["op"] != op:
            continue
        obs |= r["gold"]
        inf |= r["inferred"]
    rows = []
    for a in sorted(SDR_LEX):
        in_doc = a in obs
        local = a in inf
        if in_doc and local:
            bucket = "graph_and_document"
        elif in_doc and not local:
            bucket = "document_without_local_events"
        elif local and not in_doc:
            bucket = "event_supported_not_in_document"
        else:
            continue
        rows.append({"operator": op, "action": a, "in_document": in_doc, "local_cell": local, "bucket": bucket})
    return rows


def smoke() -> bool:
    print("SDR SMOKE", flush=True)
    rows = load_year("2024", limit=4000)
    train, test = rows[::2], rows[1::2]
    cells = mine_cells(train, min_c=2)
    done = [complete(r, cells) for r in test]
    summary = ablation(done, cells)
    print("SMOKE", {k: summary[k] for k in ("n", "observed_ca_rate", "filled_rate", "n_cells", "B3")}, flush=True)
    ok = summary["B3"]["cq2"] > 0 and summary["filled_rate"] > 0.5
    print("SMOKE_OK" if ok else "SMOKE_FAIL", flush=True)
    return ok


def named_remark_instances():
    """Proper-name attractants from strike remarks; union with OSM/docs."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from full_situation_completion import (  # noqa: E402
        component_hits,
        iter_nwsd_rows,
        load_osm,
        load_target_airports,
        normalize_airport,
        phase_bucket,
        size_guild,
        text,
    )
    DOC = json.loads((Path(__file__).resolve().parent / "fixtures" / "document_attractants.json").read_text(encoding="utf-8"))
    PROPER = re.compile(
        r"\b(?:Lake|Pond|Creek|River|Reservoir|Marsh|Landfill|Farm|Golf Course)\s+[A-Z][A-Za-z]+"
        r"|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s+(?:Lake|Pond|Creek|River|Landfill|Farm|Golf Course)\b"
    )
    target = load_target_airports()
    osm = load_osm()
    proper = defaultdict(set)
    for row in iter_nwsd_rows():
        aid = normalize_airport(row.get("AIRPORT_ID"))
        if aid not in target:
            continue
        remarks = " ".join([text(row.get("REMARKS")), text(row.get("COMMENTS"))])
        for m in PROPER.finditer(remarks):
            name = re.sub(r"\s+", " ", m.group(0)).strip()
            if len(name.split()) >= 2:
                proper[aid].add(name)
    osm_aps = {a for a, v in osm.items() if v}
    doc_aps = set(DOC["airports"])
    proper_aps = {a for a, v in proper.items() if v}
    named = osm_aps | doc_aps | proper_aps
    rec = {
        "n_target": len(target),
        "osm_airports": len(osm_aps),
        "doc_airports": len(doc_aps),
        "remark_proper_airports": len(proper_aps),
        "named_instance_airports": len(named),
        "named_share": round(100 * len(named) / max(1, len(target)), 2),
        "n_proper_names": sum(len(v) for v in proper.values()),
        "osm_objects": sum(len(v) for v in osm.values()),
        "doc_objects": sum(len(v) for v in DOC["airports"].values()),
    }
    print("NAMED", rec, flush=True)
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not smoke():
        raise SystemExit("smoke failed")
    print("SDR FULL 2023 train / 2024 test", flush=True)
    train = load_year("2023")
    test = load_year("2024")
    print("n_train", len(train), "n_test", len(test), flush=True)
    cells = mine_cells(train, min_c=2)
    done = [complete(r, cells) for r in test]
    summary = ablation(done, cells)
    ops = Counter(r["op"] for r in done)
    case_op = ops.most_common(1)[0][0]
    buckets = operator_buckets(done, case_op)
    summary["case_operator"] = case_op
    summary["case_n"] = ops[case_op]
    summary["case_buckets"] = buckets
    print("CASE", case_op, ops[case_op], buckets, flush=True)
    print("ABLATION", summary["B0"], summary["B2"], summary["B-doc"], summary["B3"], flush=True)
    named = named_remark_instances()
    summary["named"] = named
    (OUT / "sdr_situation.json").write_text(json.dumps(summary, indent=2, default=list), encoding="utf-8")
    with (OUT / "sdr_case_buckets.csv").open("w", encoding="utf-8", newline="") as handle:
        if buckets:
            w = csv.DictWriter(handle, fieldnames=list(buckets[0].keys()))
            w.writeheader()
            w.writerows(buckets)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
