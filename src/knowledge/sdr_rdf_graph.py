"""Build SDR 2024 as the same RDF + SPARQL stack as the wildlife graph.

Smoke on 2k 2024 rows with 2023-mined cells, then full year hold-out.
Every inferred action slot carries provenance (integrity of C).
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdr_situation_completion import complete, load_year, mine_cells  # noqa: E402
from smoke_situation_completion import add_slot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KNOW = Path(__file__).resolve().parent
ONTOLOGY = KNOW / "ontology" / "awhso.ttl"
SPARQL = KNOW / "sparql"
OUT = ROOT / "results" / "experiments" / "aei_situation"

AWHSO = Namespace("https://w3id.org/awhso#")
ACTION_IRI = {
    "Repair": AWHSO.ActionType_Repair,
    "Replace": AWHSO.ActionType_Replace,
    "Inspect": AWHSO.ActionType_Inspect,
    "OperationalCheck": AWHSO.ActionType_OperationalCheck,
    "Secure": AWHSO.ActionType_Secure,
}


def build(rows: list[dict], cells: dict) -> Graph:
    g = Graph()
    g.parse(ONTOLOGY.as_posix(), format="turtle")
    g.bind("awhso", AWHSO)
    ops: dict[str, URIRef] = {}
    rec_act: dict[str, set[str]] = defaultdict(set)
    doc_act: dict[str, set[str]] = defaultdict(set)
    n_inf = 0
    for rec in rows:
        done = complete(rec, cells)
        op_id = rec["op"]
        if op_id not in ops:
            node = AWHSO[f"op_{op_id}"]
            ops[op_id] = node
            g.add((node, RDF.type, AWHSO.Operator))
            g.add((node, AWHSO.operatorId, Literal(op_id)))
            g.add((node, RDFS.label, Literal(op_id)))
        op = ops[op_id]
        e = AWHSO[f"def_{rec['id']}"]
        g.add((e, RDF.type, AWHSO.DefectEvent))
        g.add((e, AWHSO.eventId, Literal(rec["id"])))
        g.add((e, AWHSO.occursAtOperator, op))
        g.add((e, AWHSO.jascCode, Literal(rec["jasc"])))
        g.add((e, AWHSO.ataChapter, Literal(rec["ata"])))
        g.add((e, AWHSO.natureCode, Literal(rec["nature"])))
        sit = AWHSO[f"sitsdr_{rec['id']}"]
        g.add((sit, RDF.type, AWHSO.Situation))
        g.add((sit, AWHSO.hasEvent, e))
        g.add((sit, AWHSO.hasArtifact, op))
        add_slot(g, sit, "event", e, "observed", "sdr", e, f"e_{rec['id']}")
        add_slot(g, sit, "artifact", op, "observed", "sdr", e, f"a_{rec['id']}")
        if rec["amm"]:
            g.add((e, AWHSO.licensedBy, AWHSO.Norm_AMM))
            add_slot(g, sit, "norm", AWHSO.Norm_AMM, "observed", "amm", AWHSO.Norm_AMM, f"n_{rec['id']}")
        else:
            add_slot(g, sit, "norm", AWHSO.EmptyFiller, "unknown", "completion", e, f"n_{rec['id']}")
        for a in rec["gold"]:
            act = ACTION_IRI[a]
            g.add((op, AWHSO.documentsAction, act))
            doc_act[op_id].add(a)
            add_slot(g, sit, "action", act, "observed", "corrective_action", AWHSO.Norm_AMM, f"d_{rec['id']}_{a}")
        for a in done["inferred"]:
            act = ACTION_IRI[a]
            g.add((e, AWHSO.supportsAction, act))
            g.add((act, AWHSO.licensedBy, AWHSO.Norm_AMM))
            rec_act[op_id].add(a)
            n_inf += 1
            add_slot(
                g, sit, "action", act, "inferred",
                f"cell_{rec['jasc']}_{rec['nature']}", AWHSO.Norm_AMM,
                f"i_{rec['id']}_{a}",
            )
        if not rec["gold"] and not done["inferred"]:
            add_slot(g, sit, "action", AWHSO.EmptyFiller, "unknown", "completion", e, f"u_{rec['id']}")
    for op_id, acts in rec_act.items():
        op = ops[op_id]
        for a in acts:
            g.add((op, AWHSO.recommendsAction, ACTION_IRI[a]))
    g._n_inf = n_inf  # type: ignore[attr-defined]
    return g


def run_cq(g: Graph) -> dict:
    out = {}
    for name in ("sdr_cq1", "sdr_cq2", "sdr_cq3", "sdr_cq4", "sdr_cq5", "prov_inferred", "slot_partition"):
        q = (SPARQL / f"{name}.rq").read_text(encoding="utf-8")
        t0 = time.perf_counter()
        rows = list(g.query(q))
        dt = round(time.perf_counter() - t0, 3)
        if name.startswith("sdr_cq") and name in {"sdr_cq1", "sdr_cq2"}:
            n = int(rows[0][0]) if rows else 0
            out[name] = {"n": n, "s": dt}
        elif name == "prov_inferred":
            inf = int(rows[0][0]) if rows else 0
            prov = int(rows[0][1]) if rows else 0
            out[name] = {"inferred_slots": inf, "with_provenance": prov, "s": dt}
        elif name == "slot_partition":
            out[name] = {"rows": [[str(c) for c in r] for r in rows], "s": dt}
        else:
            out[name] = {"n": len(rows), "rows": [[str(c) for c in r] for r in rows[:12]], "s": dt}
        print(name, out[name], flush=True)
    return out


def smoke() -> bool:
    print("SDR RDF SMOKE", flush=True)
    train = load_year("2023", limit=8000)
    test = load_year("2024", limit=2000)
    cells = mine_cells(train, min_c=2)
    g = build(test, cells)
    print("triples", len(g), flush=True)
    res = run_cq(g)
    part = {r[0]: int(float(r[1])) for r in res.get("slot_partition", {}).get("rows", [])}
    ok = (
        res["sdr_cq2"]["n"] > 0
        and res["prov_inferred"]["inferred_slots"] == res["prov_inferred"]["with_provenance"]
        and part.get("unknown", 0) > 0
    )
    print("SMOKE_OK" if ok else "SMOKE_FAIL", flush=True)
    return ok


def extract_whmp_names() -> dict:
    WHMP = ROOT / "data" / "raw" / "whmp"
    files = {
        "SEA": WHMP / "SEA_WHMP_2022.txt",
        "GIF": WHMP / "GIF_WHSV_2018.txt",
        "BJJ": WHMP / "BJJ_WHA_2014.txt",
        "0B5": WHMP / "Montague_WHMP_2022.txt",
    }
    pat = re.compile(
        r"\b(?:Lake|Pond|Creek|River|Reservoir)\s+[A-Z][A-Za-z]+"
        r"|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\s+(?:Lake|Pond|Creek|River|Golf Course|Farm|Landfill)\b"
    )
    by = {}
    for ap, path in files.items():
        blob = path.read_text(encoding="utf-8", errors="ignore")
        names = sorted({re.sub(r"\s+", " ", m.group(0)).strip() for m in pat.finditer(blob) if len(m.group(0).split()) >= 2})
        by[ap] = names
        print("WHMP names", ap, len(names), names[:8], flush=True)
    return by


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names: dict = {}
    n_whmp = 0
    if not smoke():
        raise SystemExit("smoke failed")
    print("SDR RDF FULL", flush=True)
    train = load_year("2023")
    test = load_year("2024")
    cells = mine_cells(train, min_c=2)
    print("cells", len(cells), "test", len(test), flush=True)
    g = build(test, cells)
    print("triples", len(g), flush=True)
    res = run_cq(g)
    payload = {
        "triples": len(g),
        "n_test": len(test),
        "n_cells": len(cells),
        "cq": res,
        "whmp_proper_names": {k: v for k, v in names.items()},
        "n_whmp_proper": n_whmp,
    }
    (OUT / "sdr_rdf.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
