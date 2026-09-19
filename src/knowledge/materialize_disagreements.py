"""Materialize plan-event disagreements as ABox individuals. Smoke, then full."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdr_situation_completion import complete, load_year, mine_cells  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KNOW = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiments" / "aei_situation"
ONTOLOGY = KNOW / "ontology" / "awhso.ttl"
CASE = OUT / "case_airport_actions.csv"
SPARQL = KNOW / "sparql"
AWHSO = Namespace("https://w3id.org/awhso#")

WL_ACT = {
    "DrainStandingWater": AWHSO.ActionType_DrainStandingWater,
    "GrassHeightManagement": AWHSO.ActionType_GrassHeightManagement,
    "ExclusionFencing": AWHSO.ActionType_ExclusionFencing,
    "WasteManagement": AWHSO.ActionType_WasteManagement,
    "LandUseAgreement": AWHSO.ActionType_LandUseAgreement,
    "Harassment": AWHSO.ActionType_Harassment,
    "LethalControl": AWHSO.ActionType_LethalControl,
}
SDR_ACT = {
    "Repair": AWHSO.ActionType_Repair,
    "Replace": AWHSO.ActionType_Replace,
    "Inspect": AWHSO.ActionType_Inspect,
    "OperationalCheck": AWHSO.ActionType_OperationalCheck,
    "Secure": AWHSO.ActionType_Secure,
}


def add_gap(g: Graph, site_uri: URIRef, act_uri: URIRef, kind: str, key: str) -> None:
    node = AWHSO[f"gap_{key}"]
    g.add((node, RDF.type, AWHSO.PlanEventDisagreement))
    g.add((node, AWHSO.aboutSite, site_uri))
    g.add((node, AWHSO.aboutAction, act_uri))
    g.add((node, AWHSO.disagreementKind, Literal(kind)))
    g.add((node, RDFS.label, Literal(f"{key}:{kind}")))


def wildlife_gaps(g: Graph) -> int:
    n = 0
    with CASE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            b = row["bucket"]
            if b == "graph_and_document":
                continue
            aid = row["airport_id"]
            act = row["action"]
            site = AWHSO[f"airport_{aid}"]
            g.add((site, RDF.type, AWHSO.Airport))
            g.add((site, AWHSO.airportId, Literal(aid)))
            kind = (
                "documented_without_local_events"
                if b == "document_without_local_events"
                else "event_supported_not_in_document"
            )
            add_gap(g, site, WL_ACT[act], kind, f"wl_{aid}_{act}")
            n += 1
    return n


def sdr_gaps(g: Graph, train, test) -> int:
    cells = mine_cells(train, min_c=2)
    obs: dict[str, set[str]] = defaultdict(set)
    inf: dict[str, set[str]] = defaultdict(set)
    for rec in test:
        done = complete(rec, cells)
        obs[rec["op"]] |= rec["gold"]
        inf[rec["op"]] |= done["inferred"]
    n = 0
    for op in set(obs) | set(inf):
        site = AWHSO[f"op_{op}"]
        g.add((site, RDF.type, AWHSO.Operator))
        g.add((site, AWHSO.operatorId, Literal(op)))
        for a in sorted(obs[op] - inf[op]):
            add_gap(g, site, SDR_ACT[a], "documented_without_local_events", f"sdr_{op}_{a}")
            n += 1
        for a in sorted(inf[op] - obs[op]):
            add_gap(g, site, SDR_ACT[a], "event_supported_not_in_document", f"sdr_{op}_{a}")
            n += 1
    return n


def query(g: Graph) -> dict:
    rows = list(g.query((SPARQL / "disagreement_count.rq").read_text(encoding="utf-8")))
    by = {str(r[0]): int(r[1]) for r in rows}
    total = sum(by.values())
    return {"by_kind": by, "n": total, "triples": len(g)}


def smoke() -> bool:
    g = Graph()
    g.parse(ONTOLOGY.as_posix(), format="turtle")
    g.bind("awhso", AWHSO)
    nw = wildlife_gaps(g)
    train = load_year("2024", limit=3000)
    test = load_year("2024", limit=2000)[1::2]
    ns = sdr_gaps(g, train, test)
    rec = query(g)
    print("SMOKE", nw, ns, rec, flush=True)
    ok = rec["n"] >= nw and rec["n"] > 0
    print("SMOKE_OK" if ok else "SMOKE_FAIL", flush=True)
    return ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not smoke():
        raise SystemExit("smoke failed")
    g = Graph()
    g.parse(ONTOLOGY.as_posix(), format="turtle")
    g.bind("awhso", AWHSO)
    nw = wildlife_gaps(g)
    print("wildlife disagreements", nw, flush=True)
    train = load_year("2023")
    test = load_year("2024")
    ns = sdr_gaps(g, train, test)
    print("sdr disagreements", ns, flush=True)
    rec = query(g)
    rec["wildlife_n"] = nw
    rec["sdr_n"] = ns
    print("FULL", rec, flush=True)
    (OUT / "plan_event_disagreements.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    out_ttl = OUT / "plan_event_disagreements.ttl"
    g.serialize(out_ttl.as_posix(), format="turtle")
    print("wrote", out_ttl, flush=True)


if __name__ == "__main__":
    main()
