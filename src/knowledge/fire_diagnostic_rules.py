"""Fire diagnostic rules D1, D1', D2, D3 as SPARQL CONSTRUCT. Smoke first."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdr_situation_completion import complete, load_year, mine_cells  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KNOW = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiments" / "aei_situation"
ONTOLOGY = KNOW / "ontology" / "awhso.ttl"
CASE = OUT / "case_airport_actions.csv"
DOC = json.loads((KNOW / "fixtures" / "document_attractants.json").read_text(encoding="utf-8"))
SPARQL = KNOW / "sparql"
AWHSO = Namespace("https://w3id.org/awhso#")

ATTR_CLASS = {
    "StandingWater": AWHSO.StandingWater,
    "Grassland": AWHSO.Grassland,
    "WasteFacility": AWHSO.WasteFacility,
    "Wetland": AWHSO.Wetland,
    "Agriculture": AWHSO.Agriculture,
    "PerchStructure": AWHSO.PerchStructure,
}
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
RULES = ("horn_d1", "horn_d1b", "horn_d2", "horn_d2p", "horn_d3", "horn_d3p", "horn_d3s")
HABITAT = {
    "DrainStandingWater": "StandingWater",
    "GrassHeightManagement": "Grassland",
    "WasteManagement": "WasteFacility",
    "LandUseAgreement": "Agriculture",
}
TRUE = Literal(True, datatype=XSD.boolean)


def event_log_flags() -> dict[str, bool]:
    from full_situation_completion import iter_nwsd_rows, normalize_airport  # noqa: E402
    flags = {"SEA": False, "GIF": False, "BJJ": False, "0B5": False}
    for row in iter_nwsd_rows():
        aid = normalize_airport(row.get("AIRPORT_ID"))
        if aid in flags:
            flags[aid] = True
            if all(flags.values()):
                break
    return flags


def load_facts_wildlife() -> Graph:
    g = Graph()
    g.parse(ONTOLOGY.as_posix(), format="turtle")
    g.bind("awhso", AWHSO)
    logs = event_log_flags()
    print("event_log", logs, flush=True)
    for aid, has in logs.items():
        site = AWHSO[f"airport_{aid}"]
        g.add((site, AWHSO.hasEventLog, Literal(has, datatype=XSD.boolean)))
    with CASE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            aid = row["airport_id"]
            site = AWHSO[f"airport_{aid}"]
            g.add((site, RDF.type, AWHSO.Airport))
            g.add((site, AWHSO.airportId, Literal(aid)))
            act = WL_ACT[row["action"]]
            if row["in_document"].lower() == "true":
                g.add((site, AWHSO.documentsAction, act))
            if row["local_event_support"].lower() == "true":
                ev = AWHSO[f"ev_{aid}_{row['action']}"]
                g.add((ev, RDF.type, AWHSO.StrikeEvent))
                g.add((ev, AWHSO.occursAtAirport, site))
                g.add((ev, AWHSO.supportsAction, act))
    for aid, objs in DOC["airports"].items():
        site = AWHSO[f"airport_{aid}"]
        for obj in objs:
            node = AWHSO[obj["id"]]
            g.add((node, RDF.type, ATTR_CLASS[obj["class"]]))
            g.add((node, RDFS.label, Literal(obj["name"])))
            g.add((site, AWHSO.observedAttractant, node))
    close_predicates(g, list(WL_ACT), HABITAT)
    return g


def load_facts_sdr(train, test) -> Graph:
    g = Graph()
    g.parse(ONTOLOGY.as_posix(), format="turtle")
    g.bind("awhso", AWHSO)
    cells = mine_cells(train, min_c=2)
    obs: dict[str, set[str]] = defaultdict(set)
    inf: dict[str, set[str]] = defaultdict(set)
    for rec in test:
        done = complete(rec, cells)
        obs[rec["op"]] |= rec["gold"]
        inf[rec["op"]] |= done["inferred"]
    for op in set(obs) | set(inf):
        site = AWHSO[f"op_{op}"]
        g.add((site, RDF.type, AWHSO.Operator))
        g.add((site, AWHSO.operatorId, Literal(op)))
        g.add((site, AWHSO.hasEventLog, TRUE))
        for a in obs[op]:
            g.add((site, AWHSO.documentsAction, SDR_ACT[a]))
        for a in inf[op]:
            ev = AWHSO[f"sdev_{op}_{a}"]
            g.add((ev, RDF.type, AWHSO.DefectEvent))
            g.add((ev, AWHSO.occursAtOperator, site))
            g.add((ev, AWHSO.supportsAction, SDR_ACT[a]))
    close_predicates(g, list(SDR_ACT), {})
    return g


def close_predicates(g: Graph, action_names: list[str], habitat: dict[str, str]) -> None:
    """Write positive fillers for unsupported, undocumented, logs, and attractant checks."""
    sites = set(g.subjects(RDF.type, AWHSO.Airport)) | set(g.subjects(RDF.type, AWHSO.Operator))
    for site in sites:
        documented = set(g.objects(site, AWHSO.documentsAction))
        supported = set()
        for e in set(g.subjects(AWHSO.occursAtAirport, site)) | set(g.subjects(AWHSO.occursAtOperator, site)):
            supported |= set(g.objects(e, AWHSO.supportsAction))
        g.add((site, AWHSO.hasPlan if documented else AWHSO.emptyPlan, TRUE))
        has_log = False
        for lit in g.objects(site, AWHSO.hasEventLog):
            has_log = bool(lit)
        g.add((site, AWHSO.populatedEventLog if has_log else AWHSO.emptyEventLog, TRUE))
        observed_types = set()
        for obj in g.objects(site, AWHSO.observedAttractant):
            for t in g.objects(obj, RDF.type):
                observed_types.add(t)
        for name in action_names:
            act = (WL_ACT | SDR_ACT)[name]
            if act not in documented:
                g.add((site, AWHSO.undocumentedAction, act))
            if act not in supported:
                g.add((site, AWHSO.localUnsupported, act))
            if name in habitat:
                cls = ATTR_CLASS[habitat[name]]
                if cls in observed_types:
                    g.add((site, AWHSO.preconditionObjectPresent, act))
                else:
                    g.add((site, AWHSO.preconditionObjectAbsent, act))


def fire(g: Graph) -> Graph:
    out = Graph()
    out.bind("awhso", AWHSO)
    for name in RULES:
        q = (SPARQL / f"{name}.rq").read_text(encoding="utf-8")
        constructed = g.query(q)
        for triple in constructed:
            out.add(triple)
    return out


def count_kinds(g: Graph) -> dict:
    by = defaultdict(int)
    for _, _, k in g.triples((None, AWHSO.disagreementKind, None)):
        by[str(k)] += 1
    return dict(by)


def smoke() -> bool:
    facts = load_facts_wildlife()
    conc = fire(facts)
    by = count_kinds(conc)
    print("SMOKE wildlife", by, "n", sum(by.values()), flush=True)
    ok = (
        sum(by.values()) == 16
        and "control_documented_empty_event_log" in by
        and "control_documented_pathway_mismatch" in by
        and "pathway_omitted_attractant_absent" in by
    )
    print("SMOKE_OK" if ok else "SMOKE_FAIL", flush=True)
    return ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not smoke():
        raise SystemExit("smoke failed")
    wl_facts = load_facts_wildlife()
    wl = fire(wl_facts)
    wl_by = count_kinds(wl)
    print("WILDLIFE", wl_by, sum(wl_by.values()), flush=True)
    train = load_year("2023")
    test = load_year("2024")
    sdr_facts = load_facts_sdr(train, test)
    sdr = fire(sdr_facts)
    sdr_by = count_kinds(sdr)
    print("SDR", sdr_by, sum(sdr_by.values()), flush=True)
    rec = {
        "wildlife": wl_by,
        "wildlife_n": sum(wl_by.values()),
        "sdr": sdr_by,
        "sdr_n": sum(sdr_by.values()),
        "rules": [
            "D1 habitat documented attractant present no pathway",
            "D1prime habitat documented attractant absent no pathway",
            "D2empty control documented empty event log",
            "D2mismatch control documented events present pathway mismatch",
            "D3object pathway omitted attractant present",
            "D3noobject pathway omitted attractant absent",
            "D3partial pathway omitted control plan incomplete",
            "D3silent pathway omitted control no plan",
        ],
    }
    (OUT / "diagnostic_rule_firings.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    wl.serialize((OUT / "wildlife_rule_conclusions.ttl").as_posix(), format="turtle")
    print("DONE", rec, flush=True)


if __name__ == "__main__":
    main()
