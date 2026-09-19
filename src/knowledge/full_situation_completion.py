"""Full 1990-2025 situation-completion graph + ablation + case WHMP tables."""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

from smoke_situation_completion import (
    ACTION_IRI,
    ATTR_IRI,
    AWHSO,
    COMPONENT_IRI,
    COMPONENTS,
    ONTOLOGY,
    PHASE_IRI,
    SIZE_IRI,
    add_slot,
    component_hits,
    match_rule,
    normalize_airport,
    phase_bucket,
    size_guild,
    species_unknown,
    text,
    truthy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "faa_wildlife"
AXIOMS = KNOWLEDGE_DIR / "axioms" / "action_licenses.json"
WHMP = KNOWLEDGE_DIR / "fixtures" / "whmp_actions.json"
DOC_ATTR = KNOWLEDGE_DIR / "fixtures" / "document_attractants.json"
SPARQL_DIR = KNOWLEDGE_DIR / "sparql"
COORDS = PROJECT_ROOT / "results" / "experiments" / "aei_situation" / "airport_coords.csv"
OSM = PROJECT_ROOT / "results" / "experiments" / "aei_situation" / "osm_attractants.csv"
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "aei_situation"

CASE_AIRPORTS = ["SEA", "GIF", "BJJ", "0B5"]


def load_target_airports() -> set[str]:
    wanted = set(CASE_AIRPORTS)
    with COORDS.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wanted.add(row["airport_id"])
    return wanted


def load_osm() -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    if not OSM.exists():
        return by
    with OSM.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by[row["airport_id"]].append(row)
    return by


def iter_nwsd_rows():
    for path in sorted(RAW_DIR.glob("faa_wildlife_export_*.json")):
        year = int(path.stem.rsplit("_", 1)[-1])
        if year < 1990 or year > 2025:
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for raw in payload.get("Result", []):
            row = {k.strip().upper().replace(" ", "_"): v for k, v in raw.items()}
            row["_FILE_YEAR"] = year
            yield row


def slot_rows_from_counts(n_all: int, counts: dict) -> list[dict]:
    rows = []
    for field in ["airport_id", "phase", "size", "species_known", "component_struck", "airside_object_freetext"]:
        rows.append({"slot": field, "n": n_all, "observed": counts[field], "observed_pct": round(100.0 * counts[field] / n_all, 2) if n_all else 0.0, "in_schema": True})
    for field in ["regulation_clause", "action", "whmp_role", "airside_object_structured"]:
        rows.append({"slot": field, "n": n_all, "observed": 0, "observed_pct": 0.0, "in_schema": False})
    return rows


def base_graph() -> Graph:
    g = Graph()
    g.parse(ONTOLOGY.as_posix(), format="turtle")
    g.bind("awhso", AWHSO)
    return g


def ensure_airport(g: Graph, airport_id: str, nodes: dict[str, URIRef]) -> URIRef:
    if airport_id not in nodes:
        node = AWHSO[f"airport_{airport_id}"]
        nodes[airport_id] = node
        g.add((node, RDF.type, AWHSO.Airport))
        g.add((node, AWHSO.airportId, Literal(airport_id)))
        g.add((node, RDFS.label, Literal(airport_id)))
    return nodes[airport_id]


def add_event_skeleton(g: Graph, event_uri, airport, event_id: str, year: int, size: str, phase: str, hits, damaged_any: bool) -> None:
    g.add((event_uri, RDF.type, AWHSO.StrikeEvent))
    g.add((event_uri, AWHSO.eventId, Literal(event_id)))
    g.add((event_uri, AWHSO.year, Literal(year, datatype=XSD.integer)))
    g.add((event_uri, AWHSO.occursAtAirport, airport))
    g.add((event_uri, AWHSO.hasSizeGuild, SIZE_IRI[size]))
    g.add((event_uri, AWHSO.inPhase, PHASE_IRI[phase]))
    g.add((event_uri, AWHSO.damaged, Literal(bool(damaged_any))))
    for component, _damaged in hits:
        g.add((event_uri, AWHSO.strikesComponent, COMPONENT_IRI[component]))


def add_norm_for_event(g: Graph, axioms: dict, event_uri, sit_factory, airport_id: str, size: str, phase: str, hits, inferred_actions, inferred_attractants) -> tuple[int, int]:
    pathway = 0
    links = 0
    sit = None
    for component, damaged in hits:
        for rule in axioms["pathway_rules"]:
            if not match_rule(rule, size, phase, component, damaged):
                continue
            pathway += 1
            if sit is None:
                sit = sit_factory()
            for action in rule["actions"]:
                act_uri = ACTION_IRI[action]
                g.add((event_uri, AWHSO.supportsAction, act_uri))
                inferred_actions[airport_id].add(action)
                links += 1
                add_slot(g, sit, "action", act_uri, "inferred", rule["id"], AWHSO[rule["norms"][0]], f"{text(event_uri)}_{component}_{action}")
                for norm in rule["norms"]:
                    g.add((act_uri, AWHSO.licensedBy, AWHSO[norm]))
                    g.add((sit, AWHSO.hasNorm, AWHSO[norm]))
            for attractant in rule["attractants"]:
                inferred_attractants[airport_id].add(attractant)
                g.add((sit, AWHSO.inferredAttractant, ATTR_IRI[attractant]))
    return pathway, links


def scan_and_build(axioms: dict, target: set[str], g3: Graph, gf: Graph, nodes3, nodesf) -> tuple[list[dict], dict, dict, dict, int, int]:
    n_all = 0
    n_target = 0
    counts = defaultdict(int)
    s2 = {"events": 0, "pathway_events": 0, "action_links": 0, "airports_with_actions": 0}
    s3 = {"events": 0, "pathway_events": 0, "action_links": 0}
    sf = {"events": 0, "pathway_events": 0, "action_links": 0}
    inf_act_3: dict[str, set[str]] = defaultdict(set)
    inf_attr_3: dict[str, set[str]] = defaultdict(set)
    inf_act_f: dict[str, set[str]] = defaultdict(set)
    inf_attr_f: dict[str, set[str]] = defaultdict(set)
    large_dep_engine = set()
    local_support: dict[str, set[str]] = defaultdict(set)
    for row in iter_nwsd_rows():
        n_all += 1
        aid = normalize_airport(row.get("AIRPORT_ID"))
        if aid and aid not in {"UNKNOWN", ""}:
            counts["airport_id"] += 1
        if phase_bucket(text(row.get("PHASE_OF_FLIGHT"))) != "unknown":
            counts["phase"] += 1
        if size_guild(row) != "UNKNOWN":
            counts["size"] += 1
        if not species_unknown(row):
            counts["species_known"] += 1
        hits = component_hits(row)
        if hits:
            counts["component_struck"] += 1
        remarks = " ".join([text(row.get("REMARKS")), text(row.get("COMMENTS"))])
        if any(tok in remarks.lower() for tok in ["pond", "wetland", "landfill", "grass", "perch", "attract"]):
            counts["airside_object_freetext"] += 1
        event_id = text(row.get("INDX_NR"))
        if not event_id or aid not in target:
            continue
        n_target += 1
        size = size_guild(row)
        phase = phase_bucket(text(row.get("PHASE_OF_FLIGHT")))
        year = int(row.get("INCIDENT_YEAR") or row.get("_FILE_YEAR") or 0)
        damaged_any = any(d for _c, d in hits)
        event_uri = AWHSO[f"event_{event_id}"]
        for g, nodes, stats in ((g3, nodes3, s3), (gf, nodesf, sf)):
            airport = ensure_airport(g, aid, nodes)
            add_event_skeleton(g, event_uri, airport, event_id, year, size, phase, hits, damaged_any)
            stats["events"] += 1
        s2["events"] += 1
        a3 = ensure_airport(g3, aid, nodes3)
        af = ensure_airport(gf, aid, nodesf)

        def sit3():
            uri = AWHSO[f"sit_{event_id}"]
            g3.add((uri, RDF.type, AWHSO.Situation))
            g3.add((uri, AWHSO.hasEvent, event_uri))
            g3.add((uri, AWHSO.hasArtifact, a3))
            return uri

        def sitf():
            uri = AWHSO[f"sit_{event_id}"]
            gf.add((uri, RDF.type, AWHSO.Situation))
            gf.add((uri, AWHSO.hasEvent, event_uri))
            gf.add((uri, AWHSO.hasArtifact, af))
            return uri

        p3, l3 = add_norm_for_event(g3, axioms, event_uri, sit3, aid, size, phase, hits, inf_act_3, inf_attr_3)
        pf, lf = add_norm_for_event(gf, axioms, event_uri, sitf, aid, size, phase, hits, inf_act_f, inf_attr_f)
        s3["pathway_events"] += p3
        s3["action_links"] += l3
        sf["pathway_events"] += pf
        sf["action_links"] += lf
        if size == "LARGE" and phase == "departure" and any(c == "engine" and d for c, d in hits):
            large_dep_engine.add(aid)
        local_support[aid].update(inf_act_f.get(aid, set()))

    def close_inferred(g, nodes, inf_act, inf_attr, stats):
        for airport_id, actions in inf_act.items():
            airport = nodes[airport_id]
            for action in actions:
                g.add((airport, AWHSO.recommendsAction, ACTION_IRI[action]))
                add_slot(g, airport, "action", ACTION_IRI[action], "inferred", "pathway_axiom", AWHSO.Norm_139_337_f2, f"{airport_id}_rec_{action}")
        for airport_id, attractants in inf_attr.items():
            airport = nodes[airport_id]
            for attractant in attractants:
                g.add((airport, AWHSO.inferredAttractant, ATTR_IRI[attractant]))
        stats["airports_with_actions"] = len(inf_act)

    close_inferred(g3, nodes3, inf_act_3, inf_attr_3, s3)
    close_inferred(gf, nodesf, inf_act_f, inf_attr_f, sf)
    s2["airports_with_actions"] = 0
    indexes = {
        "inf_act": {k: set(v) for k, v in inf_act_f.items()},
        "inf_attr": {k: set(v) for k, v in inf_attr_f.items()},
        "local_support": {k: set(v) for k, v in local_support.items()},
        "large_dep_engine": large_dep_engine,
    }
    return slot_rows_from_counts(n_all, counts), s2, s3, sf, n_all, n_target, indexes


def add_document_attractants(g: Graph, nodes: dict[str, URIRef]) -> int:
    spec = json.loads(DOC_ATTR.read_text(encoding="utf-8"))
    n = 0
    for airport_id, items in spec["airports"].items():
        airport = ensure_airport(g, airport_id, nodes)
        for item in items:
            node = AWHSO[item["id"]]
            cls = item["class"]
            g.add((node, RDF.type, AWHSO[cls]))
            g.add((node, RDFS.label, Literal(cls)))
            g.add((node, AWHSO.labelText, Literal(item["name"])))
            g.add((node, AWHSO.locatedAt, airport))
            g.add((airport, AWHSO.observedAttractant, node))
            add_slot(g, airport, "artifact", node, "observed", "whmp_document", AWHSO.Norm_AC_150_5200_33, item["id"])
            n += 1
    return n


def add_osm(g: Graph, osm: dict[str, list[dict]], nodes: dict[str, URIRef]) -> int:
    n = 0
    for airport_id, items in osm.items():
        if airport_id not in nodes:
            continue
        airport = nodes[airport_id]
        for item in items:
            osm_id = str(item["osm_id"])
            cls = item["attractant_class"]
            node = AWHSO[f"osm_{airport_id}_{item['osm_type']}_{osm_id}"]
            g.add((node, RDF.type, AWHSO[cls] if cls in {"StandingWater", "Wetland", "Grassland", "WasteFacility", "Agriculture", "PerchStructure"} else AWHSO.Attractant))
            g.add((node, RDFS.label, Literal(cls)))
            g.add((node, AWHSO.osmId, Literal(osm_id)))
            if item.get("name"):
                g.add((node, AWHSO.labelText, Literal(item["name"])))
            g.add((node, AWHSO.locatedAt, airport))
            g.add((airport, AWHSO.observedAttractant, node))
            add_slot(g, airport, "artifact", node, "observed", "osm", AWHSO.Norm_AC_150_5200_33, f"osm_{airport_id}_{osm_id}")
            n += 1
    return n


def add_whmp(g: Graph, whmp: dict, nodes: dict[str, URIRef]) -> int:
    n = 0
    for airport_id, spec in whmp["airports"].items():
        airport = ensure_airport(g, airport_id, nodes)
        for line in spec["lines"]:
            line_uri = AWHSO[f"whmp_{line['id']}"]
            g.add((line_uri, RDF.type, AWHSO.WHMPAction))
            g.add((line_uri, AWHSO.eventId, Literal(line["id"])))
            g.add((line_uri, RDFS.label, Literal(line["action"])))
            g.add((airport, AWHSO.documentsAction, line_uri))
            n += 1
    return n


def add_preconditions(g: Graph, axioms: dict) -> None:
    for action, attractant in axioms["action_precondition"].items():
        g.add((ACTION_IRI[action], AWHSO.preconditionAttractant, ATTR_IRI[attractant]))


def python_cqs(indexes: dict, osm: dict, whmp: dict, layer: str) -> dict[str, dict]:
    """layer: B2, B3, full."""
    t0 = time.time()
    inf_act = indexes["inf_act"] if layer != "B2" else {}
    inf_attr = indexes["inf_attr"] if layer != "B2" else {}
    local = indexes["local_support"] if layer != "B2" else {}
    pathway_airports = indexes["large_dep_engine"]
    use_osm = layer == "full"
    use_whmp = layer in {"B3", "full"}
    cq1 = []
    for aid in sorted(pathway_airports):
        if use_osm:
            for item in osm.get(aid, []):
                cq1.append({"airportId": aid, "attrClass": item["attractant_class"], "osmName": item.get("name", ""), "status": "observed", "provKind": "osm"})
        # document-named attractants count as observed in the full layer
        if layer == "full" and aid in {"SEA", "GIF", "BJJ", "0B5"}:
            pass  # added after document merge in caller if needed
        if layer == "B3":
            for cls in sorted(inf_attr.get(aid, [])):
                cq1.append({"airportId": aid, "attrClass": cls, "osmName": "", "status": "inferred", "provKind": "axiom"})
    cq2 = []
    if layer != "B2":
        for aid in sorted(pathway_airports):
            for action in sorted(inf_act.get(aid, [])):
                cq2.append({"airportId": aid, "actionLabel": action, "normLabel": "14 CFR 139.337(f)(2)", "status": "inferred"})
    cq3 = []
    if use_whmp:
        for line in whmp["airports"]["SEA"]["lines"]:
            action = line["action"]
            local_n = 1 if action in local.get("SEA", set()) else 0
            national_n = sum(1 for acts in local.values() if action in acts)
            cq3.append({"actionLabel": action, "localEvents": local_n, "nationalEvents": national_n})
    cq4 = []
    if use_whmp:
        for line in whmp["airports"]["SEA"]["lines"]:
            if line["action"] not in local.get("SEA", set()):
                cq4.append({"lineId": line["id"], "actionLabel": line["action"]})
    cq5 = []
    if layer != "B2":
        for aid in sorted(pathway_airports):
            has_water = "StandingWater" in inf_attr.get(aid, set())
            if use_osm and any(x["attractant_class"] == "StandingWater" for x in osm.get(aid, [])):
                has_water = True
            if not has_water:
                continue
            if "DrainStandingWater" in inf_act.get(aid, set()):
                cq5.append({"airportId": aid, "actionLabel": "DrainStandingWater", "supportedEvents": 1})
    elapsed = round(time.time() - t0, 3)
    packed = {}
    for name, rows in [
        ("cq1_attractants_engine_departure_large", cq1),
        ("cq2_licensed_actions", cq2),
        ("cq3_airport_vs_national_support", cq3),
        ("cq4_unsupported_whmp_lines", cq4),
        ("cq5_counterfactual_remove_water", cq5),
    ]:
        packed[name] = {"rows": rows, "seconds": elapsed, "n": len(rows)}
    return packed


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cq_answerable(cq: dict[str, dict]) -> dict[str, bool]:
    return {name: item["n"] > 0 for name, item in cq.items()}


def case_tables(g: Graph, whmp: dict) -> list[dict]:
    rows = []
    for airport_id, spec in whmp["airports"].items():
        airport = AWHSO[f"airport_{airport_id}"]
        documented = {line["action"] for line in spec["lines"]}
        recommended = {str(g.value(act, RDFS.label)) for act in g.objects(airport, AWHSO.recommendsAction)}
        local_supported = set()
        for event in g.subjects(AWHSO.occursAtAirport, airport):
            for act in g.objects(event, AWHSO.supportsAction):
                local_supported.add(str(g.value(act, RDFS.label)))
        osm_classes = {str(g.value(osm, RDFS.label)) for osm in g.objects(airport, AWHSO.observedAttractant)}
        all_actions = sorted(documented | recommended | local_supported)
        for action in all_actions:
            in_doc = action in documented
            in_graph = action in recommended or action in local_supported
            local = action in local_supported
            if in_graph and in_doc:
                bucket = "graph_and_document"
            elif in_graph and local and not in_doc:
                bucket = "graph_event_supported_not_in_document"
            elif in_doc and not local:
                bucket = "document_without_local_events"
            elif in_graph and not local:
                bucket = "graph_national_only"
            else:
                bucket = "other"
            rows.append({
                "airport_id": airport_id,
                "action": action,
                "in_document": in_doc,
                "local_event_support": local,
                "graph_recommends": action in recommended,
                "bucket": bucket,
                "osm_classes": "|".join(sorted(osm_classes)),
            })
    return rows


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    axioms = json.loads(AXIOMS.read_text(encoding="utf-8"))
    whmp = json.loads(WHMP.read_text(encoding="utf-8"))
    target = load_target_airports()
    osm = load_osm()
    if not osm:
        print("OSM attractants missing; run fetch_osm_attractants.py first", file=sys.stderr)
        return 2

    print("one-pass slot stats + B3/full graphs (B2 has no action/attractant triples)...")
    g3, gf = base_graph(), base_graph()
    add_preconditions(g3, axioms)
    add_preconditions(gf, axioms)
    nodes3, nodesf = {}, {}
    slots, s2, s3, sf, n_all, n_target, indexes = scan_and_build(axioms, target, g3, gf, nodes3, nodesf)
    write_csv(RESULT_DIR / "slot_observed_1990_2025.csv", slots)
    print(f"n_all={n_all} n_target={n_target} events_in_graph={s2['events']}", flush=True)
    add_whmp(g3, whmp, nodes3)
    n_osm = add_osm(gf, osm, nodesf)
    n_doc_attr = add_document_attractants(gf, nodesf)
    n_whmp = add_whmp(gf, whmp, nodesf)
    print("running competency queries...", flush=True)
    cq2 = python_cqs(indexes, osm, whmp, "B2")
    cq3 = python_cqs(indexes, osm, whmp, "B3")
    cqf = python_cqs(indexes, osm, whmp, "full")
    # Document-named attractants for case airports on the full layer.
    doc_attr = json.loads(DOC_ATTR.read_text(encoding="utf-8"))
    for aid, items in doc_attr["airports"].items():
        if aid not in indexes["large_dep_engine"]:
            continue
        for item in items:
            cqf["cq1_attractants_engine_departure_large"]["rows"].append({
                "airportId": aid,
                "attrClass": item["class"],
                "osmName": item["name"],
                "status": "observed",
                "provKind": "whmp_document",
            })
    cqf["cq1_attractants_engine_departure_large"]["n"] = len(cqf["cq1_attractants_engine_departure_large"]["rows"])
    print("B2", s2, {k: v["n"] for k, v in cq2.items()}, flush=True)
    print("B3", s3, {k: v["n"] for k, v in cq3.items()}, flush=True)
    print("full", sf, "osm", n_osm, "doc_attr", n_doc_attr, "whmp_lines", n_whmp, {k: v["n"] for k, v in cqf.items()}, flush=True)

    ablation = []
    for label, cq in [("B0_table", None), ("B2_events", cq2), ("B3_events_norm", cq3), ("full_osm", cqf)]:
        if cq is None:
            row = {"system": label}
            for name in ["cq1_attractants_engine_departure_large", "cq2_licensed_actions", "cq3_airport_vs_national_support", "cq4_unsupported_whmp_lines", "cq5_counterfactual_remove_water"]:
                row[name] = False
                row[f"{name}_n"] = 0
            ablation.append(row)
            continue
        ans = cq_answerable(cq)
        row = {"system": label}
        for name, ok in ans.items():
            row[name] = ok
            row[f"{name}_n"] = cq[name]["n"]
            row[f"{name}_s"] = cq[name]["seconds"]
        ablation.append(row)
    write_csv(RESULT_DIR / "ablation_cq.csv", ablation)

    for name, item in cqf.items():
        write_csv(RESULT_DIR / f"{name}.csv", item["rows"])

    cases = case_tables(gf, whmp)
    write_csv(RESULT_DIR / "case_airport_actions.csv", cases)

    osm_airports = {aid for aid, items in osm.items() if items}
    coverage = {
        "n_all_events": n_all,
        "n_target_events": n_target,
        "n_target_airports": len(target),
        "n_osm_airports_with_object": len(osm_airports),
        "osm_coverage_pct": round(100.0 * len(osm_airports) / max(len(target), 1), 2),
        "osm_objects": n_osm,
        "document_attractants": n_doc_attr,
        "triples_full": len(gf),
        "B2": s2,
        "B3": s3,
        "full": sf,
        "cq_seconds": {k: v["seconds"] for k, v in cqf.items()},
        "ablation_nonempty": {row["system"]: {k: row[k] for k in row if k.startswith("cq") and not k.endswith("_n") and not k.endswith("_s")} for row in ablation},
    }
    (RESULT_DIR / "full_run_summary.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(json.dumps(coverage, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
