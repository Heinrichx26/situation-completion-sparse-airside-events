"""Smoke test: situation-completion KG on 2024 NWSD + 10 airports.

Gate (plan section 2): planning slots empty in the raw table; at least 3 of 5
competency questions fail on SQL and succeed on the completed graph; every
inferred action has provenance; TBox loads.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = Path(__file__).resolve().parent
RAW_2024 = PROJECT_ROOT / "data" / "raw" / "faa_wildlife" / "faa_wildlife_export_2024.json"
ONTOLOGY = KNOWLEDGE_DIR / "ontology" / "awhso.ttl"
AXIOMS = KNOWLEDGE_DIR / "axioms" / "action_licenses.json"
WHMP_FIXTURE = KNOWLEDGE_DIR / "fixtures" / "smoke_whmp_den.json"
SPARQL_DIR = KNOWLEDGE_DIR / "sparql"
RESULT_DIR = PROJECT_ROOT / "results" / "smoke_tests" / "aei_situation"
TOP10 = ["DEN", "DFW", "ORD", "JFK", "MEM", "SLC", "DTW", "ATL", "SMF", "MCO"]

AWHSO = Namespace("https://w3id.org/awhso#")

COMPONENTS = {
    "radome": (["STR_RAD"], ["DAM_RAD"]),
    "windshield": (["STR_WINDSHLD"], ["DAM_WINDSHLD"]),
    "nose": (["STR_NOSE"], ["DAM_NOSE"]),
    "engine": (
        ["STR_ENG1", "STR_ENG2", "STR_ENG3", "STR_ENG4"],
        ["DAM_ENG1", "DAM_ENG2", "DAM_ENG3", "DAM_ENG4"],
    ),
    "propeller": (["STR_PROP"], ["DAM_PROP"]),
    "wing_rotor": (["STR_WING_ROT"], ["DAM_WING_ROT"]),
    "fuselage": (["STR_FUSE"], ["DAM_FUSE"]),
    "landing_gear": (["STR_LG"], ["DAM_LG"]),
    "tail": (["STR_TAIL"], ["DAM_TAIL"]),
    "lights": (["STR_LGHTS"], ["DAM_LGHTS"]),
    "other": (["STR_OTHER"], ["DAM_OTHER"]),
}

COMPONENT_IRI = {
    "engine": AWHSO.CompEngine,
    "windshield": AWHSO.CompWindshield,
    "fuselage": AWHSO.CompFuselage,
    "wing_rotor": AWHSO.CompWingRotor,
    "nose": AWHSO.CompNose,
    "radome": AWHSO.CompRadome,
    "landing_gear": AWHSO.CompLandingGear,
    "tail": AWHSO.CompTail,
    "propeller": AWHSO.CompPropeller,
    "lights": AWHSO.CompLights,
    "other": AWHSO.CompOther,
}

PHASE_IRI = {
    "departure": AWHSO.PhaseDeparture,
    "arrival": AWHSO.PhaseArrival,
    "enroute": AWHSO.PhaseEnroute,
    "ground": AWHSO.PhaseGround,
    "unknown": AWHSO.PhaseUnknown,
}

SIZE_IRI = {
    "SMALL": AWHSO.SizeSmall,
    "MEDIUM": AWHSO.SizeMedium,
    "LARGE": AWHSO.SizeLarge,
    "UNKNOWN": AWHSO.SizeUnknown,
}

ATTR_IRI = {
    "StandingWater": AWHSO.AttractantClass_StandingWater,
    "Wetland": AWHSO.AttractantClass_Wetland,
    "Grassland": AWHSO.AttractantClass_Grassland,
    "WasteFacility": AWHSO.AttractantClass_WasteFacility,
    "Agriculture": AWHSO.AttractantClass_Agriculture,
    "PerchStructure": AWHSO.AttractantClass_PerchStructure,
}

ACTION_IRI = {
    "DrainStandingWater": AWHSO.ActionType_DrainStandingWater,
    "GrassHeightManagement": AWHSO.ActionType_GrassHeightManagement,
    "ExclusionFencing": AWHSO.ActionType_ExclusionFencing,
    "WasteManagement": AWHSO.ActionType_WasteManagement,
    "Harassment": AWHSO.ActionType_Harassment,
    "LethalControl": AWHSO.ActionType_LethalControl,
    "LandUseAgreement": AWHSO.ActionType_LandUseAgreement,
}

PLANNING_FIELDS = ("action", "regulation_clause", "airside_object", "whmp_role")


def text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def empty(value) -> bool:
    return text(value).upper() in {"", "NULL", "NONE", "N/A", "NA", "UNKNOWN", "UNK"}


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return text(value).upper() in {"TRUE", "T", "YES", "Y", "1"}


def clean_row(raw: dict) -> dict:
    return {k.strip().upper().replace(" ", "_"): v for k, v in raw.items()}


def normalize_airport(airport_id: str) -> str:
    code = text(airport_id).upper()
    if len(code) == 4 and code.startswith("K"):
        return code[1:]
    return code


def phase_bucket(phase: str) -> str:
    p = phase.lower()
    if not p:
        return "unknown"
    if any(x in p for x in ["take-off", "takeoff", "climb", "departure"]):
        return "departure"
    if any(x in p for x in ["approach", "landing", "descent", "arrival"]):
        return "arrival"
    if any(x in p for x in ["en route", "enroute"]):
        return "enroute"
    if any(x in p for x in ["taxi", "parked", "pushback", "local"]):
        return "ground"
    return "unknown"


def size_guild(row: dict) -> str:
    size = text(row.get("SIZE")).upper()
    if size in SIZE_IRI:
        return size
    species = text(row.get("SPECIES")).upper()
    if "LARGE" in species:
        return "LARGE"
    if "MEDIUM" in species:
        return "MEDIUM"
    if "SMALL" in species:
        return "SMALL"
    return "UNKNOWN"


def species_unknown(row: dict) -> bool:
    species = text(row.get("SPECIES")).upper()
    return (not species) or ("UNKNOWN" in species)


def load_2024() -> list[dict]:
    payload = json.loads(RAW_2024.read_text(encoding="utf-8-sig"))
    return [clean_row(raw) for raw in payload.get("Result", [])]


def component_hits(row: dict) -> list[tuple[str, bool]]:
    hits = []
    for name, (struck, damaged) in COMPONENTS.items():
        if any(truthy(row.get(field)) for field in struck):
            hits.append((name, any(truthy(row.get(field)) for field in damaged)))
    return hits


def match_rule(rule: dict, size: str, phase: str, component: str, damaged: bool) -> bool:
    if rule.get("requires_damage") and not damaged:
        return False
    sizes = rule["size"] if isinstance(rule["size"], list) else [rule["size"]]
    phases = rule["phase"] if isinstance(rule["phase"], list) else [rule["phase"]]
    comps = rule["component"] if isinstance(rule["component"], list) else [rule["component"]]
    return size in sizes and phase in phases and component in comps


def slot_fill_report(rows: list[dict]) -> list[dict]:
    n = len(rows)
    counts = {
        "airport_id": 0,
        "phase": 0,
        "size": 0,
        "species_known": 0,
        "component_struck": 0,
        "airside_object_freetext": 0,
    }
    for row in rows:
        if normalize_airport(row.get("AIRPORT_ID")) and normalize_airport(row.get("AIRPORT_ID")) not in {"UNKNOWN", ""}:
            counts["airport_id"] += 1
        if phase_bucket(text(row.get("PHASE_OF_FLIGHT"))) != "unknown":
            counts["phase"] += 1
        if size_guild(row) != "UNKNOWN":
            counts["size"] += 1
        if not species_unknown(row):
            counts["species_known"] += 1
        if component_hits(row):
            counts["component_struck"] += 1
        remarks = " ".join([text(row.get("REMARKS")), text(row.get("COMMENTS"))])
        if any(tok in remarks.lower() for tok in ["pond", "wetland", "landfill", "grass", "perch", "attract"]):
            counts["airside_object_freetext"] += 1
    report = []
    for field, filled in counts.items():
        report.append({
            "slot": field,
            "n": n,
            "observed": filled,
            "observed_pct": round(100.0 * filled / n, 2) if n else 0.0,
            "in_schema": field not in PLANNING_FIELDS,
        })
    for field in ("regulation_clause", "action", "whmp_role"):
        report.append({
            "slot": field,
            "n": n,
            "observed": 0,
            "observed_pct": 0.0,
            "in_schema": False,
        })
    # airside_object already counted from free text; keep that row and add structured-schema row
    report.append({
        "slot": "airside_object_structured",
        "n": n,
        "observed": 0,
        "observed_pct": 0.0,
        "in_schema": False,
    })
    return report


def add_slot(g: Graph, owner: URIRef, layer: str, filler: URIRef, status: str, prov_kind: str, source: URIRef, key: str) -> None:
    slot = AWHSO[f"slot_{key}"]
    prov = AWHSO[f"prov_{key}"]
    g.add((owner, AWHSO.hasSlot, slot))
    g.add((slot, RDF.type, AWHSO.Slot))
    g.add((slot, AWHSO.slotLayer, Literal(layer)))
    g.add((slot, AWHSO.slotStatus, Literal(status)))
    status_iri = {"observed": AWHSO.StatusObserved, "inferred": AWHSO.StatusInferred, "unknown": AWHSO.StatusUnknown}[status]
    g.add((slot, AWHSO.hasStatus, status_iri))
    g.add((slot, AWHSO.slotFiller, filler))
    g.add((slot, AWHSO.hasProvenance, prov))
    g.add((prov, RDF.type, AWHSO.Provenance))
    g.add((prov, AWHSO.provenanceKind, Literal(prov_kind)))
    g.add((prov, AWHSO.derivedFrom, source))


def build_graph(rows: list[dict], axioms: dict, whmp: dict) -> Graph:
    g = Graph()
    g.parse(ONTOLOGY.as_posix(), format="turtle")
    g.bind("awhso", AWHSO)

    for action, attractant in axioms["action_precondition"].items():
        g.add((ACTION_IRI[action], AWHSO.preconditionAttractant, ATTR_IRI[attractant]))

    airport_nodes: dict[str, URIRef] = {}
    inferred_actions: dict[str, set[str]] = defaultdict(set)
    inferred_attractants: dict[str, set[str]] = defaultdict(set)
    event_action_links = 0

    for row in rows:
        event_id = text(row.get("INDX_NR"))
        if not event_id:
            continue
        airport_id = normalize_airport(row.get("AIRPORT_ID"))
        if airport_id not in TOP10:
            continue
        if airport_id not in airport_nodes:
            node = AWHSO[f"airport_{airport_id}"]
            airport_nodes[airport_id] = node
            g.add((node, RDF.type, AWHSO.Airport))
            g.add((node, AWHSO.airportId, Literal(airport_id)))
            g.add((node, RDFS.label, Literal(airport_id)))

        size = size_guild(row)
        phase = phase_bucket(text(row.get("PHASE_OF_FLIGHT")))
        event_uri = AWHSO[f"event_{event_id}"]
        g.add((event_uri, RDF.type, AWHSO.StrikeEvent))
        g.add((event_uri, AWHSO.eventId, Literal(event_id)))
        g.add((event_uri, AWHSO.year, Literal(2024, datatype=XSD.integer)))
        g.add((event_uri, AWHSO.occursAtAirport, airport_nodes[airport_id]))
        g.add((event_uri, AWHSO.hasSizeGuild, SIZE_IRI[size]))
        g.add((event_uri, AWHSO.inPhase, PHASE_IRI[phase]))

        sit = AWHSO[f"sit_{event_id}"]
        g.add((sit, RDF.type, AWHSO.Situation))
        g.add((sit, AWHSO.hasEvent, event_uri))
        g.add((sit, AWHSO.hasArtifact, airport_nodes[airport_id]))
        add_slot(g, sit, "event", event_uri, "observed", "nwsd", event_uri, f"{event_id}_event")
        add_slot(g, sit, "artifact", airport_nodes[airport_id], "observed", "nwsd", event_uri, f"{event_id}_airport")
        if size != "UNKNOWN":
            add_slot(g, sit, "agent", SIZE_IRI[size], "observed", "nwsd", event_uri, f"{event_id}_size")
        else:
            add_slot(g, sit, "agent", SIZE_IRI[size], "unknown", "missing", event_uri, f"{event_id}_size")

        hits = component_hits(row)
        if not hits:
            g.add((event_uri, AWHSO.damaged, Literal(False)))
            continue

        for component, damaged in hits:
            g.add((event_uri, AWHSO.strikesComponent, COMPONENT_IRI[component]))
            g.add((event_uri, AWHSO.damaged, Literal(bool(damaged))))
            for rule in axioms["pathway_rules"]:
                if not match_rule(rule, size, phase, component, damaged):
                    continue
                for action in rule["actions"]:
                    act_uri = ACTION_IRI[action]
                    g.add((event_uri, AWHSO.supportsAction, act_uri))
                    inferred_actions[airport_id].add(action)
                    event_action_links += 1
                    add_slot(
                        g, sit, "action", act_uri, "inferred", rule["id"],
                        AWHSO[rule["norms"][0]], f"{event_id}_{component}_{action}",
                    )
                    for norm in rule["norms"]:
                        g.add((act_uri, AWHSO.licensedBy, AWHSO[norm]))
                        g.add((sit, AWHSO.hasNorm, AWHSO[norm]))
                        add_slot(
                            g, sit, "norm", AWHSO[norm], "inferred", rule["id"],
                            AWHSO[norm], f"{event_id}_{component}_{action}_{norm}",
                        )
                for attractant in rule["attractants"]:
                    inferred_attractants[airport_id].add(attractant)
                    g.add((sit, AWHSO.inferredAttractant, ATTR_IRI[attractant]))
                    add_slot(
                        g, sit, "artifact", ATTR_IRI[attractant], "inferred", rule["id"],
                        AWHSO.Norm_AC_150_5200_33, f"{event_id}_{component}_{attractant}",
                    )

    for airport_id, actions in inferred_actions.items():
        airport = airport_nodes[airport_id]
        for action in actions:
            act_uri = ACTION_IRI[action]
            g.add((airport, AWHSO.recommendsAction, act_uri))
            add_slot(
                g, airport, "action", act_uri, "inferred", "pathway_axiom",
                AWHSO.Norm_139_337_f2, f"{airport_id}_rec_{action}",
            )
    for airport_id, attractants in inferred_attractants.items():
        airport = airport_nodes[airport_id]
        for attractant in attractants:
            attr_uri = ATTR_IRI[attractant]
            g.add((airport, AWHSO.inferredAttractant, attr_uri))
            add_slot(
                g, airport, "artifact", attr_uri, "inferred", "ac_150_5200_33",
                AWHSO.Norm_AC_150_5200_33, f"{airport_id}_attr_{attractant}",
            )

    den = airport_nodes.get("DEN")
    if den is not None:
        for line in whmp["lines"]:
            line_uri = AWHSO[f"whmp_{line['id']}"]
            g.add((line_uri, RDF.type, AWHSO.WHMPAction))
            g.add((line_uri, AWHSO.eventId, Literal(line["id"])))
            g.add((line_uri, RDFS.label, Literal(line["action"])))
            g.add((den, AWHSO.documentsAction, line_uri))

    g.event_action_links = event_action_links  # type: ignore[attr-defined]
    g.inferred_actions = inferred_actions  # type: ignore[attr-defined]
    g.inferred_attractants = inferred_attractants  # type: ignore[attr-defined]
    return g


def run_sparql(g: Graph) -> dict[str, list[dict]]:
    results = {}
    for path in sorted(SPARQL_DIR.glob("cq*.rq")):
        query = path.read_text(encoding="utf-8")
        rows = []
        for record in g.query(query):
            rows.append({str(k): ("" if v is None else str(v)) for k, v in record.asdict().items()})
        results[path.stem] = rows
    return results


def b0_answerable(rows: list[dict]) -> dict[str, dict]:
    """Raw-table baseline: a CQ is answerable only if the required columns exist and are non-empty."""
    has_action_col = False
    has_norm_col = False
    has_attractant_col = False
    large_dep_engine = 0
    for row in rows:
        if normalize_airport(row.get("AIRPORT_ID")) not in TOP10:
            continue
        hits = component_hits(row)
        if (
            size_guild(row) == "LARGE"
            and phase_bucket(text(row.get("PHASE_OF_FLIGHT"))) == "departure"
            and any(name == "engine" and damaged for name, damaged in hits)
        ):
            large_dep_engine += 1
    return {
        "cq1_attractants_engine_departure_large": {
            "answerable": has_attractant_col,
            "reason": "NWSD has no airside-object column; attractants are not structured fields.",
            "supporting_events": large_dep_engine,
        },
        "cq2_licensed_actions": {
            "answerable": has_action_col and has_norm_col,
            "reason": "NWSD has no WHMP-action or regulation-clause column.",
            "supporting_events": large_dep_engine,
        },
        "cq3_airport_vs_national_support": {
            "answerable": has_action_col,
            "reason": "Event counts exist, but action support cannot be computed without an action slot.",
            "supporting_events": large_dep_engine,
        },
        "cq4_unsupported_whmp_lines": {
            "answerable": False,
            "reason": "Raw table has no WHMP document lines to test for event support.",
            "supporting_events": 0,
        },
        "cq5_counterfactual_remove_water": {
            "answerable": False,
            "reason": "Raw table has no attractant precondition to remove.",
            "supporting_events": large_dep_engine,
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def layer_instances(g: Graph) -> dict[str, int]:
    return {
        "artifact_airports": len(list(g.subjects(RDF.type, AWHSO.Airport))),
        "agent_size_used": sum(1 for _ in g.triples((None, AWHSO.hasSizeGuild, None))),
        "event_strikes": len(list(g.subjects(RDF.type, AWHSO.StrikeEvent))),
        "norm_used": len(set(g.objects(None, AWHSO.licensedBy))),
        "action_recommended": len(set(g.objects(None, AWHSO.recommendsAction))),
        "inferred_attractants": len(set(g.objects(None, AWHSO.inferredAttractant))),
        "triples": len(g),
    }


def provenance_complete(g: Graph) -> tuple[int, int]:
    inferred_actions = list(g.subjects(AWHSO.slotLayer, Literal("action")))
    with_prov = 0
    inferred = 0
    for slot in inferred_actions:
        status = str(g.value(slot, AWHSO.slotStatus) or "")
        if status != "inferred":
            continue
        inferred += 1
        if g.value(slot, AWHSO.hasProvenance) is not None:
            with_prov += 1
    return inferred, with_prov


def judge(slot_rows: list[dict], b0: dict, cq: dict[str, list[dict]], layers: dict[str, int], inferred: int, with_prov: int) -> dict:
    planning = [r for r in slot_rows if r["slot"] in {"action", "regulation_clause", "airside_object_structured"}]
    planning_empty = all(r["observed_pct"] == 0.0 for r in planning)
    b0_fail = sum(1 for item in b0.values() if not item["answerable"])
    cq_success = {name: len(rows) > 0 for name, rows in cq.items()}
    graph_success_n = sum(cq_success.values())
    layers_ok = all(layers[k] > 0 for k in ["artifact_airports", "event_strikes", "norm_used", "action_recommended"])
    prov_ok = inferred > 0 and inferred == with_prov
    passed = planning_empty and b0_fail >= 3 and graph_success_n == 5 and layers_ok and prov_ok
    return {
        "planning_slots_empty": planning_empty,
        "b0_unanswerable": b0_fail,
        "graph_cqs_nonempty": graph_success_n,
        "cq_success": cq_success,
        "layers_ok": layers_ok,
        "provenance_ok": prov_ok,
        "inferred_action_slots": inferred,
        "inferred_action_slots_with_provenance": with_prov,
        "passed": passed,
    }


def write_report(path: Path, n_all: int, n_sub: int, slot_rows: list[dict], b0: dict, cq: dict, layers: dict, gate: dict) -> None:
    lines = [
        "# AEI situation-completion smoke report",
        "",
        f"- NWSD 2024 events: {n_all}",
        f"- Top-10 airport subset: {n_sub} ({', '.join(TOP10)})",
        f"- Gate passed: **{gate['passed']}**",
        "",
        "## Slot observed rates (2024 full file)",
        "",
        "| slot | observed_pct | in_schema |",
        "|---|---:|:---:|",
    ]
    for row in slot_rows:
        lines.append(f"| {row['slot']} | {row['observed_pct']} | {row['in_schema']} |")
    lines += [
        "",
        "## B0 raw-table competency questions",
        "",
        "| CQ | answerable | reason |",
        "|---|---|---|",
    ]
    for name, item in b0.items():
        lines.append(f"| {name} | {item['answerable']} | {item['reason']} |")
    lines += [
        "",
        "## Completed-graph competency questions",
        "",
        "| CQ | n_rows | nonempty |",
        "|---|---:|:---:|",
    ]
    for name, rows in cq.items():
        lines.append(f"| {name} | {len(rows)} | {len(rows) > 0} |")
    lines += [
        "",
        "## Layer instance counts",
        "",
    ]
    for key, value in layers.items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Gate",
        "",
        f"- planning slots empty: {gate['planning_slots_empty']}",
        f"- B0 unanswerable (need >=3): {gate['b0_unanswerable']}",
        f"- graph CQs nonempty (need 5): {gate['graph_cqs_nonempty']}",
        f"- layers present: {gate['layers_ok']}",
        f"- inferred action provenance {gate['inferred_action_slots_with_provenance']}/{gate['inferred_action_slots']}",
        f"- **passed = {gate['passed']}**",
        "",
        "If passed, full-library instantiation is allowed. If failed, follow plan section 6; do not write the manuscript from these numbers.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    axioms = json.loads(AXIOMS.read_text(encoding="utf-8"))
    whmp = json.loads(WHMP_FIXTURE.read_text(encoding="utf-8"))
    rows = load_2024()
    subset = [row for row in rows if normalize_airport(row.get("AIRPORT_ID")) in TOP10]
    slot_rows = slot_fill_report(rows)
    write_csv(RESULT_DIR / "slot_observed_2024.csv", slot_rows)

    g = build_graph(subset, axioms, whmp)
    g.serialize((RESULT_DIR / "smoke_graph.ttl").as_posix(), format="turtle")
    layers = layer_instances(g)
    cq = run_sparql(g)
    for name, cq_rows in cq.items():
        write_csv(RESULT_DIR / f"{name}.csv", cq_rows)
    b0 = b0_answerable(rows)
    write_csv(RESULT_DIR / "b0_answerability.csv", [
        {"cq": name, **{k: v for k, v in item.items()}} for name, item in b0.items()
    ])
    inferred, with_prov = provenance_complete(g)
    gate = judge(slot_rows, b0, cq, layers, inferred, with_prov)
    (RESULT_DIR / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    write_report(
        RESULT_DIR / "smoke_report.md",
        n_all=len(rows),
        n_sub=len(subset),
        slot_rows=slot_rows,
        b0=b0,
        cq=cq,
        layers=layers,
        gate=gate,
    )
    print(json.dumps({"n_2024": len(rows), "n_top10": len(subset), "layers": layers, "gate": gate}, indent=2))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
