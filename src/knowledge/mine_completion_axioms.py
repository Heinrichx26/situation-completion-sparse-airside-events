"""Mine completion axioms from AC attractant lexicon + NWSD damage cells.

Hand R1-R6 are a gold pathway set. Rules are emitted when a (size, phase,
component) cell meets tunable damage support. Attractants and actions come
from species-guild votes aligned to AC 150/5200-33 land-use classes.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

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
from smoke_situation_completion import species_unknown  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "experiments" / "aei_situation"
GBIF = Path(__file__).resolve().parents[2] / "data" / "raw" / "gbif" / "airport_month_bird_counts.csv"
DOC_ATTR = Path(__file__).resolve().parent / "fixtures" / "document_attractants.json"
GOLD_JSON = Path(__file__).resolve().parent / "axioms" / "action_licenses.json"

GUILD_ATTR = {
    "water": ["StandingWater", "Wetland"],
    "waste": ["WasteFacility", "Grassland"],
    "raptor": ["PerchStructure", "Grassland"],
    "ag": ["Agriculture", "Grassland"],
    "mammal": ["Grassland", "PerchStructure"],
    "general": ["Grassland"],
}
ATTR_ACTION = {
    "StandingWater": ["DrainStandingWater"],
    "Wetland": ["DrainStandingWater"],
    "Grassland": ["GrassHeightManagement"],
    "WasteFacility": ["WasteManagement"],
    "Agriculture": ["LandUseAgreement"],
    "PerchStructure": ["ExclusionFencing"],
}
WATER = re.compile(
    r"goose|geese|duck|swan|mallard|teal|wigeon|pintail|canvasback|redhead|"
    r"heron|egret|pelican|cormorant|loon|grebe|ibis|stork|coot|moorhen|"
    r"waterfowl|gull|tern|skimmer|pelican",
    re.I,
)
WASTE = re.compile(r"starling|blackbird|crow|raven|pigeon|rock dove|grackle|gull|vulture", re.I)
RAPTOR = re.compile(r"hawk|eagle|owl|falcon|kestrel|osprey|harrier|vulture|buzzard|kite", re.I)
AG = re.compile(r"dove|lark|meadowlark|sparrow|finch|blackbird|killdeer|horn", re.I)
MAMMAL = re.compile(r"deer|coyote|fox|dog|cat|canine|mammal|hog|pig|cow|horse|bear", re.I)

GOLD_CELLS = {
    ("LARGE", "departure", "engine"): "R1",
    ("LARGE", "arrival", "engine"): "R2",
    ("MEDIUM", "departure", "engine"): "R3",
    ("MEDIUM", "arrival", "engine"): "R4",
    ("LARGE", "ground", "engine"): "R5",
    ("LARGE", "ground", "fuselage"): "R5",
    ("LARGE", "ground", "landing_gear"): "R5",
    ("MEDIUM", "arrival", "engine"): "R6",
    ("MEDIUM", "arrival", "wing_rotor"): "R6",
    ("LARGE", "arrival", "wing_rotor"): "R6",
}


def guilds(species: str) -> list[str]:
    found = []
    if WATER.search(species):
        found.append("water")
    if WASTE.search(species):
        found.append("waste")
    if RAPTOR.search(species):
        found.append("raptor")
    if AG.search(species):
        found.append("ag")
    if MAMMAL.search(species):
        found.append("mammal")
    if not found:
        found.append("general")
    return found


def size_default(size: str) -> list[str]:
    if size == "LARGE":
        return ["water", "mammal"]
    if size == "MEDIUM":
        return ["waste", "ag"]
    return ["general"]


def emit_rule(cell, n, nd, votes) -> dict:
    size, phase, component = cell
    total = sum(votes.values()) or 1
    guilds_used = [g for g, c in votes.items() if c / total >= 0.15] or size_default(size)
    attractants = []
    for g in guilds_used:
        for a in GUILD_ATTR[g]:
            if a not in attractants:
                attractants.append(a)
    actions = ["Harassment"]
    rate = nd / n if n else 0
    if rate >= 0.10:
        actions.append("LethalControl")
    for a in attractants:
        for act in ATTR_ACTION[a]:
            if act not in actions:
                actions.append(act)
    return {
        "id": f"M_{size}_{phase}_{component}",
        "size": size,
        "phase": phase,
        "component": component,
        "n": n,
        "n_damaged": nd,
        "damage_rate": round(nd / n, 4) if n else 0,
        "attractants": attractants,
        "actions": actions,
        "guild_votes": dict(votes),
        "norms": ["Norm_139_337_f2", "Norm_AC_150_5200_33"],
        "requires_damage": True,
        "provenance": "AC150-5200-33_lexicon+NWSD_cell",
    }


def load_gbif_airports() -> set[str]:
    ids = set()
    with GBIF.open("r", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("gbif_query_ok") == "1" and int(float(row.get("gbif_bird_occurrences") or 0)) > 0:
                ids.add(row["airport_id"].upper())
    return ids


def main() -> None:
    target = load_target_airports()
    osm = load_osm()
    doc = json.loads(DOC_ATTR.read_text(encoding="utf-8"))
    gbif = load_gbif_airports()
    cells = defaultdict(lambda: {"n": 0, "nd": 0, "votes": defaultdict(int)})
    # pathway events under a rule set later
    events = []  # (aid, size, phase, component, damaged, species)

    for row in iter_nwsd_rows():
        aid = normalize_airport(row.get("AIRPORT_ID"))
        if aid not in target:
            continue
        size = size_guild(row)
        phase = phase_bucket(text(row.get("PHASE_OF_FLIGHT")))
        if size in {"UNKNOWN", ""} or phase == "unknown":
            continue
        species = text(row.get("SPECIES"))
        gs = [] if species_unknown(row) else guilds(species)
        for component, damaged in component_hits(row):
            key = (size, phase, component)
            cells[key]["n"] += 1
            if damaged:
                cells[key]["nd"] += 1
                for g in (gs or size_default(size)):
                    cells[key]["votes"][g] += 1
            events.append((aid, size, phase, component, damaged, gs))

    sweeps = []
    for min_nd in (5, 10, 20, 30, 50):
        for min_rate in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
            rules = []
            for cell, st in cells.items():
                n, nd = st["n"], st["nd"]
                if nd < min_nd or n == 0:
                    continue
                rate = nd / n
                if rate < min_rate:
                    continue
                rules.append(emit_rule(cell, n, nd, st["votes"]))
            rule_index = {(r["size"], r["phase"], r["component"]): r for r in rules}
            recovered = sorted({GOLD_CELLS[c] for c in GOLD_CELLS if c in rule_index})
            # metrics on events
            pw_events = 0
            pw_airports = set()
            action_airports = set()
            cq1_class = set()  # (aid, attr) for LARGE departure engine damaged
            cq2 = set()
            cq5 = set()
            local_support = defaultdict(set)
            evidenced = defaultdict(set)
            for aid, size, phase, component, damaged, gs in events:
                r = rule_index.get((size, phase, component))
                if r is None or (r["requires_damage"] and not damaged):
                    continue
                pw_events += 1
                pw_airports.add(aid)
                action_airports.add(aid)
                local_support[aid].update(r["actions"])
                evidenced[aid].update(r["attractants"])
                if size == "LARGE" and phase == "departure" and component == "engine" and damaged:
                    for a in r["attractants"]:
                        cq1_class.add((aid, a))
                    for act in r["actions"]:
                        cq2.add((aid, act))
                if "DrainStandingWater" in r["actions"] and "StandingWater" in r["attractants"] and damaged:
                    cq5.add(aid)
            # instance CQ1: named objects at pathway airports
            lde_airports = {aid for aid, size, phase, component, damaged, gs in events if size == "LARGE" and phase == "departure" and component == "engine" and damaged}
            cq1_inst = 0
            cq1_objects = 0
            cq1_class_obs = set()
            for aid in pw_airports:
                n_obj = len(osm.get(aid, [])) + len(doc["airports"].get(aid, []))
                if aid in gbif:
                    n_obj += 1
                    cq1_class_obs.add((aid, "Grassland"))
                cq1_objects += n_obj
                for a in evidenced.get(aid, []):
                    cq1_class_obs.add((aid, a))
            cq1_inst = 0
            for aid in lde_airports:
                n_named = len(osm.get(aid, [])) + len(doc["airports"].get(aid, []))
                if aid in gbif:
                    n_named += 1
                cq1_inst += n_named
            # gold cell event coverage
            gold_event_n = sum(cells[c]["nd"] for c in GOLD_CELLS if c in cells)
            mined_gold_n = sum(cells[c]["nd"] for c in GOLD_CELLS if c in rule_index)
            sweeps.append({
                "min_n_damaged": min_nd,
                "min_damage_rate": min_rate,
                "n_rules": len(rules),
                "gold_recovered": ",".join(recovered),
                "n_gold_recovered": len(set(recovered)),
                "gold_event_recall": round(mined_gold_n / gold_event_n, 3) if gold_event_n else 0,
                "pathway_events": pw_events,
                "pathway_airports": len(pw_airports),
                "action_airports": len(action_airports),
                "cq1_class_rows": len(cq1_class),
                "cq1_instance_rows": cq1_inst,
                "cq1_objects_pathway_airports": cq1_objects,
                "cq1_observed_class_rows": len(cq1_class_obs),
                "cq2_rows": len(cq2),
                "cq5_airports": len(cq5),
                "gbif_airports": len(gbif & target),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "axiom_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweeps[0].keys()))
        writer.writeheader()
        writer.writerows(sweeps)

    # pick: recover all 6 gold ids if possible, else max n_gold; then max cq1_instance; then max pathway_events
    def keyfn(s):
        return (s["n_gold_recovered"], s["cq1_instance_rows"], s["cq1_class_rows"], s["pathway_events"])

    # prefer configs that recover >=5 gold and have more increment than hand (4687 events, 473 class, 822 inst)
    viable = [s for s in sweeps if s["n_gold_recovered"] >= 5]
    best = max(viable or sweeps, key=keyfn)
    # also record max-increment among gold>=5
    max_inc = max(viable or sweeps, key=lambda s: (s["cq1_instance_rows"], s["pathway_events"]))

    # emit rules for best
    rules = []
    for cell, st in cells.items():
        n, nd = st["n"], st["nd"]
        if nd < best["min_n_damaged"] or n == 0:
            continue
        if nd / n < best["min_damage_rate"]:
            continue
        rules.append(emit_rule(cell, n, nd, st["votes"]))
    (OUT / "mined_axioms.json").write_text(json.dumps({"selected": best, "max_instance": max_inc, "rules": rules}, indent=2), encoding="utf-8")
    print("best", json.dumps(best, indent=2))
    print("max_instance", json.dumps(max_inc, indent=2))
    print("n_cells", len(cells), "n_gbif", len(gbif), "n_events_kept", len(events))
    print("LARGE ground cells")
    for cell, st in sorted(cells.items()):
        if cell[0] == "LARGE" and cell[1] == "ground":
            print(cell, st["n"], st["nd"], round(st["nd"] / st["n"], 3) if st["n"] else 0)


if __name__ == "__main__":
    main()
