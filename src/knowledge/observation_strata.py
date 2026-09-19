"""Three observation strata for 123 airports: catalog, event-evidenced classes, named instances."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_situation_completion import (  # noqa: E402
    COORDS,
    OSM,
    load_osm,
    load_target_airports,
    normalize_airport,
    size_guild,
    phase_bucket,
    component_hits,
    match_rule,
    iter_nwsd_rows,
    AXIOMS,
    DOC_ATTR,
)

OUT = Path(__file__).resolve().parents[2] / "results" / "experiments" / "aei_situation"
CATALOG = ["StandingWater", "Wetland", "Grassland", "WasteFacility", "Agriculture", "PerchStructure"]


def main() -> None:
    axioms = json.loads(Path(AXIOMS).read_text(encoding="utf-8"))
    target = load_target_airports()
    osm = load_osm()
    doc = json.loads(Path(DOC_ATTR).read_text(encoding="utf-8"))
    evidenced: dict[str, set[str]] = defaultdict(set)
    local_actions: dict[str, set[str]] = defaultdict(set)
    n_events = defaultdict(int)
    for row in iter_nwsd_rows():
        aid = normalize_airport(row.get("AIRPORT_ID"))
        if aid not in target:
            continue
        n_events[aid] += 1
        size = size_guild(row)
        phase = phase_bucket(str(row.get("PHASE_OF_FLIGHT") or ""))
        for component, damaged in component_hits(row):
            for rule in axioms["pathway_rules"]:
                if not match_rule(rule, size, phase, component, damaged):
                    continue
                evidenced[aid].update(rule["attractants"])
                local_actions[aid].update(rule["actions"])

    rows = []
    n_event_class = 0
    n_osm = 0
    n_doc = 0
    n_any_named = 0
    for aid in sorted(target):
        ev = sorted(evidenced.get(aid, []))
        osm_cls = sorted({x["attractant_class"] for x in osm.get(aid, [])})
        doc_cls = sorted({x["class"] for x in doc["airports"].get(aid, [])})
        named = sorted(set(osm_cls) | set(doc_cls))
        if ev:
            n_event_class += 1
        if osm_cls:
            n_osm += 1
        if doc_cls:
            n_doc += 1
        if named:
            n_any_named += 1
        rows.append({
            "airport_id": aid,
            "n_events": n_events.get(aid, 0),
            "catalog_classes": 6,
            "event_evidenced_classes": "|".join(ev),
            "n_event_classes": len(ev),
            "osm_classes": "|".join(osm_cls),
            "n_osm_objects": len(osm.get(aid, [])),
            "document_classes": "|".join(doc_cls),
            "n_document_objects": len(doc["airports"].get(aid, [])),
            "has_event_class": bool(ev),
            "has_named_instance": bool(named),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "observation_strata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_airports": len(target),
        "catalog_classes": 6,
        "airports_with_event_evidenced_class": n_event_class,
        "event_class_pct": round(100.0 * n_event_class / len(target), 2),
        "airports_with_osm": n_osm,
        "osm_pct": round(100.0 * n_osm / len(target), 2),
        "airports_with_document_instance": n_doc,
        "airports_with_any_named_instance": n_any_named,
        "named_instance_pct": round(100.0 * n_any_named / len(target), 2),
    }
    (OUT / "observation_strata_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
