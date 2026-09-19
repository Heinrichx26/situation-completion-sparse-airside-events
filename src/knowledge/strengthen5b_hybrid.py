"""Hybrid OC-NLI verify, SDR cell sweep, remark proper-names, OSM retry for LDE hubs."""
from __future__ import annotations

import csv
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
from strengthen5_ocrag import (  # noqa: E402
    ACTION_HYP,
    GOLD,
    LEXICON,
    NLI,
    NLI_NAME,
    SDR_DIR,
    SDR_LEX,
    WHMP,
    f1,
    parse_sdr_ca,
    sdr_gold_from_ca,
    sentences,
)
from fetch_osm_attractants import (  # noqa: E402
    CACHE,
    RADIUS_M,
    TAG_MAP,
    bbox,
    classify,
    elements_from_payload,
    haversine_m,
    post_overpass,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "experiments" / "aei_situation"
COORDS = OUT / "airport_coords.csv"
OSM_CSV = OUT / "osm_attractants.csv"
DOC_ATTR = json.loads((Path(__file__).resolve().parent / "fixtures" / "document_attractants.json").read_text(encoding="utf-8"))
GBIF = ROOT / "data" / "raw" / "gbif" / "airport_month_bird_counts.csv"

DEONTIC = re.compile(r"\b(recommend(?:ed|s|ation)?|shall|should|must)\b", re.I)
LOCAL = re.compile(
    r"\b(this airport|the airport|airport operator|AOA|the Port|wildlife staff|airport staff|we will)\b",
    re.I,
)
GENERIC = re.compile(r"\b(airports may|an airport|airports should|FAA preferred|in general|typically)\b", re.I)

DOCS = {
    "SEA": WHMP / "SEA_WHMP_2022.txt",
    "GIF": WHMP / "GIF_WHSV_2018.txt",
    "BJJ": WHMP / "BJJ_WHA_2014.txt",
    "0B5": WHMP / "Montague_WHMP_2022.txt",
}

PROPER = re.compile(
    r"\b(?:Lake|Pond|Creek|River|Reservoir|Marsh|Landfill|Farm|Golf Course)\s+[A-Z][A-Za-z]+"
    r"|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s+(?:Lake|Pond|Creek|River|Landfill|Farm|Golf Course)\b"
)


def gold_types():
    return {k: {ln["action"] for ln in GOLD["airports"][k]["lines"]} for k in DOCS}


def matched_deontic(blob: str) -> dict[str, list[str]]:
    sents = sentences(blob, 1200)
    out = {lab: [] for lab in LEXICON}
    local = {lab: [] for lab in LEXICON}
    for s in sents:
        if not DEONTIC.search(s):
            continue
        gen = bool(GENERIC.search(s))
        loc = bool(LOCAL.search(s))
        for lab, pats in LEXICON.items():
            if any(re.search(p, s, re.I) for p in pats):
                out[lab].append(s)
                if loc and not gen:
                    local[lab].append(s)
    return out, local


def hybrid_pred(matched, local, nli, tau, require_nli_if_no_local=True):
    pred = set()
    detail = {}
    for lab, sents in matched.items():
        n_d = len(sents)
        n_l = len(local[lab])
        mx = 0.0
        if sents and nli is not None:
            mx = float(nli.entail_probs(sents[:12], ACTION_HYP[lab]).max())
        keep = n_l >= 1
        if require_nli_if_no_local:
            keep = keep or (n_d >= 1 and mx >= tau)
        else:
            keep = keep or n_d >= 1
        if keep:
            pred.add(lab)
        detail[lab] = {"n_deont": n_d, "n_local": n_l, "nli_max": round(mx, 3), "keep": keep}
    return pred, detail


def run_hybrid():
    gold = gold_types()
    print("load nli", flush=True)
    nli = NLI()
    blobs = {ap: path.read_text(encoding="utf-8", errors="ignore") for ap, path in DOCS.items()}
    matched = {}
    local = {}
    for ap, blob in blobs.items():
        matched[ap], local[ap] = matched_deontic(blob)
        print(ap, {lab: len(matched[ap][lab]) for lab in matched[ap]}, flush=True)

    rows = []
    details = {}
    for tau in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        gset, pset = set(), set()
        details[tau] = {}
        for ap in DOCS:
            pred, det = hybrid_pred(matched[ap], local[ap], nli, tau)
            details[tau][ap] = {"pred": sorted(pred), "detail": det}
            p, r, f, tp, fp, fn = f1(gold[ap], pred)
            rec = {"trial": "hybrid_nli", "tau": tau, "airport": ap, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn, "pred": "|".join(sorted(pred))}
            rows.append(rec)
            print(rec, flush=True)
            gset |= {f"{ap}:{x}" for x in gold[ap]}
            pset |= {f"{ap}:{x}" for x in pred}
        p, r, f, tp, fp, fn = f1(gset, pset)
        rec = {"trial": "hybrid_nli", "tau": tau, "airport": "POOLED", "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn, "pred": ""}
        rows.append(rec)
        print("POOLED", rec, flush=True)

    # deontic-only baseline (no NLI)
    gset, pset = set(), set()
    for ap in DOCS:
        pred = {lab for lab, sents in matched[ap].items() if sents}
        p, r, f, tp, fp, fn = f1(gold[ap], pred)
        rec = {"trial": "deontic_lexicon", "tau": "", "airport": ap, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn, "pred": "|".join(sorted(pred))}
        rows.append(rec)
        gset |= {f"{ap}:{x}" for x in gold[ap]}
        pset |= {f"{ap}:{x}" for x in pred}
    p, r, f, tp, fp, fn = f1(gset, pset)
    rows.append({"trial": "deontic_lexicon", "tau": "", "airport": "POOLED", "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn, "pred": ""})

    with (OUT / "strengthen5b_action_f1.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "strengthen5b_hybrid_detail.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return nli, rows


def sdr_cell_sweep():
    sample = []
    path = SDR_DIR / "SDR-2024.csv"
    with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.DictReader(handle):
            disc = row.get("Discrepancy") or ""
            head, ca = parse_sdr_ca(disc)
            if not ca or len(ca) < 20:
                continue
            gold = sdr_gold_from_ca(ca)
            if not gold:
                continue
            sample.append({
                "jasc": (row.get("JASCCode") or "").strip(),
                "nature": (row.get("NatureOfConditionA") or "").strip(),
                "gold": gold,
            })
            if len(sample) >= 4000:
                break
    print("SDR n", len(sample), flush=True)
    train = sample[::2]
    test = sample[1::2]
    cells = defaultdict(lambda: defaultdict(int))
    for rec in train:
        if rec["jasc"]:
            for a in rec["gold"]:
                cells[(rec["jasc"], rec["nature"])][a] += 1
    rows = []
    for min_c in (1, 2, 3, 5, 8):
        for majority in (False, True):
            gset, pset = set(), set()
            filled = 0
            for i, rec in enumerate(test):
                key = (rec["jasc"], rec["nature"])
                counts = cells[key]
                if not counts:
                    pred = set()
                elif majority:
                    top = max(counts.values())
                    pred = {a for a, c in counts.items() if c == top and c >= min_c}
                else:
                    pred = {a for a, c in counts.items() if c >= min_c}
                if pred:
                    filled += 1
                gset |= {f"{i}:{x}" for x in rec["gold"]}
                pset |= {f"{i}:{x}" for x in pred}
            p, r, f, tp, fp, fn = f1(gset, pset)
            rec = {
                "min_c": min_c, "majority": int(majority), "n_test": len(test),
                "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                "tp": tp, "fp": fp, "filled": filled,
                "fill_rate": round(filled / max(1, len(test)), 3),
            }
            rows.append(rec)
            print("SDR", rec, flush=True)
    with (OUT / "strengthen5b_sdr_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def remark_proper_and_class():
    target = load_target_airports()
    osm = load_osm()
    gbif = set()
    if GBIF.exists():
        with GBIF.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("gbif_query_ok") == "1" and float(row.get("gbif_bird_occurrences") or 0) > 0:
                    gbif.add(row["airport_id"].upper())
    lde = set()
    proper = defaultdict(set)
    classes = defaultdict(set)
    for row in iter_nwsd_rows():
        aid = normalize_airport(row.get("AIRPORT_ID"))
        if aid not in target:
            continue
        size = size_guild(row)
        phase = phase_bucket(text(row.get("PHASE_OF_FLIGHT")))
        hits = component_hits(row)
        if size == "LARGE" and phase == "departure" and any(c == "engine" and d for c, d in hits):
            lde.add(aid)
        remarks = " ".join([text(row.get("REMARKS")), text(row.get("COMMENTS"))])
        if len(remarks) < 20:
            continue
        low = remarks.lower()
        mapping = [
            ("StandingWater", ["pond", "lake", "creek", "ditch", "stormwater", "standing water"]),
            ("Wetland", ["wetland", "marsh", "swamp"]),
            ("Grassland", ["golf course", "airfield grass"]),
            ("WasteFacility", ["landfill", "dump", "putrescible", "transfer station"]),
            ("Agriculture", ["farmland", "livestock", "hay field"]),
            ("PerchStructure", ["light pole", "perch", "roost"]),
        ]
        for cls, toks in mapping:
            if any(t in low for t in toks):
                classes[aid].add(cls)
        for m in PROPER.finditer(remarks):
            name = re.sub(r"\s+", " ", m.group(0)).strip()
            if len(name.split()) >= 2:
                cls = "StandingWater"
                low_n = name.lower()
                if "landfill" in low_n:
                    cls = "WasteFacility"
                elif "farm" in low_n:
                    cls = "Agriculture"
                elif "golf" in low_n:
                    cls = "Grassland"
                elif "marsh" in low_n:
                    cls = "Wetland"
                proper[aid].add((cls, name))
    def pack(extra_fn):
        n = 0
        for aid in lde:
            n += len(osm.get(aid, [])) + len(DOC_ATTR["airports"].get(aid, []))
            if aid in gbif:
                n += 1
            n += extra_fn(aid)
        return n
    rec = {
        "n_lde": len(lde),
        "cq1_base": pack(lambda a: 0),
        "cq1_plus_class": pack(lambda a: len(classes.get(a, ()))),
        "cq1_plus_proper": pack(lambda a: len(proper.get(a, ()))),
        "cq1_plus_class_proper": pack(lambda a: len(classes.get(a, ())) + len(proper.get(a, ()))),
        "n_class_pairs": sum(len(v) for v in classes.values()),
        "n_proper": sum(len(v) for v in proper.values()),
        "lde_with_class": sum(1 for a in lde if classes.get(a)),
        "lde_with_proper": sum(1 for a in lde if proper.get(a)),
    }
    print("CQ1", rec, flush=True)
    (OUT / "strengthen5b_cq1.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec, lde


def osm_retry_lde(lde: set[str], limit: int = 12):
    """Retry Overpass for LDE airports currently at 0 OSM objects."""
    osm = load_osm()
    missing = [a for a in sorted(lde) if not osm.get(a)]
    coords = {}
    with COORDS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("lat") and row.get("lon"):
                coords[row["airport_id"]] = (float(row["lat"]), float(row["lon"]))
    todo = [a for a in missing if a in coords][:limit]
    print("OSM retry", todo, flush=True)
    new_rows = []
    CACHE.mkdir(parents=True, exist_ok=True)
    for aid in todo:
        lat, lon = coords[aid]
        south, west, north, east = bbox(lat, lon)
        filters = "".join(
            f'way["{k}"="{v}"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});'
            for (k, v) in TAG_MAP
        )
        query = f"[out:json][timeout:25];({filters});out center tags;"
        try:
            payload = post_overpass(query)
        except Exception as exc:
            print("OSM fail", aid, exc, flush=True)
            time.sleep(6)
            continue
        (CACHE / f"{aid}.json").write_text(json.dumps(payload), encoding="utf-8")
        n = 0
        for el in elements_from_payload(payload):
            tags = el.get("tags") or {}
            cls = classify(tags)
            if not cls:
                continue
            clat = el.get("lat") or (el.get("center") or {}).get("lat")
            clon = el.get("lon") or (el.get("center") or {}).get("lon")
            if clat is None or clon is None:
                continue
            if haversine_m(lat, lon, float(clat), float(clon)) > RADIUS_M:
                continue
            new_rows.append({
                "airport_id": aid,
                "osm_id": el.get("id"),
                "class": cls,
                "name": tags.get("name") or "",
                "lat": clat,
                "lon": clon,
            })
            n += 1
        print("OSM", aid, n, flush=True)
        time.sleep(8)
    rec = {"tried": todo, "n_new_objects": len(new_rows), "airports_with_objects": len({r["airport_id"] for r in new_rows})}
    if new_rows:
        extra = OUT / "strengthen5b_osm_new.csv"
        with extra.open("w", encoding="utf-8", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=list(new_rows[0].keys()))
            w.writeheader()
            w.writerows(new_rows)
    (OUT / "strengthen5b_osm.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("OSM", rec, flush=True)
    return rec, new_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nli, hybrid_rows = run_hybrid()
    sdr_rows = sdr_cell_sweep()
    cq1, lde = remark_proper_and_class()
    osm_rec, new_rows = osm_retry_lde(lde, limit=10)
    extra_osm = sum(1 for r in new_rows if r["airport_id"] in lde)
    summary = {
        "best_hybrid": sorted([r for r in hybrid_rows if r["airport"] == "POOLED" and r["trial"] == "hybrid_nli"], key=lambda x: (-x["f1"], -x["recall"]))[0],
        "deontic": next(r for r in hybrid_rows if r["airport"] == "POOLED" and r["trial"] == "deontic_lexicon"),
        "best_gif": sorted([r for r in hybrid_rows if r["airport"] == "GIF"], key=lambda x: (-x["f1"], -x["precision"]))[0],
        "best_sdr": sorted(sdr_rows, key=lambda x: (-x["f1"], -x["fill_rate"]))[0],
        "cq1": cq1,
        "osm": osm_rec,
        "cq1_with_new_osm": cq1["cq1_plus_class_proper"] + extra_osm,
    }
    (OUT / "strengthen5b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
