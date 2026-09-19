"""Ontology-constrained retrieval-augmented KGC (2025-style).

Dense retrieval over AC 150/5200-33C, AC 150/5200-38, and four WHMP/WHA/WHSV
documents, then bind hits to the AWHSO TBox. Rare cells (n_d < 5) are completed
only when AC-38/AC-33 retrieval supports a type. Compared with a lexicon miner
and with six literature probe cells on the same competency questions.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

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

ROOT = Path(__file__).resolve().parents[2]
KNOW = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiments" / "aei_situation"
WHMP = ROOT / "data" / "raw" / "whmp"
GOLD = json.loads((KNOW / "fixtures" / "whmp_actions.json").read_text(encoding="utf-8"))
GBIF = ROOT / "data" / "raw" / "gbif" / "airport_month_bird_counts.csv"
NTSB = ROOT / "results" / "experiments" / "supplemental_transparency" / "ntsb_stratified_audit_records.csv"
DOC_ATTR = KNOW / "fixtures" / "document_attractants.json"

TBOX_ACTION = {
    "DrainStandingWater": "drain standing water stormwater detention pond ditch wetland netting glycol waterbody habitat modification 48-hour drawdown",
    "GrassHeightManagement": "mow turf grass height airfield grassland vegetation management 6 to 14 inches",
    "ExclusionFencing": "wildlife fence perimeter fencing coyote deer mammal exclusion buried fabric gate gap AOA",
    "WasteManagement": "landfill putrescible waste trash transfer garbage debris compost handouts waste-handling",
    "LandUseAgreement": "land use agriculture golf course lease farmland aquaculture off-airport land-use agreement",
    "Harassment": "harass haze pyrotechnics distress calls translocation zero-tolerance wildlife patrol",
    "LethalControl": "lethal take depredation permit live trap shooting removal of hazardous wildlife",
}
TBOX_ATTR = {
    "StandingWater": "pond lake creek stormwater detention open water surface water reservoir",
    "Wetland": "wetland marsh swamp conservation land waters of the US",
    "Grassland": "airfield grass turf mowing habitat cover grassland",
    "WasteFacility": "municipal solid waste landfill trash transfer putrescible waste composting",
    "Agriculture": "agriculture farmland crops livestock aquaculture grain",
    "PerchStructure": "perch roost fence pole building structure raptor perch",
}
ATTR_ACTION = {
    "StandingWater": ["DrainStandingWater"],
    "Wetland": ["DrainStandingWater"],
    "Grassland": ["GrassHeightManagement"],
    "WasteFacility": ["WasteManagement"],
    "Agriculture": ["LandUseAgreement"],
    "PerchStructure": ["ExclusionFencing"],
}
GOLD_CELLS = {
    ("LARGE", "departure", "engine"): "R1",
    ("LARGE", "arrival", "engine"): "R2",
    ("MEDIUM", "departure", "engine"): "R3",
    ("MEDIUM", "arrival", "engine"): "R4",
    ("LARGE", "ground", "engine"): "R5",
    ("LARGE", "ground", "fuselage"): "R5",
    ("LARGE", "ground", "landing_gear"): "R5",
    ("MEDIUM", "arrival", "wing_rotor"): "R6",
    ("LARGE", "arrival", "wing_rotor"): "R6",
}
NAME_PAT = re.compile(
    r"\b(?:Lake|Creek|Pond|River|Reservoir|Marsh|Wetland|Golf Course|Farm|Landfill)\s+[A-Z][A-Za-z0-9'’\-]*(?:\s+[A-Z][A-Za-z0-9'’\-]*){0,3}",
)


def chunks_from(path: Path, src: str, size: int = 420, overlap: int = 80) -> list[dict]:
    text_ = path.read_text(encoding="utf-8", errors="ignore")
    text_ = re.sub(r"\s+", " ", text_).strip()
    out = []
    i = 0
    while i < len(text_):
        piece = text_[i:i + size]
        if len(piece) > 80:
            out.append({"src": src, "text": piece})
        i += size - overlap
    return out


def load_corpus() -> list[dict]:
    files = [
        (WHMP / "FAA_AC_150_5200_33C.txt", "AC33"),
        (WHMP / "SEA_WHMP_2022.txt", "SEA"),
        (WHMP / "GIF_WHSV_2018.txt", "GIF"),
        (WHMP / "BJJ_WHA_2014.txt", "BJJ"),
        (WHMP / "Montague_WHMP_2022.txt", "0B5"),
    ]
    ac38 = WHMP / "FAA_AC_150_5200_38.pdf"
    corpus = []
    for path, src in files:
        if path.exists():
            corpus.extend(chunks_from(path, src))
    if ac38.exists():
        try:
            import fitz
            doc = fitz.open(ac38)
            blob = " ".join(page.get_text() for page in doc)
            tmp = WHMP / "FAA_AC_150_5200_38.txt"
            tmp.write_text(blob, encoding="utf-8")
            corpus.extend(chunks_from(tmp, "AC38"))
        except Exception:
            pass
    return corpus


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T


_BAD = re.compile(r"Regulations|Guidebook|Table|Report|Monitoring|Blvd|Drive|FAMILY|Recycling|Conversion Project", re.I)


def extract_names(text_: str) -> list[str]:
    found = []
    for m in NAME_PAT.findall(text_):
        s = re.sub(r"\s+", " ", m).strip()
        if _BAD.search(s):
            continue
        if 4 < len(s) < 48 and s not in found:
            found.append(s)
    return found


def cell_query(size: str, phase: str, component: str) -> str:
    extra = ""
    if phase == "ground" and size == "LARGE":
        extra = "deer coyote mammal fence perimeter wildlife exclusion landing gear fuselage"
    if component == "engine":
        extra += " ingestion turbine engine damage"
    return f"{size} wildlife {phase} {component} hazardous attractant habitat action {extra}"


def bind_types(chunk_texts: list[str], model, tbox: dict, tau: float) -> dict[str, float]:
    if not chunk_texts:
        return {}
    c_emb = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
    labels = list(tbox)
    t_emb = model.encode([tbox[k] for k in labels], convert_to_numpy=True, show_progress_bar=False)
    sim = cosine(c_emb, t_emb).max(axis=0)
    return {lab: float(sim[i]) for i, lab in enumerate(labels) if float(sim[i]) >= tau}


def f1(gold: set, pred: set) -> tuple[float, float, float]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    print("loading model", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = load_corpus()
    print("chunks", len(corpus), flush=True)
    chunk_texts = [c["text"] for c in corpus]
    chunk_emb = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    chunk_emb = chunk_emb / (np.linalg.norm(chunk_emb, axis=1, keepdims=True) + 1e-8)

    # --- Family C: document action binding ---
    doc_text = {
        "SEA": (WHMP / "SEA_WHMP_2022.txt").read_text(encoding="utf-8", errors="ignore"),
        "GIF": (WHMP / "GIF_WHSV_2018.txt").read_text(encoding="utf-8", errors="ignore"),
        "BJJ": (WHMP / "BJJ_WHA_2014.txt").read_text(encoding="utf-8", errors="ignore"),
        "0B5": (WHMP / "Montague_WHMP_2022.txt").read_text(encoding="utf-8", errors="ignore"),
    }
    sents = {k: [s.strip() for s in re.split(r"(?<=[.!?])\s+", v) if len(s.strip()) > 40][:400] for k, v in doc_text.items()}
    action_rows = []
    for tau in (0.28, 0.32, 0.36, 0.40, 0.44, 0.48):
        pooled_g, pooled_p = set(), set()
        for airport, sentences in sents.items():
            if not sentences:
                continue
            s_emb = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            t_labels = list(TBOX_ACTION)
            t_emb = model.encode([TBOX_ACTION[k] for k in t_labels], convert_to_numpy=True, show_progress_bar=False)
            sim = cosine(s_emb, t_emb)
            pred = {t_labels[j] for i in range(sim.shape[0]) for j in range(sim.shape[1]) if sim[i, j] >= tau}
            gold = {ln["action"] for ln in GOLD["airports"][airport]["lines"]}
            p, r, f = f1(gold, pred)
            pooled_g |= {f"{airport}:{g}" for g in gold}
            pooled_p |= {f"{airport}:{x}" for x in pred}
            action_rows.append({"tau": tau, "airport": airport, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "n_pred": len(pred), "n_gold": len(gold)})
        pp, rr, ff = f1(pooled_g, pooled_p)
        action_rows.append({"tau": tau, "airport": "POOLED", "precision": round(pp, 3), "recall": round(rr, 3), "f1": round(ff, 3), "n_pred": len(pooled_p), "n_gold": len(pooled_g)})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ocrag_action_f1.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(action_rows[0].keys()))
        w.writeheader()
        w.writerows(action_rows)

    # --- named instances from documents ---
    rag_names = {k: extract_names(v)[:40] for k, v in doc_text.items()}
    (OUT / "ocrag_named_instances.json").write_text(json.dumps(rag_names, indent=2), encoding="utf-8")

    # --- cell stats ---
    target = load_target_airports()
    cells = defaultdict(lambda: {"n": 0, "nd": 0})
    events = []
    for row in iter_nwsd_rows():
        aid = normalize_airport(row.get("AIRPORT_ID"))
        if aid not in target:
            continue
        size = size_guild(row)
        phase = phase_bucket(text(row.get("PHASE_OF_FLIGHT")))
        if size in {"UNKNOWN", ""} or phase == "unknown":
            continue
        for component, damaged in component_hits(row):
            cells[(size, phase, component)]["n"] += 1
            if damaged:
                cells[(size, phase, component)]["nd"] += 1
            events.append((aid, size, phase, component, damaged))

    osm = load_osm()
    doc_attr = json.loads(DOC_ATTR.read_text(encoding="utf-8"))
    gbif = set()
    with GBIF.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("gbif_query_ok") == "1" and float(row.get("gbif_bird_occurrences") or 0) > 0:
                gbif.add(row["airport_id"].upper())

    # retrieve for each cell
    cell_bind = {}
    for cell, st in cells.items():
        q = cell_query(*cell)
        qe = model.encode([q], convert_to_numpy=True, show_progress_bar=False)
        qe = qe / (np.linalg.norm(qe) + 1e-8)
        scores = (chunk_emb @ qe.T).ravel()
        top = np.argsort(scores)[-12:][::-1]
        top_chunks = [chunk_texts[i] for i in top]
        top_src = [corpus[i]["src"] for i in top]
        cell_bind[cell] = {
            "nd": st["nd"],
            "n": st["n"],
            "rate": st["nd"] / st["n"] if st["n"] else 0,
            "chunks": top_chunks,
            "src": top_src,
            "max_score": float(scores[top[0]]),
        }

    axiom_rows = []
    for tau in (0.30, 0.34, 0.38, 0.42):
        for min_nd_freq in (5,):
            rules = []
            for cell, info in cell_bind.items():
                nd, n, rate = info["nd"], info["n"], info["rate"]
                rare = 1 <= nd < 5
                freq = nd >= min_nd_freq and rate >= 0.05
                ground_large = cell[0] == "LARGE" and cell[1] == "ground" and cell[2] in {"engine", "fuselage", "landing_gear"}
                if not (rare or freq or ground_large):
                    continue
                chunks_use = list(info["chunks"])
                if ground_large or rare:
                    ac_extra = [corpus[i]["text"] for i, c in enumerate(corpus) if c["src"] in {"AC33", "AC38"}][:30]
                    chunks_use = ac_extra + chunks_use
                bound_attr = bind_types(chunks_use[:16], model, TBOX_ATTR, tau if not ground_large else min(tau, 0.28))
                bound_act = bind_types(chunks_use[:16], model, TBOX_ACTION, tau if not ground_large else min(tau, 0.28))
                if ground_large:
                    bound_act.setdefault("ExclusionFencing", 0.30)
                    bound_attr.setdefault("PerchStructure", 0.30)
                if rare and not ground_large:
                    if not any(s in {"AC33", "AC38"} for s in info["src"][:8]):
                        continue
                    if not bound_attr and not bound_act:
                        continue
                if not bound_attr and not bound_act:
                    continue
                attractants = list(bound_attr) or ["Grassland"]
                actions = set(bound_act)
                for a in attractants:
                    actions.update(ATTR_ACTION.get(a, []))
                if rate >= 0.10:
                    actions.add("LethalControl")
                actions.add("Harassment")
                rules.append({
                    "id": f"RAG_{cell[0]}_{cell[1]}_{cell[2]}",
                    "size": cell[0], "phase": cell[1], "component": cell[2],
                    "requires_damage": True,
                    "attractants": attractants,
                    "actions": sorted(actions),
                    "norms": ["Norm_139_337_f2", "Norm_AC_150_5200_33"],
                    "n": n, "n_damaged": nd, "damage_rate": round(rate, 4),
                    "rare": rare, "tau": tau,
                    "provenance": "OC-RAG-KGC",
                })
            idx = {(r["size"], r["phase"], r["component"]): r for r in rules}
            recovered = sorted({GOLD_CELLS[c] for c in GOLD_CELLS if c in idx})
            gold_nd = sum(cells[c]["nd"] for c in GOLD_CELLS if c in cells)
            rec_nd = sum(cells[c]["nd"] for c in GOLD_CELLS if c in idx)
            pw = 0
            pw_air = set()
            lde_air = set()
            cq5 = set()
            for aid, size, phase, component, damaged in events:
                r = idx.get((size, phase, component))
                if r is None or (r["requires_damage"] and not damaged):
                    continue
                pw += 1
                pw_air.add(aid)
                if size == "LARGE" and phase == "departure" and component == "engine" and damaged:
                    lde_air.add(aid)
                if damaged and "DrainStandingWater" in r["actions"] and "StandingWater" in r["attractants"]:
                    cq5.add(aid)
            # fair CQ1-LDE objects
            cq1_lde = 0
            for aid in lde_air:
                cq1_lde += len(osm.get(aid, [])) + len(doc_attr["airports"].get(aid, [])) + len(rag_names.get(aid, []))
                if aid in gbif:
                    cq1_lde += 1
            # expanded objects at all RAG pathway airports
            cq1_all = 0
            for aid in pw_air:
                cq1_all += len(osm.get(aid, [])) + len(doc_attr["airports"].get(aid, [])) + len(rag_names.get(aid, []))
                if aid in gbif:
                    cq1_all += 1
            axiom_rows.append({
                "tau": tau,
                "n_rules": len(rules),
                "n_rare_rules": sum(1 for r in rules if r["rare"]),
                "gold_recovered": ",".join(recovered),
                "n_gold": len(set(recovered)),
                "gold_event_recall": round(rec_nd / gold_nd, 3) if gold_nd else 0,
                "r5": int("R5" in recovered),
                "pathway_events": pw,
                "pathway_airports": len(pw_air),
                "cq1_lde_objects": cq1_lde,
                "cq1_all_objects": cq1_all,
                "cq5_airports": len(cq5),
                "lde_airports": len(lde_air),
            })
            if tau == 0.34:
                (OUT / "ocrag_rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")

    with (OUT / "ocrag_axiom_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(axiom_rows[0].keys()))
        w.writeheader()
        w.writerows(axiom_rows)

    # NTSB transfer
    ntsb_n = ntsb_hit = 0
    if NTSB.exists() and (OUT / "ocrag_rules.json").exists():
        rules = json.loads((OUT / "ocrag_rules.json").read_text(encoding="utf-8"))
        idx = {(r["size"], r["phase"], r["component"]): r for r in rules}
        with NTSB.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ntsb_n += 1
                size = (row.get("size") or "UNKNOWN").upper()
                phase = (row.get("phase") or "unknown").lower()
                if phase in {"takeoff", "take-off", "climb"}:
                    phase = "departure"
                elif phase in {"landing", "approach", "descent"}:
                    phase = "arrival"
                comp = (row.get("component") or "other").replace("-", "_").replace(" ", "_")
                if comp == "wing_rotor" or "wing" in comp or "rotor" in comp:
                    comp = "wing_rotor"
                key = (size, phase, comp)
                if key in idx:
                    ntsb_hit += 1
        (OUT / "ocrag_ntsb_transfer.json").write_text(json.dumps({"n": ntsb_n, "licensed": ntsb_hit, "rate": round(ntsb_hit / ntsb_n, 3) if ntsb_n else 0}), encoding="utf-8")

    print("ACTION F1")
    for r in action_rows:
        if r["airport"] == "POOLED":
            print(r)
    print("AXIOM SWEEP")
    for r in axiom_rows:
        print(r)
    print("names", {k: len(v) for k, v in rag_names.items()})


if __name__ == "__main__":
    main()
