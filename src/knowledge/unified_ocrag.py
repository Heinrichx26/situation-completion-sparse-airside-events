"""Unified OC-RAG: one tau, no pathway-specific branches.

Trials
  T1 lexicon + action-cue sentences
  T2 MiniLM prototypes on all sentences
  T3 MiniLM prototypes on action-cue sentences only
  T4 same as T3 with BGE-small if available
Rare axioms: nd>=1 AND a retrieved chunk is from AC33/AC38 AND bind>=tau
Frequent: nd>=5 AND rate>=0.05 AND bind>=tau
No LARGE/ground special case.
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

ROOT = Path(__file__).resolve().parents[2]
KNOW = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiments" / "aei_situation"
WHMP = ROOT / "data" / "raw" / "whmp"
GOLD = json.loads((KNOW / "fixtures" / "whmp_actions.json").read_text(encoding="utf-8"))
GBIF = ROOT / "data" / "raw" / "gbif" / "airport_month_bird_counts.csv"
NTSB = ROOT / "results" / "experiments" / "supplemental_transparency" / "ntsb_stratified_audit_records.csv"
DOC_ATTR = KNOW / "fixtures" / "document_attractants.json"

TBOX_ACTION = {
    "DrainStandingWater": "The recommended action is to drain standing water, stormwater ponds, ditches, or wetlands.",
    "GrassHeightManagement": "The recommended action is to mow turf and manage airfield grass height.",
    "ExclusionFencing": "The recommended action is to install or maintain a wildlife exclusion fence and gates.",
    "WasteManagement": "The recommended action is to control landfills, putrescible waste, trash, and debris.",
    "LandUseAgreement": "The recommended action is a land-use agreement for agriculture, golf, or off-airport habitat.",
    "Harassment": "The recommended action is to harass, haze, or translocate hazardous wildlife.",
    "LethalControl": "The recommended action is lethal take, depredation, or trapping of hazardous wildlife.",
}
TBOX_ATTR = {
    "StandingWater": "hazardous attractant: pond, lake, creek, stormwater, open water",
    "Wetland": "hazardous attractant: wetland, marsh, swamp",
    "Grassland": "hazardous attractant: airfield grass and turf",
    "WasteFacility": "hazardous attractant: landfill, trash transfer, putrescible waste",
    "Agriculture": "hazardous attractant: farmland, crops, livestock, aquaculture",
    "PerchStructure": "hazardous attractant: perch, roost, fence, pole, building",
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
    ("LARGE", "ground", "landing_gear"): "R5",
    ("MEDIUM", "arrival", "wing_rotor"): "R6",
    ("LARGE", "arrival", "wing_rotor"): "R6",
}
CUE = re.compile(
    r"\b(recommend|shall|should|must|will|action|implement|maintain|install|mow|drain|"
    r"fence|hazing|harass|lethal|trap|depredation|protocol|mitigat|manage|remove|exclude)\b",
    re.I,
)
LEXICON = {
    "DrainStandingWater": [r"\bdrain", r"ditch", r"standing water", r"stormwater", r"detention", r"glycol pond"],
    "GrassHeightManagement": [r"\bmow", r"grass height", r"turf"],
    "ExclusionFencing": [r"\bfence", r"fencing", r"gate gap", r"perimeter fence"],
    "WasteManagement": [r"\btrash\b", r"debris", r"landfill", r"garbage", r"putrescible"],
    "LandUseAgreement": [r"golf course", r"land use", r"land-use", r"agriculture", r"farmland"],
    "Harassment": [r"harass", r"hazing", r"pyrotechnic", r"zero-tolerance", r"translocat"],
    "LethalControl": [r"lethal", r"depredation", r"\btrap", r"live-trapping"],
}


def f1(gold: set, pred: set):
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


def chunks_from(path: Path, src: str, size: int = 420, overlap: int = 80):
    raw = re.sub(r"\s+", " ", path.read_text(encoding="utf-8", errors="ignore")).strip()
    out, i = [], 0
    while i < len(raw):
        piece = raw[i:i + size]
        if len(piece) > 80:
            out.append({"src": src, "text": piece})
        i += size - overlap
    return out


def load_corpus():
    files = [
        (WHMP / "FAA_AC_150_5200_33C.txt", "AC33"),
        (WHMP / "FAA_AC_150_5200_38.txt", "AC38"),
        (WHMP / "SEA_WHMP_2022.txt", "SEA"),
        (WHMP / "GIF_WHSV_2018.txt", "GIF"),
        (WHMP / "BJJ_WHA_2014.txt", "BJJ"),
        (WHMP / "Montague_WHMP_2022.txt", "0B5"),
    ]
    corpus = []
    for path, src in files:
        if path.exists():
            corpus.extend(chunks_from(path, src))
    if not any(c["src"] == "AC38" for c in corpus) and (WHMP / "FAA_AC_150_5200_38.pdf").exists():
        import fitz
        blob = " ".join(page.get_text() for page in fitz.open(WHMP / "FAA_AC_150_5200_38.pdf"))
        (WHMP / "FAA_AC_150_5200_38.txt").write_text(blob, encoding="utf-8")
        corpus.extend(chunks_from(WHMP / "FAA_AC_150_5200_38.txt", "AC38"))
    return corpus


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T


def sentences(text_, limit=500):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_) if len(s.strip()) > 40][:limit]


def lexicon_on_text(blob: str, cue_only: bool) -> set[str]:
    sents = sentences(blob, 800)
    if cue_only:
        sents = [s for s in sents if CUE.search(s)]
    joined = " ".join(sents).lower()
    hit = set()
    for lab, pats in LEXICON.items():
        if any(re.search(p, joined, re.I) for p in pats):
            hit.add(lab)
    return hit


def proto_pred(sents, model, tbox, tau):
    if not sents:
        return set()
    s_emb = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    labs = list(tbox)
    t_emb = model.encode([tbox[k] for k in labs], convert_to_numpy=True, show_progress_bar=False)
    sim = cosine(s_emb, t_emb)
    return {labs[j] for i in range(sim.shape[0]) for j in range(sim.shape[1]) if sim[i, j] >= tau}


def main():
    docs = {
        "SEA": (WHMP / "SEA_WHMP_2022.txt").read_text(encoding="utf-8", errors="ignore"),
        "GIF": (WHMP / "GIF_WHSV_2018.txt").read_text(encoding="utf-8", errors="ignore"),
        "BJJ": (WHMP / "BJJ_WHA_2014.txt").read_text(encoding="utf-8", errors="ignore"),
        "0B5": (WHMP / "Montague_WHMP_2022.txt").read_text(encoding="utf-8", errors="ignore"),
    }
    gold = {k: {ln["action"] for ln in GOLD["airports"][k]["lines"]} for k in docs}

    print("encode", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        model_bge = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("bge ok", flush=True)
    except Exception as exc:
        model_bge = None
        print("no bge", exc, flush=True)

    rows = []
    # T1 lexicon all / cue
    for cue in (False, True):
        gset, pset = set(), set()
        for ap, blob in docs.items():
            pred = lexicon_on_text(blob, cue)
            p, r, f, tp, fp, fn = f1(gold[ap], pred)
            rows.append({"trial": f"lexicon_cue{int(cue)}", "tau": "", "airport": ap, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp})
            gset |= {f"{ap}:{x}" for x in gold[ap]}
            pset |= {f"{ap}:{x}" for x in pred}
        p, r, f, tp, fp, fn = f1(gset, pset)
        rows.append({"trial": f"lexicon_cue{int(cue)}", "tau": "", "airport": "POOLED", "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp})

    all_sents = {ap: sentences(blob) for ap, blob in docs.items()}
    cue_sents = {ap: [s for s in all_sents[ap] if CUE.search(s)] for ap in docs}

    def sweep_proto(name, model_, sent_map):
        for tau in (0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56):
            gset, pset = set(), set()
            for ap, sents in sent_map.items():
                pred = proto_pred(sents, model_, TBOX_ACTION, tau)
                p, r, f, tp, fp, fn = f1(gold[ap], pred)
                rows.append({"trial": name, "tau": tau, "airport": ap, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp})
                gset |= {f"{ap}:{x}" for x in gold[ap]}
                pset |= {f"{ap}:{x}" for x in pred}
            p, r, f, tp, fp, fn = f1(gset, pset)
            rows.append({"trial": name, "tau": tau, "airport": "POOLED", "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp})

    sweep_proto("minilm_all", model, all_sents)
    sweep_proto("minilm_cue", model, cue_sents)
    if model_bge is not None:
        sweep_proto("bge_cue", model_bge, cue_sents)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "unified_action_f1.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("BEST POOLED")
    pooled = [r for r in rows if r["airport"] == "POOLED"]
    for r in sorted(pooled, key=lambda x: (-x["f1"], -x["recall"])):
        print(r)

    # --- axioms unified ---
    corpus = load_corpus()
    print("chunks", len(corpus), "ac38", sum(1 for c in corpus if c["src"] == "AC38"), flush=True)
    chunk_texts = [c["text"] for c in corpus]
    chunk_src = [c["src"] for c in corpus]
    chunk_emb = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    chunk_emb = chunk_emb / (np.linalg.norm(chunk_emb, axis=1, keepdims=True) + 1e-8)
    attr_labs = list(TBOX_ATTR)
    act_labs = list(TBOX_ACTION)
    attr_emb = model.encode([TBOX_ATTR[k] for k in attr_labs], convert_to_numpy=True, show_progress_bar=False)
    act_emb = model.encode([TBOX_ACTION[k] for k in act_labs], convert_to_numpy=True, show_progress_bar=False)
    attr_emb = attr_emb / (np.linalg.norm(attr_emb, axis=1, keepdims=True) + 1e-8)
    act_emb = act_emb / (np.linalg.norm(act_emb, axis=1, keepdims=True) + 1e-8)

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

    axiom_rows = []
    chosen_rules = None
    for tau in (0.30, 0.34, 0.38, 0.42):
        rules = []
        for cell, st in cells.items():
            nd, n = st["nd"], st["n"]
            rate = nd / n if n else 0
            q = f"{cell[0]} wildlife {cell[1]} {cell[2]} hazardous attractant recommended action"
            qe = model.encode([q], convert_to_numpy=True, show_progress_bar=False)
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            scores = (chunk_emb @ qe.T).ravel()
            top = np.argsort(scores)[-12:][::-1]
            top_src = [chunk_src[i] for i in top]
            top_txt = [chunk_texts[i] for i in top]
            te = model.encode(top_txt, convert_to_numpy=True, show_progress_bar=False)
            te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-8)
            sim_a = (te @ attr_emb.T).max(axis=0)
            sim_c = (te @ act_emb.T).max(axis=0)
            bound_attr = [attr_labs[i] for i, s in enumerate(sim_a) if s >= tau]
            bound_act = [act_labs[i] for i, s in enumerate(sim_c) if s >= tau]
            ac_hit = any(s in {"AC33", "AC38"} for s in top_src[:8])
            frequent = nd >= 5 and rate >= 0.05
            rare_ok = nd >= 1 and ac_hit
            if not ((frequent or rare_ok) and (bound_attr or bound_act)):
                continue
            actions = set(bound_act)
            for a in bound_attr:
                actions.update(ATTR_ACTION.get(a, []))
            if rate >= 0.10:
                actions.add("LethalControl")
            actions.add("Harassment")
            attractants = bound_attr or ["Grassland"]
            rules.append({
                "id": f"U_{cell[0]}_{cell[1]}_{cell[2]}",
                "size": cell[0], "phase": cell[1], "component": cell[2],
                "requires_damage": True, "attractants": attractants,
                "actions": sorted(actions), "n": n, "n_damaged": nd,
                "damage_rate": round(rate, 4), "rare": nd < 5,
                "ac_hit": ac_hit, "tau": tau,
            })
        idx = {(r["size"], r["phase"], r["component"]): r for r in rules}
        recovered = sorted({GOLD_CELLS[c] for c in GOLD_CELLS if c in idx})
        pw = 0
        pw_air = set()
        lde = set()
        cq5 = set()
        for aid, size, phase, component, damaged in events:
            r = idx.get((size, phase, component))
            if r is None or not damaged:
                continue
            pw += 1
            pw_air.add(aid)
            if size == "LARGE" and phase == "departure" and component == "engine":
                lde.add(aid)
            if "DrainStandingWater" in r["actions"] and "StandingWater" in r.get("attractants", []):
                cq5.add(aid)
        osm = load_osm()
        doc_attr = json.loads(DOC_ATTR.read_text(encoding="utf-8"))
        gbif = set()
        with GBIF.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("gbif_query_ok") == "1" and float(row.get("gbif_bird_occurrences") or 0) > 0:
                    gbif.add(row["airport_id"].upper())
        cq1 = 0
        for aid in lde:
            cq1 += len(osm.get(aid, [])) + len(doc_attr["airports"].get(aid, []))
            if aid in gbif:
                cq1 += 1
        rec = {
            "tau": tau, "n_rules": len(rules), "n_rare": sum(1 for r in rules if r["rare"]),
            "gold": ",".join(recovered), "n_gold": len(set(recovered)), "r5": int("R5" in recovered),
            "pathway_events": pw, "pathway_airports": len(pw_air), "cq1_lde": cq1, "cq5": len(cq5),
        }
        axiom_rows.append(rec)
        print("AXIOM", rec, flush=True)
        if tau == 0.34:
            chosen_rules = rules
            (OUT / "unified_rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")

    with (OUT / "unified_axiom_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(axiom_rows[0].keys()))
        w.writeheader()
        w.writerows(axiom_rows)

    ntsb_n = ntsb_hit = 0
    if chosen_rules and NTSB.exists():
        idx = {(r["size"], r["phase"], r["component"]): r for r in chosen_rules}
        with NTSB.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ntsb_n += 1
                size = (row.get("size") or "UNKNOWN").upper()
                phase = (row.get("phase") or "unknown").lower()
                if phase in {"takeoff", "take-off", "climb", "departure"}:
                    phase = "departure"
                elif phase in {"landing", "approach", "descent", "arrival"}:
                    phase = "arrival"
                elif "ground" in phase or "taxi" in phase:
                    phase = "ground"
                comp = (row.get("component") or "other").replace("-", "_").replace(" ", "_")
                if "wing" in comp or "rotor" in comp:
                    comp = "wing_rotor"
                if (size, phase, comp) in idx:
                    ntsb_hit += 1
        (OUT / "unified_ntsb.json").write_text(json.dumps({"n": ntsb_n, "hit": ntsb_hit, "rate": round(ntsb_hit / ntsb_n, 3) if ntsb_n else 0}), encoding="utf-8")
        print("NTSB", ntsb_n, ntsb_hit, flush=True)


if __name__ == "__main__":
    main()
