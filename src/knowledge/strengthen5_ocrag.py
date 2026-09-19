"""Strengthen five remaining AEI review points.

Smoke first, then full:
  P1  ontology-constrained NLI (DeBERTa) vs lexicon / MiniLM
  P3  GIF precision via recommendation-sentence entailment
  P2  same-slice CQ1-LDE objects from typed strike remarks
  P4  second-domain SDR maintenance TBox, same extractors
No pathway-specific branch. No gold-label edits.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
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
from unified_ocrag import (  # noqa: E402
    ATTR_ACTION,
    CUE,
    GOLD_CELLS,
    LEXICON,
    TBOX_ACTION,
    TBOX_ATTR,
    chunks_from,
    cosine,
    f1,
    lexicon_on_text,
    load_corpus,
    proto_pred,
    sentences,
)

ROOT = Path(__file__).resolve().parents[2]
KNOW = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiments" / "aei_situation"
WHMP = ROOT / "data" / "raw" / "whmp"
GOLD = json.loads((KNOW / "fixtures" / "whmp_actions.json").read_text(encoding="utf-8"))
DOC_ATTR = json.loads((KNOW / "fixtures" / "document_attractants.json").read_text(encoding="utf-8"))
GBIF = ROOT / "data" / "raw" / "gbif" / "airport_month_bird_counts.csv"
SDR_DIR = ROOT / "data" / "raw" / "faa_sdr"
NTSB = ROOT / "results" / "experiments" / "supplemental_transparency" / "ntsb_stratified_audit_records.csv"

NLI_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

ACTION_HYP = {
    "DrainStandingWater": "The plan recommends draining standing water, stormwater, ponds, or ditches as a management action to implement.",
    "GrassHeightManagement": "The plan recommends mowing turf or managing airfield grass height as a management action to implement.",
    "ExclusionFencing": "The plan recommends installing, repairing, or maintaining a wildlife exclusion fence or gate as a management action to implement.",
    "WasteManagement": "The plan recommends controlling landfill, trash, garbage, or putrescible waste as a management action to implement.",
    "LandUseAgreement": "The plan recommends a land-use agreement, lease change, or agricultural coordination as a management action to implement.",
    "Harassment": "The plan recommends harassing, hazing, or translocating hazardous wildlife as a management action to implement.",
    "LethalControl": "The plan recommends lethal take, depredation, or trapping of hazardous wildlife as a management action to implement.",
}

ATTR_HYP = {
    "StandingWater": "This text names a pond, lake, creek, ditch, or other standing water as an attractant.",
    "Wetland": "This text names a wetland, marsh, or swamp as an attractant.",
    "Grassland": "This text names airfield grass, turf, or a golf course as an attractant.",
    "WasteFacility": "This text names a landfill, dump, or putrescible waste site as an attractant.",
    "Agriculture": "This text names farmland, crops, or livestock as an attractant.",
    "PerchStructure": "This text names a perch, roost, fence, pole, or building as an attractant.",
}

SDR_TBOX = {
    "Repair": "The corrective action is to repair the discrepant part.",
    "Replace": "The corrective action is to replace or swap the discrepant part.",
    "Inspect": "The corrective action is to inspect or examine the part.",
    "OperationalCheck": "The corrective action is an operational or functional check.",
    "Secure": "The corrective action is to secure, reroute, or reinstall a wire or fitting.",
}

SDR_HYP = {
    "Repair": "The maintenance record recommends repairing the discrepant part as the corrective action.",
    "Replace": "The maintenance record recommends replacing the discrepant part as the corrective action.",
    "Inspect": "The maintenance record recommends inspecting or examining the part as the corrective action.",
    "OperationalCheck": "The maintenance record recommends an operational or functional check as the corrective action.",
    "Secure": "The maintenance record recommends securing, rerouting, or reinstalling a wire or fitting as the corrective action.",
}

SDR_LEX = {
    "Repair": [r"\brepair", r"repaired", r"rework"],
    "Replace": [r"\breplac", r"swapped", r"removed and install"],
    "Inspect": [r"\binspect", r"examined", r"visual check"],
    "OperationalCheck": [r"ops check", r"operational check", r"function(?:al)? check", r"ops chk"],
    "Secure": [r"\bsecur", r"reroute", r"reinstall", r"re-install"],
}

REMARK_PAT = [
    (r"\b(lora lake|miller creek|green pond|lake pleasant|[A-Z][a-z]+ lake)\b", "StandingWater"),
    (r"\b(pond|ponds|lake|lakes|creek|ditch|detention|stormwater|standing water)\b", "StandingWater"),
    (r"\b(wetland|marsh|swamp)\b", "Wetland"),
    (r"\b(golf course|turf|airfield grass|grassland)\b", "Grassland"),
    (r"\b(landfill|dump|garbage|putrescible|transfer station)\b", "WasteFacility"),
    (r"\b(farm|farmland|crop|livestock|agriculture|hay field)\b", "Agriculture"),
    (r"\b(perch|roost|light pole)\b", "PerchStructure"),
]


def load_docs():
    docs = {
        "SEA": (WHMP / "SEA_WHMP_2022.txt").read_text(encoding="utf-8", errors="ignore"),
        "GIF": (WHMP / "GIF_WHSV_2018.txt").read_text(encoding="utf-8", errors="ignore"),
        "BJJ": (WHMP / "BJJ_WHA_2014.txt").read_text(encoding="utf-8", errors="ignore"),
        "0B5": (WHMP / "Montague_WHMP_2022.txt").read_text(encoding="utf-8", errors="ignore"),
    }
    gold = {k: {ln["action"] for ln in GOLD["airports"][k]["lines"]} for k in docs}
    return docs, gold


class NLI:
    def __init__(self, name=NLI_NAME):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        labels = {int(k): v for k, v in self.model.config.id2label.items()}
        self.ent_id = next(i for i, v in labels.items() if v.lower().startswith("entail"))

    @torch.no_grad()
    def entail_probs(self, premises: list[str], hypothesis: str, bs: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(premises), bs):
            batch = premises[i:i + bs]
            enc = self.tok(
                batch,
                [hypothesis] * len(batch),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            prob = torch.softmax(logits, dim=-1)[:, self.ent_id]
            out.append(prob.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def nli_pred(sents: list[str], nli: NLI, hyp: dict[str, str], tau: float, min_hits: int = 1) -> set[str]:
    if not sents:
        return set()
    hit = set()
    for lab, h in hyp.items():
        p = nli.entail_probs(sents, h)
        if int((p >= tau).sum()) >= min_hits:
            hit.add(lab)
    return hit


def score_docs(docs, gold, pred_fn, trial, tau=""):
    rows = []
    gset, pset = set(), set()
    per = {}
    for ap, blob in docs.items():
        pred = pred_fn(ap, blob)
        p, r, f, tp, fp, fn = f1(gold[ap], pred)
        rec = {
            "trial": trial, "tau": tau, "airport": ap,
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "tp": tp, "fp": fp, "fn": fn,
            "pred": "|".join(sorted(pred)),
        }
        rows.append(rec)
        per[ap] = rec
        gset |= {f"{ap}:{x}" for x in gold[ap]}
        pset |= {f"{ap}:{x}" for x in pred}
    p, r, f, tp, fp, fn = f1(gset, pset)
    rows.append({
        "trial": trial, "tau": tau, "airport": "POOLED",
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
        "tp": tp, "fp": fp, "fn": fn, "pred": "",
    })
    return rows, per


def smoke_nli(docs, gold, nli):
    print("SMOKE NLI", flush=True)
    cue = {ap: [s for s in sentences(blob, 800) if CUE.search(s)] for ap, blob in docs.items()}
    for ap in ("GIF", "SEA"):
        print(ap, "cue_sents", len(cue[ap]), "gold", sorted(gold[ap]), flush=True)
    rows, _ = score_docs(
        docs, gold,
        lambda ap, blob: nli_pred(cue[ap], nli, ACTION_HYP, 0.7, 1),
        "smoke_nli_cue", 0.7,
    )
    for r in rows:
        print("SMOKE", r, flush=True)
    gif = next(r for r in rows if r["airport"] == "GIF")
    sea = next(r for r in rows if r["airport"] == "SEA")
    ok = sea["recall"] >= 0.85 and gif["f1"] >= 0.6
    print("SMOKE_OK" if ok else "SMOKE_WEAK", flush=True)
    return ok, cue


def sdr_gold_from_ca(ca: str) -> set[str]:
    blob = ca.lower()
    hit = set()
    for lab, pats in SDR_LEX.items():
        if any(re.search(p, blob, re.I) for p in pats):
            hit.add(lab)
    return hit


def parse_sdr_ca(disc: str) -> tuple[str, str]:
    m = re.search(r"(?:C/A|CORRECTIVE ACTION)\s*:\s*(.*)$", disc, re.I | re.S)
    if not m:
        return disc, ""
    head = disc[:m.start()].strip()
    return head, m.group(1).strip()


def load_sdr_sample(n=2500, year="2024"):
    path = SDR_DIR / f"SDR-{year}.csv"
    rows = []
    with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.DictReader(handle):
            disc = row.get("Discrepancy") or ""
            head, ca = parse_sdr_ca(disc)
            if not ca or len(ca) < 20:
                continue
            gold = sdr_gold_from_ca(ca)
            if not gold:
                continue
            rows.append({
                "jasc": (row.get("JASCCode") or "").strip(),
                "nature": (row.get("NatureOfConditionA") or "").strip(),
                "disc": head[:400],
                "ca": ca[:400],
                "gold": gold,
            })
            if len(rows) >= n:
                break
    return rows


def run_sdr(nli, mini):
    sample = load_sdr_sample()
    print("SDR sample", len(sample), flush=True)
    rows = []
    # lexicon on C/A is circular with gold (gold is lexicon). Evaluate extractors on C/A
    # vs a stricter gold: types that appear as imperative verbs in C/A.
    # Split: gold from C/A lexicon; predict from DISC-only (event text, no action).
    # That tests situation completion: infer action from sparse discrepancy.
    # Also predict from C/A with NLI (document extraction transfer).
    for trial, use_ca, pred_fn in [
        ("sdr_lexicon_ca", True, lambda t: {lab for lab, pats in SDR_LEX.items() if any(re.search(p, t.lower(), re.I) for p in pats)}),
        ("sdr_lexicon_disc", False, lambda t: {lab for lab, pats in SDR_LEX.items() if any(re.search(p, t.lower(), re.I) for p in pats)}),
    ]:
        gset, pset = set(), set()
        for i, rec in enumerate(sample):
            src = rec["ca"] if use_ca else rec["disc"]
            pred = pred_fn(src)
            gset |= {f"{i}:{x}" for x in rec["gold"]}
            pset |= {f"{i}:{x}" for x in pred}
        p, r, f, tp, fp, fn = f1(gset, pset)
        rows.append({"trial": trial, "n": len(sample), "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp})
        print("SDR", rows[-1], flush=True)

    # NLI on C/A and on DISC
    for tau in (0.5, 0.6, 0.7):
        for trial, use_ca in [("sdr_nli_ca", True), ("sdr_nli_disc", False)]:
            gset, pset = set(), set()
            texts = [(rec["ca"] if use_ca else rec["disc"]) for rec in sample]
            # sentence-level: one blob per record
            for lab, h in SDR_HYP.items():
                probs = nli.entail_probs(texts, h, bs=16)
                for i, rec in enumerate(sample):
                    pred_hit = probs[i] >= tau
                    if lab in rec["gold"]:
                        gset.add(f"{i}:{lab}")
                    if pred_hit:
                        pset.add(f"{i}:{lab}")
            p, r, f, tp, fp, fn = f1(gset, pset)
            rec = {"trial": trial, "tau": tau, "n": len(sample), "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp}
            rows.append(rec)
            print("SDR", rec, flush=True)

    # MiniLM prototype on C/A
    tbox = SDR_TBOX
    for tau in (0.35, 0.45, 0.55):
        gset, pset = set(), set()
        for i, rec in enumerate(sample):
            pred = proto_pred([rec["ca"]], mini, tbox, tau)
            gset |= {f"{i}:{x}" for x in rec["gold"]}
            pset |= {f"{i}:{x}" for x in pred}
        p, r, f, tp, fp, fn = f1(gset, pset)
        rec = {"trial": "sdr_minilm_ca", "tau": tau, "n": len(sample), "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp}
        rows.append(rec)
        print("SDR", rec, flush=True)

    # Cell mining: (jasc, nature) -> gold actions on even indices, apply to odd (held-out)
    cells = defaultdict(lambda: defaultdict(int))
    for i, rec in enumerate(sample):
        if i % 2 == 0 and rec["jasc"]:
            for a in rec["gold"]:
                cells[(rec["jasc"], rec["nature"])][a] += 1
    gset, pset = set(), set()
    n_filled = 0
    for i, rec in enumerate(sample):
        if i % 2 == 0:
            continue
        key = (rec["jasc"], rec["nature"])
        pred = {a for a, c in cells[key].items() if c >= 2}
        if pred:
            n_filled += 1
        gset |= {f"{i}:{x}" for x in rec["gold"]}
        pset |= {f"{i}:{x}" for x in pred}
    p, r, f, tp, fp, fn = f1(gset, pset)
    rec = {
        "trial": "sdr_cell_transfer", "n": sum(1 for i in range(len(sample)) if i % 2),
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
        "tp": tp, "fp": fp, "filled": n_filled,
        "fill_rate": round(n_filled / max(1, sum(1 for i in range(len(sample)) if i % 2)), 3),
    }
    rows.append(rec)
    print("SDR", rec, flush=True)

    # Observed C/A rate vs completed fill on full 2024 file (counts only)
    path = SDR_DIR / "SDR-2024.csv"
    n = n_ca = 0
    with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.DictReader(handle):
            n += 1
            d = row.get("Discrepancy") or ""
            if re.search(r"(?:C/A|CORRECTIVE ACTION)\s*:", d, re.I):
                n_ca += 1
    rows.append({"trial": "sdr_archive_ca_rate", "n": n, "precision": "", "recall": round(n_ca / n, 3) if n else 0, "f1": "", "tp": n_ca, "fp": n - n_ca})
    print("SDR archive", n, n_ca, round(n_ca / n, 3) if n else 0, flush=True)
    return rows


def remark_objects():
    """Typed unique remark objects at target airports; CQ1-LDE count."""
    target = load_target_airports()
    osm = load_osm()
    gbif = set()
    if GBIF.exists():
        with GBIF.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("gbif_query_ok") == "1" and float(row.get("gbif_bird_occurrences") or 0) > 0:
                    gbif.add(row["airport_id"].upper())
    lde = set()
    objects = defaultdict(set)  # airport -> set of (class, name)
    n_remarks = 0
    for row in iter_nwsd_rows():
        aid = normalize_airport(row.get("AIRPORT_ID"))
        if aid not in target:
            continue
        size = size_guild(row)
        phase = phase_bucket(text(row.get("PHASE_OF_FLIGHT")))
        hits = component_hits(row)
        damaged_eng = any(c == "engine" and d for c, d in hits)
        if size == "LARGE" and phase == "departure" and damaged_eng:
            lde.add(aid)
        remarks = " ".join([text(row.get("REMARKS")), text(row.get("COMMENTS"))])
        if len(remarks) < 20:
            continue
        low = remarks.lower()
        if not any(tok in low for tok in ["pond", "lake", "wetland", "landfill", "grass", "golf", "farm", "ditch", "marsh", "dump", "perch", "creek"]):
            continue
        n_remarks += 1
        for pat, cls in REMARK_PAT:
            for m in re.finditer(pat, remarks, re.I):
                name = re.sub(r"\s+", " ", m.group(0)).strip().title()
                objects[aid].add((cls, name))
    cq1_base = 0
    cq1_plus = 0
    n_lde_remark = 0
    for aid in lde:
        base = len(osm.get(aid, [])) + len(DOC_ATTR["airports"].get(aid, []))
        if aid in gbif:
            base += 1
        extra = objects.get(aid, set())
        cq1_base += base
        cq1_plus += base + len(extra)
        if extra:
            n_lde_remark += 1
    n_obj = sum(len(v) for v in objects.values())
    rec = {
        "n_lde_airports": len(lde),
        "cq1_lde_osm_doc_gbif": cq1_base,
        "cq1_lde_plus_remarks": cq1_plus,
        "remark_objects_all_target": n_obj,
        "airports_with_remark_objects": sum(1 for v in objects.values() if v),
        "lde_airports_with_remark_objects": n_lde_remark,
        "n_remarks_scanned": n_remarks,
    }
    print("CQ1", rec, flush=True)
    (OUT / "strengthen5_cq1_remarks.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    # save a sample
    sample = []
    for aid, items in list(objects.items())[:40]:
        for cls, name in sorted(items)[:8]:
            sample.append({"airport": aid, "class": cls, "name": name, "lde": aid in lde})
    with (OUT / "strengthen5_remark_objects.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["airport", "class", "name", "lde"])
        w.writeheader()
        w.writerows(sample)
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    docs, gold = load_docs()
    print("load nli", flush=True)
    nli = NLI()
    print("nli ok", nli.device, flush=True)
    ok, cue = smoke_nli(docs, gold, nli)
    print("load minilm", flush=True)
    mini = SentenceTransformer("all-MiniLM-L6-v2")

    all_rows = []
    # lexicon baselines
    for cue_only in (False, True):
        rows, _ = score_docs(
            docs, gold,
            lambda ap, blob, c=cue_only: lexicon_on_text(blob, c),
            f"lexicon_cue{int(cue_only)}",
        )
        all_rows.extend(rows)

    # MiniLM cue sweep (short)
    for tau in (0.40, 0.48, 0.56):
        rows, _ = score_docs(
            docs, gold,
            lambda ap, blob, t=tau: proto_pred(cue[ap], mini, TBOX_ACTION, t),
            "minilm_cue", tau,
        )
        all_rows.extend(rows)

    # NLI cue sweep
    for tau in (0.50, 0.60, 0.70, 0.80, 0.90):
        for min_hits in (1, 2):
            rows, per = score_docs(
                docs, gold,
                lambda ap, blob, t=tau, m=min_hits: nli_pred(cue[ap], nli, ACTION_HYP, t, m),
                f"nli_cue_h{min_hits}", tau,
            )
            all_rows.extend(rows)
            pooled = next(r for r in rows if r["airport"] == "POOLED")
            gif = next(r for r in rows if r["airport"] == "GIF")
            print(f"NLI tau={tau} h={min_hits} pooled_f1={pooled['f1']} gif_f1={gif['f1']} sea_r={next(r for r in rows if r['airport']=='SEA')['recall']}", flush=True)

    # NLI on all sentences (no cue) at best-looking tau later; run 0.7
    all_sents = {ap: sentences(blob, 800) for ap, blob in docs.items()}
    for tau in (0.70, 0.85):
        rows, _ = score_docs(
            docs, gold,
            lambda ap, blob, t=tau: nli_pred(all_sents[ap], nli, ACTION_HYP, t, 1),
            "nli_all", tau,
        )
        all_rows.extend(rows)

    with (OUT / "strengthen5_action_f1.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print("BEST POOLED", flush=True)
    pooled = [r for r in all_rows if r["airport"] == "POOLED"]
    for r in sorted(pooled, key=lambda x: (-x["f1"], -x["recall"], -x.get("precision", 0)))[:12]:
        print(r, flush=True)
    print("GIF rows", flush=True)
    for r in sorted([r for r in all_rows if r["airport"] == "GIF"], key=lambda x: (-x["f1"], -x["precision"]))[:8]:
        print(r, flush=True)

    print("CQ1 remarks", flush=True)
    cq1 = remark_objects()

    print("SDR domain", flush=True)
    sdr_rows = run_sdr(nli, mini)
    with (OUT / "strengthen5_sdr.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=sorted({k for r in sdr_rows for k in r}))
        w.writeheader()
        w.writerows(sdr_rows)

    summary = {
        "smoke_ok": ok,
        "best_pooled": sorted(pooled, key=lambda x: (-x["f1"], -x["recall"]))[0],
        "best_gif": sorted([r for r in all_rows if r["airport"] == "GIF"], key=lambda x: (-x["f1"], -x["precision"]))[0],
        "cq1": cq1,
    }
    (OUT / "strengthen5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
