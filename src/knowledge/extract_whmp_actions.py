"""Lexicon extraction of WHMP action types from public plan text (Family C)."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
KNOWLEDGE = Path(__file__).resolve().parent
WHMP_DIR = PROJECT / "data" / "raw" / "whmp"
GOLD = json.loads((KNOWLEDGE / "fixtures" / "whmp_actions.json").read_text(encoding="utf-8"))
OUT = PROJECT / "results" / "experiments" / "aei_situation"

LEXICON = {
    "DrainStandingWater": [
        r"\bdrain", r"ditch", r"standing water", r"stormwater", r"detention",
        r"glycol pond", r"open water", r"wetland modification", r"netting",
        r"floating ball", r"48-hour", r"48 hour",
    ],
    "GrassHeightManagement": [
        r"\bmow", r"grass height", r"turf", r"cut airfield grass", r"tall grass",
        r"7-10 inch", r"7 to 10", r"2 and 10 inch", r"2–10",
    ],
    "ExclusionFencing": [
        r"\bfence", r"fencing", r"gate gap", r"perimeter fence", r"coyote-deterrent",
        r"wildlife fence", r"exclusion gate",
    ],
    "WasteManagement": [
        r"\btrash\b", r"debris", r"handout", r"waste", r"landfill", r"garbage",
        r"prey-base", r"prey base",
    ],
    "LandUseAgreement": [
        r"golf course", r"land use", r"land-use", r"lease", r"agriculture",
        r"grain crop", r"hay production", r"water department",
    ],
    "Harassment": [
        r"harass", r"hazing", r"pyrotechnic", r"zero-tolerance", r"zero tolerance",
        r"translocat", r"distress call",
    ],
    "LethalControl": [
        r"lethal", r"live-trapping", r"live trapping", r"depredation", r"\btrap",
        r"take canada goose", r"shotgun",
    ],
}

DOCS = {
    "SEA": WHMP_DIR / "SEA_WHMP_2022.txt",
    "GIF": WHMP_DIR / "GIF_WHSV_2018.txt",
    "BJJ": WHMP_DIR / "BJJ_WHA_2014.txt",
    "0B5": WHMP_DIR / "Montague_WHMP_2022.txt",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def extract_types(text: str) -> dict[str, int]:
    blob = normalize(text)
    hits = {}
    for action, pats in LEXICON.items():
        n = 0
        for pat in pats:
            n += len(re.findall(pat, blob, flags=re.I))
        if n:
            hits[action] = n
    return hits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    type_rows = []
    for airport, path in DOCS.items():
        text = path.read_text(encoding="utf-8", errors="ignore")
        extracted = extract_types(text)
        gold_types = {line["action"] for line in GOLD["airports"][airport]["lines"]}
        extracted_types = set(extracted)
        tp = gold_types & extracted_types
        fp = extracted_types - gold_types
        fn = gold_types - extracted_types
        prec = len(tp) / len(extracted_types) if extracted_types else 0.0
        rec = len(tp) / len(gold_types) if gold_types else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({
            "airport_id": airport,
            "gold_types": "|".join(sorted(gold_types)),
            "extracted_types": "|".join(sorted(extracted_types)),
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "hit_counts": json.dumps(extracted, sort_keys=True),
        })
        for action in sorted(gold_types | extracted_types):
            type_rows.append({
                "airport_id": airport,
                "action": action,
                "in_gold": action in gold_types,
                "in_extract": action in extracted_types,
                "lexicon_hits": extracted.get(action, 0),
            })
    with (OUT / "family_c_extraction.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "family_c_extraction_types.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(type_rows[0].keys()))
        writer.writeheader()
        writer.writerows(type_rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
