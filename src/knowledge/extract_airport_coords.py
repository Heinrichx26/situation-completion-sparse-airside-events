"""Median lat/lon for top-120 plus case-study airports from NWSD 2024."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw" / "faa_wildlife" / "faa_wildlife_export_2024.json"
TOP120 = PROJECT_ROOT / "data" / "metadata" / "top_nwsd_airports_120.txt"
OUT = PROJECT_ROOT / "results" / "experiments" / "aei_situation" / "airport_coords.csv"

CASE = ["SEA", "BJJ", "GIF", "0B5"]


def text(v) -> str:
    return "" if v is None else str(v).strip()


def normalize_airport(airport_id: str) -> str:
    code = text(airport_id).upper()
    if len(code) == 4 and code.startswith("K") and code[1:].isalpha():
        return code[1:]
    return code


def main() -> None:
    wanted = set(TOP120.read_text(encoding="utf-8").strip().split(","))
    wanted.update(CASE)
    payload = json.loads(RAW.read_text(encoding="utf-8-sig"))
    lats: dict[str, list[float]] = defaultdict(list)
    lons: dict[str, list[float]] = defaultdict(list)
    names: dict[str, str] = {}
    for raw in payload.get("Result", []):
        row = {k.strip().upper().replace(" ", "_"): v for k, v in raw.items()}
        aid = normalize_airport(row.get("AIRPORT_ID"))
        if aid not in wanted:
            continue
        try:
            lat = float(row.get("AIRPORT_LATITUDE") or 0)
            lon = float(row.get("AIRPORT_LONGITUDE") or 0)
        except (TypeError, ValueError):
            continue
        if abs(lat) < 1 or abs(lon) < 1:
            continue
        lats[aid].append(lat)
        lons[aid].append(lon)
        if row.get("AIRPORT"):
            names[aid] = text(row.get("AIRPORT"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for aid in sorted(wanted):
        if aid not in lats:
            rows.append({"airport_id": aid, "lat": "", "lon": "", "n": 0, "name": names.get(aid, ""), "in_nwsd_2024": False})
            continue
        lat_s = sorted(lats[aid])
        lon_s = sorted(lons[aid])
        mid = len(lat_s) // 2
        rows.append({
            "airport_id": aid,
            "lat": f"{lat_s[mid]:.6f}",
            "lon": f"{lon_s[mid]:.6f}",
            "n": len(lat_s),
            "name": names.get(aid, ""),
            "in_nwsd_2024": True,
        })
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["airport_id", "lat", "lon", "n", "name", "in_nwsd_2024"])
        writer.writeheader()
        writer.writerows(rows)
    found = sum(1 for r in rows if r["in_nwsd_2024"])
    print(f"wrote {OUT} airports={len(rows)} with_coords={found}")


if __name__ == "__main__":
    main()
