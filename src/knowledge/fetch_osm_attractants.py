"""Fetch OSM attractants in 10,000-ft boxes. Batch Overpass queries to avoid rate limits."""
from __future__ import annotations

import csv
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COORDS = PROJECT_ROOT / "results" / "experiments" / "aei_situation" / "airport_coords.csv"
CACHE = PROJECT_ROOT / "data" / "raw" / "osm_attractants"
OUT = PROJECT_ROOT / "results" / "experiments" / "aei_situation" / "osm_attractants.csv"

RADIUS_M = 3048
OVERPASS = "https://overpass-api.de/api/interpreter"
BATCH = 5
SLEEP = 8.0

TAG_MAP = {
    ("natural", "water"): "StandingWater",
    ("natural", "wetland"): "Wetland",
    ("water", "lake"): "StandingWater",
    ("water", "pond"): "StandingWater",
    ("water", "reservoir"): "StandingWater",
    ("landuse", "reservoir"): "StandingWater",
    ("landuse", "basin"): "StandingWater",
    ("landuse", "landfill"): "WasteFacility",
    ("amenity", "waste_transfer_station"): "WasteFacility",
    ("landuse", "farmland"): "Agriculture",
    ("leisure", "golf_course"): "Grassland",
}


def classify(tags: dict) -> str | None:
    for (k, v), cls in TAG_MAP.items():
        if tags.get(k) == v:
            return cls
    return None


def bbox(lat: float, lon: float) -> tuple[float, float, float, float]:
    dlat = RADIUS_M / 111320.0
    dlon = RADIUS_M / (111320.0 * max(math.cos(math.radians(lat)), 0.2))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def post_overpass(query: str) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(OVERPASS, data=data, headers={"User-Agent": "awhso-batch/0.2"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_coords() -> list[dict]:
    priority = ["SEA", "GIF", "BJJ", "0B5", "FWA", "ATL", "DEN", "ORD"]
    with COORDS.open("r", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("lat") and row.get("lon")]
    rank = {k: i for i, k in enumerate(priority)}
    rows.sort(key=lambda r: (rank.get(r["airport_id"], 99), r["airport_id"]))
    return rows


def elements_from_payload(payload: dict) -> list[dict]:
    out = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        cls = classify(tags)
        if not cls:
            continue
        if "lat" in el:
            elat, elon = float(el["lat"]), float(el["lon"])
        elif "center" in el:
            elat, elon = float(el["center"]["lat"]), float(el["center"]["lon"])
        else:
            continue
        out.append({
            "attractant_class": cls,
            "osm_type": el.get("type", ""),
            "osm_id": el.get("id", ""),
            "lat": elat,
            "lon": elon,
            "name": tags.get("name", ""),
        })
    return out


def assign(airports: list[dict], elements: list[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {a["airport_id"]: [] for a in airports}
    seen: dict[str, set] = {a["airport_id"]: set() for a in airports}
    for el in elements:
        best_aid = None
        best_d = RADIUS_M + 1
        for a in airports:
            d = haversine_m(float(a["lat"]), float(a["lon"]), el["lat"], el["lon"])
            if d < best_d:
                best_d = d
                best_aid = a["airport_id"]
        if best_aid is None:
            continue
        key = (el["attractant_class"], round(el["lat"], 5), round(el["lon"], 5))
        if key in seen[best_aid]:
            continue
        seen[best_aid].add(key)
        by[best_aid].append({
            "airport_id": best_aid,
            "attractant_class": el["attractant_class"],
            "osm_type": el["osm_type"],
            "osm_id": el["osm_id"],
            "lat": f"{el['lat']:.6f}",
            "lon": f"{el['lon']:.6f}",
            "name": el["name"],
        })
    return by


def fetch_batch(airports: list[dict]) -> dict[str, list[dict]]:
    parts = []
    for a in airports:
        s, w, n, e = bbox(float(a["lat"]), float(a["lon"]))
        parts.append(
            f'way["natural"="water"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});'
            f'way["natural"="wetland"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});'
            f'way["landuse"="landfill"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});'
            f'way["landuse"="basin"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});'
            f'way["leisure"="golf_course"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});'
        )
    query = "[out:json][timeout:30];(\n" + "\n".join(parts) + "\n);out center tags 800;"
    payload = post_overpass(query)
    return assign(airports, elements_from_payload(payload))


def write_cache(airport_id: str, rows: list[dict]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {"elements": [
        {
            "type": r["osm_type"],
            "id": r["osm_id"],
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "tags": {"name": r["name"], "_class": r["attractant_class"]},
        }
        for r in rows
    ], "awhso_rows": rows}
    (CACHE / f"{airport_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def read_cache(airport_id: str) -> list[dict] | None:
    path = CACHE / f"{airport_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "awhso_rows" in payload:
        return payload["awhso_rows"]
    rows = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        cls = tags.get("_class") or classify(tags)
        if not cls:
            continue
        if "lat" in el:
            elat, elon = el["lat"], el["lon"]
        elif "center" in el:
            elat, elon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        rows.append({
            "airport_id": airport_id,
            "attractant_class": cls,
            "osm_type": el.get("type", ""),
            "osm_id": el.get("id", ""),
            "lat": f"{float(elat):.6f}",
            "lon": f"{float(elon):.6f}",
            "name": tags.get("name", ""),
        })
    return rows


def main() -> None:
    coords = load_coords()
    pending = []
    done: dict[str, list[dict]] = {}
    for row in coords:
        cached = read_cache(row["airport_id"])
        if cached is not None:
            done[row["airport_id"]] = cached
        else:
            pending.append(row)
    print(f"cached={len(done)} pending={len(pending)}", flush=True)

    for i in range(0, len(pending), BATCH):
        chunk = pending[i:i + BATCH]
        ids = [c["airport_id"] for c in chunk]
        print(f"batch {i // BATCH + 1} {ids}", flush=True)
        try:
            assigned = fetch_batch(chunk)
            for aid, rows in assigned.items():
                write_cache(aid, rows)
                done[aid] = rows
                classes = sorted({r["attractant_class"] for r in rows})
                print(f"  {aid} objects={len(rows)} {classes}", flush=True)
        except Exception as exc:
            print(f"  FAIL {ids} {exc}", flush=True)
            time.sleep(SLEEP)
            continue
        time.sleep(SLEEP)

    all_rows = []
    coverage = []
    for row in coords:
        aid = row["airport_id"]
        items = done.get(aid, [])
        all_rows.extend(items)
        coverage.append({
            "airport_id": aid,
            "n_objects": len(items),
            "classes": "|".join(sorted({x["attractant_class"] for x in items})),
            "ok": aid in done,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["airport_id", "attractant_class", "osm_type", "osm_id", "lat", "lon", "name"])
        writer.writeheader()
        writer.writerows(all_rows)
    with OUT.with_name("osm_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["airport_id", "n_objects", "classes", "ok"])
        writer.writeheader()
        writer.writerows(coverage)
    n_ok = sum(1 for r in coverage if r["ok"] and int(r["n_objects"]) > 0)
    print(f"airports_with_attractant={n_ok}/{len(coverage)} objects={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
