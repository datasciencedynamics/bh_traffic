#!/usr/bin/env python
"""Pre-release check of the top-ranked locations.

Before the hotspot list goes to anyone, every location that makes a top 10
(under any ranking the app offers, for either outcome) should be checked by
eye: is the pin on the right corner, and is the location real rather than an
artifact of how BHPD spelled the address?

For each such location this writes one row with
    - its rank under each ranking and outcome
    - how many reports were merged into it, and the raw BHPD spellings
      (Location + From Street) behind them, most common first
    - the median and 90th-percentile distance from the intersection recorded
      on the reports (Feet From); large values mean a corridor, not a corner
    - its coordinates, where they came from (OSM or manual override), and
      the OSM spread when several nodes matched
    - flags for anything that needs a closer look (no or approximate
      coordinates, a pin far from the city, OSM matches far apart, mid-block,
      most reports far from the corner). Unflagged rows still need the
      two-link check; flags only say where to look first.
    - two links: the pin itself, and a Google Maps search for the address.
      They should land on the same corner.
    - an empty "checked_ok" column to fill in

Outputs (data/ is gitignored)
-------
    data/audit/top_locations_audit.csv
    data/audit/top_locations_audit.html   same table, with clickable links

Fix a wrong pin by adding a row (location_key, lat, lon) to
data/external/location_coords_manual.csv and re-running `make export_dash`.

Run after `make export_dash`, or via `make audit_top`.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import typer

from core.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from core.constants import location_key_var

app = typer.Typer(add_completion=False, help=__doc__)

## Beverly Hills city center (same as export_dash); pins farther out get flagged
BH_CENTER = (34.0736, -118.4004)
FAR_KM = 3.0
## Median report distance from the corner beyond which a "hotspot" is really a
## stretch of street (most intersection crashes sit within ~200 ft)
FAR_FEET = 250

## The rankings the app offers ("Rank by"), per outcome tag
RANKINGS = {
    "eb": "expected per year",
    "excess": "vs. similar",
    "obs": "last 12 months",
    "ml": "next 12 months",
}
OUTCOME_TAGS = {"inj": "all injury", "vru_inj": "ped/bike/moto"}


def _col(tag: str, ranking: str) -> str:
    return {
        "eb": f"eb_{tag}_per_year",
        "excess": f"excess_{tag}_per_year",
        "obs": f"obs_{tag}_last12",
        "ml": f"ml_{tag}_next12",
    }[ranking]


def _pretty(key: str) -> str:
    """'LA CIENEGA BLVD / WILSHIRE BLVD' -> 'La Cienega Blvd & Wilshire Blvd'."""
    return " & ".join(part.strip().title() for part in key.split("/"))


def _km(lat: float, lon: float) -> float:
    dy = (lat - BH_CENTER[0]) * 111.0
    dx = (lon - BH_CENTER[1]) * 111.0 * np.cos(np.radians(BH_CENTER[0]))
    return float(np.hypot(dx, dy))


@app.command()
def run(
    dash_dir: Path = typer.Option(PROCESSED_DATA_DIR.parent / "dash"),
    out_dir: Path = typer.Option(PROCESSED_DATA_DIR.parent / "audit"),
    top: int = typer.Option(10, "--top"),
    variants: int = typer.Option(4, "--variants", help="raw spellings to list"),
) -> None:
    """Write the top-location audit table (CSV and HTML)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hot = pd.read_csv(dash_dir / "hotspots.csv")
    meta = json.loads((dash_dir / "meta.json").read_text())

    ## Ranks under every ranking and outcome; keep any location in a top N
    ranks = pd.DataFrame({location_key_var: hot[location_key_var]})
    for tag, tag_label in OUTCOME_TAGS.items():
        for rk, rk_label in RANKINGS.items():
            c = _col(tag, rk)
            if c not in hot.columns:
                continue
            order = hot[c].rank(ascending=False, method="first").astype(int)
            ranks[f"rank {tag_label}, {rk_label}"] = order
    rank_cols = [c for c in ranks.columns if c.startswith("rank ")]
    ranks["best_rank"] = ranks[rank_cols].min(axis=1)
    keep = ranks[ranks["best_rank"] <= top].copy()

    ## Raw spellings and distances behind each key
    raw = pd.read_parquet(PROCESSED_DATA_DIR / "df.parquet").set_index("ObjectId")
    crashes = pd.read_parquet(PROCESSED_DATA_DIR / "df_sans_zero.parquet")
    crashes = crashes.set_index("object_id")[[location_key_var]]
    j = crashes.join(raw[["Location", "From Street", "Feet From"]])
    j = j[j[location_key_var].isin(keep[location_key_var])]
    j["spelling"] = (
        j["Location"].fillna("").str.split().str.join(" ")
        + " | "
        + j["From Street"].fillna("").str.split().str.join(" ")
    )
    j["feet"] = pd.to_numeric(j["Feet From"], errors="coerce")

    def spellings(g: pd.DataFrame) -> str:
        vc = g["spelling"].value_counts()
        shown = "; ".join(f"{s} ({n})" for s, n in vc.head(variants).items())
        more = len(vc) - variants
        return shown + (f"; +{more} more" if more > 0 else "")

    by_key = j.groupby(location_key_var)
    info = pd.DataFrame(
        {
            "reports": by_key.size(),
            "n_spellings": by_key["spelling"].nunique(),
            "raw_spellings": by_key[["spelling"]].apply(spellings),
            "feet_median": by_key["feet"].median(),
            "feet_p90": by_key["feet"].quantile(0.9),
        }
    ).reset_index()

    ## Coordinates and their source
    coords = hot[[location_key_var, "lat", "lon", "approx"]]
    cache = EXTERNAL_DATA_DIR / "location_coords.csv"
    manual = EXTERNAL_DATA_DIR / "location_coords_manual.csv"
    spread = (
        pd.read_csv(cache)[[location_key_var, "spread_m"]]
        if cache.exists()
        else pd.DataFrame(columns=[location_key_var, "spread_m"])
    )
    manual_keys = (
        set(pd.read_csv(manual)[location_key_var]) if manual.exists() else set()
    )

    out = (
        keep.merge(info, how="left")
        .merge(coords, how="left")
        .merge(spread, how="left")
        .sort_values(["best_rank", location_key_var])
    )
    out.insert(1, "location", out[location_key_var].map(_pretty))
    out["coord_source"] = np.where(
        out[location_key_var].isin(manual_keys),
        "manual",
        np.where(out["lat"].notna(), "OpenStreetMap", "none"),
    )

    ## Flags: anything a person should look at before trusting the pin
    def flags(r: pd.Series) -> str:
        f = []
        if pd.isna(r["lat"]):
            f.append("no coordinates")
        else:
            if _km(r["lat"], r["lon"]) > FAR_KM:
                f.append(f"pin {_km(r['lat'], r['lon']):.1f} km from city center")
            if bool(r.get("approx")):
                f.append("position approximate")
            if pd.notna(r.get("spread_m")) and r["spread_m"] > 60:
                f.append(f"OSM matched points {r['spread_m']:.0f} m apart")
        if str(r[location_key_var]).endswith("(MIDBLOCK)"):
            f.append("mid-block: a stretch of street, not a corner")
        if pd.notna(r["feet_median"]) and r["feet_median"] >= FAR_FEET:
            f.append(f"half the reports are {r['feet_median']:.0f}+ ft from the corner: "
                     "more a stretch of street than an intersection")
        return "; ".join(f)

    out["flags"] = out.apply(flags, axis=1).fillna("")
    out["pin_link"] = [
        f"https://www.google.com/maps?q={la:.6f},{lo:.6f}" if pd.notna(la) else ""
        for la, lo in zip(out["lat"], out["lon"])
    ]
    out["search_link"] = [
        "https://www.google.com/maps/search/?api=1&query="
        + quote_plus(f"{_pretty(k)}, Beverly Hills, CA")
        for k in out[location_key_var]
    ]
    out["checked_ok"] = ""
    out["notes"] = ""

    cols = [
        "best_rank", "location", location_key_var, "flags", "reports",
        "n_spellings", "raw_spellings", "feet_median", "feet_p90", "lat", "lon",
        "coord_source", "spread_m", "pin_link", "search_link", *rank_cols,
        "checked_ok", "notes",
    ]
    out = out[cols]
    out.round({"feet_median": 0, "feet_p90": 0, "lat": 6, "lon": 6, "spread_m": 0}).to_csv(
        out_dir / "top_locations_audit.csv", index=False
    )

    ## HTML: same rows, clickable, flagged rows highlighted
    def cell(v) -> str:
        if isinstance(v, float) and np.isnan(v):
            return ""
        if isinstance(v, float):
            return f"{v:.0f}" if abs(v) >= 100 else f"{v:g}"
        return html.escape(str(v))

    show = [
        "best_rank", "location", "flags", "reports", "raw_spellings", "feet_median",
        "coord_source",
    ]
    head = "".join(f"<th>{html.escape(c.replace('_', ' '))}</th>" for c in show)
    rows = []
    for _, r in out.iterrows():
        links = (
            (f'<a href="{r.pin_link}" target="_blank">pin</a> &middot; '
             if r.pin_link else "")
            + f'<a href="{r.search_link}" target="_blank">address search</a>'
        )
        cls = ' class="flag"' if r.flags else ""
        rows.append(
            f"<tr{cls}>" + "".join(f"<td>{cell(r[c])}</td>" for c in show)
            + f"<td>{links}</td><td><input type='checkbox'></td></tr>"
        )
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Top locations audit</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1c1d1f; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left;
          vertical-align: top; }}
th {{ background: #f1f5f9; position: sticky; top: 0; }}
tr.flag td {{ background: #fdf6e6; }}
td:nth-child(5) {{ max-width: 420px; color: #55585d; }}
p {{ max-width: 860px; line-height: 1.5; }}
</style></head><body>
<h2>Top locations audit, crashes through {meta['last_month']}</h2>
<p>Every location in a top {top} under any ranking the app offers. For each row,
open both links: the pin and the address search should land on the same corner.
Rows highlighted in yellow have a flag worth reading first. Fix a wrong pin by
adding <code>location_key, lat, lon</code> to
<code>data/external/location_coords_manual.csv</code>, then re-run
<code>make export_dash</code>.</p>
<table><thead><tr>{head}<th>check on map</th><th>ok</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    (out_dir / "top_locations_audit.html").write_text(doc)

    n_flag = int((out["flags"] != "").sum())
    print(f"{len(out)} locations in a top {top}; {n_flag} flagged")
    print(f"wrote {out_dir / 'top_locations_audit.csv'}")
    print(f"wrote {out_dir / 'top_locations_audit.html'}")


if __name__ == "__main__":
    app()
