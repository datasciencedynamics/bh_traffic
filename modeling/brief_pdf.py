#!/usr/bin/env python
"""One-page PDF brief for city decision makers.

Reads only the files `make export_dash` writes to data/dash, so the brief and
the Dash app always show the same numbers. Plain language, no model jargon:
the top 10 locations, what stands out at each, the concentration finding, the
method in two sentences, the main limits, and the link to the app.

Output: data/brief/bh_<outcome>_<ranking>_top<N>_brief_<last month>.pdf

Run after `make export_dash`, or via `make brief`. The same file is copied into
flask_apps/bh_traffic/bh_brief.py, where the app's "Download PDF brief" button
calls `build_brief` with the current map settings. The module needs only
pandas and reportlab (typer and the project config load in the CLI only), so
keep the two copies identical.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

## Crash types and rankings the app offers, with the columns behind them
OUTCOMES = {
    "inj_count": dict(
        tag="inj", noun="injury crashes",
        title="Where injury crashes concentrate in Beverly Hills"),
    "vru_inj_count": dict(
        tag="vru_inj", noun="injury crashes involving a pedestrian, cyclist, or "
        "motorcyclist",
        title="Where pedestrian, cyclist and motorcyclist injury crashes concentrate "
              "in Beverly Hills"),
}
RANK_BY = {
    "eb": ("expected crashes per year", "Expected per year", "eb_{t}_per_year",
           "{:.1f}"),
    "excess": ("crashes above or below similar intersections", "vs. similar per year",
               "excess_{t}_per_year", "{:+.1f}"),
    "obs12": ("crashes in the last 12 months", "Last 12 months", "obs_{t}_last12",
              "{:.0f}"),
    "ml": ("the 12-month forecast", "Next 12 months", "ml_{t}_next12", "{:.1f}"),
}


################################################################################
# Data Science Dynamics logo, drawn as vectors (from the site's
# data_science_dynamics_logo.svg, viewBox 33 42 380 104), so the brief needs no
# image file and stays sharp at any zoom
################################################################################
DSD_BLUE = "#185FA5"
## (cx, cy, r, opacity), relative to the hub at (90, 85)
_HUB = (90, 85)
_NODES = [
    (0, 0, 13, 1.0),
    (-30, -30, 7, 0.8), (20, -34, 6, 0.6), (-50, -12, 5, 0.45),
    (-30, 30, 7, 0.8), (20, 34, 6, 0.6), (-50, 12, 5, 0.45),
    (36, -10, 4.5, 0.4), (36, 10, 4.5, 0.4),
]
## (x1, y1, x2, y2, width, opacity), relative to the hub
_SPOKES = [
    (0, 0, -30, -30, 1.5, 0.6), (0, 0, 20, -34, 1.2, 0.55), (0, 0, -50, -12, 1, 0.5),
    (0, 0, -30, 30, 1.5, 0.6), (0, 0, 20, 34, 1.2, 0.55), (0, 0, -50, 12, 1, 0.5),
    (0, 0, 36, -10, 1, 0.55), (0, 0, 36, 10, 1, 0.55),
    (-30, -30, 20, -34, 0.7, 0.4), (-30, -30, -50, -12, 0.7, 0.4),
    (-30, 30, 20, 34, 0.7, 0.4), (-30, 30, -50, 12, 0.7, 0.4),
    (36, -10, 20, -34, 0.7, 0.35), (36, 10, 20, 34, 0.7, 0.35),
]
## absolute coordinates
_LINKS = [(126, 75, 166, 66, 1.2, 0.55), (126, 95, 166, 106, 1.2, 0.55)]
_DOTS = [(166, 66, 2.5, 0.55), (166, 86, 3, 0.65), (166, 106, 2.5, 0.55)]
_WAVE = [(103, 88), (112, 87), (119, 78), (126, 94), (133, 74), (140, 90), (147, 76),
         (154, 88), (159, 74), (166, 86)]
_WORDS = [("DATA", 68), ("SCIENCE", 98), ("DYNAMICS", 128)]
_VB_X, _VB_Y, _VB_W, _VB_H = 33, 42, 380, 104


class DSDLogo(Flowable):
    """The DSD logo at a given width (height follows the 380 x 104 viewBox)."""

    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = width * _VB_H / _VB_W
        self.k = width / _VB_W

    def wrap(self, *_):
        return self.width, self.height

    def _xy(self, x, y):
        ## SVG (y down, viewBox offset) -> PDF (y up, from this flowable's corner)
        return (x - _VB_X) * self.k, (_VB_H - (y - _VB_Y)) * self.k

    def draw(self):
        c, k = self.canv, self.k
        blue = colors.HexColor(DSD_BLUE)
        c.saveState()
        c.setFillColor(blue)
        c.setStrokeColor(blue)
        c.setLineCap(1)
        c.setLineJoin(1)
        hx, hy = _HUB
        for x1, y1, x2, y2, w, a in _SPOKES:
            c.setStrokeAlpha(a)
            c.setLineWidth(w * k)
            c.line(*self._xy(hx + x1, hy + y1), *self._xy(hx + x2, hy + y2))
        for cx, cy, r, a in _NODES:
            c.setFillAlpha(a)
            c.circle(*self._xy(hx + cx, hy + cy), r * k, stroke=0, fill=1)
        for x1, y1, x2, y2, w, a in _LINKS:
            c.setStrokeAlpha(a)
            c.setLineWidth(w * k)
            c.line(*self._xy(x1, y1), *self._xy(x2, y2))
        c.setStrokeAlpha(0.65)
        c.setLineWidth(1.8 * k)
        path = c.beginPath()
        path.moveTo(*self._xy(*_WAVE[0]))
        for pt in _WAVE[1:]:
            path.lineTo(*self._xy(*pt))
        c.drawPath(path, stroke=1, fill=0)
        for cx, cy, r, a in _DOTS:
            c.setFillAlpha(a)
            c.circle(*self._xy(cx, cy), r * k, stroke=0, fill=1)
        c.setFillAlpha(1)
        for word, y in _WORDS:
            t = c.beginText(*self._xy(172, y))
            t.setFont("Helvetica-Bold", 34 * k)
            t.setCharSpace(5 * k)
            t.textOut(word)
            c.drawText(t)
        c.restoreState()

INK = colors.HexColor("#1c1d1f")
INK2 = colors.HexColor("#55585d")
RULE = colors.HexColor("#d9d9d4")
BLUE = colors.HexColor("#1f5f8b")
WARM = colors.HexColor("#c2410c")
BAND = colors.HexColor("#f5f5f2")
AMBER = colors.HexColor("#eda100")

## Same thresholds as the app's "What stands out"
STANDOUT_MIN_N, STANDOUT_MIN_SHARE, STANDOUT_MIN_RATIO = 4, 0.2, 1.5


## Pattern -> what an engineering review would usually look at first. Drawn
## from FHWA's Proven Safety Countermeasures; starting points, not prescriptions.
REVIEW = {
    "Left turn, failed to yield": "protected left-turn phasing (a green arrow) and "
                                  "sight lines for turning drivers",
    "Ran stop sign": "stop-sign visibility and advance warning; whether the "
                     "intersection warrants an all-way stop or a signal",
    "Ran red light": "yellow and all-red signal timing, and signal-head visibility "
                     "(backplates with reflective borders)",
    "Cyclist involved": "bike lane separation and markings through the intersection",
    "Rear-end": "signal timing and coordination, and advance warning of the signal",
    "Unsafe turn or lane change": "lane markings and turn-lane design on the approaches",
    "Failed to yield (other)": "right-of-way control and sight lines at the corner",
    "Failed to yield to pedestrian": "leading pedestrian intervals and crosswalk "
                                     "visibility",
    "Vehicle hits pedestrian": "leading pedestrian intervals and crosswalk visibility",
    "Pedestrian involved": "leading pedestrian intervals and crosswalk visibility",
    "Broadside (T-bone)": "signal timing and sight lines for crossing traffic",
    "Unsafe speed": "speed management on the approaches",
    "Motorcyclist involved": "lane-change and turning conflicts with motorcycles",
}


def _pretty(key: str) -> str:
    return " & ".join(part.strip().title() for part in key.split("/"))


def _standout_row(profile: pd.DataFrame, key: str):
    p = profile[
        (profile.location_key == key)
        & (profile.n >= STANDOUT_MIN_N)
        & (profile.share >= STANDOUT_MIN_SHARE)
        & (profile.ratio >= STANDOUT_MIN_RATIO)
        & ~profile.category.isin(["Other", "Other or unknown"])
    ]
    if p.empty:
        return None
    return p.sort_values(["ratio", "n"], ascending=False).iloc[0]


def _standout(profile: pd.DataFrame, key: str) -> str:
    r = _standout_row(profile, key)
    if r is None:
        return "No single type or cause stands out"
    return f"{r.category}: {r.share:.0%} of injury crashes ({r.ratio:.1f}\u00d7 city)"


def _month(ym: str) -> str:
    return pd.Period(ym, freq="M").strftime("%B %Y")


def build_brief(
    dash_dir: Path,
    outcome: str = "inj_count",
    by: str = "eb",
    top: int = 10,
    url: str = "apps.datasciencedynamics.com/bh_traffic/",
    author: str = "Leonid Shpaner, Data Science Dynamics",
    contact: str = "lshpaner@datasciencedynamics.com",
    logo: bool = True,
    logo_width: float = 1.4,
) -> tuple[bytes, str, float]:
    """One-page brief as PDF bytes, plus a file name and the layout scale used.

    outcome: "inj_count" or "vru_inj_count"; by: "eb", "excess", "obs12", "ml"
    (the app's "Rank by"); top: number of locations listed (5 to 15).
    """
    dash_dir = Path(dash_dir)
    o = OUTCOMES[outcome]
    t = o["tag"]
    by_long, by_head, by_col, by_fmt = RANK_BY[by]
    by_col = by_col.format(t=t)
    last5 = f"obs_{t}_last60"
    top = int(max(5, min(top, 15)))
    meta = json.loads((dash_dir / "meta.json").read_text())
    hot = pd.read_csv(dash_dir / "hotspots.csv")
    profile = pd.read_csv(dash_dir / "hotspot_profile.csv")
    conc = pd.read_csv(dash_dir / "concentration.csv").set_index(["outcome", "test"])

    top_col = [c for c in conc.columns if c.startswith("top_") and c != "top_locations"]
    top_col = [c for c in top_col if not c.startswith("top_locations")][0]
    oos = conc.loc[(outcome, "out-of-sample, panel"), top_col]
    oracle = conc.loc[(outcome, "out-of-sample, panel"), "oracle_top_share"]
    cw = conc.loc[(outcome, "out-of-sample, citywide")]
    city_share = cw[top_col]
    k = int(cw["top_locations"])
    k_share = cw["top_locations_share_of_all"]
    cut = meta["train_end"][:4]
    years = meta["eb_window_months"] // 12
    last = _month(meta["last_month"])
    first = _month(meta["first_month"])

    d = hot.sort_values([by_col, f"eb_{t}_per_year"], ascending=False).head(top)
    noun = o["noun"]
    short = "injury crashes" if outcome == "inj_count" else "ped/bike/moto injury crashes"

    ## Build at full size; if anything spills onto a second page, shrink type
    ## and spacing a little and rebuild, so the brief is always one page
    name = f"bh_{outcome}_{by}_top{top}_brief_{meta['last_month']}.pdf"

    def build(scale: float, buf) -> int:
        ## ---------------------------------------------------------------- styles
        def st(name, **kw):
            base = dict(fontName="Helvetica", fontSize=9, leading=12, textColor=INK,
                        alignment=TA_LEFT)
            base.update(kw)
            base["fontSize"] *= scale
            base["leading"] *= scale
            return ParagraphStyle(name, **base)

        TITLE = st("title", fontName="Helvetica-Bold", fontSize=18, leading=22)
        SUB = st("sub", fontSize=10, leading=14, textColor=INK2)
        H = st("h", fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=4)
        BODY = st("body", fontSize=9, leading=12.5, textColor=INK2)
        BIG = st("big", fontName="Helvetica-Bold", fontSize=22, leading=24, textColor=INK)
        BIGL = st("bigl", fontSize=8.5, leading=11, textColor=INK2)
        TH = st("th", fontName="Helvetica-Bold", fontSize=7.5, leading=9, textColor=INK2)
        TD = st("td", fontSize=8.5, leading=10.5)
        TDN = st("tdn", fontSize=8.5, leading=10.5, alignment=1)
        THC = st("thc", fontName="Helvetica-Bold", fontSize=7.5, leading=9,
                 textColor=INK2, alignment=1)
        SMALL = st("small", fontSize=7.5, leading=10, textColor=INK2)

        story = []
        ## a slightly smaller title when the logo shares its row, so it stays on one line
        title_style = TITLE if not logo else st(
            "title_l", fontName="Helvetica-Bold", fontSize=16, leading=20)
        title_block = [
            Paragraph(o["title"], title_style),
            Spacer(1, 3 * scale),
            Paragraph(
                f"An independent look at {meta['n_crashes']:,} BHPD collision reports, "
                f"{first} to {last}: where injury crashes keep happening, and what "
                "kind.", SUB),
        ]
        if logo:
            ## logo top-right beside the title
            w = logo_width * inch
            head = Table([[title_block, DSDLogo(w)]],
                         colWidths=[7.3 * inch - w - 0.2 * inch, w + 0.2 * inch],
                         hAlign="LEFT")
            head.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            story.append(head)
        else:
            story.extend(title_block)
        story.append(Spacer(1, 10 * scale))

        ## ------------------------------------------------------ headline numbers
        stats = [
            (f"{city_share:.0%}",
             f"of {short} after {cut} happened at just {k} intersections, "
             f"{k_share:.1%} of all crash locations, picked using data through {cut}."),
            (f"{oos:.0%}",
             f"of later {short} at the city\u2019s {meta['n_panel_locations']} "
             f"recurring crash sites happened at the fifth of them ranked highest on "
             f"history alone. Perfect hindsight: {oracle:.0%}."),
            (f"{meta['n_injury']:,}",
             f"reported crashes with an injury or death since {meta['first_month'][:4]}; "
             f"{meta['n_vru_injury']:,} involved a pedestrian, cyclist, or motorcyclist."),
        ]
        cells = [[Paragraph(v, BIG), Paragraph(t, BIGL)] for v, t in stats]
        stat_tbl = Table(
            [[Table([[c[0]], [c[1]]], colWidths=[2.25 * inch],
                    style=[("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 1),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1)])
              for c in cells]],
            colWidths=[2.4333 * inch] * 3, hAlign="LEFT",
        )
        stat_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEABOVE", (0, 0), (0, 0), 3, WARM),
            ("LINEABOVE", (1, 0), (1, 0), 3, BLUE),
            ("LINEABOVE", (2, 0), (2, 0), 3, INK2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(stat_tbl)
        story.append(Spacer(1, 12 * scale))

        ## -------------------------------------------------------------- top 10
        story.append(Paragraph(f"The top {top}, and what stands out at each", H))
        story.append(Paragraph(
            f"Ranked by {by_long}, counting {noun}. \u201cWhat stands out\u201d is the "
            "type of crash or cause most over-represented at that location compared with "
            f"the whole city, over the last {years} years. It is where a fix would start.",
            BODY))
        story.append(Spacer(1, 5 * scale))
        ## number columns: header and values centered on each other
        rows = [[Paragraph("#", TH), Paragraph("Intersection", TH),
                 Paragraph(by_head, THC),
                 Paragraph(f"Last {years} yrs", THC),
                 Paragraph("What stands out", TH)]]
        for i, (_, r) in enumerate(d.iterrows(), start=1):
            rows.append([
                Paragraph(str(i), TD),
                Paragraph(_pretty(r.location_key), TD),
                Paragraph(by_fmt.format(r[by_col]), TDN),
                Paragraph(f"{int(r[last5])}", TDN),
                Paragraph(_standout(profile, r.location_key), TD),
            ])
        tbl = Table(rows, colWidths=[0.3 * inch, 2.15 * inch, 0.85 * inch, 0.7 * inch,
                                     3.3 * inch], repeatRows=1, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 9 * scale))

        ## ------------------------------------------------------- where to start
        groups: dict[str, list[str]] = {}
        for i, (_, r) in enumerate(d.iterrows(), start=1):
            so = _standout_row(profile, r.location_key)
            if so is not None and so.category in REVIEW:
                groups.setdefault(so.category, []).append(f"#{i}")
        if groups:
            story.append(Paragraph("Where an engineering review could start", H))
            story.append(Paragraph(
                f"The top {top} grouped by what stands out (numbers refer to the table above), "
                "with what a traffic engineer would usually check first, from the Federal "
                "Highway Administration\u2019s Proven Safety Countermeasures. Starting "
                "points for a site review, not recommendations.", BODY))
            story.append(Spacer(1, 4 * scale))
            g_rows = []
            for cat, locs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                g_rows.append([
                    Paragraph(f"<b>{cat}</b> <font color='#55585d'>({', '.join(locs)})"
                              "</font>", TD),
                    Paragraph(f"Review {REVIEW[cat]}.", TD),
                ])
            g_tbl = Table(g_rows, colWidths=[2.75 * inch, 4.55 * inch], hAlign="LEFT")
            g_tbl.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(g_tbl)
            story.append(Spacer(1, 8 * scale))

        ## ------------------------------------------------------ method + limits
        method = [
            Paragraph("How the list is made", H),
            Paragraph(
                f"Each intersection's expected crashes per year blends its last {years} "
                "years with what similar intersections see (the Highway Safety "
                "Manual\u2019s Empirical Bayes screening method), so one unlucky year does "
                f"not put a place at the top. Tested fairly: rankings built only on data "
                f"through {cut} were checked against the crashes that came after.", BODY),
        ]
        limits = [
            Paragraph("Keep in mind", H),
            Paragraph(
                "Locations are ranked by the number of injury crashes, not by crashes per "
                "car. The police data has no traffic counts, so the busiest intersections "
                "rank high partly because of volume; with the city\u2019s counts this could "
                "be re-ranked as crash rates. Only reported crashes are included, and the "
                "cause is the one the officer recorded.", BODY),
        ]
        two = Table([[method, limits]], colWidths=[3.55 * inch, 3.75 * inch],
                    hAlign="LEFT")
        two.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (0, 0), 0),
            ("RIGHTPADDING", (0, 0), (0, 0), 14),
            ("LINEBEFORE", (1, 0), (1, 0), 3, AMBER),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#fdf6e6")),
            ("LEFTPADDING", (1, 0), (1, 0), 9),
            ("RIGHTPADDING", (1, 0), (1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(two)
        story.append(Spacer(1, 9 * scale))

        ## ---------------------------------------------------------------- close
        story.append(Paragraph(
            f"<b>Interactive version</b> (map, every intersection, history, forecast, "
            f"methods): <font color='#1f5f8b'><link href='https://{url}'>{url}</link>"
            "</font>", st("link", fontSize=9.5, leading=13)))
        story.append(Spacer(1, 6 * scale))
        who = author + (f" \u00b7 {contact}" if contact else "")
        story.append(Paragraph(
            f"{who}. Data: City of Beverly Hills Open Data, Historic Collision Records "
            "(BHPD Traffic Bureau); street geometry \u00a9 OpenStreetMap contributors. "
            "Independent analysis, not affiliated with or endorsed by the City of Beverly "
            "Hills or the Beverly Hills Police Department.", SMALL))

        doc = SimpleDocTemplate(
            buf, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
            topMargin=0.5 * inch, bottomMargin=0.45 * inch,
            title=o["title"], author=author,
        )
        doc.build(story)
        return doc.page

    for scale in (1.0, 0.96, 0.92, 0.88, 0.85, 0.82, 0.78):
        buf = io.BytesIO()
        if build(scale, buf) == 1:
            break
    return buf.getvalue(), name, scale


################################################################################
# Command line (make brief)
################################################################################
if __name__ == "__main__":
    import typer

    from core.config import PROCESSED_DATA_DIR

    cli = typer.Typer(add_completion=False, help=__doc__)

    @cli.command()
    def run(
        dash_dir: Path = typer.Option(PROCESSED_DATA_DIR.parent / "dash"),
        out_dir: Path = typer.Option(PROCESSED_DATA_DIR.parent / "brief"),
        outcome: str = typer.Option("inj_count", help="inj_count or vru_inj_count"),
        rank_by: str = typer.Option("eb", help="eb, excess, obs12, or ml"),
        top: int = typer.Option(10, help="locations listed, 5 to 15"),
        url: str = typer.Option("apps.datasciencedynamics.com/bh_traffic/"),
        author: str = typer.Option("Leonid Shpaner, Data Science Dynamics"),
        contact: str = typer.Option("lshpaner@datasciencedynamics.com"),
        logo: bool = typer.Option(True, "--logo/--no-logo"),
        logo_width: float = typer.Option(1.4, help="inches"),
    ) -> None:
        """Write the one-page brief."""
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf, name, scale = build_brief(dash_dir, outcome, rank_by, top, url, author,
                                       contact, logo, logo_width)
        (out_dir / name).write_bytes(pdf)
        print(f"layout scale {scale:.2f}")
        print(f"wrote {out_dir / name}")

    cli()
