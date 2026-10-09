#!/usr/bin/env python
"""Write the one-page PDF brief to data/brief (`make brief`).

The PDF itself is built by modeling/bh_brief.py, the same file the Dash app in
flask_apps uses for its "PDF brief" button. This script only adds the command
line and the project paths.

Output: data/brief/bh_<outcome>_<ranking>_top<N>_brief_<last month>.pdf
"""

from __future__ import annotations

from pathlib import Path

import typer

from core.config import PROCESSED_DATA_DIR
from modeling.bh_brief import build_brief

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def run(
    dash_dir: Path = typer.Option(PROCESSED_DATA_DIR.parent / "dash"),
    out_dir: Path = typer.Option(PROCESSED_DATA_DIR.parent / "brief"),
    outcome: str = typer.Option("inj_count", help="inj_count or vru_inj_count"),
    rank_by: str = typer.Option("eb", help="eb, excess, obs12, or ml"),
    top: int = typer.Option(10, help="locations listed, 5 to 15"),
    url: str = typer.Option("apps.datasciencedynamics.com/bh_traffic/"),
    author: str = typer.Option("Leon Shpaner, M.S., Data Science Dynamics"),
    contact: str = typer.Option("lshpaner@datasciencedynamics.com"),
    logo: bool = typer.Option(True, "--logo/--no-logo"),
    logo_width: float = typer.Option(1.4, help="inches"),
) -> None:
    """Write the one-page brief."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, name, scale = build_brief(
        dash_dir, outcome, rank_by, top, url, author, contact, logo, logo_width
    )
    (out_dir / name).write_bytes(pdf)
    print(f"layout scale {scale:.2f}")
    print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    app()
