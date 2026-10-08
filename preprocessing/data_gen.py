#!/usr/bin/env python3
"""
data_gen.py
-----------
Reads the raw Beverly Hills Police Department (BHPD) collision CSV export and
writes a Parquet copy (plus an optional CSV backup) for reproducible downstream
processing.

Source: City of Beverly Hills Open Data, "Historic Collision Records of
Beverly Hills (New)", records since January 1, 2015.

Example usage:
    python data_gen.py \
        --input-data-file ./data/raw/bh_collisions.csv \
        --output-data-file ./data/processed/df.parquet \
        --csv-backup
"""

from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_data_file: Path = typer.Option(
        Path("data/raw/bh_collisions.csv"),
        "--input-data-file",
        "-i",
        help="Path to the raw CSV export.",
    ),
    output_data_file: Path = typer.Option(
        Path("data/processed/df.parquet"),
        "--output-data-file",
        "-o",
        help="Path where the Parquet file will be written.",
    ),
    csv_backup: bool = typer.Option(
        False,
        "--csv-backup/--no-csv-backup",
        help="Also write a CSV alongside the Parquet file.",
    ),
) -> None:
    """Fetch the raw dataset and save it to Parquet, with optional CSV backup."""

    if not input_data_file.is_file():
        typer.secho(f"Input file not found: {input_data_file}", fg="red", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Reading {input_data_file} ...")
    ## utf-8-sig strips the byte-order mark the portal prepends to "Location"
    df = pd.read_csv(input_data_file, encoding="utf-8-sig")
    typer.echo(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    ## resolve() so a bare filename still yields a valid parent directory
    output_data_file = output_data_file.resolve()
    output_data_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_data_file, index=False)
    typer.echo(f"Saved dataset to {output_data_file}")

    if csv_backup:
        csv_path = output_data_file.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        typer.echo(f"Saved CSV backup to {csv_path}")

    typer.echo("Data generation complete.")


if __name__ == "__main__":
    app()
