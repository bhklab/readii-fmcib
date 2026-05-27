# %% [markdown]
# # Run original and READII negative control CTs through FMCIB
# 
# CT images cropped around a Gross Tumour Volume (GTV), resized, and saved as nifti files are fed into an instance of FMCIB for inference and all features are saved.


# %%
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict

import yaml
import click

# Make 'code/' importable (for infer.runFMCIBInfer)
sys.path.append("code")
from infer import runFMCIBInfer


def run_fmcib_pipeline(
    config_path: Path | str = "config/RADCURE.yaml",
    csv_root: Path | str | None = None,
    results_root: Path | str = "procdata",
    weights_path: Optional[Path | str] = None,
    precropped: bool = True,
) -> Dict[str, int]:
    """
    Run FMCIB inference over all CSVs under fmcib_input for the dataset in `config_path`.

    - Reads dataset_name from YAML.
        - Scans: <csv_root or srcdata>/<dataset>/features/fmcib/cube_50_50_50/*.csv (flat)
            If none are found at the top level, will also scan one directory level below for legacy layouts.
        - Writes features to: <results_root>/<dataset>/fmcib_features/ (flat). If using legacy subfolders,
            features are written under a matching subdirectory to avoid filename collisions.
    - Skips CSVs whose feature file already exists.

    Returns: {"processed": N, "skipped": M}
    """
    config_path = Path(config_path)
    cfg = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    dataset_name = cfg["dataset_name"]

    fmcib_csv_dir = (
        Path("srcdata") / dataset_name / "features" / "fmcib" / "cube_50_50_50"
        if csv_root is None else Path(csv_root)
    )
    results_root = Path(results_root)

    if isinstance(weights_path, str):
        weights_path = Path(weights_path)
    # Resolve weights (use local file if present, else None)
    # weight_file_path = Path("model_weights.torch")
    if weights_path.exists():
        fmcib_weights_path = weights_path
    else:
        fmcib_weights_path = None

    processed = 0
    skipped = 0

    # Base output directory
    features_base = results_root / dataset_name / "fmcib_features"
    features_base.mkdir(parents=True, exist_ok=True)

    # Prefer flat layout: fmcib_input/*.csv
    top_level_csvs = sorted(fmcib_csv_dir.glob("*.csv"))

    if top_level_csvs:
        for fmcib_csv_path in top_level_csvs:
            name = fmcib_csv_path.name
            suffix = name.removeprefix("fmcib_input_") if name.startswith("fmcib_input_") else name
            feature_file_save_path = features_base / f"fmcib_features_{suffix}"

            if feature_file_save_path.exists():
                skipped += 1
                continue

            runFMCIBInfer(
                feature_file_save_path=str(feature_file_save_path),
                csv_path=str(fmcib_csv_path),
                weights_path=fmcib_weights_path,
                precropped=precropped,
            )
            processed += 1
    else:
        # Legacy fallback: scan subdirectories one level deep
        for subdir in sorted(p for p in fmcib_csv_dir.iterdir() if p.is_dir()):
            features_dir = features_base / subdir.name
            features_dir.mkdir(parents=True, exist_ok=True)
            for fmcib_csv_path in sorted(subdir.glob("*.csv")):
                name = fmcib_csv_path.name
                suffix = name.removeprefix("fmcib_input_") if name.startswith("fmcib_input_") else name
                feature_file_save_path = features_dir / f"fmcib_features_{suffix}"

                # Only run inference if the output does not already exist
                if not feature_file_save_path.exists():
                    runFMCIBInfer(
                        feature_file_save_path=str(feature_file_save_path),
                        csv_path=str(fmcib_csv_path),
                        weights_path=fmcib_weights_path,
                        precropped=True,
                    )
                    processed += 1
                else:
                    skipped += 1

    return {"processed": processed, "skipped": skipped}


@click.command(context_settings={"show_default": True})
@click.option("--config-path", type=click.Path(path_type=Path), default=Path("config/RADCURE.yaml"), help="Path to dataset YAML config")
@click.option("--csv-root", type=click.Path(path_type=Path), default=None, help="Root dir containing CSVs. If omitted, uses srcdata/<dataset>/features/fmcib/cube_50_50_50 from the config’s dataset_name.")
@click.option("--results-root", type=click.Path(path_type=Path), default=Path("procdata"), help="Root dir to write features under <dataset>/fmcib_features/ (default: procdata)")
@click.option("--weights-path", type=click.Path(path_type=Path), default=None, help="Optional path to FMCIB weights file")
@click.option("--precropped/--no-precropped", default=True, help="Whether inputs are pre-cropped around GTV")
def cli(config_path: Path, csv_root: Optional[Path], results_root: Path, weights_path: Optional[Path], precropped: bool) -> None:
    """Run FMCIB inference over discovered CSVs and save feature files."""
    summary = run_fmcib_pipeline(
        config_path=config_path,
        csv_root=csv_root,
        results_root=results_root,
        weights_path=weights_path,
        precropped=precropped,
    )
    click.echo(
        f"FMCIB inference complete. Processed: {summary['processed']}, Skipped (already existed): {summary['skipped']}"
    )


if __name__ == "__main__":
    cli()
