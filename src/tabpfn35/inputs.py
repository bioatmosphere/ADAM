"""What the pipeline needs on disk, and what can be rebuilt from where.

The raw sources behind the predictor stack are large (TerraClimate is ~9 GB for
2001-2010, the soil-moisture archive ~9 GB) and one of them is not a public
download, so the stack itself is the unit that travels with this submission.
With the training table and the stack present, every command except ``stack``
runs; ``check`` says which of those two levels the current checkout is at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Requirement:
    name: str
    path: Path
    needed_for: str
    shipped: bool          # travels with the repository
    how_to_get: str
    is_directory_glob: str | None = None

    def present(self) -> bool:
        if self.is_directory_glob:
            return self.path.is_dir() and any(self.path.glob(self.is_directory_glob))
        return self.path.exists() and self.path.stat().st_size > 0

    def detail(self) -> str:
        if not self.present():
            return "missing"
        if self.is_directory_glob:
            return f"{len(list(self.path.glob(self.is_directory_glob)))} files"
        size_mb = self.path.stat().st_size / 1e6
        return f"{size_mb:,.1f} MB"


def requirements() -> list[Requirement]:
    return [
        Requirement(
            "training table", config.TRAINING_CSV, "validate, global", True,
            "committed with this repository",
        ),
        Requirement(
            "predictor stack", config.STACK_NC, "global", True,
            "committed with this repository; rebuild with `run.py stack`",
        ),
        Requirement(
            "SoilGrids 5 km rasters", config.SOILGRIDS_5KM_DIR, "stack", False,
            "`run.py fetch-soil` (72 MB from files.isric.org)",
            is_directory_glob="*.tif",
        ),
        Requirement(
            "TerraClimate 2001-2010", config.TERRACLIMATE_DIR, "stack", False,
            "`python src/ancillary/terraclimate.py --mode download --variables aet pet "
            "ppt tmax tmin vpd --start-year 2001 --end-year 2010` (~9 GB)",
            is_directory_glob="TerraClimate_*_20*.nc",
        ),
        Requirement(
            "soil moisture (EC ORS)", config.SOIL_MOISTURE_NC, "stack", False,
            "not a public download: provided by the dataset authors "
            "(contact in the file's attributes), ~9 GB",
        ),
        Requirement(
            "elevation 0.5 deg", config.ELEVATION_NC, "stack", False,
            "`python src/ancillary/download_global_elevation.py`",
        ),
    ]


def check(verbose: bool = True) -> dict[str, bool]:
    """Report which inputs are present and which commands can therefore run."""
    reqs = requirements()
    present = {r.name: r.present() for r in reqs}

    if verbose:
        print("Inputs")
        for r in reqs:
            mark = "ok     " if present[r.name] else "MISSING"
            ships = "shipped" if r.shipped else "fetched"
            print(f"  [{mark}] {r.name:26s} {ships}  {r.detail():>12s}  (for {r.needed_for})")
            if not present[r.name]:
                print(f"            get it: {r.how_to_get}")

    can_validate = present["training table"]
    can_global = can_validate and present["predictor stack"]
    can_stack = all(
        present[r.name] for r in reqs if r.needed_for == "stack"
    )
    token = bool(os.environ.get("TABPFN_TOKEN"))

    if verbose:
        print("\nCommands")
        for label, ready, extra in [
            ("validate", can_validate and token, "needs the training table and TABPFN_TOKEN"),
            ("global", can_global and token, "needs the stack as well"),
            ("figures", True, "reads existing outputs only"),
            ("stack", can_stack, "needs all four raw sources"),
        ]:
            status = "ready" if ready else "blocked"
            print(f"  [{status:7s}] {label:9s} {extra}")
        if not token:
            print("\nTABPFN_TOKEN is not set; see src/tabpfn35/README.md")

    return {
        "validate": can_validate and token,
        "global": can_global and token,
        "stack": can_stack,
        "token": token,
    }
