"""Command line entry point for the TabPFN-3.5 BNPP pipeline.

    python -m src.tabpfn35.run check          # what is on disk and what can run
    python -m src.tabpfn35.run fetch-soil     # download the SoilGrids 5 km rasters
    python -m src.tabpfn35.run stack          # build the 0.5 deg predictor stack + QC
    python -m src.tabpfn35.run validate       # random / site / spatial-block validation
    python -m src.tabpfn35.run global         # global prediction with uncertainty
    python -m src.tabpfn35.run figures        # redraw figures from existing outputs
    python -m src.tabpfn35.run all            # stack (if needed) -> validate -> global -> figures

Everything that touches a model goes through the Prior Labs API and needs a token:

    export TABPFN_TOKEN="<your-token>"
"""

from __future__ import annotations

import argparse
import sys

from . import config


def _build_stack(args) -> None:
    from . import stack as stack_module

    built = stack_module.build_stack(resolution=args.resolution)
    stack_module.save_stack(built)
    qc = stack_module.check_provenance(built)
    config.STACK_QC_CSV.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(config.STACK_QC_CSV, index=False)
    print(f"Provenance table written to {config.STACK_QC_CSV}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="tabpfn35",
        description="TabPFN-3.5 belowground productivity pipeline (Prior Labs API only)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="report which inputs are present and what can run")
    sub.add_parser("fetch-soil", help="download the SoilGrids 2.0 5 km rasters")

    stack_parser = sub.add_parser("stack", help="build the global predictor stack")
    stack_parser.add_argument("--resolution", type=float, default=config.GRID_RESOLUTION)

    validate_parser = sub.add_parser("validate", help="run the validation schemes")
    validate_parser.add_argument(
        "--schemes", nargs="+", default=["random", "site", "spatial", "site_mean"],
        choices=["random", "site", "spatial", "site_mean"],
    )
    validate_parser.add_argument("--folds", type=int, default=config.N_FOLDS)

    global_parser = sub.add_parser("global", help="apply the model to the global stack")
    global_parser.add_argument("--all-land", action="store_true",
                               help="predict every land cell, not only in-domain cells")
    global_parser.add_argument("--use-records", action="store_true",
                               help="fit on all records instead of one row per coordinate")
    global_parser.add_argument("--batch-size", type=int, default=None)
    global_parser.add_argument("--limit", type=int, default=None,
                               help="predict at most this many cells (for a cheap test run)")
    global_parser.add_argument("--dry-run", action="store_true",
                               help="prepare everything but make no API calls")

    sub.add_parser("figures", help="redraw figures from existing outputs")

    all_parser = sub.add_parser("all", help="stack (if needed), validate, global, figures")
    all_parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    all_parser.add_argument("--all-land", action="store_true")
    all_parser.add_argument("--batch-size", type=int, default=None)
    all_parser.add_argument("--resolution", type=float, default=config.GRID_RESOLUTION)

    args = parser.parse_args(argv)

    if args.command == "check":
        from . import inputs

        inputs.check()
        return 0

    if args.command == "fetch-soil":
        from . import stack as stack_module

        stack_module.fetch_soilgrids()
        return 0

    if args.command == "stack":
        _build_stack(args)
        return 0

    if args.command == "validate":
        from . import figures, validation

        validation.run(schemes=tuple(args.schemes), n_folds=args.folds)
        figures.validation_figure()
        return 0

    if args.command == "global":
        from . import apply_global, figures

        result = apply_global.run(
            all_land=args.all_land,
            use_records=args.use_records,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        if result is not None:
            figures.prediction_maps(result)
            figures.latitudinal_profile(result)
            figures.domain_figure(result)
        return 0

    if args.command == "figures":
        from . import figures

        figures.all_figures()
        return 0

    if args.command == "all":
        from . import apply_global, figures, validation

        if not config.STACK_NC.exists():
            _build_stack(args)
        validation.run(n_folds=args.folds)
        apply_global.run(all_land=args.all_land, batch_size=args.batch_size)
        figures.all_figures()
        return 0

    parser.error(f"unknown command {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
