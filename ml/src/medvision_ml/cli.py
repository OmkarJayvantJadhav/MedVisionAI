"""Command-line entry point: ``medvision-ml <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from medvision_ml.config import ConfigError, ProjectConfig, load_config

LOGGER = logging.getLogger("medvision_ml")


def _groups_for(explicit: Path | None, folder: Path) -> dict[str, str] | None:
    from medvision_ml.data.groups import read_groups
    from medvision_ml.data.isic2019 import GROUPS_FILE

    path = explicit or folder / GROUPS_FILE
    if explicit is None and not path.is_file():
        return None
    return read_groups(path)


def _check_leakage(args: argparse.Namespace, config: ProjectConfig) -> int:
    from medvision_ml.data.images import scan_split_dataset
    from medvision_ml.data.leakage import check_leakage

    data_dir = args.data_dir or config.data_dir
    max_distance = (
        args.max_distance
        if args.max_distance is not None
        else config.data.near_duplicate_max_distance
    )
    groups = _groups_for(args.groups, data_dir)
    report = check_leakage(scan_split_dataset(data_dir), max_distance, groups)
    output = args.output or config.output_dir / "leakage_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.summary(), indent=2))
    print(f"Full report: {output}")
    if report.has_leakage:
        print("Leakage found: duplicate images appear in more than one split.", file=sys.stderr)
        return 1
    return 0


def _split(args: argparse.Namespace, config: ProjectConfig) -> int:
    from medvision_ml.data.dedupe import deduplicate
    from medvision_ml.data.groups import group_keys, merge_groups, write_groups
    from medvision_ml.data.images import scan_class_dataset, scan_split_dataset
    from medvision_ml.data.isic2019 import GROUPS_FILE
    from medvision_ml.data.leakage import check_leakage, fingerprint_all, group_duplicates
    from medvision_ml.data.splitting import assign_splits, materialise_splits

    records = scan_class_dataset(args.source)
    fingerprints, failed = fingerprint_all(records)
    if failed:
        print(f"Skipping {len(failed)} unreadable images.", file=sys.stderr)
    fingerprints, removed = deduplicate(fingerprints)
    print(json.dumps(removed.summary(), indent=2))
    report_path = config.output_dir / "split_removed.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(vars(removed), indent=2), encoding="utf-8")
    readable = [f.record for f in fingerprints]
    known = _groups_for(args.groups, args.source) or {}
    groups = merge_groups(
        group_duplicates(fingerprints, config.data.near_duplicate_max_distance),
        group_keys(readable, known),
    )
    assignments = assign_splits(readable, groups, config.data.split_ratios, config.seed)
    destination = args.dest or config.data_dir
    counts = materialise_splits(assignments, destination, overwrite=args.overwrite)
    if known:
        names = {r.path.name for r in readable}
        write_groups({k: v for k, v in known.items() if k in names}, destination / GROUPS_FILE)
    print(json.dumps(counts, indent=2))
    report = check_leakage(
        scan_split_dataset(destination), config.data.near_duplicate_max_distance, known or None
    )
    print(json.dumps(report.summary(), indent=2))
    return 1 if report.has_leakage else 0


def _prepare_isic2019(args: argparse.Namespace, config: ProjectConfig) -> int:
    from medvision_ml.data.isic2019 import prepare_isic2019

    counts = prepare_isic2019(
        args.images, args.ground_truth, args.metadata, args.dest, link=not args.copy
    )
    print(json.dumps(counts, indent=2))
    print(f"Next: medvision-ml split --source {args.dest}")
    return 0


def _train(args: argparse.Namespace, config: ProjectConfig) -> int:
    from medvision_ml.training.trainer import train

    result = train(config, run_name=args.run_name)
    print(f"Model: {result.model_path}")
    print(
        f"Best validation loss {result.best_val_loss:.4f}, accuracy {result.best_val_accuracy:.4f}"
    )
    return 0


def _evaluate(args: argparse.Namespace, config: ProjectConfig) -> int:
    from medvision_ml.evaluation.evaluate import evaluate

    metrics = evaluate(
        config,
        args.model,
        split=args.split,
        output_dir=args.output,
        trust_legacy_lambda=args.trust_legacy_model,
    )
    keys = ("accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr_macro", "ece")
    print(
        json.dumps(
            {k: metrics[k] for k in keys} | {"malignant_group": metrics.get("malignant_group")},
            indent=2,
        )
    )
    return 0


def _gradcam(args: argparse.Namespace, config: ProjectConfig) -> int:
    from PIL import Image

    from medvision_ml.explainability.gradcam import gradcam, load_image, overlay
    from medvision_ml.models.io import load_class_names, load_model

    model = load_model(args.model, trust_legacy_lambda=args.trust_legacy_model)
    class_names = load_class_names(args.model.with_name(f"{args.model.stem}_class_names.json"))
    image = load_image(args.image, config.data.image_size)
    result = gradcam(model, image, args.class_index)
    output = args.output or config.output_dir / "gradcam" / f"{args.image.stem}_overlay.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay(image, result.heatmap)).save(output)
    top = int(result.probabilities.argmax())
    print(f"Prediction: {class_names[top]} ({result.probabilities[top]:.1%})")
    print(f"Explained class: {class_names[result.class_index]}")
    print(f"Overlay: {output}")
    print("The heatmap shows what drove the model's score; it is not a diagnosis.")
    return 0


def _export(args: argparse.Namespace, config: ProjectConfig) -> int:
    from medvision_ml.export.onnx_export import export_model
    from medvision_ml.models.io import load_class_names, load_model

    model = load_model(args.model, trust_legacy_lambda=args.trust_legacy_model)
    class_names = load_class_names(args.model.with_name(f"{args.model.stem}_class_names.json"))
    output = args.output or config.model_dir / "export" / args.version
    result = export_model(
        model,
        output,
        class_names=class_names,
        malignant_classes=config.evaluation.malignant_classes,
        model_version=args.version,
        backbone=args.backbone or config.model.backbone,
        image_size=config.data.image_size,
    )
    print(f"Exported to {result.directory}")
    print(f"Max ONNX/Keras probability difference: {result.max_probability_diff:.2e}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medvision-ml", description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML config (default: configs/skin.yaml)"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    leak = commands.add_parser("check-leakage", help="find duplicate images across splits")
    leak.add_argument("--data-dir", type=Path, default=None)
    leak.add_argument("--max-distance", type=int, default=None, help="dHash distance threshold")
    leak.add_argument("--output", type=Path, default=None, help="JSON report path")
    leak.add_argument(
        "--groups", type=Path, default=None, help="image,group CSV (default: <data-dir>/groups.csv)"
    )
    leak.set_defaults(handler=_check_leakage)

    split = commands.add_parser("split", help="build duplicate-aware train/validation/test splits")
    split.add_argument(
        "--source", type=Path, required=True, help="folder with one sub-folder per class"
    )
    split.add_argument(
        "--dest", type=Path, default=None, help="output root (default: paths.data_dir)"
    )
    split.add_argument("--overwrite", action="store_true")
    split.add_argument(
        "--groups", type=Path, default=None, help="image,group CSV (default: <source>/groups.csv)"
    )
    split.set_defaults(handler=_split)

    isic = commands.add_parser("prepare-isic2019", help="lay out the ISIC 2019 training release")
    isic.add_argument("--images", type=Path, required=True, help="ISIC_2019_Training_Input folder")
    isic.add_argument("--ground-truth", type=Path, required=True)
    isic.add_argument("--metadata", type=Path, required=True)
    isic.add_argument("--dest", type=Path, required=True)
    isic.add_argument("--copy", action="store_true", help="copy files instead of hard-linking")
    isic.set_defaults(handler=_prepare_isic2019)

    train = commands.add_parser("train", help="train a classifier")
    train.add_argument("--run-name", default=None)
    train.set_defaults(handler=_train)

    for name, handler, help_text in (
        ("evaluate", _evaluate, "evaluate a saved model on a split"),
        ("gradcam", _gradcam, "explain one image with Grad-CAM"),
        ("export", _export, "export a model to ONNX serving artifacts"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--output", type=Path, default=None)
        sub.add_argument(
            "--trust-legacy-model",
            action="store_true",
            help="allow the prototype's Lambda-layer model format (trusted files only)",
        )
        sub.set_defaults(handler=handler)
        if name == "evaluate":
            sub.add_argument("--split", default="Test", choices=["Train", "Validation", "Test"])
        elif name == "export":
            sub.add_argument("--version", required=True, help="e.g. skin-effb0-v1.0.0")
            sub.add_argument("--backbone", default=None, help="recorded in the metadata")
        else:
            sub.add_argument("--image", type=Path, required=True)
            sub.add_argument("--class-index", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_config(args.config)
        code: int = args.handler(args, config)
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return code


if __name__ == "__main__":
    sys.exit(main())
