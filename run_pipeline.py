#!/usr/bin/env python3
"""
Convenience orchestrator that runs the full embedding → clustering → classification pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full title clustering pipeline.")
    parser.add_argument("--input", required=True, help="Path to CSV with columns 'title' and 'group'.")
    parser.add_argument(
        "--output-dir",
        default="pipeline_output",
        help="Directory where intermediate and final artifacts will be written.",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "jsonl"],
        default="parquet",
        help="Format for the prepared embeddings dataset.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["sentence-transformers", "gemini"],
        default="sentence-transformers",
        help="Embedding provider passed to prepare_embeddings.py and classify_title.py.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Model identifier for sentence-transformers provider.",
    )
    parser.add_argument(
        "--gemini-model",
        help="Gemini embedding model identifier.",
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Gemini API key (falls back to GEMINI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file shared across embedding/classification steps.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of clusters for K-Means and neighbors for KNN (default 3).",
    )
    parser.add_argument(
        "--batch",
        help="Optional CSV/JSONL of titles to classify after clustering (must include 'title' column).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a 2D visualization of embeddings colored by original group.",
    )
    parser.add_argument(
        "--visualize-output",
        help="Override path for visualization PNG (defaults to <output-dir>/embedding_plot.png).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity for this orchestrator script.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_command(cmd: List[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    base_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_base = output_dir / "embeddings"
    embeddings_base.parent.mkdir(parents=True, exist_ok=True)

    prepared_dataset = embeddings_base.with_suffix(f".{args.format}")
    manifest_path = embeddings_base.with_suffix(".manifest.json")

    provider = config.get("provider", args.embedding_provider)
    embedding_model = config.get("model", args.embedding_model)
    gemini_model = config.get("gemini_model", args.gemini_model)
    gemini_api_key = config.get("gemini_api_key", args.gemini_api_key)

    # Step 1: Prepare embeddings
    prepare_cmd = [
        sys.executable,
        str(base_dir / "prepare_embeddings.py"),
        "--input",
        str(Path(args.input).resolve()),
        "--output",
        str(embeddings_base),
        "--format",
        args.format,
        "--provider",
        provider,
    ]
    if args.config:
        prepare_cmd.extend(["--config", args.config])
    if provider == "sentence-transformers":
        if embedding_model:
            prepare_cmd.extend(["--model", embedding_model])
    else:
        if gemini_model:
            prepare_cmd.extend(["--gemini-model", gemini_model])
        if gemini_api_key:
            prepare_cmd.extend(["--gemini-api-key", gemini_api_key])
    run_command(prepare_cmd)

    # Resolve dataset path after format selection
    dataset_path = prepared_dataset

    # Step 2: Run K-Means clustering
    clustering_output = output_dir / "clustering_results.csv"
    report_path = output_dir / "kmeans_report.md"
    centroids_path = output_dir / "centroids.json"

    kmeans_cmd = [
        sys.executable,
        str(base_dir / "run_kmeans.py"),
        "--data",
        str(dataset_path),
        "--k",
        str(args.k),
        "--output",
        str(clustering_output),
        "--report",
        str(report_path),
        "--centroids",
        str(centroids_path),
    ]
    if manifest_path.exists():
        kmeans_cmd.extend(["--manifest", str(manifest_path)])
    run_command(kmeans_cmd)

    # Step 3: Optional batch classification
    if args.batch:
        predictions_path = output_dir / "batch_predictions.csv"
        classify_cmd = [
            sys.executable,
            str(base_dir / "classify_title.py"),
            "--assignments",
            str(clustering_output),
            "--k",
            str(args.k),
            "--batch",
            str(Path(args.batch).resolve()),
            "--output",
            str(predictions_path),
            "--provider",
            provider,
        ]
        if args.config:
            classify_cmd.extend(["--config", args.config])
        if provider == "sentence-transformers":
            if embedding_model:
                classify_cmd.extend(["--model", embedding_model])
        else:
            if gemini_model:
                classify_cmd.extend(["--gemini-model", gemini_model])
            if gemini_api_key:
                classify_cmd.extend(["--gemini-api-key", gemini_api_key])
        if manifest_path.exists():
            classify_cmd.extend(["--manifest", str(manifest_path)])
        run_command(classify_cmd)
        logging.info("Batch predictions written to %s", predictions_path)
    else:
        logging.info("Batch file not provided; skip classification step. Run classify_title.py manually for interactive mode.")

    if args.visualize:
        visualize_path = Path(args.visualize_output).resolve() if args.visualize_output else output_dir / "embedding_plot.png"
        visualize_cmd = [
            sys.executable,
            str(base_dir / "visualize_clusters.py"),
            "--data",
            str(dataset_path),
            "--assignments",
            str(clustering_output),
            "--output",
            str(visualize_path),
            "--title",
            f"Embeddings by Group (k={args.k})",
        ]
        run_command(visualize_cmd)
        logging.info("Visualization saved to %s", visualize_path)

    logging.info("Pipeline completed. Artifacts stored in %s", output_dir)


if __name__ == "__main__":
    main()
