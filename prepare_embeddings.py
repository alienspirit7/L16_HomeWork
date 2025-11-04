#!/usr/bin/env python3
"""
Prepare embeddings for title data using sentence-transformers/all-MiniLM-L6-v2.
Reads a two-column CSV (title, group), produces raw and normalized embeddings,
and persists outputs alongside a metadata manifest.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SUPPORTED_FORMATS = {"parquet", "jsonl", "csv"}
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RunManifest:
    script: str
    version: str
    timestamp_utc: str
    input_path: str
    input_rows: int
    processed_rows: int
    skipped_rows: int
    model_id: str
    embedding_dim: int
    normalization: str
    seed: Optional[int]
    elapsed_seconds: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embeddings and normalized vectors for CSV title data."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV with columns 'title' and 'group'.")
    parser.add_argument("--output", required=True, help="Base path (without extension for parquet/csv) for processed dataset.")
    parser.add_argument(
        "--format",
        choices=sorted(SUPPORTED_FORMATS),
        default="parquet",
        help="Output format for processed dataset.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional path to write manifest JSON. Defaults to <output>.manifest.json.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer model identifier.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate titles in the input data.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def read_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected_cols = {"title", "group"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df = df[list(expected_cols)]
    df.rename(columns={"group": "original_group"}, inplace=True)
    df["title"] = df["title"].astype(str).str.strip()
    df["original_group"] = df["original_group"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["original_group"] != "")]
    return df.reset_index(drop=True)


def enforce_uniqueness(df: pd.DataFrame, allow_duplicates: bool) -> pd.DataFrame:
    if allow_duplicates:
        return df
    duplicate_titles = df[df.duplicated("title", keep=False)]["title"].unique()
    if duplicate_titles.size > 0:
        raise ValueError(
            "Duplicate titles detected. Re-run with --allow-duplicates to proceed. "
            f"Examples: {duplicate_titles[:5]!r}"
        )
    return df


def embed_titles(model_id: str, titles: List[str]) -> np.ndarray:
    logging.info("Loading model %s", model_id)
    model = SentenceTransformer(model_id)
    logging.info("Encoding %d titles", len(titles))
    embeddings = model.encode(titles, convert_to_numpy=True, show_progress_bar=len(titles) > 10)
    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def serialize_vector(vec: np.ndarray) -> List[float]:
    return vec.astype(float).tolist()


def write_dataset(
    df: pd.DataFrame,
    output_path: Path,
    fmt: str,
) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        path = output_path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        path = output_path.with_suffix(".csv")
        df.to_csv(path, index=False)
    elif fmt == "jsonl":
        path = output_path.with_suffix(".jsonl")
        with path.open("w", encoding="utf-8") as fh:
            for record in df.to_dict(orient="records"):
                fh.write(json.dumps(record) + "\n")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    logging.info("Wrote dataset to %s", path)
    return path


def write_manifest(manifest_path: Path, manifest: RunManifest) -> None:
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    logging.info("Wrote manifest to %s", manifest_path)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    start_time = time.time()
    input_path = Path(args.input)
    output_base = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_base.with_suffix(".manifest.json")

    logging.info("Reading input %s", input_path)
    raw_df = read_input_csv(input_path)
    input_rows = len(raw_df)
    logging.info("Loaded %d rows", input_rows)

    filtered_df = enforce_uniqueness(raw_df, args.allow_duplicates)
    processed_rows = len(filtered_df)
    skipped_rows = input_rows - processed_rows

    if processed_rows == 0:
        raise ValueError("No valid rows to process after applying filters.")

    embeddings = embed_titles(args.model, filtered_df["title"].tolist())
    normalized = normalize_embeddings(embeddings)

    embedding_dim = embeddings.shape[1]
    logging.info("Embedding dimension: %d", embedding_dim)

    dataset = filtered_df.copy()
    dataset["embedding"] = [serialize_vector(vec) for vec in embeddings]
    dataset["normalized_embedding"] = [serialize_vector(vec) for vec in normalized]

    output_path = write_dataset(dataset, output_base, args.format)

    manifest = RunManifest(
        script="prepare_embeddings.py",
        version="1.0.0",
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        input_path=str(input_path.resolve()),
        input_rows=input_rows,
        processed_rows=processed_rows,
        skipped_rows=skipped_rows,
        model_id=args.model,
        embedding_dim=embedding_dim,
        normalization="l2",
        seed=None,
        elapsed_seconds=round(time.time() - start_time, 3),
    )
    write_manifest(manifest_path, manifest)

    logging.info(
        "Processing complete. processed=%d skipped=%d elapsed=%.2fs output=%s",
        processed_rows,
        skipped_rows,
        manifest.elapsed_seconds,
        output_path,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Failed: %s", exc)
        sys.exit(1)
