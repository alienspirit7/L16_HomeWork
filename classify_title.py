#!/usr/bin/env python3
"""
Classify new titles into K-Means-derived clusters using KNN over normalized embeddings.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Neighbor:
    title: str
    original_group: str
    cluster_id: int
    distance: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify titles based on K-Means clustering results using KNN (k=3 by default)."
    )
    parser.add_argument("--assignments", required=True, help="Record-level clustering output from run_kmeans.py.")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors for KNN voting (default 3).")
    parser.add_argument("--model", help="Override embedding model identifier. Defaults to manifest or MiniLM.")
    parser.add_argument("--batch", help="Optional CSV/JSONL with column 'title' for batch classification.")
    parser.add_argument("--output", help="Optional path to write batch predictions (CSV).")
    parser.add_argument("--manifest", help="Manifest JSON path from prepare_embeddings.py for metadata.")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"])
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_assignments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Assignments file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported assignments format: {ext}")
    required_cols = {"title", "original_group", "normalized_embedding", "cluster_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Assignments missing required columns: {sorted(missing)}")
    return df.reset_index(drop=True)


def ensure_vector_list(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            from ast import literal_eval

            parsed = literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected list-like string, got {type(parsed)}")
        return [float(v) for v in parsed]
    if isinstance(value, tuple):
        return [float(v) for v in value]
    raise TypeError(f"Unsupported vector type: {type(value)}")


def dataframe_to_matrix(df: pd.DataFrame, column: str) -> np.ndarray:
    vectors = [ensure_vector_list(v) for v in df[column]]
    return np.asarray(vectors, dtype=float)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def load_manifest(assignments_path: Path, manifest_override: Optional[Path]) -> Optional[Dict[str, Any]]:
    if manifest_override:
        candidate = manifest_override
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
        logging.warning("Manifest override not found: %s", candidate)
        return None
    candidate = assignments_path.with_suffix(".manifest.json")
    if candidate.exists():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logging.warning("Failed to parse manifest %s: %s", candidate, exc)
    return None


def resolve_model_id(args_model: Optional[str], manifest: Optional[Dict[str, Any]]) -> str:
    if args_model:
        return args_model
    if manifest and manifest.get("model_id"):
        return manifest["model_id"]
    return DEFAULT_MODEL


def load_model(model_id: str) -> SentenceTransformer:
    logging.info("Loading model %s", model_id)
    return SentenceTransformer(model_id)


def build_index(vectors: np.ndarray, metric: str = "cosine") -> NearestNeighbors:
    logging.info("Building NearestNeighbors index over %d vectors", vectors.shape[0])
    nn = NearestNeighbors(metric=metric, n_neighbors=min(50, vectors.shape[0]))
    nn.fit(vectors)
    return nn


def predict_cluster(
    title: str,
    vector: np.ndarray,
    index: NearestNeighbors,
    all_vectors: np.ndarray,
    clusters: np.ndarray,
    titles: List[str],
    original_groups: List[str],
    k: int,
) -> Tuple[int, float, List[Neighbor]]:
    distances, indices = index.kneighbors(vector.reshape(1, -1), n_neighbors=min(k, all_vectors.shape[0]))
    distances = distances[0]
    indices = indices[0]

    votes: Dict[int, List[float]] = {}
    neighbors: List[Neighbor] = []
    for idx, dist in zip(indices, distances):
        cluster_id = int(clusters[idx])
        votes.setdefault(cluster_id, []).append(dist)
        neighbors.append(
            Neighbor(
                title=titles[idx],
                original_group=original_groups[idx],
                cluster_id=cluster_id,
                distance=float(dist),
            )
        )

    # Determine winning cluster: highest vote count, tie broken by lowest average distance.
    winner = None
    winner_score = (-1, float("inf"))
    for cluster_id, dist_list in votes.items():
        count = len(dist_list)
        avg_dist = sum(dist_list) / count
        score = (count, -avg_dist)
        if score > winner_score:
            winner = cluster_id
            winner_score = score

    confidence = len(votes[winner]) / sum(len(v) for v in votes.values())
    return winner, confidence, neighbors


def embed_titles(model: SentenceTransformer, titles: List[str]) -> np.ndarray:
    embeddings = model.encode(titles, convert_to_numpy=True, show_progress_bar=len(titles) > 10)
    return normalize_vectors(embeddings)


def classify_interactive(
    model: SentenceTransformer,
    index: NearestNeighbors,
    base_vectors: np.ndarray,
    clusters: np.ndarray,
    titles: List[str],
    original_groups: List[str],
    k: int,
) -> None:
    logging.info("Entering interactive mode. Press Enter on empty input to exit.")
    while True:
        try:
            user_input = input("Enter title (blank to quit): ").strip()
        except EOFError:
            print()  # ensure newline
            break
        if not user_input:
            break
        embedded = embed_titles(model, [user_input])[0]
        winner, confidence, neighbors = predict_cluster(
            user_input,
            embedded,
            index,
            base_vectors,
            clusters,
            titles,
            original_groups,
            k,
        )
        print(f"Predicted cluster: {winner} (confidence {confidence:.2f})")
        print("Nearest neighbors:")
        for neighbor in neighbors[:k]:
            print(
                f"  - '{neighbor.title}' | original_group={neighbor.original_group} | "
                f"cluster={neighbor.cluster_id} | distance={neighbor.distance:.4f}"
            )
        print("")


def load_batch_titles(path: Path) -> List[str]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "title" not in df.columns:
            raise ValueError("Batch CSV must contain a 'title' column.")
        titles = df["title"].astype(str).str.strip().tolist()
    elif path.suffix.lower() == ".jsonl":
        titles = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "title" not in record:
                    raise ValueError("Each JSONL record must contain a 'title' field.")
                titles.append(str(record["title"]).strip())
    else:
        raise ValueError("Unsupported batch format. Provide CSV or JSONL.")
    return [title for title in titles if title]


def write_batch_results(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logging.info("Wrote batch results to %s", path)
    return path


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    assignments_path = Path(args.assignments)
    assignments = load_assignments(assignments_path)
    base_vectors = dataframe_to_matrix(assignments, "normalized_embedding")
    clusters = assignments["cluster_id"].to_numpy(dtype=int)
    titles = assignments["title"].tolist()
    original_groups = assignments["original_group"].tolist()

    if args.k <= 0:
        raise ValueError("k must be positive.")
    if args.k > len(assignments):
        logging.warning("Requested k=%d exceeds dataset size; using k=%d", args.k, len(assignments))

    manifest = load_manifest(assignments_path, Path(args.manifest) if args.manifest else None)
    model_id = resolve_model_id(args.model, manifest)
    model = load_model(model_id)

    index = build_index(base_vectors)

    if args.batch:
        batch_titles = load_batch_titles(Path(args.batch))
        logging.info("Classifying %d titles (batch mode)", len(batch_titles))
        embedded = embed_titles(model, batch_titles)
        rows: List[Dict[str, Any]] = []
        for title, vec in zip(batch_titles, embedded):
            cluster_id, confidence, neighbors = predict_cluster(
                title,
                vec,
                index,
                base_vectors,
                clusters,
                titles,
                original_groups,
                args.k,
            )
            rows.append(
                {
                    "title": title,
                    "predicted_cluster": cluster_id,
                    "confidence": round(confidence, 4),
                    "nearest_titles": "; ".join(n.title for n in neighbors[:args.k]),
                    "nearest_original_groups": "; ".join(n.original_group for n in neighbors[:args.k]),
                    "nearest_distances": "; ".join(f"{n.distance:.4f}" for n in neighbors[:args.k]),
                }
            )
        if args.output:
            write_batch_results(Path(args.output), rows)
        else:
            for row in rows:
                logging.info(
                    "Title='%s' predicted_cluster=%s confidence=%.2f nearest=%s",
                    row["title"],
                    row["predicted_cluster"],
                    row["confidence"],
                    row["nearest_titles"],
                )
    else:
        classify_interactive(
            model,
            index,
            base_vectors,
            clusters,
            titles,
            original_groups,
            args.k,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed: %s", exc)
        sys.exit(1)
