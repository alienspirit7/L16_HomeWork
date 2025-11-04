# Title Embedding & Clustering Toolkit

This project processes a CSV of content titles (`title`, `group`) to generate embeddings, run K-Means clustering, and classify new titles with a KNN vote. The toolchain uses the `sentence-transformers/all-MiniLM-L6-v2` model to keep everything local and reproducible.

## File Structure
````text
.
├── TitlesforL16HomeWork.csv      # Sample input file
├── README.md                     # Project overview and usage instructions
├── embeddings_clustering_prd.md  # Product requirements document
├── requirements.txt              # Python dependencies
├── prepare_embeddings.py         # Step 1: embed & normalize titles
├── run_kmeans.py                 # Step 2: run K-Means and generate reports
├── classify_title.py             # Step 3: classify titles via KNN
├── visualize_clusters.py         # Optional: generate PCA scatter plots
└── run_pipeline.py               # Orchestrates the full end-to-end flow
````

## Prerequisites
- Python 3.10 or newer
- Recommended: virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`)
- Optional: Google Gemini API key exported as `GEMINI_API_KEY` if you plan to embed with Gemini
- Optional: JSON config file (e.g., `config/embedding.json`) to store provider, model IDs, and API keys

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start (Orchestrated Pipeline)
`run_pipeline.py` ties all scripts together. It prepares embeddings, runs K-Means, and optionally classifies a batch of new titles.

```bash
python run_pipeline.py \
  --input TitlesforL16HomeWork.csv \
  --output-dir pipeline_output \
  --format parquet \
  --embedding-provider sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --config config/embedding.json \  # optional shared config
  --k 3 \
  --batch new_titles.csv \
  --visualize
```

Outputs (within `pipeline_output/` by default):
- `embeddings.parquet` (or `.csv` / `.jsonl`) and `embeddings.manifest.json`
- `clustering_results.csv`
- `kmeans_report.md`
- `centroids.json`
- `batch_predictions.csv` (only when `--batch` is supplied)
- `embedding_plot.png` (only when `--visualize` is supplied)

Flags `--batch`, `--visualize`, and `--embedding-model` are optional; omit them if you do not need batch classification, a visualization, or a custom model override. Use `--visualize-output` to customize the image path.

Switch to Gemini embeddings by replacing the provider flags, for example:

```bash
python run_pipeline.py \
  --input TitlesforL16HomeWork.csv \
  --output-dir pipeline_output \
  --format parquet \
  --embedding-provider gemini \
  --gemini-model models/text-embedding-004 \
  --gemini-api-key "$GEMINI_API_KEY" \
  --k 3 \
  --batch new_titles.csv \
  --visualize
```

If `--batch` is omitted, the pipeline skips classification and reminds you how to run the interactive classifier manually.

## Script-by-Script Usage

### 1. Generate Embeddings
```bash
python prepare_embeddings.py \
  --input TitlesforL16HomeWork.csv \
  --output processed/embeddings \
  --format parquet \
  --config config/embedding.json
```
Key options:
- `--provider` selects the embedding backend (`sentence-transformers` or `gemini`).
- `--model` to change the SentenceTransformer model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--gemini-model` to choose a Gemini embedding model (default: `models/text-embedding-004`)
- `--gemini-api-key` (or `GEMINI_API_KEY` env var) supplies credentials when using Gemini
- `--config` supplies a JSON file with any of the above fields (e.g., `{"provider": "gemini", "gemini_api_key": "..."}`)
- `--allow-duplicates` to process duplicate titles
- `--manifest` to override where the JSON manifest is written

### 2. Run K-Means Clustering
```bash
python run_kmeans.py \
  --data processed/embeddings.parquet \
  --output processed/clustering_results.csv \
  --report processed/kmeans_report.md \
  --centroids processed/centroids.json \
  --k 3 \
  --seed 42
```
Outputs:
- Record-level assignments (`clustering_results.csv`) listing title, original group, assigned cluster, and distance to centroid
- Aggregate report (CSV or Markdown) with cluster sizes, majority groups, mismatch rate, and silhouette score
- Centroid export for reuse (`centroids.json`)

### 3. Classify New Titles (KNN)
Interactive mode:
```bash
python classify_title.py \
  --assignments processed/clustering_results.csv \
  --k 3 \
  --provider sentence-transformers  # omit to use manifest provider
```

Batch mode:
```bash
python classify_title.py \
  --assignments processed/clustering_results.csv \
  --k 3 \
  --batch new_titles.csv \
  --output processed/batch_predictions.csv
```
Batch file must contain a `title` column (CSV) or field (JSONL). Output lists predicted clusters, confidence, nearest neighbor titles, their original groups, K-Means clusters, and distances. Add `--provider gemini --gemini-api-key ...` when classifying with Gemini-based embeddings (model defaults to the manifest value).
In batch outputs, the `nearest_cluster_distances` column pairs each neighbor's cluster ID with its distance (e.g., `0:0.1234; 1:0.4567`).

### 4. Visualize Embeddings
```bash
python visualize_clusters.py \
  --data processed/embeddings.parquet \
  --assignments processed/clustering_results.csv \
  --output processed/embedding_plot.png
```
Generates a PCA scatter plot where marker shapes reflect the original groups (circle, triangle, square, etc.) and colors represent K-Means clusters (0, 1, 2), with centroids annotated for quick visual inspection of separation.

## Notes & Tips
- All scripts accept `--log-level` for verbose logging (`debug`, `info`, etc.).
- Manifests generated by `prepare_embeddings.py` include model metadata and are automatically reused by the orchestrator to keep embeddings and clustering in sync.
- `run_kmeans.py` computes cosine-based silhouette scores when enough samples are available; otherwise, the metric is reported as `N/A`.
- For reproducibility, control seeds with `--seed` (K-Means) and run on consistent dependency versions defined in `requirements.txt`.
- Gemini usage requires the `google-generativeai` dependency (included here) and an active API key/quota.
