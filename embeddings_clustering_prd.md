# Embedding & Clustering Pipeline PRD

## 1. Product Overview
- Provide an end-to-end toolkit for clustering content titles from a two-column CSV (`title`, `group`), preserving legacy grouping while surfacing K-Means-derived insights and enabling fast classification of new titles.
- Deliver three complementary CLI scripts: data preparation and embedding, K-Means clustering audit, and KNN-based classification for user-supplied titles.
- Core user value: faster understanding of thematic groupings, drift detection against original labels, and repeatable classification of fresh titles.

## 2. Goals & Success Metrics
- 100% of valid CSV rows produce normalized embeddings while retaining original group metadata, using either local SentenceTransformer or hosted Gemini embeddings.
- K-Means outputs highlight agreement/drift between historical and algorithmic groupings, including mismatch rates, distances, and silhouette scores.
- `classify_title.py` responds within 3 seconds per title on a laptop-class machine and achieves ≥80% accuracy on a validation set curated by stakeholders, exposing nearest-neighbour context (original group + cluster IDs).
- Clear CLI ergonomics (`--help`, actionable errors) and optional JSON config support to reduce repeated flag inputs for analysts.
- Pipeline orchestration (`run_pipeline.py`) finishes end-to-end processing (embedding → clustering → optional classification + visualization) without manual intervention under 5 minutes for 10k records.

## 3. Primary Users & Key Flows
1. **Data Analyst** loads CSV → runs embedding preparation → hands normalized dataset to clustering script.
2. **Product Owner** runs K-Means script → reviews side-by-side table of original vs new clusters → exports report for discussions.
3. **Support Agent** runs classifier → inputs single or batch titles → receives predicted cluster with supporting nearest neighbors for routing decisions.

## 4. Functional Requirements
### 4.1 `prepare_embeddings.py`
- CLI arguments: `--input`, `--output`, `--format`, `--provider` (`sentence-transformers` or `gemini`), `--model`, `--gemini-model`, `--gemini-api-key`, `--config`, `--allow-duplicates`, `--log-level`.
- Accept JSON config overrides (e.g., `config/embedding.json`) to supply provider, model ID, and API key; CLI flags take precedence.
- Validate inputs case-insensitively for `title` / `group`; trim whitespace, drop empty rows, optionally halt on duplicates.
- Generate embeddings via SentenceTransformers (default `all-MiniLM-L6-v2`) or Google Gemini `text-embedding-004` using `google-generativeai`.
- Apply L2 normalization and persist both raw + normalized vectors. Default output is Parquet; JSONL or CSV available.
- Emit manifest containing provider, model_id, embedding_dim, normalization, counts, timestamps. Auto-create directories for outputs.
- Structured logging summarises processed/skipped counts and elapsed time; explicit errors for missing API key or package when using Gemini.

### 4.2 `run_kmeans.py`
- Inputs: dataset path, `--k`, `--seed`, `--output`, `--report`, `--centroids`, optional `--manifest`, `--log-level`.
- Robustly parse vector columns (JSON, list, numpy string dumps); ensure normalized embeddings present.
- Execute scikit-learn K-Means (`n_init="auto"`) with reproducible seeding; compute distances to centroids.
- Produce record-level CSV including `title`, `original_group`, `embedding`, `normalized_embedding`, `cluster_id`, `distance_to_centroid`.
- Generate aggregate metrics: cluster sizes, majority original group, mismatch rate, inertia, silhouette (cosine) where feasible.
- Export Markdown or CSV reports; persist centroid JSON with provider/model metadata from manifest.
- Auto-create directories prior to writing outputs; log human-readable summaries.

### 4.3 `classify_title.py`
- CLI arguments: `--assignments`, `--k`, `--provider`, `--model`, `--gemini-model`, `--gemini-api-key`, `--config`, `--batch`, `--output`, `--manifest`, `--log-level`.
- Load clustering assignments (CSV/Parquet/JSONL) with serialized embeddings; parse vectors robustly.
- Resolve provider/model-id from CLI → config → manifest. Support Gemini embeddings with `google-generativeai`.
- Build cosine NearestNeighbors index on normalized embeddings; interactive mode embeds titles on the fly.
- Batch mode outputs CSV with predicted cluster, confidence, nearest neighbour titles, original groups, cluster IDs, and cluster-distance pairs.
- Clear error messaging for missing files, incompatible provider/model, or absent API credentials.

### 4.4 `visualize_clusters.py`
- Required inputs: `--data` (normalized dataset), `--assignments`, `--output`; optional `--title`, `--alpha`, `--log-level`.
- Reduce embeddings to 2D via PCA; scatter plot uses marker shapes for original groups and colors for K-Means clusters.
- Annotate centroid labels and provide dual legends (groups vs clusters). Save PNG to specified path.

### 4.5 `run_pipeline.py`
- Orchestrates embedding → clustering → optional batch classification → optional visualization.
- Accepts shared config file, provider/model overrides, batch classification input, visualization toggle.
- Handles directory creation, passes resolved provider/model/API key downstream, and logs each command executed.

## 5. Data & Model Specifications
- Input CSV schema: `title` (string), `group` (string). Reject rows with missing or purely whitespace values unless explicitly overridden.
- Embedding providers:
  - Default: `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
  - Optional: Google Gemini `text-embedding-004` (3072-dim) via `google-generativeai`.
- Manifest captures provider, model ID, embedding dimension, normalization, row counts, elapsed time.
- Batch classification outputs storing neighbour metadata: original group, cluster ID, distance.

## 6. Non-Functional Requirements
- Scripts run with Python 3.9+ on macOS/Linux. Dependencies managed via `requirements.txt` (numpy, pandas, pyarrow, scikit-learn, sentence-transformers, matplotlib, google-generativeai).
- Provide structured logging to stdout at INFO level; enable DEBUG with `--log-level debug`.
- Support reproducible results: respect `--seed`; log it in manifest.
- Ensure runtime for 10k titles remains under 2 minutes for embedding script on reference hardware (Apple M1 or equivalent).
- Handle offline environments gracefully: informative error when Gemini provider requested without installed package or API key.

## 7. Error Handling & Validation
- Fail fast on missing columns, unreadable files, or unsupported formats; return non-zero exit codes.
- Collect and report skipped rows (e.g., empty titles, embedding failures) in a side file when `--skip-report` provided.
- Anticipate SentenceTransformers download failures; retry once, then provide remediation steps.
- Detect missing `google-generativeai` package or API key when provider=gemini and surface actionable error.
- Validate normalized dataset schema before clustering or classification; abort with clear instructions if mismatched.

## 8. Testing Strategy
- Unit tests: CSV validation, embedding normalization function, K-Means parameter validation, KNN vote aggregation.
- Integration tests: End-to-end run using sample CSV to confirm pipelines interoperate and produce deterministic outputs with fixed seed.
- Regression tests: classify known titles and assert predicted clusters remain stable across changes.
- Use pytest with fixtures for sample data; include golden files for K-Means summary outputs.

## 9. Out of Scope
- Web UI or dashboard layers.
- Automated scheduling or monitoring.
- Incremental or streaming clustering updates.
- Model fine-tuning or experimentation beyond the all-MiniLM-L6-v2 baseline.

## 10. Open Questions
- Preferred storage format for normalized dataset and reports (Parquet vs JSONL vs CSV) — currently defaulting to Parquet.
- Destination for generated artifacts (local repo vs shared storage) and retention policy.
- Need for anonymization or PII handling policies for content titles.
- Target testing strategy (unit + integration) and CI integration timeline.
