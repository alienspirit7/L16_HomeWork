# Embedding & Clustering Pipeline PRD

## 1. Product Overview
- Provide an end-to-end toolkit for clustering content titles from a two-column CSV (`title`, `group`), preserving legacy grouping while surfacing K-Means-derived insights and enabling fast classification of new titles.
- Deliver three complementary CLI scripts: data preparation and embedding, K-Means clustering audit, and KNN-based classification for user-supplied titles.
- Core user value: faster understanding of thematic groupings, drift detection against original labels, and repeatable classification of fresh titles.

## 2. Goals & Success Metrics
- 100% of valid CSV rows produce normalized embeddings while retaining their original group metadata.
- K-Means output highlights alignment and mismatches between historical and algorithmic groupings, including quantitative cluster metrics.
- `classify_title.py` responds within 3 seconds per title on a laptop-class machine and achieves ≥80% accuracy on a validation set curated by stakeholders.
- Clear CLI ergonomics (`--help`, actionable errors) enabling analysts to run scripts without engineering support.

## 3. Primary Users & Key Flows
1. **Data Analyst** loads CSV → runs embedding preparation → hands normalized dataset to clustering script.
2. **Product Owner** runs K-Means script → reviews side-by-side table of original vs new clusters → exports report for discussions.
3. **Support Agent** runs classifier → inputs single or batch titles → receives predicted cluster with supporting nearest neighbors for routing decisions.

## 4. Functional Requirements
### 4.1 `prepare_embeddings.py`
- CLI arguments: `--input` (path to CSV), `--output` (path for normalized dataset), optional `--model` defaulting to `sentence-transformers/all-MiniLM-L6-v2`, `--allow-duplicates`, `--log-level`.
- Validate inputs: confirm columns `title`, `group`; enforce UTF-8, trim whitespace, reject blank titles unless `--allow-duplicates`.
- Generate embeddings via the all-MiniLM-L6-v2 model (using SentenceTransformers); store original `title`, `group`, raw embedding vector.
- Apply L2 normalization to each vector; persist structured output (default Parquet, `--format` allows JSONL/CSV) plus metadata manifest (timestamp, model ID, normalization details, script version).
- Emit run summary: processed count, skipped rows with reasons, elapsed time, and output locations.

### 4.2 `run_kmeans.py`
- Inputs: normalized dataset path, optional `--k` default 3, `--seed` for reproducibility, `--report` path, `--log-level`.
- Load normalized vectors only for clustering; retain original title/group metadata for join.
- Execute K-Means (scikit-learn) with deterministic seed; store cluster centroids, inertia, silhouette score.
- Produce record-level output: `title`, `original_group`, `cluster_id`, distance to assigned centroid, top contributing dimensions (optional debug flag).
- Generate aggregate metrics: cluster sizes, mismatch rate vs original groups, silhouette score, centroid norms.
- Export detailed report in CSV by default; Markdown option summarizing metrics and sample records; optionally persist centroids to disk for reuse.

### 4.3 `classify_title.py`
- CLI arguments: `--data` (normalized dataset), `--centroids` (optional path), `--k 3` for KNN votes, `--batch` for CSV/JSONL inputs, interactive mode when no batch file provided.
- Load embeddings dataset, metadata manifest, and precomputed centroids; validate models align (same dimensions, normalization method).
- Embed new title(s) using all-MiniLM-L6-v2 (same preprocessing pipeline) and L2-normalize.
- Perform KNN (scikit-learn or faiss-lite) over stored vectors; output predicted cluster, confidence (vote ratio), top 3 nearest neighbor titles with original groups and distances.
- For batch runs, write structured results to stdout or file (`--output`).
- Provide graceful error messages for missing artifacts, incompatible model versions, or empty inputs.

## 5. Data & Model Specifications
- Input CSV schema: `title` (string), `group` (string). Reject rows with missing or purely whitespace values unless explicitly overridden.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, loaded locally via SentenceTransformers; document model checksum/version in metadata.
- Normalization: L2 norm per embedding vector; store normalization flag and vector dimensionality (384) in manifest.
- Metadata manifest fields: script name/version, run timestamp (UTC ISO8601), input file hash, model id, normalization type, row counts.

## 6. Non-Functional Requirements
- Scripts run with Python 3.10+ on macOS/Linux. Manage dependencies via `requirements.txt` (sentence-transformers, scikit-learn, pandas, pyarrow, numpy).
- Provide structured logging to stdout at INFO level; enable DEBUG with `--log-level debug`.
- Support reproducible results: respect `--seed`; log it in manifest.
- Ensure runtime for 10k titles remains under 2 minutes for embedding script on reference hardware (Apple M1 or equivalent).

## 7. Error Handling & Validation
- Fail fast on missing columns, unreadable files, or unsupported formats; return non-zero exit codes.
- Collect and report skipped rows (e.g., empty titles, embedding failures) in a side file when `--skip-report` provided.
- Anticipate SentenceTransformers download failures; retry once, then provide remediation steps.
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
- Preferred storage format for normalized dataset and reports (Parquet vs JSONL vs CSV).
- Destination for persisted artifacts (local folder layout, cloud bucket?).
- Need for anonymization or PII handling policies for content titles.
- Desired frequency of reruns and whether automation hooks are needed.

## 11. Next Steps
1. Confirm artifact storage formats and directory conventions.
2. Define CLI UX details (argument names, default paths).
3. Prototype `prepare_embeddings.py` against `TitlesforL16HomeWork.csv`.
4. Draft testing plan and integrate into CI (if applicable).
