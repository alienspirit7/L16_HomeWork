# Embedding & Clustering Results Analysis

## Overview
Two end-to-end runs were evaluated on `TitlesforL16HomeWork.csv`:

1. **SentenceTransformer baseline** using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Gemini-enhanced** pipeline targeting `models/text-embedding-004` (captured in the shared screenshots).

Both flows produced normalized embeddings, K-Means cluster allocations (`k=3`), and KNN-based classification for the hold-out title “Inter Miami Announce Lionel Messi Long-Term Contract Renewal.”

---

## 1. K-Means Behaviour with SentenceTransformer Embeddings

- **Mismatch rate:** 1/30 titles (3.3%) disagreed with their original label.  
  - The outlier (`"‘A Calculated Risk’: New Angel City Star Nealy Martin Signs with WSL Giants"`) belongs to *Women Soccer* yet landed in cluster `2` (the NBA-dominated cluster).
- **Why it drifted:** the MiniLM embedding leans heavily on surface words (“star”, “giants”) shared with NBA headlines. With only 384 dimensions, context cues such as “Angel City” (NWSL) and “WSL” did not outweigh generic athletic terms, so cosine distance pulled it toward the NBA centroid.
- **Nearest-neighbour influence:** KNN lists Women Soccer and NBA articles with similar phrasing, showing that lexical overlap still traps the model despite normalization.
- **Visual cues (Regular PCA plot):** The scatter shows overlapping clouds for the two soccer segments; cluster boundaries are fuzzy, making single misassignments expected.

**Takeaway:** Conventional embeddings give acceptable accuracy and are lightweight/offline, but subtle domain phrases can overshadow semantic intent, letting K-Means allocate a minority of women’s football stories to the NBA cluster.

---

## 2. K-Means with Gemini Embeddings

- **Observed behaviour (per screenshots):** Gemini’s 3,072-dimension vectors separate the three topical regions cleanly—the K-Means summary shows near-zero mismatch between cluster IDs and original groups.
- **Why Gemini helps:** larger embedding space captures richer semantic relations (club names, league acronyms, player references) and better differentiates women’s football from NBA jargon, so centroids align with the intended topics.
- **Dependencies:** requires `google-generativeai` and a valid API key; runs are cloud-dependent and slower due to network latency.

**However, new-title classification still struggled:**

- The hold-out Messi headline aligns with *Soccer Transfers* in both embedding runs (SentenceTransformer and Gemini) because one of the nearest neighbours is a transfer article (“Barcelona’s Potential Lewandowski Replacement Wants Transfer ‘Finalized’ in January”), so the vote favours cluster `0`.  
- **Residual risk:** the other two neighbours are Women Soccer headlines with similar “star/record move” language, meaning any small shift (different `k`, added data, or altered normalization) could tip the result toward the wrong class. The example highlights how fragile KNN can be when clusters have overlapping semantics and imbalanced counts.  
- **Additional factors:**  
  - The transfers cluster remains the smallest segment; adding more MLS or men’s transfer stories would solidify its centroid.  
  - With `k=3`, a future dataset update might deliver two women’s soccer neighbours, causing a misclassification even though the embeddings themselves are high quality.

---

## 3. Comparing the Embedding Models

| Aspect | SentenceTransformer | Gemini |
| --- | --- | --- |
| **Availability** | Offline, no API key required | Requires network + `google-generativeai` + API key |
| **Embedding Dimensionality** | 384 | 3072 |
| **K-Means Alignment** | Small mismatch (1/30) due to lexical overlap | Near-perfect cluster alignment in sample runs |
| **KNN New Title** | Correctly assigns Messi headline to *Soccer Transfers* (confidence 0.67) | Also assigns Messi headline to *Soccer Transfers* (confidence 0.67), but neighbours include Women Soccer titles |
| **Runtime** | Fast, lightweight | Slower per request, subject to API rate limits |
| **Explainability** | Easier to reason about drift (lexical) | Higher semantic nuance but harder to debug without centroid introspection |

**Dependency on Embedding Quality:**  
K-Means clustering quality is tightly coupled to the embedding manifold. Better embeddings → tighter intra-cluster cohesion → cleaner centroid assignments. Yet downstream KNN still depends on local neighbour labels; if the training corpus is imbalanced or lacks representative samples, even high-quality embeddings can propagate the wrong class.

---

## 4. Lessons & Recommendations

1. **Augment the dataset:** add more *Soccer Transfers* headlines, especially MLS-focused, so KNN has balanced examples and K-Means learns a distinct centroid for transfers.
2. **Tune KNN parameters:**  
   - Lower `k` to 1 or weight votes inversely with distance to reduce cross-cluster contamination.  
   - Consider distance-threshold rejection for low-confidence predictions.
3. **Leverage cluster summaries:** Export centroid keywords (e.g., via PCA loadings or top TF-IDF tokens) to interpret why neighbours sway decisions.
4. **Hybrid approach:** Use Gemini for primary embeddings when connectivity is available; fall back to SentenceTransformer offline, but adjust post-processing (e.g., manual rules for certain leagues) to catch known failure modes.
5. **Monitor new-title outcomes:** store predicted clusters and nearest neighbours for review, flagging cases where confidence < 0.7 or where neighbour labels disagree to prompt manual checks.

Overall, Gemini embeddings yield superior cluster separation, but KNN classification accuracy remains contingent on balanced, representative training data. Improving data coverage or adjusting voting logic is essential to translate embedding quality into correct real-world routing for new content.
