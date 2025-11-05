# Embedding & Clustering Results Analysis

## Overview
Two end-to-end runs were evaluated on `TitlesforL16HomeWork.csv`:

1. **SentenceTransformer baseline** using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Gemini-enhanced** pipeline targeting `models/text-embedding-004` (captured in the shared screenshots).

Both flows produced normalized embeddings, K-Means cluster allocations (`k=3`), and KNN-based classification for the hold-out title “Inter Miami Announce Lionel Messi Long-Term Contract Renewal.”

---

## 1. K-Means Behaviour with SentenceTransformer Embeddings

- **Mismatch rate:** Significantly higher than Gemini (see “Regular Model K-Means clustering results” screenshot). Multiple Women Soccer headlines spilled into the NBA or transfers clusters, and a few transfer stories drifted towards women’s soccer.
- **Why so many drifts:** the 384-dimension MiniLM vector space is dominated by overlapping lexical cues (“star”, “loan”, “record-breaking”), so cosine similarity groups titles by shared buzzwords rather than league context. Sparse representation of MLS and WSL references makes it difficult for K-Means to form clean boundaries.
- **Nearest-neighbour influence:** KNN recommends titles from mixed sports segments, reinforcing the confusion—many neighbours for women’s football and NBA stories look interchangeable at the sentence level.
- **Visual cues (Regular PCA plot):** Clusters bleed into each other, especially between Women Soccer and Soccer Transfers, demonstrating that the embedding manifold itself lacks separation.

**Takeaway:** The SentenceTransformer baseline is serviceable offline but introduces numerous cluster mis-allocations due to vocabulary overlap, which in turn raises the risk of downstream misclassifications.

---

## 2. K-Means with Gemini Embeddings

- **Observed behaviour (per screenshots):** Gemini’s 3,072-dimension embeddings yield nearly perfect alignment—only a single title falls outside its original group, and the rest of the cluster assignments mirror the ground truth.
- **Why Gemini helps:** the larger, semantically richer embedding space captures league-specific cues (“Angel City”, “Chelsea”, “NBA”) far better, letting K-Means isolate coherent thematic regions.
- **Dependencies:** requires `google-generativeai` and a valid API key; runs are cloud-dependent and slower due to network latency.

**However, new-title classification diverged:**

- Using SentenceTransformer embeddings, the Messi headline is correctly routed to *Soccer Transfers* because one of the nearest neighbours is a transfer article (“Barcelona’s Potential Lewandowski Replacement Wants Transfer ‘Finalized’ in January”), giving cluster `0` the majority vote.
- With Gemini embeddings, the same headline was pulled into the *Women Soccer* cluster despite sharper overall separation. Two of the closest neighbours are women’s soccer stories (“USWNT Icon Christen Press…” and “Alyssa Thompson…”) whose language (“star”, “record-breaking move”) dominates the vote, overwhelming the lone transfers neighbour.
- **Why Gemini still misses:** KNN only considers local neighbours. Even though the Gemini manifold separates clusters cleanly at a global level, sparse representation of MLS/men’s transfer headlines means nearest neighbours remain women’s soccer articles, leading to misclassification.
- **Additional factors:**  
  - The transfers cluster is the smallest cohort; more diverse transfer headlines would strengthen its centroid and neighbour pool.  
  - The default `k=3` and equal weighting make the prediction susceptible to class imbalance—two women’s articles can outvote one transfer article even with higher-quality embeddings.

---

## 3. Comparing the Embedding Models

| Aspect | SentenceTransformer | Gemini |
| --- | --- | --- |
| **Availability** | Offline, no API key required | Requires network + `google-generativeai` + API key |
| **Embedding Dimensionality** | 384 | 3072 |
| **K-Means Alignment** | Higher mismatch rate (multiple cross-cluster allocations) due to lexical overlap | Near-perfect cluster alignment (only one mismatch observed) |
| **KNN New Title** | Correctly assigns Messi headline to *Soccer Transfers* (confidence 0.67) | Misclassifies Messi headline to *Women Soccer* (confidence ~0.67) because nearest neighbours skew female soccer |
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
5. **Data enrichment:** Continue expanding the training corpus—especially transfer-focused and MLS-specific titles—so future embeddings have richer neighbour contexts for borderline cases.
6. **Monitor new-title outcomes:** store predicted clusters and nearest neighbours for review, flagging cases where confidence < 0.7 or where neighbour labels disagree to prompt manual checks.

Overall, Gemini embeddings yield superior cluster separation, but KNN classification accuracy remains contingent on balanced, representative training data. The Messi title misclassification under Gemini highlights that even a high-fidelity embedding space cannot compensate for skewed nearest neighbours. Improving data coverage or adjusting voting logic (e.g., reweighting votes, enriching the transfers cohort) is essential to translate embedding quality into correct real-world routing for new content.
