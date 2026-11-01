# Santander Product Recommendation

Predict which banking products a customer will acquire next month, given their current portfolio, demographics, and history.

**Kaggle competition**: [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation)

## Results (Private Leaderboard)

| Version | Approach | Private Score |
|---------|----------|---------------|
| Naive popularity | Top-7 historical additions, masked | 0.01813 |
| Collaborative only | LightFM, no side features | 0.01945 |
| Hybrid v1 | LightFM + simple demographics | 0.02269 |
| **Final (Model 3)** | **LightFM + ordinal encoding + rare labels + Option A interactions** | **0.02502** |

Improvement over popularity baseline: **~38%**.

## Key Insight

The competition asks for **acquisitions**, not holdings. Only ~3.2–3.8% of customers add a product in any given month. The score is therefore driven entirely by ranking quality on the small set of buyers.

## Interaction Matrix (Option A vs Option C)

| Construction | Private Score |
|--------------|---------------|
| **Option A** (historical rating > 0) | **0.02491** |
| Option C (binary state at cutoff) | 0.02392 |

Abandoned products supply useful co-occurrence signal. Theoretical alignment with inference-time masking does **not** outweigh the value of historical interactions. Final models use Option A.

## Repository Contents

| File | Description |
|------|-------------|
| `santander_f_v3.ipynb` | Full pipeline using **Polars** (original production version, memory-efficient) |
| `README.md` | This file |

## Requirements

```bash
pip install lightfm-next polars pandas numpy scikit-learn
```

- **Polars notebook**: recommended for full 13.6 M-row runs (Colab High-RAM or equivalent).
- **Pandas notebook**: easier to read and modify; still requires substantial RAM for the complete dataset.

## Data

Download from Kaggle:

```bash
kaggle competitions download -c santander-product-recommendation
unzip santander-product-recommendation.zip
unzip train_ver2.csv.zip test_ver2.csv.zip sample_submission.csv.zip
```

## Pipeline Overview

1. **Schema diagnostics & curation** – type inference driven by data facts, not assumptions.
2. **Target characterisation** – user variability, products-per-user, monthly additions, net flow, rating continuum.
3. **Interaction matrix** – Option A (empirically superior).
4. **Models**
   - Popularity baseline
   - Collaborative filtering only
   - Hybrid LightFM (demographics + ordinal encoding + rare labels)
5. **Temporal validation** (April → May)
6. **Final submission** (May → June)

## What Worked / What Did Not

**Worked**
- Option A interactions
- No sample weights (`rating_frecuencia` is anti-correlated with the target)
- Ordinal encoding of all categoricals + rare-label grouping (1%)
- No numerical binning
- Outlier clipping + flag

**Did not work**
- Diagnostic-derived features guided by dataset understanding rather than target alignment
- Trajectory features (mostly zeros → embedding collapse in LightFM)
- One-hot encoding
- Using test-month features at inference

## Path Beyond 0.025

LightFM’s shared latent space is a bottleneck with only 24 products. The most promising route could possible be to  **per-product binary classifiers** (XGBoost / LightGBM), which remove the shared-space constraint and can exploit sparse trajectory signals more effectively.

## License

Code released for educational purposes. Competition data remains subject to Kaggle terms.
