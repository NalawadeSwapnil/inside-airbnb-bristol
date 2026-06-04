# StayPriceML — Dynamic Pricing & Revenue Analysis for Short-Term Rentals

Short-term rental platforms operate in a highly dynamic pricing environment where the difference
between a well-calibrated nightly rate and an arbitrary one can represent tens of thousands of
pounds in annual revenue per property. This project develops a data-driven revenue analysis
framework using InsideAirbnb Bristol listings data to segment properties by performance tier,
diagnose why low performers underperform, and quantify the key drivers of price and occupancy.
The methodology is designed to support a dynamic pricing engine that can recommend and adjust
nightly rates in response to seasonal demand signals, day-of-week patterns, property attributes,
and competitive market position.

---

## Key Findings

*(To be completed after full analysis run — placeholder below)*

- Top 20% of listings generate a disproportionate share of market RevPAR; the long tail is the
  primary optimisation opportunity.
- ~25% of Bristol listings are **Underpriced** (high occupancy, below-market rate) — direct
  revenue capture opportunity.
- **Occupancy is the primary differentiator** between Best Seller and Low Seller tiers — demand
  activation matters more than price-setting alone.
- Listings requiring ≥ 7 minimum nights show materially lower occupancy than those requiring ≤ 3.
- `bedrooms` and `accommodates` are the dominant SHAP features for price prediction; premium
  amenities (pool, hot tub) are the strongest positive amenity price signals.

---

## Methodology

### Notebook 1 — EDA & Revenue Segmentation (`01_eda_revenue_segmentation.ipynb`)

Loads and cleans the listings and calendar datasets, computes **RevPAR** (occupancy rate ×
average nightly price) per listing, and segments the market into three performance tiers using
the 20th and 80th RevPAR percentiles. Produces a **Profit Gap Matrix** that maps every listing
into one of four strategic quadrants (Revenue Winners, Underpriced, Overpriced & Empty, Dead
Listings) — directly informing where the pricing engine should intervene and how.

### Notebook 2 — Pricing Drivers (`02_pricing_drivers.ipynb`)

Quantifies the factors that drive price and occupancy variation: seasonal demand patterns by
month, weekend vs weekday occupancy premiums, the impact of minimum-night policy on occupancy,
amenity-level price correlations, and a **CatBoost machine learning model** with **SHAP
explainability** that identifies which property features most strongly determine nightly rates.
These drivers feed directly into the feature weights and multiplier structure of a dynamic
pricing algorithm.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine learning | `catboost`, `scikit-learn` |
| Explainability | `shap` |
| Environment | `jupyter`, Python 3.9+ |

---

## How to Run

```bash
# 1. Clone / navigate to the project root
cd staypriceml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw data files in data/raw/
#    Required: listings.csv.gz, calendar.csv.gz

# 4. Launch Jupyter and run notebooks in order
jupyter notebook notebooks/01_eda_revenue_segmentation.ipynb
jupyter notebook notebooks/02_pricing_drivers.ipynb

# All chart outputs are saved to outputs/figures/
```

---

## Data Source

Data sourced from [InsideAirbnb](http://insideairbnb.com/) under a Creative Commons licence.
Bristol data used as a methodological proof of concept with direct transferability to the
Cornish luxury rental market.

> Cornwall-specific data has been formally requested from InsideAirbnb. This analysis uses
> Bristol data as a methodological proof of concept with direct transferability to the Cornish
> luxury rental market.

---

## Author

**Swapnil Nalawade** — MSc Data Science (Distinction), University of Essex

Portfolio project aligned with a KTP Associate application for a Revenue and Systems Analyst
role at University of Essex × Unique Homestays Ltd.
