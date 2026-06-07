"""
Fix CatBoost cell in NB2 to remove price outliers before training.
Run from project root: python scripts/fix_catboost.py
"""
import json

nb_path = "notebooks/02_pricing_drivers.ipynb"

NEW_CATBOOST = "\n".join([
    "from catboost import CatBoostRegressor",
    "import shap",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.metrics import mean_squared_error, r2_score",
    "",
    "# Top 10 amenities by |price correlation| for model features",
    "top10_amenities = price_corr.abs().nlargest(10).index.tolist()",
    "",
    "base_features = [",
    "    'bedrooms', 'accommodates', 'minimum_nights',",
    "    'number_of_reviews', 'review_scores_rating'",
    "]",
    "all_features = base_features + top10_amenities",
    "",
    "# Build feature matrix",
    "X = listings[base_features].join(amenity_df[top10_amenities])",
    "y = listings['price']",
    "",
    "# Drop rows with any NaN",
    "valid_mask = X.notna().all(axis=1) & y.notna()",
    "X, y = X[valid_mask].copy(), y[valid_mask].copy()",
    "X.columns = all_features",
    "",
    "# Remove extreme price outliers above 99th percentile before training",
    "# The Bristol dataset has listings up to £10,502 — these destroy model performance",
    "# by making variance enormous and preventing the model from learning normal patterns",
    "price_cap = y.quantile(0.99)",
    "outlier_mask = y <= price_cap",
    "X, y = X[outlier_mask], y[outlier_mask]",
    "print(f'Price cap (99th percentile): £{price_cap:.0f}')",
    "print(f'Removed {(~outlier_mask).sum()} extreme outliers')",
    "print(f'Price range in model: £{y.min():.0f} to £{y.max():.0f}  |  mean: £{y.mean():.0f}')",
    "",
    "X_train, X_test, y_train, y_test = train_test_split(",
    "    X, y, test_size=0.2, random_state=42",
    ")",
    "print(f'Train: {len(X_train):,}  |  Test: {len(X_test):,}')",
    "",
    "# Train CatBoost",
    "model = CatBoostRegressor(",
    "    iterations=500, learning_rate=0.05, depth=6,",
    "    loss_function='RMSE', random_seed=42, verbose=0",
    ")",
    "model.fit(X_train, y_train)",
    "",
    "# Evaluate on test set",
    "y_pred = model.predict(X_test)",
    "rmse = np.sqrt(mean_squared_error(y_test, y_pred))",
    "r2   = r2_score(y_test, y_pred)",
    "print(f'\\nCatBoost Regressor — Test Set Performance')",
    "print(f'  RMSE: £{rmse:.2f}')",
    "print(f'  R2:   {r2:.4f}')",
    "",
    "# SHAP summary plot",
    "explainer   = shap.TreeExplainer(model)",
    "shap_values = explainer.shap_values(X_test)",
    "",
    "plt.figure(figsize=(10, 7))",
    "shap.summary_plot(shap_values, X_test, show=False, plot_type='dot',",
    "                  color_bar_label='Feature Value')",
    "",
    "fig = plt.gcf()",
    "fig.patch.set_facecolor(DARK_BG)",
    "for axis in fig.axes:",
    "    axis.set_facecolor(DARK_BG)",
    "    for spine in axis.spines.values():",
    "        spine.set_edgecolor(GRID_COLOR)",
    "    axis.tick_params(colors=TEXT_COLOR)",
    "    axis.xaxis.label.set_color(TEXT_COLOR)",
    "    axis.yaxis.label.set_color(TEXT_COLOR)",
    "",
    "plt.title('SHAP Feature Importance — Price Prediction Model',",
    "          color=ACCENT, fontsize=14, pad=12)",
    "plt.tight_layout()",
    "plt.savefig('outputs/figures/shap_summary.png', dpi=150,",
    "            bbox_inches='tight', facecolor=DARK_BG)",
    "plt.show()",
    "print('Saved: outputs/figures/shap_summary.png')",
])

with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if "CatBoostRegressor" in src:
            cell["source"] = NEW_CATBOOST
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"Fixed cell {i}")
            break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Saved:", nb_path)
