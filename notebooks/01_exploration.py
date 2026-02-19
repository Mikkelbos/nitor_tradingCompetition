"""
EDA script: generates exploration plots and prints summary statistics.
Run: python3 notebooks/01_exploration.py
Outputs saved to notebooks/ directory.
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# seaborn removed — version incompatible with installed matplotlib

from src.data_preprocessing import load_data, impute_missing, NUMERIC_FEATURES, TARGET, MARKET, DELIVERY_START

# ── Configuration ────────────────────────────────────────────────────────────
plt.rcParams.update({"axes.grid": True, "font.size": 11})
PLOT_DIR = "notebooks"

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data …")
train, test = load_data()
train = impute_missing(train)
train_with_target = train.dropna(subset=[TARGET]).copy()

print(f"Train shape (all):        {train.shape}")
print(f"Train shape (w/ target):  {train_with_target.shape}")
print(f"Test shape:               {test.shape}")

# ── 1. Target distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(train_with_target[TARGET], bins=100, edgecolor="black", alpha=0.7)
axes[0].set_title("Target Distribution (all markets)")
axes[0].set_xlabel("Target (electricity price)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(train_with_target[TARGET].median(), color="red", ls="--", label=f"median={train_with_target[TARGET].median():.1f}")
axes[0].legend()

# Clip extreme outliers for a cleaner view
clipped = train_with_target[TARGET].clip(-100, 200)
axes[1].hist(clipped, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
axes[1].set_title("Target Distribution (clipped to [-100, 200])")
axes[1].set_xlabel("Target (electricity price)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  01_target_distribution.png")

# ── 2. Target vs delivery_start (time series) ───────────────────────────────
fig, ax = plt.subplots(figsize=(18, 6))
for market, grp in train_with_target.groupby(MARKET):
    ax.plot(grp[DELIVERY_START], grp[TARGET], alpha=0.4, linewidth=0.5, label=market)
ax.set_title("Target vs Delivery Start (by Market)")
ax.set_xlabel("Delivery Start")
ax.set_ylabel("Target")
ax.legend(loc="upper right")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_target_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  02_target_timeseries.png")

# ── 3. Target by market (boxplot) ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
train_with_target.boxplot(column=TARGET, by=MARKET, ax=ax, showfliers=False)
ax.set_title("Target Distribution by Market (no outliers)")
ax.set_xlabel("Market")
ax.set_ylabel("Target")
plt.suptitle("")  # remove pandas auto-title
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_target_by_market.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  03_target_by_market.png")

# ── 4. Hourly pattern ───────────────────────────────────────────────────────
train_with_target["hour"] = train_with_target[DELIVERY_START].dt.hour
fig, ax = plt.subplots(figsize=(12, 5))
hourly = train_with_target.groupby(["hour", MARKET])[TARGET].mean().unstack()
hourly.plot(ax=ax, marker="o", markersize=3)
ax.set_title("Average Target by Hour of Day (per Market)")
ax.set_xlabel("Hour")
ax.set_ylabel("Mean Target")
ax.legend(title="Market")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_hourly_pattern.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  04_hourly_pattern.png")

# ── 5. Correlation heatmap (top features vs target) ─────────────────────────
numeric_cols = [c for c in NUMERIC_FEATURES if c in train_with_target.columns]
corr = train_with_target[numeric_cols + [TARGET]].corr()
target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)

print("\n── Feature correlations with target ──")
print(target_corr.to_string())

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr.columns, fontsize=7)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=5)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Correlation Heatmap (Numeric Features + Target)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  05_correlation_heatmap.png")

# ── 6. Monthly pattern ──────────────────────────────────────────────────────
train_with_target["month"] = train_with_target[DELIVERY_START].dt.month
fig, ax = plt.subplots(figsize=(12, 5))
monthly = train_with_target.groupby(["month", MARKET])[TARGET].mean().unstack()
monthly.plot(ax=ax, marker="o", markersize=4)
ax.set_title("Average Target by Month (per Market)")
ax.set_xlabel("Month")
ax.set_ylabel("Mean Target")
ax.legend(title="Market")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_monthly_pattern.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  06_monthly_pattern.png")

# ── 7. Key feature scatter plots ────────────────────────────────────────────
key_features = ["load_forecast", "wind_forecast", "solar_forecast",
                "air_temperature_2m", "wind_speed_10m"]
fig, axes = plt.subplots(1, len(key_features), figsize=(20, 4))
for ax, feat in zip(axes, key_features):
    sample = train_with_target.sample(min(5000, len(train_with_target)), random_state=42)
    ax.scatter(sample[feat], sample[TARGET], alpha=0.3, s=5)
    ax.set_xlabel(feat)
    ax.set_ylabel(TARGET)
    ax.set_title(f"{feat}\nvs target")
plt.suptitle("Key Feature Scatter Plots (5k sample)", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_feature_scatters.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  07_feature_scatters.png")

# ── 8. Missing values overview ──────────────────────────────────────────────
raw_train, raw_test = load_data()
print("\n── Missing values (raw train) ──")
missing = raw_train.isnull().sum()
missing = missing[missing > 0]
print(missing.to_string())

print("\n── Missing values (raw test) ──")
missing_test = raw_test.isnull().sum()
missing_test = missing_test[missing_test > 0]
print(missing_test.to_string())

# ── Summary stats ────────────────────────────────────────────────────────────
print("\n── Descriptive statistics (train, with target) ──")
print(train_with_target[numeric_cols + [TARGET]].describe().T.to_string())

print(f"\n✅  All EDA plots saved to {PLOT_DIR}/")
