import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading and cleaning the data

df = pd.read_csv("globalterrorismdb_0718dist.csv", encoding="latin-1", low_memory=False)
df = df[(df.iyear >= 2000)].copy()
n_raw = len(df)

df = df.dropna(
    subset=[
        "nkill",
        "nwound",
        "multiple",
        "ishostkid",
        "targtype1_txt",
        "region_txt",
        "attacktype1_txt",
        "weaptype1_txt",
    ]
)

df = df[df.ishostkid != -9]
df = df[df.multiple != -9]


for col in ["targtype1", "weaptype1"]:
    top8 = df[f"{col}_txt"].value_counts().head(8).index
    df[f"{col}_grouped"] = df[f"{col}_txt"].where(df[f"{col}_txt"].isin(top8), "Other")

# defining total casualties and log transformation

df["total_casualties"] = df.nkill + df.nwound
df["log_casualties"] = np.log1p(df.total_casualties)

# Plotting distributions

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, title, c in [
    (axes[0], "targtype1_grouped", "Top 5 Target Types", "red"),
    (axes[1], "weaptype1_grouped", "Top 5 Weapon Types", "blue"),
]:
    counts = df[col].value_counts().head(5)
    ax.bar(counts.index, counts.values, color=c)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    for l in ax.get_xticklabels():
        l.set_ha("right")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(df.total_casualties, bins=50, range=(0, 100), color="red")
axes[0].set_title("Raw Total Casualties")
axes[1].hist(df.log_casualties, bins=50, color="blue")
axes[1].set_title("log(Total Casualties + 1)")
plt.tight_layout()
plt.show()

# Preparing features and train/val/test splits

features = [
    "targtype1_grouped",
    "region_txt",
    "attacktype1_txt",
    "weaptype1_grouped",
    "multiple",
    "ishostkid",
]
cat_cols = ["targtype1_grouped", "region_txt", "attacktype1_txt", "weaptype1_grouped"]
X = pd.get_dummies(df[features], columns=cat_cols, drop_first=True)
y = df.log_casualties

train, val, test = df.iyear <= 2014, df.iyear == 2015, df.iyear >= 2016
X_train, X_val, X_test = X[train], X[val], X[test]
y_train, y_val, y_test = y[train], y[val], y[test]

print(" Summary ")
print(f"Total attacks (2000-2017): {n_raw}")
print(f"After cleaning: {len(df)} ({100 * len(df) / n_raw:.1f}% retained)")
print(
    f"Skewness — raw: {df.total_casualties.skew():.0f}, log: {df.log_casualties.skew():.2f}"
)
print(f"Train: {train.sum()} ({100 * train.sum() / len(df):.1f}%)")
print(f"Val:   {val.sum()} ({100 * val.sum() / len(df):.1f}%)")
print(f"Test:  {test.sum()} ({100 * test.sum() / len(df):.1f}%)")
