# ============================================================
# Project      : Titanic Data Visualization Dashboard
# Internship   : CodeAlpha Data Analytics Internship
# Author       : Ganesh Jadhav
#
# Description:
# This project creates a visual dashboard from the Titanic
# dataset using Pandas, Matplotlib, and Seaborn.
# ============================================================


# ============================================================
# 1. Import Required Libraries
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 2. Configure Visualization Theme
# ============================================================

sns.set_theme(style="whitegrid")


# ============================================================
# 3. Define Project Paths
# ============================================================

CURRENT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = CURRENT_FOLDER.parent
DATASET_PATH = PROJECT_FOLDER / "Dataset" / "Titanic-Dataset.csv"
OUTPUT_PATH = CURRENT_FOLDER / "dashboard.png"


# ============================================================
# 4. Load the Dataset
# ============================================================

print("=" * 70)
print("TITANIC DATA VISUALIZATION DASHBOARD")
print("=" * 70)

try:
    df = pd.read_csv(DATASET_PATH)

except FileNotFoundError:
    print("\nError: Titanic dataset file was not found.")
    print(f"Expected location: {DATASET_PATH}")
    raise SystemExit(1)

except Exception as error:
    print("\nAn unexpected error occurred while loading the dataset.")
    print(f"Error details: {error}")
    raise SystemExit(1)

print("\nDataset loaded successfully.")


# ============================================================
# 5. Data Cleaning
# ============================================================

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("Missing values handled successfully.")


# ============================================================
# 6. Create Dashboard
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

fig.suptitle(
    "Titanic Dataset Visualization Dashboard",
    fontsize=20,
    fontweight="bold"
)


# ------------------------------------------------------------
# Graph 1: Survival Count
# ------------------------------------------------------------

survival_plot = sns.countplot(
    data=df,
    x="Survived",
    hue="Survived",
    palette=["#e76f51", "#2a9d8f"],
    legend=False,
    ax=axes[0, 0]
)

axes[0, 0].set_title(
    "Passenger Survival Count",
    fontsize=14,
    fontweight="bold"
)
axes[0, 0].set_xlabel(
    "Survival Status (0 = Did Not Survive, 1 = Survived)"
)
axes[0, 0].set_ylabel("Number of Passengers")

for container in survival_plot.containers:
    survival_plot.bar_label(container)


# ------------------------------------------------------------
# Graph 2: Gender Distribution
# ------------------------------------------------------------

gender_plot = sns.countplot(
    data=df,
    x="Sex",
    hue="Sex",
    palette=["#457b9d", "#f4a6c1"],
    legend=False,
    ax=axes[0, 1]
)

axes[0, 1].set_title(
    "Gender Distribution",
    fontsize=14,
    fontweight="bold"
)
axes[0, 1].set_xlabel("Gender")
axes[0, 1].set_ylabel("Number of Passengers")

for container in gender_plot.containers:
    gender_plot.bar_label(container)


# ------------------------------------------------------------
# Graph 3: Passenger Class Distribution
# ------------------------------------------------------------

class_plot = sns.countplot(
    data=df,
    x="Pclass",
    hue="Pclass",
    palette="viridis",
    legend=False,
    ax=axes[1, 0]
)

axes[1, 0].set_title(
    "Passenger Class Distribution",
    fontsize=14,
    fontweight="bold"
)
axes[1, 0].set_xlabel("Passenger Class")
axes[1, 0].set_ylabel("Number of Passengers")

for container in class_plot.containers:
    class_plot.bar_label(container)


# ------------------------------------------------------------
# Graph 4: Age Distribution
# ------------------------------------------------------------

sns.histplot(
    data=df,
    x="Age",
    bins=20,
    kde=True,
    color="#5dade2",
    edgecolor="black",
    ax=axes[1, 1]
)

axes[1, 1].set_title(
    "Age Distribution",
    fontsize=14,
    fontweight="bold"
)
axes[1, 1].set_xlabel("Age")
axes[1, 1].set_ylabel("Frequency")


# ============================================================
# 7. Save Dashboard
# ============================================================

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. Dashboard Insights
# ============================================================

print("\n" + "=" * 70)
print("DASHBOARD INSIGHTS")
print("=" * 70)

print("1. Most passengers did not survive.")
print("2. Male passengers were more than female passengers.")
print("3. Passenger Class 3 had the highest number of passengers.")
print("4. Most passengers were young adults.")


# ============================================================
# 9. Completion Message
# ============================================================

print("\n" + "=" * 70)
print("DASHBOARD CREATED SUCCESSFULLY")
print("=" * 70)

print(f"\nDashboard saved at:\n{OUTPUT_PATH}")