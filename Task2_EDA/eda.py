# ============================================================
# Project      : Exploratory Data Analysis - Titanic Dataset
# Internship   : CodeAlpha Data Analytics Internship
# Author       : Ganesh Jadhav
#
# Description:
# This project performs data loading, data inspection,
# data cleaning, exploratory analysis, and visualization
# on the Titanic passenger dataset.
# ============================================================


# ============================================================
# 1. Import Required Libraries
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 2. Configure Display and Visualization Settings
# ============================================================

# Display all dataset columns in the terminal.
pd.set_option("display.max_columns", None)

# Display wider output without unnecessary line breaks.
pd.set_option("display.width", 120)

# Apply a clean visualization theme.
sns.set_theme(style="whitegrid")


# ============================================================
# 3. Define Project Paths
# ============================================================

# Location of the current Python file.
CURRENT_FOLDER = Path(__file__).resolve().parent

# Main project directory.
PROJECT_FOLDER = CURRENT_FOLDER.parent

# Dataset file path.
DATASET_PATH = PROJECT_FOLDER / "Dataset" / "Titanic-Dataset.csv"


# ============================================================
# 4. Load the Dataset
# ============================================================

print("=" * 70)
print("TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

try:
    df = pd.read_csv(DATASET_PATH)

except FileNotFoundError:
    print("\nError: Titanic dataset file was not found.")
    print(f"Expected dataset location: {DATASET_PATH}")
    raise SystemExit(1)

except Exception as error:
    print("\nAn unexpected error occurred while loading the dataset.")
    print(f"Error details: {error}")
    raise SystemExit(1)

print("\nDataset loaded successfully.")
print(f"Dataset location: {DATASET_PATH}")


# ============================================================
# 5. Inspect the Dataset
# ============================================================

print("\n" + "=" * 70)
print("FIRST FIVE RECORDS")
print("=" * 70)
print(df.head())


print("\n" + "=" * 70)
print("DATASET DIMENSIONS")
print("=" * 70)
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")


print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for column_number, column_name in enumerate(df.columns, start=1):
    print(f"{column_number}. {column_name}")


print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)
df.info()


print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)
print(df.describe())


print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 70)
print(df.isnull().sum())


print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")


# ============================================================
# 6. Data Cleaning
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)


# Fill missing Age values with the average passenger age.
age_mean = df["Age"].mean()
df["Age"] = df["Age"].fillna(age_mean)

print(
    f"Missing Age values filled using mean age: "
    f"{age_mean:.2f}"
)


# Fill missing Embarked values with the most common category.
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)

print(
    f"Missing Embarked values filled using mode: "
    f"{embarked_mode}"
)


# Remove Cabin because it contains a very large number
# of missing values and is not required for this analysis.
if "Cabin" in df.columns:
    df = df.drop(columns=["Cabin"])
    print("Cabin column removed successfully.")


# Remove duplicate records, if any exist.
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"{duplicate_count} duplicate rows removed.")
else:
    print("No duplicate rows were found.")


# ============================================================
# 7. Verify the Cleaned Dataset
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES AFTER CLEANING")
print("=" * 70)
print(df.isnull().sum())


print("\n" + "=" * 70)
print("CLEANED DATASET DIMENSIONS")
print("=" * 70)
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")


print("\nData cleaning completed successfully.")
# ============================================================
# 8. Create Visualizations
# ============================================================

# ------------------------------------------------------------
# Visualization 1: Passenger Survival Count
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

survival_plot = sns.countplot(
    data=df,
    x="Survived",
    hue="Survived",
    palette=["#e76f51", "#2a9d8f"],
    legend=False
)

plt.title(
    "Passenger Survival Count",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Survival Status (0 = Did Not Survive, 1 = Survived)")
plt.ylabel("Number of Passengers")

for bar in survival_plot.containers:
    survival_plot.bar_label(bar)

plt.tight_layout()
plt.savefig(
    CURRENT_FOLDER / "survival_count.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# Visualization 2: Gender Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

gender_plot = sns.countplot(
    data=df,
    x="Sex",
    hue="Sex",
    palette=["#457b9d", "#f4a6c1"],
    legend=False
)

plt.title(
    "Gender Distribution",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

for bar in gender_plot.containers:
    gender_plot.bar_label(bar)

plt.tight_layout()
plt.savefig(
    CURRENT_FOLDER / "gender_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# Visualization 3: Passenger Class Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

class_plot = sns.countplot(
    data=df,
    x="Pclass",
    hue="Pclass",
    palette="viridis",
    legend=False
)

plt.title(
    "Passenger Class Distribution",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

for bar in class_plot.containers:
    class_plot.bar_label(bar)

plt.tight_layout()
plt.savefig(
    CURRENT_FOLDER / "passenger_class.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# Visualization 4: Age Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=20,
    kde=True,
    color="#5dade2",
    edgecolor="black"
)

plt.title(
    "Age Distribution of Titanic Passengers",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(
    CURRENT_FOLDER / "age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# Visualization 5: Fare Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Fare",
    bins=30,
    kde=True,
    color="#f4a261",
    edgecolor="black"
)

plt.title(
    "Fare Distribution of Titanic Passengers",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(
    CURRENT_FOLDER / "fare_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# Visualization 6: Correlation Heatmap
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include="number")
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Correlation Value"}
)

plt.title(
    "Correlation Heatmap",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    CURRENT_FOLDER / "heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


print("\nAll six visualizations were created successfully.")
# ============================================================
# 9. Key Insights
# ============================================================

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

print("1. Most passengers did not survive the Titanic disaster.")
print("2. Passenger Class 3 had the highest number of passengers.")
print("3. Male passengers outnumbered female passengers.")
print("4. Most passengers were between 20 and 35 years old.")
print("5. Fare distribution is positively skewed because a few passengers paid very high fares.")
print("6. The dataset was successfully cleaned before analysis.")


# ============================================================
# 10. Project Summary
# ============================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"Dataset Records Analysed : {len(df)}")
print(f"Total Features           : {len(df.columns)}")
print("Visualizations Created   : 6")
print("Missing Values Handled   : Yes")
print("Duplicate Records Checked: Yes")


# ============================================================
# 11. Completion Message
# ============================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Generated Files:

✔ survival_count.png
✔ gender_distribution.png
✔ passenger_class.png
✔ age_distribution.png
✔ fare_distribution.png
✔ heatmap.png

All graphs have been saved successfully.
""")

print("=" * 70)
print("Thank you for using this EDA Project!")
print("=" * 70)