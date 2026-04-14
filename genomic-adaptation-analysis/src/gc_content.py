from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# CONFIG / PATHS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ORGANISMS = [
    # Extremophiles
    ("Thermus aquaticus", DATA_DIR / "thermus.fna", "extremophile"),
    ("Deinococcus radiodurans", DATA_DIR / "deinococcus.fna", "extremophile"),
    ("Halobacterium salinarum", DATA_DIR / "halobacterium.fna", "extremophile"),
    ("Thermotoga maritima", DATA_DIR / "thermotoga.fna", "extremophile"),
    ("Sulfolobus solfataricus", DATA_DIR / "sulfolobus.fna", "extremophile"),
    ("Pyrococcus furiosus", DATA_DIR / "pyrococcus.fna", "extremophile"),

    # Normal
    ("E. coli", DATA_DIR / "ecoli.fna", "normal"),
    ("Bacillus subtilis", DATA_DIR / "bacillus.fna", "normal"),
    ("Pseudomonas aeruginosa", DATA_DIR / "pseudomonas.fna", "normal"),
    ("Staphylococcus aureus", DATA_DIR / "staph.fna", "normal"),
    ("Lactobacillus acidophilus", DATA_DIR / "lacto.fna", "normal"),
    ("Salmonella enterica", DATA_DIR / "salmonella.fna", "normal"),
]


# DATA FUNCTIONS

def read_fasta(file_path):
    """Read a FASTA file and return the DNA sequence as a single string."""
    sequence = ""

    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith(">"):
                sequence += line.strip()

    return sequence


def gc_content(sequence):
    """Calculate GC content as a fraction."""
    g_count = sequence.count("G")
    c_count = sequence.count("C")
    return (g_count + c_count) / len(sequence)


def analyze_genome(file_path, organism_name):
    """Analyze one genome and return summary statistics."""
    sequence = read_fasta(file_path)

    return {
        "name": organism_name,
        "length": len(sequence),
        "gc": gc_content(sequence),
    }


def safe_analyze_genome(name, path):
    """Safely analyze a genome if the file exists."""
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None

    return analyze_genome(str(path), name)


def build_dataframe():
    """Load all available genomes and build the analysis DataFrame."""
    results = []

    for name, path, group in ORGANISMS:
        result = safe_analyze_genome(name, path)
        if result is not None:
            result["group"] = group
            results.append(result)

    df = pd.DataFrame(results)

    if df.empty:
        raise ValueError("No genome files were loaded. Check your data folder and filenames.")

    df["at"] = 1 - df["gc"]
    df["length_scaled"] = df["length"] / df["length"].max()
    df["label"] = df["group"].map({"normal": 0, "extremophile": 1})

    return df.sort_values("gc").reset_index(drop=True)


# STATISTICS

def print_summary_statistics(df):
    """Print summary tables for GC content and genome length."""
    print("\nLoaded organisms:\n")
    print(df[["name", "group", "length", "gc", "at", "length_scaled"]])

    gc_stats = df.groupby("group")["gc"].agg(["mean", "std", "count"])
    length_stats = df.groupby("group")["length"].agg(["mean", "std", "count"])

    print("\nGC Content Group Statistics:\n")
    print(gc_stats)

    print("\nGenome Length Group Statistics:\n")
    print(length_stats)

    return gc_stats, length_stats


# MACHINE LEARNING

def train_models(df):
    """Train logistic regression and decision tree models."""
    X = df[["gc", "length_scaled"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_test)
    tree_predictions = tree_model.predict(X_test)

    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    tree_accuracy = accuracy_score(y_test, tree_predictions)

    print("\nModel Comparison:")
    print(f"Logistic Regression: {logistic_accuracy:.2f}")
    print(f"Decision Tree:       {tree_accuracy:.2f}")
    
    
    test_sample = pd.DataFrame(
        [[0.65, 0.50]],
        columns=["gc", "length_scaled"],
    )

    sample_prediction = logistic_model.predict(test_sample)[0]
    prediction_label = "Extremophile" if sample_prediction == 1 else "Normal"

    print(f"\nTest sample prediction: {prediction_label}")

    return logistic_model, tree_model, logistic_accuracy, tree_accuracy


# PLOTTING

def plot_gc_by_organism(df):
    plt.figure(figsize=(12, 6))

    colors = ["red" if group == "extremophile" else "green" for group in df["group"]]

    plt.bar(df["name"], df["gc"], color=colors)
    plt.title("GC Content Across Bacterial Genomes")
    plt.ylabel("GC Content (fraction)")
    plt.xlabel("Organism")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(df["gc"].min() - 0.05, df["gc"].max() + 0.05)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gc_by_organism.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_group_gc_comparison(gc_stats):
    plt.figure(figsize=(7, 5))

    means = gc_stats["mean"]
    errors = gc_stats["std"]

    plt.bar(means.index, means.values, yerr=errors.values, capsize=5)
    plt.title("Average GC Content by Group (with variability)")
    plt.ylabel("GC Content")
    plt.xlabel("Group")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gc_group_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_feature_scatter(df):
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        df["gc"],
        df["length_scaled"],
        c=df["label"],
        cmap="bwr",
    )

    plt.xlabel("GC Content")
    plt.ylabel("Scaled Genome Length")
    plt.title("GC vs Genome Length (Colored by Class)")
    plt.colorbar(scatter, label="Class (0 = Normal, 1 = Extremophile)")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gc_vs_length_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_decision_tree_model(tree_model):
    plt.figure(figsize=(10, 6))

    plot_tree(
        tree_model,
        feature_names=["gc", "length_scaled"],
        class_names=["normal", "extremophile"],
        filled=True,
    )

    plt.title("Decision Tree Model")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "decision_tree.png", dpi=300, bbox_inches="tight")
    plt.show()


# MAIN

def main():
    print("Saving results to:", RESULTS_DIR)
    df = build_dataframe()
    gc_stats, _ = print_summary_statistics(df)

    df.to_csv(RESULTS_DIR / "genome_features.csv", index=False)

    _, tree_model, accuracy, tree_accuracy = train_models(df)

    summary_text = f"""
Model Results

Logistic Regression Accuracy: {accuracy:.2f}
Decision Tree Accuracy: {tree_accuracy:.2f}

Features used:
- GC content
- Scaled genome length

Dataset size: {len(df)} organisms
"""

    with open(RESULTS_DIR / "model_summary.txt", "w") as f:
        f.write(summary_text)

    plot_gc_by_organism(df)
    plot_group_gc_comparison(gc_stats)
    plot_feature_scatter(df)
    plot_decision_tree_model(tree_model)


if __name__ == "__main__":
    main()