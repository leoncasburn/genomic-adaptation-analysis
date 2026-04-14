# Genomic Signatures of Environmental Adaptation

## Overview

This project investigates whether simple genomic features can be used to distinguish extremophile organisms from non-extremophile bacteria. Using complete genome sequences, I extracted biologically meaningful features and applied both statistical analysis and machine learning models to explore patterns in environmental adaptation.

The project combines:

* genome sequence processing
* feature extraction (GC content, genome length)
* statistical comparison
* data visualization
* machine learning classification

---

## Motivation

Extremophiles are organisms that thrive in environments such as high temperature, high salinity, or radiation. These organisms are of particular interest in astrobiology, as they provide insight into the limits of life and the potential for life in extraterrestrial environments.

This project explores the question:

> Can simple genomic features such as GC content and genome size help predict whether an organism is adapted to extreme environments?

---

## Dataset

Genome sequences were obtained from NCBI in FASTA (`.fna`) format.

### Extremophiles

* Thermus aquaticus
* Deinococcus radiodurans
* Halobacterium salinarum
* Thermotoga maritima
* Sulfolobus solfataricus
* Pyrococcus furiosus

### Non-extremophiles

* Escherichia coli
* Bacillus subtilis
* Pseudomonas aeruginosa
* Staphylococcus aureus
* Lactobacillus acidophilus
* Salmonella enterica

---

## Methods

### 1. Genome Processing

FASTA files were parsed to extract full DNA sequences for each organism.

### 2. Feature Extraction

For each genome, the following features were computed:

* **Genome length** (total number of bases)
* **GC content** (fraction of G and C nucleotides)
* **AT content** (derived as `1 - GC`)
* **Scaled genome length** (normalized for machine learning)

### 3. Statistical Analysis

Organisms were grouped into extremophiles and non-extremophiles. Mean GC content and genome length were compared between groups, including variability using standard deviation.

### 4. Machine Learning

Two models were trained using GC content and scaled genome length:

* Logistic Regression
* Decision Tree Classifier

Performance was evaluated using a train/test split.

---

## Results

### GC Content Patterns

GC content varied across organisms. Some extremophiles exhibited high GC content, but this was not universal, indicating that GC alone is not a definitive marker of environmental adaptation.

### Group-Level Differences

On average, extremophiles in this dataset showed different GC content patterns compared to non-extremophiles. However, overlap between groups was observed, suggesting that multiple factors contribute to adaptation.

### Genome Length Observations

Several extremophiles in this dataset exhibited relatively smaller genome sizes compared to many non-extremophile bacteria, although variability and overlap were present.

### Machine Learning Performance

Both models achieved similar performance:

* Logistic Regression accuracy: ~0.75
* Decision Tree accuracy: ~0.75

This indicates that GC content and genome length contain some predictive signal, but are insufficient for perfect classification.

---

## Interpretation

This project demonstrates that simple genomic features can reveal meaningful biological patterns, but also highlights the limitations of using a small number of features to explain complex biological phenomena.

From an astrobiology perspective, this suggests that potential biosignatures may be informative but not definitive. Detecting life in extreme environments—on Earth or elsewhere—will likely require integrating multiple lines of evidence.

---

## Project Structure

```
bio_project/
├── data/        # Raw genome files (.fna)
├── results/     # Generated outputs
│   ├── genome_features.csv
│   ├── gc_by_organism.png
│   ├── gc_group_comparison.png
│   ├── gc_vs_length_scatter.png
│   ├── decision_tree.png
│   └── model_summary.txt
├── src/
│   └── gc_content.py
└── README.md
```

---

## How to Run

1. Install dependencies:

```
pip install pandas matplotlib scikit-learn
```

2. Place genome `.fna` files in the `data/` folder

3. Run:

```
python src/gc_content.py
```

Outputs will be saved automatically in the `results/` directory.

---

## Future Improvements

* Expand dataset to include more organisms
* Add additional genomic features (e.g. k-mer frequencies, GC skew)
* Compare additional machine learning models
* Perform cross-validation for more robust evaluation
* Incorporate evolutionary or phylogenetic analysis

---

## Skills Demonstrated

* Python programming
* FASTA parsing and genome processing
* Feature engineering
* Statistical analysis
* Data visualization
* Machine learning classification
* Scientific interpretation of computational results

---

## Author

This project was developed as part of a transition into bioinformatics, with the long-term goal of pursuing research in astrobiology.
