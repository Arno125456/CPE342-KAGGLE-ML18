# CPE342-KAGGLE-ML18
Repos containing Kaggle Competition files for CPE 342 Machine Learning 

# Task 1: Anti-Cheat Pre-Filter

## The Problem

Karena’s automated system flags **2,000 accounts daily**, but **60-70% are false positives** (highly skilled players). With a manual review capacity of only **200 cases/day**, an 8,000+ backlog has formed.

**Goal:** Build a binary classifier to filter true cheaters from skilled players, ensuring the manual team only reviews high-probability offenders.

## Solution Approach

* **Model:** **CatBoostClassifier** — chosen for its superior handling of missing performance data (e.g., `reaction_time_ms`) and categorical features.
* **Preprocessing:** Removed non-predictive IDs, handled 25% missing data in skill-based columns, and used Stratified K-Fold validation.
* **Optimization:** Focused on **Recall**, ensuring actual cheaters are rarely missed, even at the cost of some precision.

## Performance Metrics

The model was evaluated using the **Macro  Score**, which weights recall twice as heavily as precision.

| Metric | Score |
| --- | --- |
| **Macro  Score** | **0.8654** |
| **Recall (Cheaters)** | **0.9120** |
| **Precision (Cheaters)** | **0.7240** |

## Key Impact

* **Backlog Clearance:** Automates the rejection of ~65% of false positive flags.
* **Efficiency:** The manual review team can now focus on the 91% of true cheaters identified by the model, effectively eliminating the review bottleneck.

## Quick Start

1. **Install dependencies:** `pip install catboost scikit-learn pandas`
2. **Run the notebook:** `task1.ipynb` processes the raw data and outputs the final predictions.
