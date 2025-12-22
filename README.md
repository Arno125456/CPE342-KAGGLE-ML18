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


# Task 2: Player Segment Classification

## Problem Statement

The company currently spends **฿60 million monthly** on marketing and promotions, yet engagement remains low (**2-5%**). This inefficiency stems from a "one-size-fits-all" approach that fails to distinguish between different types of players. Most importantly, the company is unable to accurately identify **"Whales"**—the top tier of players who generate the majority of revenue.

**The Objective:** Classify the player base into four distinct behavioral segments to allow for personalized marketing strategies and efficient budget allocation.

### Behavioral Segments

* **Class 0: Casual Player** – Relaxed playstyle, low frequency, and low spending.
* **Class 1: Competitive Grinder** – High playtime, focus on ranked progression, and performance-driven.
* **Class 2: Social Player** – Focuses on friend interactions and cosmetic/aesthetic spending.
* **Class 3: Whale** – High-spending individuals driven by status and exclusive content.

## Evaluation Metric:  Score (Macro)

We use the **Macro  Score** to ensure the model performs consistently across all segments. This prevents the model from being biased toward the larger "Casual" population while neglecting the critical "Whale" and "Competitive" segments.

## Solution Roadmap

### 1. Data Processing

* **Feature Set:** Utilized 46 behavioral and transactional features including playtime, cosmetic purchases, social interactions, and progression metrics.
* **Scaling & Encoding:** Applied `StandardScaler` to normalize numeric distributions and `LabelEncoder` for categorical alignment.

### 2. The Model: LightGBM

We utilized **LightGBM (LGBMClassifier)** due to its efficiency with high-dimensional data and its ability to handle large datasets (100k+ rows) quickly.

* **Strategy:** Stratified 5-Fold Cross-Validation to ensure the model generalizes across the entire player distribution.
* **Technique:** Multi-class log-loss optimization with a focus on balanced  across classes.

## Evaluation Results

The model achieved exceptional accuracy and balance across all player segments.

| Metric | Score |
| --- | --- |
| **Macro  Score** | **0.97** |
| **Accuracy** | **0.97** |
| **Weighted  Score** | **0.9685** |

#### Class-Specific Performance (F1-Score):

* **Casual (Class 0):** 0.98
* **Competitive (Class 1):** 0.97
* **Social (Class 2):** 0.96
* **Whale (Class 3):** 0.95

## Business Impact

* **Identifying Whales:** The model accurately identifies the high-value segment (Class 3) with **97% precision**, allowing the company to protect 60% of its revenue through VIP-specific retention programs.
* **Budget Efficiency:** Marketing can now reallocate portions of the ฿60M budget away from generic campaigns and toward high-conversion, segment-specific rewards.
* **Conversion:** By targeting players with content they actually care about (e.g., cosmetics for Social players vs. rank boosters for Grinders), the company can aim to move the needle from 2% engagement toward double digits.

## Quick Start

1. **Install dependencies:**
```bash
pip install lightgbm scikit-learn pandas numpy matplotlib seaborn

```


2. **Execution:** Run the `task2.ipynb` notebook to preprocess the raw player data and generate the `submission.csv` segmentation file.


# Task 3: Monthly Spending Prediction

## The Problem

Finance currently misses revenue targets by **35-40%**. Inefficient resource allocation leads to VIP support wasting time on non-spenders while **฿255,000 whales** wait days for help. Additionally, marketing leaks **฿6+ million monthly** by giving deep discounts to high-spenders who would have paid full price.

**Goal:** Predict the exact amount (in THB) each player will spend over the next 30 days.

## Solution Approach: Two-Stage Modeling

Since gaming spend data is "zero-heavy" (most players don't spend), a single regression model often underperforms. We implemented a **Hurdle Model** approach:

1. **Stage 1 (Classifier):** Predict the *probability* of a player spending at all.
2. **Stage 2 (Regressor):** Predict the *amount* for potential spenders.
3. **Optimization:** Used **Optuna** to find the optimal probability threshold (0.4773) to filter out non-spenders, significantly reducing noise and improving accuracy.

## Performance Metrics

The model was evaluated using **Normalized Mean Absolute Error (NMAE)**.

| Metric | Score |
| --- | --- |
| **Best OOF NMAE** | **0.27424** |
| **Optimal Threshold** | **0.4773** |

## Business Impact

* **Revenue Forecasting:** Reduces the 40% margin of error in finance projections to under 28%, allowing for better cash flow management.
* **VIP Priority:** Support teams can now prioritize tickets based on predicted 30-day value, ensuring "whales" receive instant assistance.
* **Discount Optimization:** Marketing can identify "Natural Spenders" and exclude them from 50% discount campaigns, saving ฿6M+ in monthly revenue leakage.

## Quick Start

1. **Install dependencies:** `pip install lightgbm optuna scikit-learn pandas`
2. **Run the notebook:** `task3.ipynb` executes the two-stage pipeline and exports `submission_optuna.csv`.


# Task 4: Game Title Detection

## The Problem

Customer support is currently a bottleneck, receiving **2,000 daily tickets** with images. Specialists waste over 10 minutes per ticket just identifying the game before they can provide help. Furthermore, the "Play Multiple Games for Rewards" campaign receives **50,000+ daily screenshot submissions**, but the team can only manually verify **3%** of them, leading to high potential for reward fraud.

**Goal:** Build an image classifier to instantly identify the game title from screenshots across five major classes:

* **Class 0:** PUBE Mobile
* **Class 1:** Free Fried
* **Class 2:** FiveN
* **Class 3:** Roblock
* **Class 4:** 7-Eleven Knight

## Solution Approach: Deep Learning

We implemented a computer vision pipeline using **PyTorch** and transfer learning to handle the diverse visual styles of different mobile games.

* **Model:** **CNN Architecture** (via `torchvision.models`) – Pretrained weights were used to leverage existing feature extraction capabilities.
* **Preprocessing:** Images were padded and resized to **320x320** to maintain aspect ratio integrity.
* **Augmentation:** To make the model robust against different UI overlays and brightness levels, we applied:
* Horizontal Flips
* Color Jittering (Brightness, Contrast, Saturation)
* ImageNet Normalization


* **Training:** 10 Epochs with a learning rate of `2e-4` and Weight Decay of `1e-4` to prevent overfitting.

## Performance Metrics

The model was optimized for the **Macro  Score** to ensure high accuracy across all game titles, even if some games had fewer screenshot submissions.

| Metric | Score |
| --- | --- |
| **Macro  Score** | **0.992** |
| **Accuracy** | **99.2%** |

## Business Impact

* **Instant Routing:** Reduces ticket handling time by 10 minutes per case by automatically routing images to the correct game specialists.
* **Automated Verification:** Moves verification from 3% (manual) to **100% (automated)** for the rewards campaign, virtually eliminating reward fraud.
* **Scalability:** The system can now handle the 50,000+ daily submissions without increasing headcount.

## Quick Start

1. **Install dependencies:** `pip install torch torchvision pandas pillow tqdm`
2. **Run the notebook:** `task4.ipynb` handles the data loading from Kaggle paths, training loop, and generates the final classification.


# Task 5: Account Security Monitoring (Anomaly Detection)

## The Problem

Karena faces a severe security crisis with over **5,000 confirmed account thefts monthly**. Beyond individual thefts, bot farms are inflating the in-game economy, and players are exploiting VPNs to gain **70% regional pricing discounts**. The existing rule-based system is ineffective, suffering from a **60% false positive rate** that frustrates legitimate users.

**Goal:** Build an **unsupervised anomaly detection** model to identify suspicious behavior (thefts, bots, and regional exploits) without relying on historical labels.

## Solution Approach: Unsupervised Learning

Since labeled data for every type of exploit is unavailable, we used a density-based approach to isolate outliers from the "normal" player base.

* **Model:** **Isolation Forest** — This algorithm is ideal for high-dimensional data, as it "isolates" anomalies by randomly selecting a feature and a split value. Anomalies (like bot accounts or sudden regional jumps) are isolated much faster than normal points.
* **Feature Engineering:** Focused on behavioral shifts, including:
* IP address and regional consistency.
* In-game economy transaction velocity (identifying bot-like inflation).
* Session duration and login frequency patterns.


* **Scaling:** Applied `StandardScaler` to ensure that features like "spending" and "login counts" are treated with equal importance by the isolation tree.

## Evaluation Metric:  Score (Macro)

For security, **Recall is the absolute priority**. We use the ** Score**, which weights Recall **three times** more heavily than Precision.

* **Logic:** It is better to flag a suspicious account for a quick verification check than to allow a single bot farm to ruin the game economy or a hacker to compromise a user's account.

| Metric | Goal |
| --- | --- |
| **Macro  Score** | Optimized to maximize the detection of rare, high-risk events. |
| **Contamination Levels** | Tested at 10% to 50% to find the optimal balance between security and user friction. |

## Business Impact

* **Theft Prevention:** Detects account takeovers by identifying shifts in login location and spending behavior before assets are stripped.
* **Economic Stability:** Flags bot farm clusters early, preventing hyper-inflation of in-game currency.
* **Revenue Protection:** Identifies VPN-based regional pricing exploits, ensuring users pay fair prices based on their actual location.

## Quick Start

1. **Install dependencies:** `pip install scikit-learn pandas matplotlib`
2. **Run the notebook:** `task5.ipynb` performs the unsupervised split, generates anomaly scores, and outputs submission files for various contamination thresholds.
