# 🧠 Stroke Analysis Using EEG Signals for Paralysis Side, Laterality, and Severity Classification

## Project Overview

Stroke is one of the leading causes of disability worldwide, and rapid evaluation of its laterality and severity is crucial for timely clinical intervention. This project presents an integrated, machine-learning–driven framework for analyzing 33-channel Electroencephalography (EEG) recordings collected from 50 acute stroke patients to predict three critical stroke characteristics: **Paralysis Side**, **Stroke-Affected Hemisphere (Laterality)**, and **Stroke Severity**.

Our novel approach combines traditional signal processing, advanced **Topological Data Analysis (TDA)** to capture global connectivity, and a variety of machine learning models and a deep neural network.

## ✨ Key Findings

* **TDA Improves Prediction:** Incorporating **Topological Data Analysis (TDA)** features significantly improves both paralysis-side and stroke-side prediction performance.
* **Best Laterality Predictors:** Random Forest and XGBoost achieved the highest accuracy (**0.818**) for stroke side prediction *after* TDA was included.
* **Severity Classification:** A **Bi-LSTM** (Bidirectional LSTM) architecture was trained to effectively classify stroke severity, outperforming classical machine learning approaches by effectively capturing temporal dynamics in EEG data.
* **Clinical Relevance:** Severity prediction achieved an accuracy of **0.8000** when defined by both the NIHSS ($\ge 5$) and mRS ($\ge 3$) thresholds *after* TDA.

---

## 🛠️ Methodology Pipeline

### 1. Dataset & Preparation
* **Data Source:** Publicly available, high-temporal-resolution EEG recordings from **50 acute stroke patients**.
* **Features:** Raw and processed signals with detailed metadata such as the affected hemisphere and paralysis side.
* **Data Splitting:** Subject-wise split (75% training, 25% testing) was used, ensuring all epochs belonging to the same patient were kept together to prevent data leakage.

### 2. EEG Preprocessing
The pipeline ensures the EEG signals retain only physiologically meaningful components.
* **Filtering:** A **0.5–45 Hz bandpass filter** and a **50 Hz notch filter** were applied to remove slow drifts, high-frequency muscle artifacts, and powerline interference.
* **Re-referencing:** **Common Average Reference (CAR)** technique was used, where each channel is referenced to the average of all channels, helping reduce spatial bias.
* **Epoching:** Data was segmented into **5-second windows** with a **4.5-second step** (90% overlap) for fine-grained temporal analysis.

### 3. Feature Extraction

| Feature Type | Calculation Method | Role/Description |
| :--- | :--- | :--- |
| **Spectral Features** | Power Spectral Density (PSD) via Welch method | Bandpower features extracted across Delta (0.5–4 Hz), Theta, Alpha, and Beta bands, capturing important neural oscillatory patterns. |
| **Hemispheric Asymmetry Features** | Left–right electrode pair asymmetry indices | Quantify differences in spectral power between corresponding electrodes (e.g., F3–F4, C3–C4, P3–P4) for identifying the affected hemisphere. |

### 4. Novelty: Topological Data Analysis (TDA)

**Topological Data Analysis (TDA)** is a mathematical framework that uses topology—the study of shape and connectivity—to uncover global patterns in complex data.

| TDA Step | Topological Concept | Purpose |
| :--- | :--- | :--- |
| **Point Cloud Generation** | Geometric shape of the time-series | Converts physiological features (like power features) into a collection of points in a metric space. |
| **Vietoris-Rips Filtration** | Simplicial complexes | Gradually builds a sequence of geometric shapes by connecting points closer than a radius $\epsilon$. |
| **Persistent Homology** | Betti numbers ($\beta_0$ for clusters, $\beta_1$ for loops) | Tracks the 'birth' and 'death' of topological features (connected components, holes, loops) as the filtration scale increases. |
| **Statistical Feature Extraction** | Persistence Diagrams | Vectorizes the diagrams by computing statistical summaries (Count, Max/Mean/Total Persistence, Shannon Entropy) for $H_0$ and $H_1$. |
| **Feature Fusion** | Combined Vector | The 10 TDA features (5 for H0 and 5 for H1) are concatenated with the spectral Band Power features for classifier input. |

### 5. Classification Models

#### A. Paralysis Side & Stroke Laterality
* **Models Evaluated:** Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting, and both Linear and RBF SVMs.
* **Evaluation Metric:** Predictions were first generated at the epoch level and then aggregated at the **patient level** using majority voting. Performance was evaluated using accuracy, F1 scores, precision, and recall.

#### B. Severity Classification
* **Severity Thresholds:**
    * **NIHSS (National Institute of Health Stroke Scale):** Score of **$\ge 5$** to differentiate minor from moderate or severe stroke.
    * **mRS (modified Rankin Scale):** Score of **$\ge 3$** to define a poor functional outcome (dependence) versus a good outcome.
* **Model Used:** **Bidirectional LSTM (Bi-LSTM)** network.
* **Input Shape:** Data formatted into an input shape of **(142, 300)**.
* **Training:** Trained for **50 epochs** using a **weighted cross-entropy loss** function to handle class imbalance.

---

## 📊 Experimental Results

### Paralysis Side Prediction Accuracy

| Model | Accuracy (Before TDA) | Accuracy (After TDA) |
| :--- | :--- | :--- |
| **RBF SVM** | 0.769 | 0.692 |
| **Linear SVM** | 0.692 | 0.692 |
| **AdaBoost** | 0.615 | 0.692 |

### Stroke Side Prediction Accuracy

| Model | Accuracy (Before TDA) | Accuracy (After TDA) |
| :--- | :--- | :--- |
| **Random Forest** | 0.818 | 0.818 |
| **XGBoost** | 0.727 | 0.818 |
| **Linear SVM** | 0.727 | 0.727 |

### Severity Classification Accuracy (Bi-LSTM)

| Basis | Accuracy (Before TDA) | Accuracy (After TDA) | F1-score (After TDA) |
| :--- | :--- | :--- | :--- |
| **NIHSS ($\ge 5$)** | 0.6923 | **0.8000** | 0.6667 |
| **mRS ($\ge 3$)** | 0.7692 | **0.8000** | 0.7500 |

---

## Future Work

1.  **Stroke Prediction:** Develop a diagnostic model to predict acute stroke occurrence using EEG signals, comparing EEG features between stroke patients and non-stroke healthy controls.
2.  **Deep Learning and Explainable AI (XAI):** Enhance accuracy and clinical interpretability using Deep Learning (CNN/RNN) on spatiotemporal EEG data and XAI techniques (LIME/SHAP).
3.  **Integrative Multi-Modal Data Fusion:** Design a Multi-Modal Machine Learning framework to systematically fuse EEG features, clinical scores (NIHSS/mRS), and neuroimaging metrics.
4.  **Longitudinal Analysis for Recovery Monitoring:** Conduct a longitudinal study collecting repeated EEG/clinical data (e.g., 3, 6, 12 months) to track and predict functional recovery over time.

## References
[1]  F. H. Alghamedy *et al.*, “EEG-Driven Machine Learning for Stroke Detection in High-Risk Patients,” *IEEE Access*, vol. 13, pp. 166593–166608, 2025.
[2] Cleveland Clinic, “NIH Stroke Scale (NIHSS) – Test for Stroke Severity,”
[3] European Stroke Organisation (ESO), “Outcome Measures in Stroke: Modified Rankin Scale (mRS),”
[4] C. Y.-F. Ling, P. Phang, and S.-H. Liew, “Topological data analysis in EEG signal processing: A review,”
[5] Figshare, “EEG datasets of stroke patients,” 2024. Available:

## Collaborators
1. Rajat Badaria
2. Samyak Mahapatra
3. Pranav Reddy Pedaballe
4. Raunak Gupta
