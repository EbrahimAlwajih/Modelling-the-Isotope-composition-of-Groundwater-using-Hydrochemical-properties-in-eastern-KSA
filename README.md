## Modelling the Isotope composition of Groundwater using Hydrochemical properties in eastern Saudi Arabia: Implementation of Innovative Data Intelligence Techniques

This repository contains the code and models developed to predict the isotope composition (δ¹⁸O and δ²H) of coastal groundwater using Artificial Intelligence (AI) algorithms. The study focuses on understanding the geochemical evolution of groundwater and assessing the impact of seawater intrusion in the Al-Qatif coastal region of eastern Saudi Arabia.

## Study Overview

- **Study Region**: The research was conducted in the Al-Qatif coastal region, which is an arid area with limited surface water resources and vulnerable to seawater intrusion.
- **Study Focus**: The study models the prediction of isotopic composition of groundwater using AI algorithms, leveraging a hydrochemical dataset to understand the mixing processes and impact of seawater intrusion on groundwater quality.

## Key AI Algorithms Used

Eight AI algorithms were utilized to model and predict the isotopic composition of δ¹⁸O and δ²H:

1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Regression (SVR)**
3. **Random Forest (RF)**
4. **Extra Trees (ET)**
5. **Bagging**
6. **AdaBoost (AdaBt)**
7. **Gradient Boosting (GB)**
8. **Classification and Regression Trees (CAT)**

Additionally, stacking ensemble models were applied, which outperformed individual algorithms.

## Model Performance

- **δ¹⁸O (O_M1) Model Performance**:
  - R²: 0.9858
  - Mean Absolute Error (MAE): 0.0440
  - Pearson Correlation: 0.9941

- **δ²H (H_M1) Model Performance**:
  - R²: 0.9317
  - Mean Absolute Error (MAE): 0.5334
  - Pearson Correlation: 0.9658

The study demonstrates that the variation in groundwater chemistry is largely driven by mixing processes, with seawater intrusion being a primary factor affecting groundwater composition.

## Installation

### Prerequisites
Ensure the following libraries and tools are installed:

- Python 3.x
- Pip (Python package installer)

### Dependencies
You can install the required dependencies by running:

```bash
pip install -r requirements.txt
