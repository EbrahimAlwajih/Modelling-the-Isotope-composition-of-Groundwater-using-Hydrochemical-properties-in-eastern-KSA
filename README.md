# Modeling the Isotope Composition of Groundwater Using Hydrochemical Properties in Eastern Saudi Arabia: Implementation of Innovative Data Intelligence Techniques

## Overview

This project focuses on the modeling and prediction of the isotope composition (δ¹⁸O and δ²H) of coastal groundwater in the Al-Qatif region of eastern Saudi Arabia. The region, being arid with limited surface water resources, is highly vulnerable to seawater intrusion, which affects groundwater quality. The project leverages **Artificial Intelligence (AI) models** and **data intelligence techniques** to predict groundwater isotope compositions using a dataset of hydrochemical properties from various wells in the region.

The study provides insights into the geochemical evolution of groundwater and the impact of seawater intrusion in arid coastal environments.

## Key Features

- **AI Algorithms**: Utilizes various AI models, including K-Nearest Neighbors (KNN), Support Vector Regression (SVR), Random Forest (RF), Extra Trees (ET), AdaBoost (AdaBt), Gradient Boosting (GRB), and CatBoost.
- **Ensemble Models**: Implements Stacking Regressor to improve model accuracy by combining different base models.
- **Performance Metrics**: Achieves high prediction accuracy with the optimal models:
  - **δ¹⁸O (O_M1)**: R² = 0.9858, MAE = 0.0440, Pearson correlation = 0.9941.
  - **δ²H (H_M1)**: R² = 0.9317, MAE = 0.5334, Pearson correlation = 0.9658.
- **Insights on Groundwater Chemistry**: Identifies the primary impact of seawater intrusion on groundwater geochemistry, as evidenced by the relationship between hydrochemical parameters and isotopic compositions.

## Requirements

To replicate this study or use the models in your own research, you need the following dependencies.

### Installing Dependencies

1. **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**:

    - On **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    - On **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### `requirements.txt`

```plaintext
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
catboost>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
joblib>=1.0.0



## How to Run

### 1. Clone the repository:

```bash
git clone https://github.com/EbrahimAlwajih/Modelling-the-Isotope-composition-of-Groundwater-using-Hydrochemical-properties-in-eastern-KSA.git
cd groundwater-ai-prediction

### 2. Set up a virtual environment and install dependencies:

1. **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**:
    - On **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    - On **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the main script:

After setting up the virtual environment and installing the dependencies, you can run the main script to load the dataset, clean the data, train the models, and output the results:

```bash
python main.py
