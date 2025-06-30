# NHL Goal-Scoring Prediction & Shot Analysis

![Hockey Analytics](./milestone%201/nhl_rink.png)

## Goal: predict the probability of a goal for any given shot in the National Hockey League (NHL), creating a robust 'Expected Goals' (xG) model.

## Our Amazing Team:
- [@gandatchabana](https://github.com/gandatchabana)
- [@linkai-aries-ma](https://github.com/linkai-aries-ma)
- [@liyuanmontreal](https://github.com/liyuanmontreal)

## 🎯 Project Overview

In modern hockey analytics, moving beyond traditional statistics like goals and assists is essential for a deeper understanding of player performance and team strategy. The core challenge is that not all shots are created equal. This project tackles this challenge by building an end-to-end system to quantify shot quality by predicting the probability of a goal for every shot taken.

The result is a powerful 'Expected Goals' (xG) model that provides a more nuanced way to evaluate performance, identifying players who generate high-quality chances regardless of whether they result in a goal.

## ✨ Key Features

* **Automated Data Collection:** Gathers raw, event-level game data directly from the NHL's public API.
* **In-Depth EDA:** Includes visualizations of shot and goal distributions on a rink plot to uncover patterns.
* **Advanced Feature Engineering:** Creates high-impact predictive features like shot distance, angle, and rebound status.
* **Predictive Modeling:** Implements, evaluates, and compares multiple machine learning models (Logistic Regression, SVM, XGBoost) to predict goal probability.
* **Model Evaluation:** Uses a comprehensive suite of metrics including Accuracy, Precision, Recall, F1-Score, and AUC-ROC to handle the imbalanced nature of the dataset.

---

## ⚙️ Methodology

This project follows a structured data science workflow from data acquisition to model deployment.

#### 1. Data Collection
-   Game data was programmatically collected for over 3,000 NHL games using a Python script that queries the official **NHL public API**.
-   The raw JSON response was parsed to extract event-level shot data.

#### 2. Data Cleaning & Preprocessing
-   Handled missing or inconsistent data points to ensure data quality.
-   Transformed data types and structured the data into a clean, usable format for analysis.

#### 3. Exploratory Data Analysis (EDA)
-   Performed extensive EDA to understand the characteristics of goal-scoring events.
-   Key visualizations included rink plots showing the spatial distribution of all shots versus shots that resulted in goals, confirming that the majority of goals are scored from the high-danger "slot" area.

#### 4. Feature Engineering
-   Based on EDA and domain knowledge, the following critical features were engineered to improve model performance:
    -   `Shot Distance`: The Euclidean distance from the shot coordinates to the center of the net.
    -   `Shot Angle`: The angle of the shot relative to the center of the net. A wider angle is more difficult for a goalie.
    -   `Is Rebound`: A binary flag to indicate if a shot was taken within seconds of a preceding shot.
    -   `Power Play`: A binary flag indicating if the shot was taken during a power play.
    -   `Shot Type`: Categorical data for shot types (e.g., Wrist Shot, Slap Shot, etc.).

#### 5. Model Training & Evaluation
-   Multiple classification models were trained to predict the binary outcome (Goal vs. No Goal):
    1.  **Logistic Regression:** Served as a baseline model.
    2.  **Support Vector Machine (SVM):** A more complex, non-linear model.
    3.  **XGBoost Classifier:** A powerful gradient-boosting algorithm known for high performance.
-   Models were rigorously evaluated, with a focus on the **AUC-ROC score** due to the severe class imbalance (goals are rare events).

---

## 📈 Results & Key Findings

After training and evaluating several models, the **XGBoost Classifier** emerged as the champion, demonstrating the strongest predictive performance.

#### Champion Model: XGBoost
With the [selected features](Milestone%202/results/Xgboost/selected_features.png), the final XGBoost model achieved an **AUC score of 0.76**, indicating a strong capability to distinguish between goals and non-goal shots. The detailed performance diagnostics for this model are shown below.

![XGBoost Model Performance](Milestone%202/results/Xgboost/Xgboost_selected_features.png)

---

#### Model Comparison
To ensure a rigorous evaluation, we compared the performance of several algorithms. The Area Under the Curve (AUC) score for each model is summarized below, confirming that XGBoost was the most effective choice.

| Model | AUC Score |
| :--- | :---: |
| [**XGBoost Classifier**](Milestone%202/results/Xgboost/Xgboost_selected_features.png) | **0.76**  |
| [Decision Tree Classifier](Milestone%202/results/Decision_Tree.png)|0.73|
| [Gaussian Naive Bayes](Milestone%202/results/Naive%_Bayes.png) |0.68|
| [Neural network](Milestone%202/results/Neural_Network.png) |0.76|
---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation & Analysis:** pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **API Interaction:** `requests` library
* **Development Environment:** Jupyter Notebooks
