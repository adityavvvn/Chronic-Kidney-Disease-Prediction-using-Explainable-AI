
# Chronic Kidney Disease Prediction using Explainable AI

## Overview

This project focuses on the early detection of Chronic Kidney Disease (CKD) through machine learning models enhanced with Explainable AI (XAI) techniques. The goal is to not only predict the presence of CKD but also to provide clear insights into the factors influencing each prediction, thereby aiding healthcare professionals in decision-making processes.([github.com][1])

## Features

* **Data Preprocessing**: Handling missing values, encoding categorical variables, and normalizing data.
* **Model Development**: Implementing classification algorithms such as Logistic Regression, Random Forest, and Support Vector Machines.
* **Model Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
* **Explainability**: Applying SHAP (SHapley Additive exPlanations) to interpret model predictions and understand feature importance.

## Dataset

The dataset used contains various medical attributes relevant to CKD diagnosis. It includes features such as age, blood pressure, specific gravity, albumin levels, and more. The dataset is preprocessed to ensure quality and consistency for model training.([pmc.ncbi.nlm.nih.gov][2])

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/adityavvvn/Chronic-Kidney-Disease-Prediction-using-Explainable-AI.git
   cd Chronic-Kidney-Disease-Prediction-using-Explainable-AI
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

   Open the notebook file and execute the cells sequentially to preprocess data, train models, and visualize explanations.

2. **Interpret Results**:
   Use the SHAP plots generated to understand how each feature contributes to the model's predictions.

## Results

The implemented models achieved satisfactory performance metrics, indicating their potential effectiveness in early CKD detection. The use of SHAP provided valuable insights into feature importance, enhancing the transparency of the predictive models.([mdpi.com][3], [github.com][1])


## License

This project is licensed under the MIT License.

---

For more details, visit the [GitHub repository](https://github.com/adityavvvn/Chronic-Kidney-Disease-Prediction-using-Explainable-AI).

[1]: https://github.com/asthasoni22/Disease-Prediction-XAI-publication?utm_source=chatgpt.com "asthasoni22/Disease-Prediction-XAI-publication - GitHub"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12025083/?utm_source=chatgpt.com "Explainable AI for Chronic Kidney Disease Prediction in Medical IoT"
[3]: https://www.mdpi.com/1999-4893/17/10/443?utm_source=chatgpt.com "Explainable Machine Learning Model for Chronic Kidney Disease ..."
