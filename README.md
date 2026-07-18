# 🚀 GitHub Community Adoption Prediction

## Project Overview

This project predicts whether a GitHub repository will achieve **High Adoption** or **Low Adoption** based on repository characteristics.

The prediction is performed using Machine Learning algorithms and deployed as an interactive Streamlit web application.

---

## Problem Statement

The goal is to identify whether a GitHub repository is likely to become highly adopted by analyzing factors such as:

- Programming language
- Number of forks
- Open issues
- Contributors
- Repository size
- License information
- Repository age
- Recent activity

---

## Dataset

Dataset:

`github_repo_dataset.csv`

Total Records:

`46,201 repositories`

Target Variable:

- High Adoption
- Low Adoption

High Adoption is defined as repositories in the top 20% based on GitHub stars.

---

# Machine Learning Workflow

The project follows these steps:

1. Data Loading
2. Data Cleaning
3. Missing Value Handling
4. Feature Engineering
5. Feature Encoding
6. Feature Scaling
7. Model Training
8. Model Evaluation
9. Model Deployment

---

# Algorithms Used

The following classification algorithms were tested:

| Algorithm | Accuracy |
|----------|----------|
| Logistic Regression | 86.31% |
| KNN | 85.89% |
| LightGBM | 88.75% |

---

# Best Model

## LightGBM Classifier

Selected because it achieved the best performance.

Performance:

- Accuracy: 88.75%
- Precision: 91.17%
- Recall: 95.12%
- F1 Score: 93.10%

---

# Features Used

The model uses:

- Language
- Forks
- Open Issues
- Contributors
- Size (KB)
- License
- Project Age
- Days Since Last Push
- Description Length
- Created Year
- Pushed Year

---

# Deployment

The model is deployed using Streamlit.

The application allows users to:

1. Enter a GitHub repository URL.
2. Fetch repository information using GitHub API.
3. Predict adoption level.
4. Display confidence percentage.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>