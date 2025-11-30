# Credit Risk Analysis - Final Project

## 📋 Project Description

This project aims to develop a complete credit risk analysis system using Machine Learning techniques. The system will evaluate the probability of customer default and make informed decisions about credit approval or rejection.

**Dataset**: PAKDD2010 - Credit Risk Analysis Dataset

## 🎯 Objectives

1. **EDA (Exploratory Data Analysis)**: Perform a complete exploratory analysis of the customer and credit dataset.
2. **Preprocessing Pipeline**: Design a standard preprocessing pipeline for the entire team.
3. **Model Training**: Train and compare various ML models for credit risk (logistic regression, decision trees, ensembles, etc.).
4. **Model Selection**: Choose a final model based on evaluation metrics.
5. **API Deployment**: Expose the model through a REST API using FastAPI.
6. **UI Demo**: Build a simple interface (Streamlit) to demonstrate how a "bank" would use the model to approve/reject credits.

## 📁 Project Structure

```
Credit-Risk-Analysis-Final-Project/
├── data/
│   ├── raw/              # Original PAKDD2010 dataset unprocessed
│   │   ├── PAKDD2010_Modeling_Data.txt
│   │   ├── PAKDD2010_Prediction_Data.txt
│   │   ├── PAKDD2010_VariablesList.XLS
│   │   └── ...
│   └── processed/        # Clean and preprocessed dataset (to be created)
├── notebooks/
│   ├── EDA/              # Exploratory data analysis notebooks
│   │   └── simon_EDA.ipynb
│   └── preprocessing/    # Preprocessing notebooks (to be created)
├── src/                  # Python source code (to be developed)
│   └── __init__.py
├── api/                  # FastAPI (to be developed)
├── ui/                   # Streamlit UI (to be developed)
├── models/               # Saved trained models (to be created)
├── Dockerfile            # Docker for API + model
├── docker-compose.yml    # Service orchestration
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## 🚀 Installation and Setup

### 1. Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import pandas, sklearn, fastapi; print('Dependencies installed correctly')"
```

## 📝 Current Project Status

The project is under active development. The base structure is established and the team is working on:

- **Folder structure**: Project organization defined
- **Dataset**: PAKDD2010 data loaded in `data/raw/`
- **EDA**: Exploratory analysis in progress (`notebooks/EDA/`)
- **Preprocessing**: Preprocessing pipeline (to be developed)
- **Models**: Training and comparison (to be developed)
- **API**: FastAPI for deployment (to be developed)
- **UI**: Streamlit interface (to be developed)

## 📝 Next Steps

1. **Complete EDA**:

   - Analyze variable distributions
   - Identify null values and outliers
   - Study correlations between features
   - Document findings in `notebooks/EDA/`

2. **Develop Preprocessing Pipeline**:

   - Implement cleaning and transformation functions
   - Define imputation and encoding strategies
   - Save processed data in `data/processed/`

3. **Train Models**:

   - Implement baseline model
   - Compare multiple models (Logistic Regression, Decision Tree, Random Forest, etc.)
   - Select final model based on metrics relevant for credit risk

4. **Develop API**:

   - Implement FastAPI in `api/`
   - Connect with trained model
   - Expose `/predict` endpoint

5. **Develop UI**:
   - Implement Streamlit interface in `ui/`
   - Connect with API for real-time predictions

## 👥 Team

This project is being developed by a team of 6 people.
