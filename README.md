# Employee-Attrition-Prediction-and-Analysis

## Overview

This Streamlit app provides a user-friendly interface for predicting employee attrition and exploring HR data. It allows users to filter and visualize data, build and evaluate predictive models, and discover insights about employee attrition within their organization.

## Features

- Data Filtering: Select a specific department to filter the data and focus on a particular segment of the organization.
- Data Exploration: Explore summary statistics, data distributions, and relationships between various HR features.
- Data Preprocessing: Encode categorical variables for machine learning model training.
- Employee Attrition Prediction: Train a Random Forest classifier to predict employee attrition and evaluate the model's performance.
- Most Risky Employees: Identify the employees at the highest risk of attrition based on predicted probabilities.
- Data Visualization: Visualize data through various charts, including box plots, bar charts, and pie charts.
- Custom Data Exploration: You can further enhance the app with custom data analysis and visualization.

## Installation

To run this Streamlit app, you'll need Python and a few libraries. Here's how to set it up:

1. Clone or download this repository to your local machine.
2. Open a terminal and navigate to the project directory.
3. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
```

Install the required packages:
```bash
pip install -r requirements.txt
```

Run the Streamlit app:
```bash
streamlit run prediction.py
```

The app will launch in your web browser, and you can start exploring and predicting employee attrition.

## Usage

1. Select a department from the sidebar to filter the data.
   
2.Explore data in various sections:

   - Data Exploration: View sample data, data shape, information, and summary statistics.
   - Data Preprocessing: Encode categorical variables for machine learning.
   - Employee Attrition Prediction: Predict attrition using the Random Forest model.
   - Most Risky Employees: Identify the most risky employees based on predicted probabilities.
   - Data Visualization: Explore data through various charts.
    
3. Customize the app by adding your own analysis and visualization code.
