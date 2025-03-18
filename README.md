# Machine Learning Web App

## Overview
This web application allows users to upload datasets and train machine learning models using a user-friendly interface powered by Streamlit. Users can choose between classification and regression models, configure hyperparameters, and visualize model performance metrics.

## Features
- Upload datasets in CSV or Excel format
- Select between Classification and Regression models
- Choose from various machine learning models:
  - **Classification**: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, Decision Tree, Logistic Regression
  - **Regression**: Linear Regression, Gradient Boosting, Support Vector Regression (SVR)
- Hyperparameter tuning options for different models
- Automatic handling of categorical data
- Dataset statistics and visualization
- Performance metrics evaluation and visualizations
- Model fit analysis for detecting overfitting/underfitting

## Installation
To run this web application locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/rayxxee/Machine-learning-web-app.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Machine-learning-web-app
   ```
  
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Upload your dataset (CSV or Excel format)
2. Select the model type (Classification or Regression)
3. Choose a machine learning model
4. Configure hyperparameters (if applicable)
5. Select feature and target columns
6. Train the model and view performance metrics
7. Analyze the results with visualizations

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib


## Contribution
Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request with your improvements.

## Author
[Rayyan](https://github.com/rayxxee)

