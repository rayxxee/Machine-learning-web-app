import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error, r2_score)
import seaborn as sns
import matplotlib.pyplot as plt

#Main headings
st.title("Machine Learning Model Trainer")
st.sidebar.header("Upload Dataset and Set Parameters")

#asking user to Upload a file
dataset_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])

#asking what kinf of model the user wants to train and then asking him to select the model of that type from
#the dropdown menu
model_type=st.sidebar.radio("Select Model Type",("Classifiction","Regression"),key="Model")
if model_type=="Classifiction":
    model_choice = st.sidebar.selectbox("Select Model", ["KNN", "SVM", "Random Forest", "Decision Tree", "Logistic Regression"])
elif model_type=="Regression":
    model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Gradient Boosting","SVR"])

# based on what model user selected giving him parameters to adjust whose values will be stored in variables
#which will be used later onwards
if model_choice == "KNN":
    k_value = st.sidebar.slider("Select number of neighbors (k)", min_value=1, max_value=20, value=5, step=1)
elif model_choice == "SVM":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
    gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
elif model_choice == "Random Forest":
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
    max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
elif model_choice == "Decision Tree":
    st.sidebar.subheader("Model Hyperparameters")
    max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
    criterion = st.sidebar.radio("Criterion", ("gini", "entropy"), key='criterion')
    splitter = st.sidebar.radio("Splitter", ("best", "random"), key='splitter')
elif model_choice == "Logistic Regression":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
    max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
elif model_choice == "Linear Regression":
    st.sidebar.subheader("No Hyperparameters for Linear Regression")
elif model_choice == "Gradient Boosting":
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input("The number of boosting stages", 100, 5000, step=10, key='n_estimators_gb')
    learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.0, step=0.01, key='learning_rate')
    max_depth = st.sidebar.number_input("The maximum depth of the individual estimators", 1, 20, step=1, key='max_depth_gb')
elif model_choice=="SVR":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVR')
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear", "poly"), key='kernel_svr')
    gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma_svr')
    epsilon = st.sidebar.slider("Epsilon (Tolerance for error)", 0.01, 1.0, step=0.01, key='epsilon')

# providing a slider to the user to select the size of two datasets train and test
test_size = st.sidebar.slider("Select test data size for testing (as a percentage)", min_value=10, max_value=50, value=20, step=5) / 100

if dataset_file:
    # Loading dataset based on whether its a csv file or a exel spreadsheet and showing 1st few rows
    if dataset_file.name.endswith('csv'):
        data = pd.read_csv(dataset_file)
    elif dataset_file.name.endswith('xlsx'):
        data = pd.read_excel(dataset_file)
    st.write("#### Dataset Preview")
    st.dataframe(data.head())

    # Displaying dataset statistics of each column
    st.write("#### Dataset Statistics")
    st.write(data.describe(include='all'))

    # asking user to select features and target columns according to his need
    # feature and target columns are used to train data
    st.sidebar.subheader("Feature Selection")
    features = st.sidebar.multiselect("Select feature columns", options=data.columns, default=data.columns[:-1])
    target = st.sidebar.selectbox("Select target column", options=data.columns, index=len(data.columns)-1)

    if len(features) > 0 and target:
        X = data[features]
        y = data[target]

        # if user selects a target column that has a specific number of unique 
        # stringor unsupported values for model training then to coonvert them
        #into a form they can be used by the model
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            classes = le.classes_
        else:
            classes = sorted(set(y))
        
        #if user selects a column that has string data in it and has less than 10 unique then that column is converted 
        #into numbers so it can be processed by the model
        cat_features = X.select_dtypes(include=['object']).columns.tolist()  # Find categorical feature columns
        for col in cat_features:
            if X[col].nunique() <= 10:  # Encoding only if a limited number of unique values
                st.write(f"Encoding categorical feature: **{col}**")
                X[col] = LabelEncoder().fit_transform(X[col])
            else:
                st.warning(f"Feature `{col}` has too many unique values; consider preprocessing it outside.")


        # Scaling features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        #splitting data into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=50)

        # Initialize model with the type of model to be used
        if model_choice == "KNN":
            model = KNeighborsClassifier(n_neighbors=10)
        elif model_choice == "SVM":
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap == 'True', random_state=50)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, splitter=splitter, random_state=42)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=50)
        elif model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=50)
        elif model_choice== "SVR":
            model = SVR(C=C, kernel=kernel, gamma=gamma, epsilon=epsilon)

        # Training the model using library function in sklearn
        model.fit(X_train, y_train)

        # Predicting values using the model that was trained for diff. test values
        y_pred = model.predict(X_test)
        

        # showing diff. metrics based on the type of model selected(regression/classification)
        if model_type=="Classifiction":
            #calculating metrics using inbuilt functions
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Display metrics calculated
            st.write("## Model Performance Metrics")
            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.write(f"**Precision:** {precision:.2f}")
            st.write(f"**Recall (Sensitivity):** {recall:.2f}")
            st.write(f"**F1 Score:** {f1:.2f}")

            # showing heatmap of confusion matrix 
            st.write("**Confusion Matrix:**")
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis", xticklabels=np.unique(y), yticklabels=np.unique(y))
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix Heatmap")
            st.pyplot(plt)

            #showing scatter plot of predicted values by the model
            if len(features) >= 2:
                st.write("### Scatter Plot of Predictions")
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=[classes[i] for i in y_pred], style=[classes[i] for i in y_test], palette="magma",s=100)
                plt.title("Scatter Plot of Predictions")
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                st.pyplot(plt)
            else:
                st.warning("Please select valid features and target column.")


        elif model_type=="Regression":
            #calculating metrics for regression models
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Display metrics calculated
            st.write("## Regression Model Performance Metrics")
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")

            # Scatter plot for regression model based on predicted values by model
            st.write("**Actual vs Predicted Scatter Plot:**")
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.7, color='b')
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Actual vs Predicted Values")
            st.pyplot(plt)

        # showing Feature importance (A special metric for  Random Forest model)
        if model_choice == "Random Forest":
            st.write("**Feature Importance:**")
            feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.bar_chart(feature_importance)

        # Highlight underfitting/overfitting trends
        st.write("**Model Fit Analysis:**")
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        st.write(f"**Training Score:** {train_score:.2f}")
        st.write(f"**Testing Score:** {test_score:.2f}")
        if train_score > test_score + 0.1:
            st.warning("The model might be overfitting. Consider adjusting the hyperparameters.")
        elif test_score > train_score + 0.1:
            st.warning("The model might be underfitting. Consider increasing model complexity.")

    else:
        st.warning("Please select valid features and target column.")
#end of program