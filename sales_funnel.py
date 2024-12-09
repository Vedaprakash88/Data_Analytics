# Import necessary libraries
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Seed for reproducibility
np.random.seed(42)

# Step 1: Data Simulation or Loading
def simulate_data():
    """Simulate a dataset for the A/B testing project."""
    data = pd.DataFrame({
        'customer_id': range(1, 501),
        'variant': np.random.choice(['A', 'B'], size=500),
        'sales_script': np.random.choice(['script1', 'script2'], size=500),
        'cta_button': np.random.choice(['button1', 'button2'], size=500),
        'customer_segment': np.random.choice(['segment1', 'segment2'], size=500),
        'conversion': np.random.choice([0, 1], size=500, p=[0.7, 0.3])
    })
    return data

data = simulate_data()
print(data.head())

# Step 2: Exploratory Data Analysis (EDA)
print("Summary Statistics:\n", data.describe())
print("Missing Values:\n", data.isnull().sum())

# Step 3: A/B Testing
def ab_testing(data, group_col, metric_col):
    """Conduct A/B testing for given groups and metrics."""
    group_a = data[data[group_col] == 'A'][metric_col]
    group_b = data[data[group_col] == 'B'][metric_col]
    
    # T-test for means
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    print(f"T-Test Results for {metric_col}:\nT-Stat: {t_stat:.2f}, P-Value: {p_value:.4f}")
    
    # Proportions Z-Test if binary data
    if data[metric_col].nunique() == 2:
        count = [group_a.sum(), group_b.sum()]
        nobs = [len(group_a), len(group_b)]
        z_stat, p_val = stats.proportions_ztest(count, nobs)
        print(f"Z-Test Results for {metric_col}:\nZ-Stat: {z_stat:.2f}, P-Value: {p_val:.4f}")

# Example: A/B test for conversion
ab_testing(data, group_col='variant', metric_col='conversion')

# Step 4: Statistical Data Analysis (Regression)
def regression_analysis(data, predictors, target):
    """Perform regression analysis."""
    X = data[predictors]
    y = data[target]
    
    # Linear Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(f"Regression Coefficients: {model.coef_}")
    print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")

# Example: Regression for conversion
regression_analysis(data, predictors=['sales_script', 'cta_button', 'customer_segment'], target='conversion')

# Step 5: Predictive Modeling
def predictive_model(data, predictors, target):
    """Build and evaluate a predictive model."""
    X = data[predictors]
    y = data[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Regressor
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(f"Random Forest R2 Score: {r2_score(y_test, predictions):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")

# Example: Predict conversion
predictive_model(data, predictors=['sales_script', 'cta_button', 'customer_segment'], target='conversion')
