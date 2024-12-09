# Import necessary libraries
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, r2_score

# Seed for reproducibility
np.random.seed(42)

# Step 1: Data Simulation or Loading
# Replace this with your actual dataset
def simulate_data():
    """Simulate a dataset for the A/B testing project."""
    data = pd.DataFrame({
        'customer_id': range(1, 501),
        'variant': np.random.choice(['A', 'B'], size=500),
        'price': np.random.choice([49.99, 59.99], size=500),  # Pricing example
        'email_opened': np.random.choice([0, 1], size=500, p=[0.7, 0.3]),  # Email campaign
        'clicked': np.random.choice([0, 1], size=500, p=[0.8, 0.2]),  # Sales funnel
        'insurance_claim': np.random.choice([0, 1], size=500, p=[0.9, 0.1]),  # Insurance risk
        'conversion_rate': np.random.rand(500) * 100,  # Cross-domain conversion
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
    
    # Proportions Z-Test
    if data[metric_col].nunique() == 2:  # Binary data
        count = [group_a.sum(), group_b.sum()]
        nobs = [len(group_a), len(group_b)]
        z_stat, p_val = proportions_ztest(count, nobs)
        print(f"Z-Test Results for {metric_col}:\nZ-Stat: {z_stat:.2f}, P-Value: {p_val:.4f}")

# Example: A/B test for email_opened
ab_testing(data, group_col='variant', metric_col='email_opened')

# Step 4: Statistical Data Analysis (Regression and ANOVA)
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

# Example: Regression for conversion_rate
regression_analysis(data, predictors=['price'], target='conversion_rate')

# Step 5: Advanced Statistical Methods (ANOVA)
def perform_anova(data, group_col, metric_col):
    """Perform ANOVA analysis."""
    groups = [data[data[group_col] == grp][metric_col] for grp in data[group_col].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"ANOVA Results for {metric_col} by {group_col}:\nF-Stat: {f_stat:.2f}, P-Value: {p_value:.4f}")

# Example: ANOVA for conversion_rate across variants
perform_anova(data, group_col='variant', metric_col='conversion_rate')

# Step 6: Predictive Modeling
def predictive_model(data, predictors, target):
    """Build and evaluate a predictive model."""
    X = data[predictors]
    y = data[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(f"Random Forest R2 Score: {r2_score(y_test, predictions):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")

# Example: Predict conversion_rate
predictive_model(data, predictors=['price'], target='conversion_rate')

# Step 7: Recommendation Framework
def recommend_pricing(data):
    """Provide recommendations based on analysis."""
    price_groups = data.groupby('price')['conversion_rate'].mean()
    print("Conversion Rates by Price:\n", price_groups)
    optimal_price = price_groups.idxmax()
    print(f"Optimal Price Recommendation: {optimal_price}")

recommend_pricing(data)



'''

How to Build Upon This Code:
Data Integration: Replace the simulated data with your domain-specific datasets.
Hypothesis Testing: Customize A/B testing metrics (e.g., open rates, claim rates).
Feature Engineering: Add relevant features, like demographics or customer segments.
Advanced Analysis: Incorporate Bayesian analysis, confidence intervals, or chi-square tests.
Domain-Specific Insights: Tailor insights to specific business needs like pricing elasticity or segmentation strategies.


'''
