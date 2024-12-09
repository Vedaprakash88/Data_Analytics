# Import necessary libraries
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Simulating Data for Price Sensitivity A/B Test
def simulate_price_sensitivity_data():
    """Simulate data for pricing strategies."""
    np.random.seed(42)
    data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'price': np.random.choice([49.99, 54.99, 59.99, 64.99], size=1000),
        'conversion': np.random.choice([0, 1], size=1000, p=[0.75, 0.25]),
    })
    # Adjust conversion probabilities based on price (lower prices, higher conversion)
    data.loc[data['price'] == 49.99, 'conversion'] = np.random.choice([0, 1], size=(data['price'] == 49.99).sum(), p=[0.6, 0.4])
    data.loc[data['price'] == 54.99, 'conversion'] = np.random.choice([0, 1], size=(data['price'] == 54.99).sum(), p=[0.7, 0.3])
    data.loc[data['price'] == 59.99, 'conversion'] = np.random.choice([0, 1], size=(data['price'] == 59.99).sum(), p=[0.8, 0.2])
    data.loc[data['price'] == 64.99, 'conversion'] = np.random.choice([0, 1], size=(data['price'] == 64.99).sum(), p=[0.85, 0.15])
    return data

data = simulate_price_sensitivity_data()
print(data.head())

# Step 2: Exploratory Data Analysis (EDA)
print("Data Overview:\n", data.describe())
conversion_rates = data.groupby('price')['conversion'].mean()
print("Conversion Rates by Price:\n", conversion_rates)

# Step 3: A/B Testing for Pricing Strategies
def ab_test_conversion_rates(data):
    """Conduct A/B testing between different price groups."""
    price_groups = data['price'].unique()
    for i in range(len(price_groups)):
        for j in range(i + 1, len(price_groups)):
            group_a = data[data['price'] == price_groups[i]]['conversion']
            group_b = data[data['price'] == price_groups[j]]['conversion']
            t_stat, p_value = stats.ttest_ind(group_a, group_b)
            print(f"Comparing {price_groups[i]} vs {price_groups[j]}:")
            print(f"  T-Stat: {t_stat:.2f}, P-Value: {p_value:.4f}\n")

ab_test_conversion_rates(data)

# Step 4: Determine Optimal Pricing Point
optimal_price = conversion_rates.idxmax()
print(f"Optimal Price Based on Conversion Rates: {optimal_price}")

# Step 5: Predictive Modeling for Price Elasticity
def build_price_elasticity_model(data):
    """Build a regression model to predict conversion rates based on price."""
    # Prepare the data
    data['price_scaled'] = data['price'] / data['price'].max()  # Scale prices
    X = data[['price_scaled']]
    y = data['conversion']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and evaluate the model
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    elasticity = model.coef_[0]
    
    print("Regression Model Coefficient (Price Elasticity):", elasticity)
    print("R2 Score:", r2_score(y_test, predictions))
    print("Mean Squared Error:", mean_squared_error(y_test, predictions))

    # Recommendation: Negative elasticity indicates demand decreases as price increases
    if elasticity < 0:
        print("Demand is elastic: Lowering prices will increase conversions.")
    else:
        print("Demand is inelastic: Raising prices may not significantly affect conversions.")

build_price_elasticity_model(data)

# Step 6: Visualization (Optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Conversion rate vs price
sns.barplot(x=conversion_rates.index, y=conversion_rates.values)
plt.title('Conversion Rates by Price')
plt.xlabel('Price')
plt.ylabel('Conversion Rate')
plt.show()


'''

Explanation of Steps
Data Simulation: Creates a dataset with four price tiers and corresponding probabilities of conversion. Adjusts conversion probabilities to simulate real-world price sensitivity.

EDA: Provides insights into conversion rates at different price points.

A/B Testing: Conducts pairwise t-tests to check statistical differences between conversion rates for various price tiers.

Optimal Pricing Point: Identifies the price with the highest conversion rate.

Price Elasticity Modeling: Builds a regression model to predict the relationship between price and conversion. Outputs a price elasticity coefficient to guide pricing strategy.

Visualization: Optional barplot for quick visualization of conversion rates by price.

'''