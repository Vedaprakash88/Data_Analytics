import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# ---------------------- Data Collection ----------------------

# Load S&P 500 Company Data from Wikipedia
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_df = pd.read_html(sp500_url, header=0)[0]

# Keep Relevant Columns
sp500_df = sp500_df[['Symbol', 'Security', 'GICS Sector']]

# Function to Fetch Financial Data from Yahoo Finance
def fetch_financial_data(symbols):
    data = {}
    for symbol in symbols:
        try:
            # Download historical stock data
            stock = yf.Ticker(symbol)
            history = stock.history(period="5y")
            
            # Calculate average returns and volatility
            data[symbol] = {
                'Close': history['Close'],
                'Returns': history['Close'].pct_change().mean(),
                'Volatility': history['Close'].pct_change().std()
            }
        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {e}")
    return data

# Fetch Data for the First 50 Companies (for demo purposes)
financial_data = fetch_financial_data(sp500_df['Symbol'][:50])

# Convert Financial Data to DataFrame
financial_df = pd.DataFrame(financial_data).T.reset_index().rename(columns={'index': 'Symbol'})

# Simulate ESG Data (replace this with real ESG API data if available)
np.random.seed(42)
sp500_df['ESG_Score'] = np.random.uniform(50, 100, len(sp500_df))

# ---------------------- Data Analysis ----------------------

# Merge ESG and Financial Data
merged_df = pd.merge(sp500_df, financial_df, on='Symbol', how='inner')

# Calculate Correlations Between ESG and Financial Metrics
correlations = merged_df[['ESG_Score', 'Returns', 'Volatility']].corr()
print("Correlation Matrix:")
print(correlations)

# Sector-wise ESG and Financial Performance Summary
sector_avg = merged_df.groupby('GICS Sector').agg({
    'ESG_Score': 'mean',
    'Returns': 'mean',
    'Volatility': 'mean'
}).reset_index()

print("\nSector-wise ESG Performance Summary:")
print(sector_avg)

# ---------------------- Visualization ----------------------

# Scatter Plot: ESG Score vs. Returns
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='ESG_Score', y='Returns', hue='GICS Sector')
plt.title('ESG Score vs Returns')
plt.xlabel("ESG Score")
plt.ylabel("Average Returns")
plt.legend(title="Sector", bbox_to_anchor=(1, 1))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap Between ESG and Financial Metrics")
plt.show()

# Sector-wise ESG Heatmap
pivot_df = merged_df.pivot_table(index='GICS Sector', values='ESG_Score', aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_df, cmap='YlGnBu', annot=True, linewidths=.5)
plt.title('Average ESG Score by Sector')
plt.show()

# ---------------------- Export Results ----------------------

# Save Results to CSV for Dashboard Integration
merged_df.to_csv('esg_financial_analysis.csv', index=False)
print("\nData exported to 'esg_financial_analysis.csv' for dashboard creation.")