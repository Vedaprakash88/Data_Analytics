import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import tensorflow as tf
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 1. Data Ingestion
def load_data(file_paths):
    """
    Load data from multiple sources with error handling
    """
    try:
        data_sources = {}
        for name, path in file_paths.items():
            try:
                data_sources[name] = pd.read_csv(path)
                logger.info(f"Successfully loaded {name} from {path}")
            except FileNotFoundError:
                logger.error(f"File not found: {path}")
                data_sources[name] = pd.DataFrame()
        
        return data_sources
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise

# 2. Data Preprocessing and Cleaning
def preprocess_data(data_sources):
    """
    Advanced preprocessing with more robust feature engineering
    """
    try:
        order_data = data_sources['order_data']
        vendor_data = data_sources['vendor_data']
        gps_data = data_sources['gps_data']
        weather_data = data_sources['weather_data']
        traffic_data = data_sources['traffic_data']

        # Enhanced preprocessing
        order_data['timestamp'] = pd.to_datetime(order_data['timestamp'])
        
        # Advanced feature engineering
        order_data['hour_of_day'] = order_data['timestamp'].dt.hour
        order_data['day_of_week'] = order_data['timestamp'].dt.dayofweek
        order_data['is_weekend'] = order_data['day_of_week'].isin([5, 6]).astype(int)
        order_data['month'] = order_data['timestamp'].dt.month
        order_data['season'] = pd.cut(order_data['month'], 
                                       bins=[0, 3, 6, 9, 12], 
                                       labels=['Winter', 'Spring', 'Summer', 'Autumn'])

        # Merge datasets with more robust approach
        data = order_data.merge(vendor_data, on='vendor_id', how='left')
        data = data.merge(gps_data, on='order_id', how='left')
        data = data.merge(weather_data, on='timestamp', how='left')
        data = data.merge(traffic_data, on='timestamp', how='left')

        return data
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise

# 3. Exploratory Data Analysis (EDA)
def perform_eda(data):
    """
    Comprehensive Exploratory Data Analysis
    """
    try:
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()

        # Distribution of delivery times
        plt.figure(figsize=(10, 6))
        sns.histplot(data['delivery_time'], kde=True)
        plt.title('Distribution of Delivery Times')
        plt.savefig('delivery_time_distribution.png')
        plt.close()

        logger.info("EDA visualizations saved successfully")
    except Exception as e:
        logger.error(f"Error in EDA: {e}")

# 4. Model Training with Advanced Techniques
def train_model(data):
    """
    Advanced model training with preprocessing pipeline
    """
    try:
        # Separate features and target
        X = data.drop('delivery_time', axis=1)
        y = data['delivery_time']

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])

        # Create model pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
        ])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Model Performance - MAE: {mae}, RMSE: {rmse}, R2: {r2}")

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
        logger.info(f"Cross-validation MAE: {-cv_scores.mean()}")

        # Save model
        joblib.dump(pipeline, 'food_delivery_model.joblib')
        logger.info("Model saved successfully")

        return pipeline
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

# Main execution
def main():
    file_paths = {
        'order_data': 'order_data.csv',
        'vendor_data': 'vendor_data.csv',
        'gps_data': 'gps_data.csv',
        'weather_data': 'weather_data.csv',
        'traffic_data': 'traffic_data.csv'
    }
    
    try:
        data_sources = load_data(file_paths)
        processed_data = preprocess_data(data_sources)
        perform_eda(processed_data)
        model = train_model(processed_data)
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()