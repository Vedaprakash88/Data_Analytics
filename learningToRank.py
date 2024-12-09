import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import lightgbm as lgb

class Letor_Converter(object):    
    '''
    Class Converter implements parsing from original letor txt files to
    pandas data frame representation.
    '''
    
    def __init__(self, path):
        
        '''
        Arguments:
            path: path to letor txt file
        '''
        self._path = path
        
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, p):
        self._path = p
        
    def _load_file(self):
        '''        
        Loads and parses raw letor txt file.
        
        Return:
            letor txt file parsed to csv in raw format
        '''
        return pd.read_csv(str(self._path), sep=" ", header=None)
        
    def _drop_col(self, df):
        '''
        Drops last column, which was added in the parsing procedure due to a
        trailing white space for each sample in the text file
        
        Arguments:
            df: pandas dataframe
        Return:
            df: original df with last column dropped
        '''
        return df.drop(df.columns[-1], axis=1)
    
    def _split_colon(self, df):
        '''
        Splits the data on the colon and transforms it into a tabular format
        where columns are features and rows samples. Cells represent feature
        values per sample.
        
        Arguments:
            df: pandas dataframe object
        Return:
            df: original df with string pattern ':' removed; columns named appropriately
        '''
        for col in range(1,len(df.columns)):
            df.loc[:,col] = df.loc[:,col].apply(lambda x: str(x).split(':')[1])
        df.columns = ['rel', 'qid'] + [str(x) for x in range(1,len(df.columns)-1)] # renaming cols
        return df
    
    def convert(self):
        '''
        Performs final conversion.
        
        Return:
            fully converted pandas dataframe
        '''
        df_raw = self._load_file()
        df_drop = self._drop_col(df_raw)
        return self._split_colon(df_drop)
    
conv = Letor_Converter("web30k/Fold1/test.txt")
df_train_fold1 = conv.convert()
print (df_train_fold1.shape)
print(df_train_fold1.head())

# Load dataset (example using MSLR-WEB10K)
data = pd.read_csv('MSLR-WEB10K.csv')

# Preprocess data
# Assuming the dataset has columns: 'query_id', 'product_id', 'features', 'relevance'
X = data.drop(columns=['query_id', 'product_id', 'relevance'])
y = data['relevance']
query_ids = data['query_id']

# Split data into training and testing sets
X_train, X_test, y_train, y_test, query_train, query_test = train_test_split(X, y, query_ids, test_size=0.2, random_state=42)

# Prepare data for LightGBM
train_data = lgb.Dataset(X_train, label=y_train, group=query_train.groupby(query_train).size().values)
test_data = lgb.Dataset(X_test, label=y_test, group=query_test.groupby(query_test).size().values, reference=train_data)

# Set parameters for LambdaMART
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'max_bin': 255,
    'boosting': 'gbdt',
    'verbose': -1
}

# Train the model
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000, early_stopping_rounds=50)

# Predict relevance scores for the test set
y_pred = model.predict(X_test)

# Calculate NDCG score
ndcg = ndcg_score([y_test], [y_pred], k=5)
print(f'NDCG Score: {ndcg}')

# Example query
query_id = 1
query_data = X_test[query_test == query_id]

# Predict relevance scores for the query
query_pred = model.predict(query_data)

# Rank products based on predicted relevance scores
ranked_products = query_data.assign(predicted_relevance=query_pred).sort_values(by='predicted_relevance', ascending=False)
print(ranked_products)
