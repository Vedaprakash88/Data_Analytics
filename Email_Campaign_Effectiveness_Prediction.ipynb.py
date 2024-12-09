


# **Project Name**    - Email Campaign Effectiveness Prediction


##### **Project Type**    - Classification
##### **Contribution**    - Individual
##### **Team Member 1 -**  Dipak Balram Patil

# **Project Summary -**



# **GitHub Link -**



# **Problem Statement**




# **General Guidelines** : -  

# Chart visualization code


# ***Let's Begin !***

## ***1. Know Your Data***

### Import Libraries and defining important functions
# Import Libraries
import warnings
warnings.filterwarnings("ignore")

import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# import pickle
# define a function to calculate the upper and lower outlier boundary
# returns a tuple (upper_outlier_bound, lower_outlier_bound)
def get_outlier_boundaries(dataframe, column):
  per_25 = dataframe[column].quantile(0.25)
  per_75 = dataframe[column].quantile(0.75)
  percentile_25 = np.nanpercentile(dataframe[column],25)
  percentile_75 = np.nanpercentile(dataframe[column],75)
  iqr = per_75 - per_25
  upper_outlier_bound = per_75 + 1.5*iqr
  lower_outlier_bound = per_25 - 1.5*iqr
  return (upper_outlier_bound, lower_outlier_bound)

def get_outlier_count_and_percentage(dataframe,column):
  upper_outlier_bound, lower_outlier_bound = get_outlier_boundaries(dataframe, column)

  count = 0
  for i in dataframe[column]:
    if i > upper_outlier_bound or i < lower_outlier_bound:
      count += 1
    percentage = round(count/dataframe.shape[0]*100,2)

  return(count, percentage)
# define a function to calculate the vif of all features
# returns a dataframe
def calculate_vif_of_all_features(dataframe):
  temp_df = pd.DataFrame()
  temp_df['Feature'] = dataframe.columns
  temp_df['VIF'] = [variance_inflation_factor(dataframe.values, column_index) for column_index in range(dataframe.shape[1])]
  return temp_df
#define function to find number and percentage of missing values in the dataframe
def get_missing_count_and_percenetage(dataframe):
  num = 0
  for column in dataframe.columns:
    count = dataframe[column].isnull().sum()
    percentage = count/dataframe.shape[0]*100
    if percentage > 0:
      num += 1
      print(f"{column} contains \033[4m{count}\033[0m null values. Percentage wise it is \033[1m{round(percentage,2)}% \033[0m")
  if num == 0:
    print("No missing values in the dataframe")
# get count of each unique value from a particular column
# and output its as a dataframe
def get_count_from_column(df, column_label):
  df_grpd = df[column_label].value_counts()
  df_grpd = pd.DataFrame({'index':df_grpd.index, 'count':df_grpd.values})
  return df_grpd

# add value to the top of each bar
def add_value_label(x_list,y_list):
    for i in range(1, len(x_list)+1):
        plt.text(i-1,y_list[i-1],y_list[i-1], ha="center", fontweight='bold')

# plot bar graph from grouped data
def plot_bar_graph_from_column(df, column_label):
  df_grpd = get_count_from_column(df, column_label)

  df_grpd.plot(x='index', y='count', kind='bar', figsize=(3*df_grpd.shape[0], 6))
  add_value_label(df_grpd['index'].tolist(), df_grpd['count'].tolist())
  plt.xlabel(column_label)
  plt.ylabel("Count")
  plt.xticks(rotation='horizontal')
  plt.gca().legend_.remove()
  plt.show()
# define a function to generate a count plot
def generate_count_plot(dataframe):
  plt.rcParams["figure.fisize"] = [8,4]
  plt.rcParams['figure.autolayout'] = True

  fig, ax = plt.subplots()
  dataframe.value_counts().plot(ax = x, kind = 'bar')
  plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

def plot_categorical_influence_with_percentage(dataframe, categorical_column, target_column='Email_Status'):
    if categorical_column not in dataframe.columns or target_column not in dataframe.columns:
        raise ValueError("Invalid column name(s). Please check the column names in the DataFrame.")

    # Create a count plot with percentages
    plt.figure(figsize=(10, 6))
    sns.countplot(x=categorical_column, data=dataframe, palette='viridis')

    # Add percentage labels
    total = len(dataframe)
    for p in plt.gca().patches:
        height = p.get_height()
        plt.text(p.get_x() + p.get_width() / 2., height + 3, f'{height/total:.1%}', ha="center")

    plt.title(f'Influence of {categorical_column} on {target_column}')
    plt.xlabel(categorical_column)
    plt.ylabel('Percentage')
    plt.show()

# Example Usage:
# Assuming you have a DataFrame named 'df' with columns including 'Email_Status' and categorical features like 'Category'
# plot_categorical_influence_with_percentage(df, 'Category')

def generate_horizontal_box_plot(dataframe, x_feature, y_feature=None):
  sns.set_theme(rc={'figure.figsize': (8,4)},style='whitegrid',palette='muted')
  # plt.figure(figsize=(8, 6))
  if y_feature != None:
    ax = sns.boxplot(x=dataframe[x_feature], y=dataframe[y_feature])
  else:
    ax = sns.boxplot(x=dataframe[x_feature], y=None)
  ax.grid(False)

# define a function to plot a stacked bar to show percentage of a feature in a grouped parameter
# so that it can be used in the later stages also

# get the count of unique values in secondary column
# segmented by each unique value in primary column
def get_count_of_unique_values(df, pri_column_label, sec_column_label):
  # finding unique values in secondary column for grouping
  values = sorted([x for x in df[sec_column_label].unique() if str(x) != 'nan'])

  # creating a list of dataframes that gives the value of each unique value in primary column
  # a dataframe is created for each unique value in secondary column
  list_of_counts_df = [df[df[sec_column_label] == value].groupby(pri_column_label)[sec_column_label].count().reset_index(name=f'{value}')
                       for value in values]

  # merge all dataframes into one dataframe
  df_merged = list_of_counts_df[0]
  for i in range(1, len(list_of_counts_df)):
    df_merged = pd.merge(df_merged, list_of_counts_df[i], how='inner', on=pri_column_label)

  return df_merged

# plotting a stacked bar graph to represent the count of unique values in secondary column
# segmented by each unique value in primary column
def stacked_bar_graph_with_count(df, pri_column_label, sec_column_label):
   # computing the percentage of unique values in secondary column contributed by each unique value in primary column
  df_merged = get_count_of_unique_values(df, pri_column_label, sec_column_label)

  ax = df_merged.plot(x=pri_column_label, kind='bar', stacked=True, figsize=(12,6))
  for bar in ax.patches:
    height = bar.get_height()
    width = bar.get_width()
    x = bar.get_x()
    y = bar.get_y()
    label_text = str(height)
    label_x = x + width / 2
    label_y = y + height / 2
    ax.text(label_x, label_y, label_text, ha='center', va='center', fontweight='bold')
  plt.xticks(rotation='horizontal')
  plt.title(f"Percentage of different categories of {sec_column_label} across each {pri_column_label} category")
  plt.show()

# get the % of unique values in secondary column
# segmented by each unique value in primary column
def get_percentage_of_unique_values(df, pri_column_label, sec_column_label):
  # finding unique values in secondary column for grouping
  values = [x for x in df[sec_column_label].unique() if str(x) != 'nan']

  # creating a dataframe that gives the count of each unique value in primary column
  df_merged = get_count_of_unique_values(df, pri_column_label, sec_column_label)

  # computing the percentage of unique values in secondary column contributed by each unique value in primary column
  df_merged['total_count'] = df_merged.sum(axis=1, numeric_only=True)
  for value in values:
    df_merged[f'{value}'] = round(df_merged[f'{value}'] / df_merged['total_count'] * 100)
  df_merged.drop('total_count', axis=1, inplace=True)

  return df_merged

# plotting a stacked bar graph to represent the % of unique values in secondary column
# segmented by each unique value in primary column
def stacked_bar_graph_with_percentage(df, pri_column_label, sec_column_label):
   # computing the percentage of unique values in secondary column contributed by each unique value in primary column
  df_merged = get_percentage_of_unique_values(df, pri_column_label, sec_column_label)

  ax = df_merged.plot(x=pri_column_label, kind='bar', stacked=True, figsize=(12,6))
  for bar in ax.patches:
    height = bar.get_height()
    width = bar.get_width()
    x = bar.get_x()
    y = bar.get_y()
    label_text = str(height) + " %"
    label_x = x + width / 2
    label_y = y + height / 2
    ax.text(label_x, label_y, label_text, ha='center', va='center', fontweight='bold')
  plt.xticks(rotation='horizontal')
  plt.title(f"Percentage of different categories of {sec_column_label} across each {pri_column_label} category")
  plt.show()
# define a function to generate density plot for feature in dataframe
def generate_density_plot(dataframe, feature):
  plt.figure(figsize = (8,4))
  sns.distplot(dataframe[feature])
  plt.show()

# generate density plot for all features in dataframe
def density_plot_of_all_features(dataframe):
  columns = dataframe.describe().columns.tolist()

  columns_num = 3
  rows_num = math.ceil(len(columns)/columns_num)
  fig, axes = plt.subplots(rows_num, columns_num, figsize=(10*columns_num, 8*rows_num))

  row = -1
  column = columns_num - 1
  for feature in columns:
    if column == (columns_num - 1):
      row += 1
      column = 0
    else:
      column += 1
    sns.distplot(ax=axes[row, column], a=dataframe[feature])
    axes[row, column].set_title(f"{feature} Distribution")

# define a function to calculate metrics
# returns a dictionary
def calculate_model_metrics(trained_model, X_train, y_train, X_test, y_test):

  # print best parameter values and score
  print("The best parameters: ")
  for key, value in trained_model.best_params_.items():
    print(f"{key}={value}")
  print(f"\nBest score: {trained_model.best_score_}\n")

  # predict train and test data
  y_train_pred = trained_model.predict(X_train)
  y_test_pred= trained_model.predict(X_test)

  # probabilities of train and test data
  train_prob = trained_model.predict_proba(X_train)
  test_prob = trained_model.predict_proba(X_test)

  metrics_dict = {}

  metrics_dict['Train_Accuracy'] = accuracy_score(y_train, y_train_pred) * 100
  metrics_dict['Test_Accuracy'] = accuracy_score(y_test, y_test_pred) * 100
  metrics_dict['Train_Precision'] = precision_score(y_train, y_train_pred, average='weighted') * 100
  metrics_dict['Test_Precision'] = precision_score(y_test, y_test_pred, average='weighted') * 100
  metrics_dict['Train_Recall'] = recall_score(y_train, y_train_pred, average='weighted') * 100
  metrics_dict['Test_Recall'] = recall_score(y_test, y_test_pred, average='weighted') * 100
  metrics_dict['Train_F1_Score'] = f1_score(y_train, y_train_pred, average='weighted') * 100
  metrics_dict['Test_F1_Score'] = f1_score(y_test, y_test_pred, average='weighted') * 100
  metrics_dict['Train_ROC_AUC'] = roc_auc_score(y_train, train_prob, average='weighted', multi_class='ovr')
  metrics_dict['Test_ROC_AUC'] = roc_auc_score(y_test, test_prob, average='weighted', multi_class='ovr')

  # print the results of model evaluation
  print(f"Training Data")
  print(f"Accuracy  : {round(metrics_dict['Train_Accuracy'], 6)} %")
  print(f"Precision : {round(metrics_dict['Train_Precision'], 6)} %")
  print(f"Recall    : {round(metrics_dict['Train_Recall'], 6)} %")
  print(f"F1 Score  : {round(metrics_dict['Train_F1_Score'], 6)} %")
  print(f"ROC AUC   : {round(metrics_dict['Train_ROC_AUC'], 6)}\n")
  print(f"Testing Data")
  print(f"Accuracy  : {round(metrics_dict['Test_Accuracy'], 6)} %")
  print(f"Precision : {round(metrics_dict['Test_Precision'], 6)} %")
  print(f"Recall    : {round(metrics_dict['Test_Recall'], 6)} %")
  print(f"F1 Score  : {round(metrics_dict['Test_F1_Score'], 6)} %")
  print(f"ROC AUC   : {round(metrics_dict['Test_ROC_AUC'], 6)}\n")

  # plot ROC curve
  fpr = {}
  tpr = {}
  thresh ={}
  no_of_class=3
  for i in range(no_of_class):
      fpr[i], tpr[i], thresh[i] = metrics.roc_curve(y_test, test_prob[:,i], pos_label=i)
  plt.figure(figsize=(12, 6))
  plt.plot(fpr[0], tpr[0], linestyle='--',color='blue', label='Ignored vs Others'+" AUC="+str(round(metrics_dict['Test_ROC_AUC'], 6)))
  plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Read vs Others'+" AUC="+str(round(metrics_dict['Test_ROC_AUC'], 6)))
  plt.plot(fpr[2], tpr[2], linestyle='--',color='orange', label='Acknowledged vs Others'+" AUC="+str(round(metrics_dict['Test_ROC_AUC'], 6)))
  plt.title("ROC curve")
  plt.ylabel("True Positive Rate")
  plt.xlabel("False Positive Rate")
  plt.legend(loc=4)
  plt.show()

  # plot confusion matrix
  cf_matrix = confusion_matrix(y_test_pred, y_test)
  print("\n")
  sns.heatmap(cf_matrix, annot=True, cmap='Blues')

  return metrics_dict
# define a function to plot bar graph with three features
# prints a bar graph
def plot_bar_graph_with_three_features(dataframe, x_feature, y_feature, z_feature, y_label):
  plt.figure(figsize=(26, 6))

  X = dataframe[x_feature].tolist()
  Y = dataframe[y_feature].tolist()
  Z = dataframe[z_feature].tolist()

  X_axis_length = np.arange(len(X))

  plt.bar(X_axis_length - 0.2, Y, 0.4, label = y_feature)
  plt.bar(X_axis_length + 0.2, Z, 0.4, label = z_feature)

  min_limit = 0.9 * min(dataframe[y_feature].min(), dataframe[z_feature].min())
  max_limit = 1.1 * max(dataframe[y_feature].max(), dataframe[z_feature].max())
  plt.ylim(min_limit, max_limit)

  plt.xticks(X_axis_length, X)
  plt.xlabel(x_feature)
  plt.ylabel(y_label)
  plt.legend()
  plt.show()

### Dataset Loading
# Load Dataset
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv("/content/drive/MyDrive/CD/Data science /Projects and all/data_email_campaign.csv")

### Dataset First View
# Dataset First Look
df.head()
df.tail()
df.describe() # Provides statistical summary of data set

### Dataset Rows & Columns count
# Dataset Rows & Columns count
print(f"Number of rows in the dataset: {df.shape[0]}\nNumber of columns in the dataset: {df.shape[1]}")


### Dataset Information
# Dataset Info
df.info()
# Missing Values/Null Values Count
df.isnull().sum()

#### Duplicate Values
# Dataset Duplicate Value Count
duplicated = df[df.duplicated()].shape[0]
print(f"Total number of duplicate rows are: {duplicated}")

### What did you know about your dataset?



## ***2. Understanding Your Variables***
# Dataset Columns
df.columns.to_list()
# Dataset Describe
df.describe(include = 'all') #Summary statistics for Dataframe

### Variables Description



### Check Unique Values for each variable.
# Check Unique Values for each variable.
# unique values in each column of the dataframe
df.apply(lambda x: x.unique())

## 3. ***Data Wrangling***

# Write your code to make your dataset analysis ready.


#### Missing Values/Null Values
# Missing Values/Null Values Count
get_missing_count_and_percenetage(df)
# Visualizing the missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
# Create a bar plot
plt.figure(figsize=(14, 6))
missing_percentage.plot(kind='bar', color='skyblue')
plt.xlabel('Columns')
plt.ylabel('Percentage of Missing Values')
plt.title('Missing Values in DataFrame - Bar Plot')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


df.head()
# create a density plot to show the distribution of values in Total_Past_Communications
generate_density_plot(df, 'Total_Past_Communications')


# generate box plots to show the distribution of Total_Past_Communications in each category of Email_Type and Email_Source_Type
fig, axes = plt.subplots(1,2, figsize=(18,6))
sns.boxplot(ax=axes[0], data=df, x='Total_Past_Communications', y="Email_Type", orient='h')
sns.boxplot(ax =axes[1], data=df, x='Total_Past_Communications', y='Email_Source_Type', orient='h')
# Let's check outliers in Total_Past_Communications
outliers, percentage = get_outlier_count_and_percentage(df,'Total_Past_Communications')
print(f"Count of outliers in Total_Past_Communications is {outliers} \nPercentage wise it is {percentage}%")
generate_horizontal_box_plot(df, 'Total_Past_Communications')


df['Total_Past_Communications'].fillna(value=df['Total_Past_Communications'].mean(), inplace=True)


# create a density plot to show the distribution of values in Total_Links
generate_density_plot(df, 'Total_Links')
outliers, percentage = get_outlier_count_and_percentage(df,'Total_Links')
print(f"Count of outliers in Total_Links is {outliers} \nPercentage wise it is {percentage}%")


# filling the missing values with median
df.Total_Links.fillna(value=df.Total_Links.median(), inplace=True)


generate_density_plot(df,'Total_Images')
outliers, percentage = get_outlier_count_and_percentage(df,'Total_Images')
print(f"Count of outliers in Total_Images is {outliers} \nPercentage wise it is {percentage}%")
df.Total_Images.nunique()   #number of unique value in Total_Images column


# filling the missing values with mode
df.Total_Images.fillna(value=df.Total_Images.median(), inplace=True)


get_missing_count_and_percenetage(pd.DataFrame(df.Customer_Location))
df.Customer_Location.value_counts()



## Conversion of Column Datatype

# Present data type of columns in the dataframe
df.dtypes
# convert Total_Past_Communications, Total_Links, Total_Images
df = df.astype({'Total_Past_Communications':int, 'Total_Links':int, 'Total_Images':int}) # from float to int
# datatypes of columns in the dataframe
df.dtypes

### What all manipulations have you done and insights you found?



## ***4. Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables***

#### Chart - 1


# Chart - 1 visualization code
# plot bar graph to show the count of each category in Email_Status
plot_bar_graph_from_column(df, 'Email_Status')

##### 1. Why did you pick the specific chart?



##### 2. What is/are the insight(s) found from the chart?



#### Chart - 2


# # Chart - 2 visualization code
# # plot the distribution of all features
density_plot_of_all_features(df)

##### 1. Why did you pick the specific chart?



##### 2. What is/are the insight(s) found from the chart?



#### Chart - 3


# Chart - 3 visualization code
# plot stacked bar graphs to show the percentage of e-mails in each feature category for every Email_Status
categorical_features = ['Email_Type', 'Email_Source_Type', 'Customer_Location',
                        'Email_Campaign_Type', 'Time_Email_sent_Category']

for feature in categorical_features:
  stacked_bar_graph_with_percentage(df, 'Email_Status', feature)

##### 1. Why did you pick the specific chart?



##### 2. What is/are the insight(s) found from the chart?



#### Chart - 4


# Chart - 4 visualization code
# plot stacked bar graphs to show the percentage of e-mails in each Email_Status for every campaign type
stacked_bar_graph_with_percentage(df, 'Email_Campaign_Type', 'Email_Status')



##### Why did you pick the specific chart?



#### Chart - 5
# Chart - 5 visualization code
# generate box plots to show the distribution of numerical features in each category of Email_Status
numerical_features = ['Subject_Hotness_Score', 'Total_Past_Communications',
                      'Word_Count', 'Total_Links', 'Total_Images']

fig, axes = plt.subplots(5, 1, figsize=(18, 36))

for row, feature in enumerate(numerical_features):
  sns.boxplot(ax=axes[row], data=df, x=feature, y='Email_Status', orient='h')
  axes[row].set_title(f"Distribution of {feature} across each Email_Status category")

##### 1. Why did you pick the specific chart?



##### 2. What is/are the insight(s) found from the chart?



##### 3. Will the gained insights help creating a positive business impact?




#### Chart - 6


print(f"size of the dataframe is: {df.shape[0]}")
# Chart - 6 visualization code
# generate a correlation matrix using all features in the dataframe
corr_mat = df.corr().abs()

# plot heatmap using correlation matrix
plt.figure(figsize=(12,12))
sns.heatmap(corr_mat, annot = True, fmt='.2f', annot_kws={'size': 10},  vmax=.8, square=True, cmap='Blues');

##### 1. Why did you pick the specific chart?



##### 2. What is/are the insight(s) found from the chart?



## ***6. Feature Engineering & Data Pre-processing***
# exploring the head of the dataframe
print(f'Shape of dataframe is: {df.shape}')
df.head()

### 1. Handling Missing Values
# Handling Missing Values & Missing Value Imputation
df.isna().sum()


# drop Customer_Location
df.drop('Customer_Location', axis=1, inplace=True)

# Handling Multicollinearity



# calculate VIF of all numerical features
numerical_independant_features = ['Subject_Hotness_Score', 'Total_Past_Communications', 'Word_Count', 'Total_Links', 'Total_Images']
calculate_vif_of_all_features(df[[column for column in df.describe().columns if column in numerical_independant_features]])


# create a new feature by combining Total_Links & Total_Images
df['Total_Links_Images'] = df['Total_Links'] + df['Total_Images']
df.drop(['Total_Links', 'Total_Images'], axis=1, inplace=True)
numerical_independant_features.remove('Total_Links')
numerical_independant_features.remove('Total_Images')
numerical_independant_features.append('Total_Links_Images')

# calculate VIF of all numerical features
calculate_vif_of_all_features(df[[column for column in df.describe().columns if column in numerical_independant_features]])



### 2. Handling Outliers

## Subject Hotness Score


# Handling Outliers & Outlier treatments
# generate a box plot for Subject_Hotness_Score
generate_horizontal_box_plot(df, 'Subject_Hotness_Score')

# count and percentage of outliers in Subject_Hotness_Score
count, perc = get_outlier_count_and_percentage(df, 'Subject_Hotness_Score')
print(f"Outliers in Subject_Hotness_Score : {count} ({perc}%)")


# remove outliers
upper_boundary, lower_boundary = get_outlier_boundaries(df, 'Subject_Hotness_Score')
df = df[(df['Subject_Hotness_Score'] > lower_boundary) & (df['Subject_Hotness_Score'] < upper_boundary)]

## Total Past Communications


# generate a box plot for Total_Past_Communications
generate_horizontal_box_plot(df, 'Total_Past_Communications')

# count and percentage of outliers in Total_Past_Communications
count, perc = get_outlier_count_and_percentage(df, 'Total_Past_Communications')
print(f"Outliers in Total_Past_Communications : {count} ({perc}%)")


# remove outliers
upper_boundary, lower_boundary = get_outlier_boundaries(df, 'Total_Past_Communications')
df = df[(df['Total_Past_Communications'] > lower_boundary) & (df['Total_Past_Communications'] < upper_boundary)]

##Word Count


# generate a box plot for Word_Count
generate_horizontal_box_plot(df, 'Word_Count')

# count and percentage of outliers in Word_Count
count, perc = get_outlier_count_and_percentage(df, 'Word_Count')
print(f"Outliers in Word_Count : {count} ({perc}%)")



##Total Links & Images


df.head()  #checking data view
# generate a box plot for Total_Links_Images
generate_horizontal_box_plot(df, 'Total_Links_Images')

# count and percentage of outliers in Total_Links_Images
count, perc = get_outlier_count_and_percentage(df, 'Total_Links_Images')
print(f"Outliers in Total_Links_Images : {count} ({perc}%)")


# generate a box plot for Total_Links_Images in majority class
generate_horizontal_box_plot(df[df['Email_Status'] == 0], 'Total_Links_Images')

# count and percentage of outliers in Total_Links_Images in majority class
count, perc = get_outlier_count_and_percentage(df[df['Email_Status'] == 0], 'Total_Links_Images')
print(f"Outliers in Total_Links_Images : {count} ({perc}%)")
# generate a box plot for Total_Links_Images in minority classes
generate_horizontal_box_plot(df[(df['Email_Status'] == 1) | (df['Email_Status'] == 2)], 'Total_Links_Images')

# count and percentage of outliers in Total_Links_Images in minority classes
count, perc = get_outlier_count_and_percentage(df[(df['Email_Status'] == 1) | (df['Email_Status'] == 2)], 'Total_Links_Images')
print(f"Outliers in Total_Links_Images : {count} ({perc}%)")



### 3. Categorical Encoding
#identify categorical features
df.apply(lambda x: x.unique())


# Encode your categorical columns
categorical_features = ['Email_Type', 'Email_Source_Type', 'Email_Campaign_Type']

ohe = OneHotEncoder(sparse=False, dtype=int)
ohe.fit(df[categorical_features])
encoded_features = list(ohe.get_feature_names_out(categorical_features))
df[encoded_features] = ohe.transform(df[categorical_features])
df.drop(categorical_features, axis=1, inplace=True)


# find the correlation between encoded features & Sales
corr_mat = df.loc[:, ['Email_Campaign_Type_1', 'Email_Campaign_Type_2',
                               'Email_Campaign_Type_3', 'Email_Status']].corr().abs()
f, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(corr_mat, annot = True, fmt='.3f', annot_kws={'size': 10},  vmax=.8, square=True, cmap="Blues");


# drop Email_Campaign_Type_1
df.drop(['Email_Type_2', 'Email_Source_Type_2', 'Email_Campaign_Type_1'], axis=1, inplace=True)

### 4. Feature Manipulation & Selection

#### Feature Selection


# Select your features wisely to avoid overfitting
df.drop('Email_ID', axis=1, inplace=True)


# drop Time_Email_sent_Category
df.drop('Time_Email_sent_Category', axis=1, inplace=True)
# now again exploring the head of the dataframe
print(f'Shape of dataframe is: {df.shape}')
df.head()

### 5. Data Transformation

#### Do you think that your data needs to be transformed? If yes, which transformation have you used. Explain Why?


# Transform Your data
# numerical features
numerical_features = ['Total_Past_Communications', 'Word_Count', 'Total_Links_Images','Subject_Hotness_Score']

# generate density plot for numerical features
for feature in numerical_features:
  plt.figure(figsize=(9, 6))
  sns.distplot(df[feature]).set(title=f'{feature} Distribution')
  plt.show()


df['Total_Links_Images'] = np.log(df['Total_Links_Images'])
df['Subject_Hotness_Score'] = np.sqrt(df['Subject_Hotness_Score'])

for feature in ['Subject_Hotness_Score', 'Total_Links_Images']:
  plt.figure(figsize=(9, 6))
  sns.distplot(df[feature]).set(title=f'{feature} Distribution')



### 6. Data Splitting


# independant features
X = df.drop('Email_Status', axis=1)

# dependant feature
y = df['Email_Status']
# Split your data to train and test. Choose Splitting ratio wisely.
# split the datasets to training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 0)
print(X_train.shape)
print(X_test.shape)

##### What data splitting ratio have you used and why?



### 7. Data Scaling


# Scaling your data
# standardization of independant training and testing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##### Which method have you used to scale you data and why?



### 8. Handling Imbalanced Dataset

##### Do you think the dataset is imbalanced? Explain Why.


# use undersampling to eliminate data imbalance
rus = RandomUnderSampler(replacement=True)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# use oversampling to eliminate data imbalance
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# generating a count plot to check data imbalance
plt.rcParams['figure.figsize'] = [20, 4]
fig, axes = plt.subplots(1, 3)

# Plot count plots
sns.countplot(data=df, x=y_train, ax=axes[0])
axes[0].set_title('Count Plot - y_train')

sns.countplot(data=df, x=y_train_rus, ax=axes[1])
axes[1].set_title('Count Plot - y_train_rus')

sns.countplot(data=df, x=y_train_smote, ax=axes[2])
axes[2].set_title('Count Plot - y_train_smote')



##### What technique did you use to handle the imbalance dataset and why? (If needed to be balanced)



# **Create a table to store metrics related to models**
# create a dataframe to store metrics related to models
metrics_table = pd.DataFrame(columns=['Model', 'Sampling', 'Train_Accuracy', 'Test_Accuracy',
       'Train_Precision', 'Test_Precision', 'Train_Recall', 'Test_Recall',
       'Train_F1Score', 'Test_F1Score', 'Train_ROC_AUC', 'Test_ROC_AUC'])

## ***7. ML Model Implementation***

## Logistic Regression with Hyperparameter Tuning


### ML Model - 1
# ML Model - 1 Implementation
# initialize hyperparameters for logistic regression
log_reg = LogisticRegression(multi_class='multinomial', class_weight='balanced')
parameters = {'solver':['lbfgs', 'newton-cg', 'saga'],
              'C':[0.01, 0.1, 1],
              'max_iter':[50, 80, 100]}
# train data with logistic regression on random undersampling
log_reg_rus = RandomizedSearchCV(log_reg, parameters, cv=5, n_iter=10)
log_reg_rus.fit(X_train_rus, y_train_rus)

# model evaluation
model_evaluation = calculate_model_metrics(log_reg_rus, X_train_rus, y_train_rus, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['Logistic Regression', 'RandomUnderSampling',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]
# train data with logistic regression on SMOTE
log_reg_smote = RandomizedSearchCV(log_reg, parameters, cv=5, n_iter=10)
log_reg_smote.fit(X_train_smote, y_train_smote)

# model evaluation
model_evaluation = calculate_model_metrics(log_reg_smote, X_train_smote, y_train_smote, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['Logistic Regression', 'SMOTE',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]


## Decision Tree with Hyperparameter Tuning

### ML Model - 2
# initialize hyperparameters for decision tree classifier
decision_tree = DecisionTreeClassifier()
parameters = {'max_depth': [5, 10, None],
              'min_samples_leaf': [1, 2, 5],
              'min_samples_split': [2, 5, 10],
              'max_leaf_nodes': [5, 20, 100],
              'max_features': ['auto', 'sqrt', 'log2']}
# Visualizing evaluation Metric Score chart
#train data with decision tree on random undersampling
dt_rus = RandomizedSearchCV(decision_tree, parameters, cv=5, n_iter=10)
dt_rus.fit(X_train_rus, y_train_rus)

# model evaluation
model_evaluation = calculate_model_metrics(dt_rus, X_train_rus, y_train_rus, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['Decision Tree', 'RandomUnderSampling',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]
# train data with decision tree on SMOTE
dt_smote = RandomizedSearchCV(decision_tree, parameters, cv=5, n_iter=10)
dt_smote.fit(X_train_smote, y_train_smote)

# model evaluation
model_evaluation = calculate_model_metrics(dt_smote, X_train_smote, y_train_smote, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['Decision Tree', 'SMOTE',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]

## Random Forest with Hyperparameter Tuning

### ML Model - 3
# ML Model - 3 Implementation
# initialize hyperparameters for random forest classifier
random_forest = RandomForestClassifier()
parameters = {'max_depth': [5, 10, None],
              'min_samples_leaf': [1, 2, 5],
              'min_samples_split': [2, 5, 10],
              'max_leaf_nodes': [5, 20, 100],
              'max_features': ['auto', 'sqrt', 'log2']}
# train data with random forest on random undersampling
rf_rus = RandomizedSearchCV(random_forest, parameters, cv=5, n_iter=10)
rf_rus.fit(X_train_rus, y_train_rus)

# model evaluation
model_evaluation = calculate_model_metrics(rf_rus, X_train_rus, y_train_rus, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['Random Forest', 'RandomUnderSampling',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]

# train data with random forest on SMOTE
rf_smote = RandomizedSearchCV(random_forest, parameters, cv=5, n_iter=10)
rf_smote.fit(X_train_smote, y_train_smote)

# model evaluation
model_evaluation = calculate_model_metrics(rf_smote, X_train_smote, y_train_smote, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['Random Forest', 'SMOTE',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]

##XGBoost with Hyperparameter Tuning
# initialize hyperparameters for XGBoost classifier
xgboost = xgb.XGBClassifier(objective='multi:softmax', verbosity=0)
parameters = {'max_depth': [2, 5, 10],
              'learning_rate': [0.05, 0.1, 0.2],
              'min_child_weight': [1, 2, 5],
              'gamma': [0, 0.1, 0.3],
              'colsample_bytree': [0.3, 0.5, 0.7]}
# train data with XGBoost on random undersampling
xgb_rus = RandomizedSearchCV(xgboost, parameters, cv=5, n_iter=10)
xgb_rus.fit(X_train_rus, y_train_rus)

# model evaluation
model_evaluation = calculate_model_metrics(xgb_rus, X_train_rus, y_train_rus, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['XGBoost', 'RandomUnderSampling',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]
# train data with XGBoost on SMOTE
xgb_smote = RandomizedSearchCV(xgboost, parameters, cv=5, n_iter=10)
xgb_smote.fit(X_train_smote, y_train_smote)

# model evaluation
model_evaluation = calculate_model_metrics(xgb_smote, X_train_smote, y_train_smote, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['XGBoost', 'SMOTE',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]

##KNN with Hyperparameter Tuning

# initialize hyperparameters for knn classifier
knn = KNeighborsClassifier()
parameters = {'n_neighbors':[5, 10, 15],
              'weights':['uniform','distance'],
              'metric':['minkowski','euclidean','manhattan'],
              'leaf_size':[10, 20, 30]}
# train data with knn on random undersampling
knn_rus = RandomizedSearchCV(knn, parameters, cv=5, n_iter=10)
knn_rus.fit(X_train_rus, y_train_rus)

# model evaluation
model_evaluation = calculate_model_metrics(knn_rus, X_train_rus, y_train_rus, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['KNN', 'RandomUnderSampling',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]
# train data with knn on SMOTE
knn_smote = RandomizedSearchCV(knn, parameters, cv=5, n_iter=10)
knn_smote.fit(X_train_smote, y_train_smote)

# model evaluation
model_evaluation = calculate_model_metrics(knn_smote, X_train_smote, y_train_smote, X_test, y_test)

# add metrics to metrics table
metrics_table.loc[len(metrics_table.index)] = ['KNN', 'SMOTE',
                                                model_evaluation['Train_Accuracy'], model_evaluation['Train_Precision'],
                                                model_evaluation['Train_Recall'], model_evaluation['Train_F1_Score'],
                                                model_evaluation['Train_ROC_AUC'], model_evaluation['Test_Accuracy'],
                                                model_evaluation['Test_Precision'], model_evaluation['Test_Recall'],
                                                model_evaluation['Test_F1_Score'], model_evaluation['Test_ROC_AUC']]

#Model Comparison
# print metrics table
metrics_table


# plot bar graph to show F1 scores
plot_bar_graph_with_three_features(metrics_table, 'Model', 'Train_F1Score', 'Test_F1Score', 'F1 Score')



###Explain the model which you have used and the feature importance using any model explainability tool?



# **Conclusion**

### Conclusions from EDA:
### Recommendations:
### Conclusions from Modeling:


### ***Hurrah! You have successfully completed your Machine Learning Capstone Project !!!***
