import pandas as pd
from sklearn.preprocessing import StandardScaler
from consts import rating_map

# LOAD AND INSPECT DATA
df = pd.read_excel('Artificial_Data.xlsx')
print("First 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())

# Basic descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# DATA PREPROCESSING

# There are no missing values in this file as shown by the above print statement. 
# But I am still including a step for handling missing values and cleaning the data

# Drop columns with more than 50% missing values
threshold = 0.5
df = df.loc[:, df.isnull().mean() < threshold]

# Impute missing numeric values with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Check remaining missing values. None in this case
print("Missing Values After Imputation:")
print(df.isnull().sum())

# Finding Pearson's correlation matrix for numerical features

corr_matrix = df[numeric_cols].corr()
print("Pearson's Correlation Matrix:")
print(corr_matrix)

# Using the rating_map we defined earlier to convert rating into a numerical variable
df['rating_numerical'] = df['Rating'].map(rating_map)

# Check if any ratings are unmapped and ignore them. Not needed in this case but it's a nice-to-have.
unmapped_ratings = df[df['rating_numerical'].isna()]['Rating'].unique()
if len(unmapped_ratings) > 0:
    print(f"Warning: These ratings are not in the mapping: {unmapped_ratings}")

# Convert rating_type to categorical variable. Use get_dummies to do one-hot encoding
df = pd.get_dummies(df, columns=['RATING_TYPE'], prefix='rating_type', drop_first=False)

# Normalize the numerical features
numeric_features = df.select_dtypes(include=['float64', 'int64']).drop('Rating_encoded', axis=1, errors='ignore')
scaler = StandardScaler()
df[numeric_features.columns] = scaler.fit_transform(numeric_features)

# Check final dataframe after transformations
print(df.head())
