# This file takes the given dataset and inspects it first. 
# Then, it focuses on showing basic descriptive statistics and distribution of our independent variable Rating.
# Then, preprocessing is done - Normalization of the numeircal data, one-hot encoding of Rating_type and Rating is converted into a numerical variable.


import pandas as pd
from sklearn.preprocessing import StandardScaler
from consts import rating_map
from warnings import simplefilter

# To remove annoying performance warning that doesn't really affect performance as much
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

print("Loading data:\n")
# LOAD AND INSPECT DATA
df = pd.read_excel('Artificial_Data.xlsx')
print(f"First 5 rows: {df.head()}")
print(f"\nData Info:{df.info()}" )
print(f"\nData shape: {df.shape}")

# Basic descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

print("\nRating distribution:")
print(df['Rating'].value_counts().sort_index())

print("\nRating Type distribution:")
print(df['RATING_TYPE'].value_counts())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)
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

# Using the rating_map we defined earlier to convert rating into a numerical variable
df['rating_numerical'] = df['Rating'].map(rating_map)

# Check if any ratings are unmapped and ignore them. Not needed in this case but it's a nice-to-have.
unmapped_ratings = df[df['rating_numerical'].isna()]['Rating'].unique()
if len(unmapped_ratings) > 0:
    print(f"Warning: These ratings are not in the mapping: {unmapped_ratings}")

# Convert rating_type to categorical variable. Use get_dummies to do one-hot encoding
df = pd.get_dummies(df, columns=['RATING_TYPE'], prefix='rating_type', drop_first=False)

# Normalize the numerical features
numeric_features = df.select_dtypes(include=['float64', 'int64']).drop('rating_numerical', axis=1, errors='ignore')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)
scaled_df = pd.DataFrame(scaled_features, columns=[f'scaled_{feature}' for feature in numeric_features])
processed_df = pd.concat([df[["rating_type_Fitch", "rating_type_Moody's", 'rating_type_S&P', 'rating_numerical', 'string_values', 'Rating']], 
scaled_df
], axis=1)


# Finding Pearson's correlation for numerical features with numerical rating
print("\nTop 10 features most correlated with numerical rating:")
numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
ratingCorr = processed_df[numeric_cols].corr()['rating_numerical'].sort_values(ascending=False)
print(ratingCorr[1:11])

# Check final dataframe after transformations and save file
print(f"Final look of the dataframe after transformations: {processed_df.head()}")
print("\nSaving processed data to processed_data.csv")
processed_df.to_csv('processed_data.csv', index=False)
