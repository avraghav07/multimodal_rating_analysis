# MULTIMODAL RATING ANALYSIS #

This repo demonstrates a data science solution that leverages both structured and textual data to explain and predict the target variable (Rating in this case).

## STRUCTURE ##

There are 6 Python scripts in this project. Four tackle the given problem statements, and the other two are just for utility, maintained for the sake of better structure and code readability.
I have tried to add helpful comments explaining each file in all the scripts. 

These are all the steps covered:

### DATA LOADING AND PREPROCESSING ###
- The data_preprocess.py script explores the given dataset then promptly moves to pre processing.
- There are a lot of numerical features, which are first normalized.
- The Rating is converted to a numerical variable, and one-hot encoding is used on the Rating Type.
- The final scaled and converted dataset is saved as processed_data.csv.

### NLP FEATURE ENGINEERING ###
- The dataset has string values which are used for evaluation and prediction. For that, necessary NLP features have to be engineered using the text. The nlp_features.py script generates these features.
- The text is first cleaned (removing any punctuations, stopwords, and by performing lemmatization).
- Word embeddings are created using GloVe wiki gigaword. These embeddings are then scaled similar to the numerical features.
- Sentiment analysis is conducted to calculate polarity and subjectivity. Positive and Negative words were manually listed out and used to calculate sentiment ratio for each string value.
- A few other features like 'average word count', 'character count', and 'average word length' were also added for each string value.
- The final text and structural feature concatenated dataset is saved as combined_features.csv

### PREDICTIVE MODELING ###
- The predictive_modeling.py script trains and evaluates the dataset on three classification models: Logistic Regression, Random Forest Classifier, and XGoost.
- Multimodal analysis is conducted to determine predictibility of Rating. Text only, Numerical only (Structured), and the Combined data's performance is measured using the above three algorithms.
- The models' accuracy, F1-score, and cross validation scores are compared.
- The final results are saved as model_results.csv.
- Feature importance is also compared in case of Random Forest classifier, and saved as feature_importance.csv.

### VISUALIZATION ###
- The plots.py script outputs the rating distribution, model comparison, and sentiment analysis plots.

## HOW TO RUN ##

1. Install venv using command 'python -m venv env'.
2. Enter the venv, then install requirements using 'pip install -r requirements.txt'.
3. Run the data_preprocess.py file to have a look at the data preprocessing steps taken.
4. Run the nlp_features.py file to take a look at text preprocessing, word embeddings and the steps taken to get there, along with the final feature set.
5. Run the predictive_modeling.py file to get the evaluation resultse, along with the feature importance.
6. Run the plots.py file to get plots.
