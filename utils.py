import pandas as pd
import numpy as np
import nltk
import string

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from textblob import TextBlob
from consts import rating_map

# For text processing
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Data preprocessing util function
def dataPreprocessor(df):

    df['rating_numerical'] = df['Rating'].map(rating_map)
    unmapped_ratings = df[df['rating_numerical'].isna()]['Rating'].unique()

    if len(unmapped_ratings) > 0:
        print(f"Warning: These ratings are not in the mapping: {unmapped_ratings}")

    df = pd.get_dummies(df, columns=['RATING_TYPE'], prefix='rating_type', drop_first=False)

    numeric_features = df.select_dtypes(include=['float64', 'int64']).drop('rating_numerical', axis=1, errors='ignore')

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    scaled_df = pd.DataFrame(scaled_features, columns=[f'scaled_{feature}' for feature in numeric_features])
    df = pd.concat([df[["rating_type_Fitch", "rating_type_Moody's", 'rating_type_S&P', 'rating_numerical', 'string_values', 'Rating']], 
    scaled_df
    ], axis=1)

    return df

# Text Preprocessing using stopword removal, punctuation removal, lowercasing, and lemmatization.
def preprocessText(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


# Create word embeddings using above embedding model
def createWordEmbeddings(tokens, word_vectors):
    embeddings = []
    for token in tokens:
        if token in word_vectors:
            embeddings.append(word_vectors[token])
    
    if embeddings:
        # Average the word vectors
        return np.mean(embeddings, axis=0)
    else:
        # Return zero vector if no words found
        return np.zeros(word_vectors.vector_size)
    
# Get polarity and subjectivity
def getSentiment(text):
    if not text:
        return 0, 0
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# To count positive and negative keywords in string_value
def countKeywords(text, keywords):
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for word in keywords if word in text_lower)

# Train and Eval function used in predictive modeling
def trainAndEval(X_train, X_test, y_train, y_test, feature_type):
    """Train multiple models and evaluate performance"""
    print(f"\n--- {feature_type.upper()} FEATURES ---")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    results = {}
    best_score = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': accuracy,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_model_name = name
            best_predictions = y_pred
    
    return results, best_model, best_model_name, best_predictions