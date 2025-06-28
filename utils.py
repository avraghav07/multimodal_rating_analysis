import pandas as pd
import numpy as np
import nltk
import string
from sklearn.preprocessing import StandardScaler
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
    
    df = df.drop('Rating', axis=1)

    df = pd.get_dummies(df, columns=['RATING_TYPE'], prefix='rating_type', drop_first=False)

    numeric_features = df.select_dtypes(include=['float64', 'int64']).drop('Rating_encoded', axis=1, errors='ignore')

    scaler = StandardScaler()
    df[numeric_features.columns] = scaler.fit_transform(numeric_features)

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