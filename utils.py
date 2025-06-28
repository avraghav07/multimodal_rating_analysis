import pandas as pd
import numpy as np
import nltk
import string
from sklearn.preprocessing import StandardScaler

# For converting rating to numerical
rating_order = [
        'AAA', 'AA+', 'AA', 'A+', 'A',
        'BBB+', 'BBB', 'BB+', 'BB',
        'B+', 'B'
    ]
rating_map = {rating: idx for idx, rating in enumerate(rating_order)}

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Data preprocessing util function
def dataPreprocessor(df: pd.DataFrame) -> pd.DataFrame:

    df['rating_numerical'] = df['Rating'].map(rating_map)
    unmapped_ratings = df[df['rating_numerical'].isna()]['Rating'].unique()

    if len(unmapped_ratings) > 0:
        print(f"Warning: These ratings are not in the mapping: {unmapped_ratings}")

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