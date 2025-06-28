import pandas as pd
import numpy as np
import nltk
import string
import gensim.downloader as api

from dataPreprocess import dataPreprocessor
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler


nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Word embeddings using GloVe/Word2Vec
try:
    # Load pre-trained word vectors (smaller model for speed)
    print("Loading GloVe gigaword 100 model")
    word_vectors = api.load('glove-wiki-gigaword-100') 
except:
    # Load different model if error loading
    print("Error loading GloVe, trying Word2Vec")
    word_vectors = api.load('word2vec-google-news-300')

# Text Preprocessing using stopword removal, punctuation removal, lowercasing, and lemmatization
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


# Using our data preprocessor to refine the df
unrefinedDf = pd.read_excel('Artificial_Data.xlsx')
df = dataPreprocessor(unrefinedDf)

df['cleaned_text'] = df['string_values'].apply(preprocessText)
print("Sample Cleaned Text:")
print(df['cleaned_text'].head())

# Create word embeddings and scale
print("\nCreating document embeddings:")
embeddings = np.array([createWordEmbeddings(tokens, word_vectors) for tokens in df['cleaned_text']])
scaler = StandardScaler()
embedding_scaled = scaler.fit_transform(embeddings)
embedding_df = pd.DataFrame(embedding_scaled, columns=[f'embed_{i}' for i in range(embeddings.shape[1])])

# Calculate embedding statistics
print("\nEmbedding statistics:")
print(f" Non-zero embeddings: {np.sum(np.any(embeddings != 0, axis=1))}/{len(embeddings)}")
print(f" Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.3f}")

# Sentiment analysis and inserting sentiment score into df
df['sentiment'] = df['string_values'].apply(lambda x: TextBlob(x).sentiment.polarity)
print("Sample Sentiment Scores:")
print(df[['string_values', 'sentiment']].head())

