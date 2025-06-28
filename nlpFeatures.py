import pandas as pd
import numpy as np
import nltk
import gensim.downloader as api

from utils import dataPreprocessor, preprocessText, createWordEmbeddings, getSentiment, countKeywords
from consts import positive_words, negative_words
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

# Using our data preprocessor to refine the df
unrefinedDf = pd.read_excel('Artificial_Data.xlsx')
df = dataPreprocessor(unrefinedDf)

# Preprocess text using the preprocessText function in utils
df['cleaned_text'] = df['string_values'].apply(preprocessText)
print("Sample Cleaned Text:")
print(df['cleaned_text'].head())

# Create word embeddings using the createWordEmbeddings function in utils and scale
print("\nCreating document embeddings:")
embeddings = np.array([createWordEmbeddings(tokens, word_vectors) for tokens in df['cleaned_text']])
scaler = StandardScaler()
embedding_scaled = scaler.fit_transform(embeddings)
embedding_df = pd.DataFrame(embedding_scaled, columns=[f'embed_{i}' for i in range(embeddings.shape[1])])

# Calculate embedding statistics
print("\nEmbedding statistics:")
print(f" Non-zero embeddings: {np.sum(np.any(embeddings != 0, axis=1))}/{len(embeddings)}")
print(f" Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1))}")

# Getting average polarity and subjectivity using sentiment analysis
sentiments = df['string_values'].apply(lambda x: getSentiment(x))
df['polarity'] = sentiments.apply(lambda x: x[0])
df['subjectivity'] = sentiments.apply(lambda x: x[1])

print(f"Average polarity: {df['polarity'].mean()}")
print(f"Average subjectivity: {df['subjectivity'].mean()}")

# Simple keyword features
df['positive_words'] = df['string_values'].apply(lambda x: countKeywords(x, positive_words))
df['negative_words'] = df['string_values'].apply(lambda x: countKeywords(x, negative_words))
df['sentiment_ratio'] = (df['positive_words'] - df['negative_words']) / (df['positive_words'] + df['negative_words'] + 1)

print(f"\nFinal feature set shape: {df.shape}")

# Save combined features
df.to_csv('combined_features.csv', index=False)
print("\nCombined features saved to 'combined_features.csv'")

