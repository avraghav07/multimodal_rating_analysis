# This script focuses on NLP Feature Engineering.
# The string values given in the data are first cleaned by removing stopwords, punctuation, and by using lemmatization.
# Then ,GloVe is used to find word embeddings, and also find the polarity and subjectivity of each word using sentiment analysis.
# A few other nlp features are engineered using positive and negative keywords, and general word and character count.

import pandas as pd
import numpy as np
import nltk
import gensim.downloader as api

from utils import preprocess_text, create_word_embeddings, get_sentiment, count_keywords
from consts import positive_words, negative_words
from sklearn.preprocessing import StandardScaler
from warnings import simplefilter

# To remove annoying performance warning that doesn't really affect performance as much
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Word embeddings using GloVe/Word2Vec
try:
    # Load pre-trained word vectors (smaller model for speed)
    print("Loading GloVe gigaword 100 model\n")
    word_vectors = api.load('glove-wiki-gigaword-100') 
except:
    # Load different model if error loading
    print("Error loading GloVe, trying Word2Vec\n")
    word_vectors = api.load('word2vec-google-news-300')

# Load the output of data_preprocess
try:
    print("Loading processed data:")
    df = pd.read_csv('processed_data.csv')
    print(f"Data shape: {df.shape}")
except:
    print("Error loading processed_data.csv. Make sure you run data_preprocess.py first.")

# Preprocess text using the preprocess_text function in utils
print("\n" + "="*50)
print("TEXT PREPROCESSING")
print("="*50)
df['cleaned_text'] = df['string_values'].apply(preprocess_text)
print("Sample Cleaned Text:")
print(df['cleaned_text'].head(), "\n")

print("\nText preprocessing examples:")
for i in range(3):
    print(f"\nOriginal: {df['string_values'].iloc[i][:100]}")
    print(f"Tokens: {df['cleaned_text'].iloc[i]}")

# Create word embeddings using the create_word_embeddings function in utils and scale
print("\n" + "="*50)
print("FEATURE EXTRACTION")
print("="*50)
print("Creating word embeddings\n")
embeddings = np.array([create_word_embeddings(tokens, word_vectors) for tokens in df['cleaned_text']])
scaler = StandardScaler()
embedding_scaled = scaler.fit_transform(embeddings)
embedding_df = pd.DataFrame(embedding_scaled, columns=[f'embed_{i}' for i in range(embeddings.shape[1])])

# Calculate embedding statistics
print("Embedding statistics:")
print(f"Non-zero embeddings: {np.sum(np.any(embeddings != 0, axis=1))}/{len(embeddings)}")
print(f"Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1))}")

# Getting average polarity and subjectivity using sentiment analysis
print("\nExtracting sentiment features:")
sentiments = df['string_values'].apply(lambda x: get_sentiment(x))
df['polarity'] = sentiments.apply(lambda x: x[0])
df['subjectivity'] = sentiments.apply(lambda x: x[1])

print(f"Average polarity: {df['polarity'].mean()}")
print(f"Average subjectivity: {df['subjectivity'].mean()}")

# Extracting keyword features and text statistics, then combining these with our word embeddings and our scaled dataframe
print("\nExtracting keyword features:")
df['positive_words'] = df['string_values'].apply(lambda x: count_keywords(x, positive_words))
df['negative_words'] = df['string_values'].apply(lambda x: count_keywords(x, negative_words))
df['sentiment_ratio'] = (df['positive_words'] - df['negative_words']) / (df['positive_words'] + df['negative_words'] + 1)

print("\nExtracting text statistics:")
df['word_count'] = df['cleaned_text'].apply(len)
df['char_count'] = df['string_values'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
df['avg_word_length'] = df['string_values'].apply(
    lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) and str(x).split() else 0
)

print("\n" + "="*50)
print("COMBINING FEATURES")
print("="*50)

nlp_features = ['polarity', 'subjectivity', 'positive_words', 'negative_words', 
                'sentiment_ratio', 'word_count', 'char_count', 'avg_word_length']
final_df = pd.concat([df[nlp_features], 
                df[['Rating', 'rating_numerical', "rating_type_Fitch", "rating_type_Moody's", 'rating_type_S&P']],
                df[[col for col in df.columns if col.startswith('scaled_')]],
                embedding_df], axis=1)
print(f"\nFinal feature set shape: {df.shape}")

# Feature summary
print("\nFeature Summary:")
print(f"- Numeric features (scaled): {len([col for col in final_df.columns if col.startswith('scaled_')])}")
print(f"- NLP features: {len(nlp_features)}")
print(f"- Word embedding features: {embedding_df.shape[1]}")
print(f"- Total: {final_df.shape[1] - 3}")

# Save combined features
df.to_csv('combined_features.csv', index=False)
print("\nCombined features saved to 'combined_features.csv'")

