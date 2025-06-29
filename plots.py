# This script outputs all the plots. Make sure you run this at the end.

import pandas as pd
import matplotlib.pyplot as plt
import os

print('Creating rating distribution plot:\n')
if os.path.exists('processed_data.csv'):
    df = pd.read_csv('processed_data.csv')
    
    plt.figure(figsize=(10, 6))
    rating_order = ['AAA', 'AA+', 'AA', 'A+', 'A', 'BBB+', 'BBB', 'BB', 'B']
    existing_ratings = [r for r in rating_order if r in df['Rating'].unique()]
    rating_counts = df['Rating'].value_counts()[existing_ratings]
    
    bars = plt.bar(range(len(rating_counts)), rating_counts.values, color='skyblue', edgecolor='navy')
    plt.xticks(range(len(rating_counts)), rating_counts.index, rotation=45)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Distribution of Ratings')
    
    # Add value labels on bars
    for i, (idx, count) in enumerate(rating_counts.items()):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created file: rating_distribution.png')
else:
    print('processed_data.csv not found. Run data_preprocess.py first')

print('Creating sentiment analysis plots:\n')
if os.path.exists('combined_features.csv'):
    df_features = pd.read_csv('combined_features.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Polarity by rating
    df_features.boxplot(column='polarity', by='Rating', ax=axes[0])
    axes[0].set_title('Sentiment Polarity by Rating')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Polarity') 
    
    # Average sentiment by rating
    sentiment_avg = df_features.groupby('Rating')[['polarity', 'positive_words', 'negative_words']].mean()
    sentiment_avg.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Average Sentiment Metrics by Rating')
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Average Value')
    axes[1].legend(['Polarity', 'Positive Words', 'Negative Words'])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created file: sentiment_analysis.png')
else:
    print('combined_features.csv not found. Run nlp_features.py first')

# ========== 3. MODEL COMPARISON ==========
print('Creating model comparison plot:\n')
if os.path.exists('model_results.csv'):
    comparison_df = pd.read_csv('model_results.csv')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_pivot = comparison_df.pivot(index='Model', columns='Feature Type', values='Accuracy')
    comparison_pivot.plot(kind='bar', ax=ax)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.legend(title='Feature Type')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created file: model_comparison.png')
else:
    print('model_results.csv not found. Run predictive_modeling.py first')