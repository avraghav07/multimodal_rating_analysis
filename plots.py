import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("\n2. Creating sentiment analysis plots...")
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
    print("Created file: sentiment_analysis.png")
else:
    print("combined_features.csv not found - run nlpFeatures.py first")

# ========== 3. MODEL COMPARISON ==========
print("\n3. Creating model comparison plot...")
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
    print("Created file: model_comparison.png")
else:
    print("model_results.csv not found - run predictiveModeling.py first")