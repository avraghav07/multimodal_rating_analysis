import pandas as pd
import warnings

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from utils import trainAndEval

# Ignoring warnings for better readability (they do not affect performance)
warnings.filterwarnings('ignore')

print("Loading combined features:\n")
df = pd.read_csv('combined_features.csv')
print(f"Data shape: {df.shape}")

# Preparing structured (no text) and text features (embeddings + nlp features)
print("\n" + "="*50)
print("PREPARING FEATURE SETS")
print("="*50)

structured_features = [col for col in df.columns if col.startswith('scaled_')]
nlp_features = ['polarity', 'subjectivity', 'positive_words', 'negative_words', 
                'sentiment_ratio', 'word_count', 'char_count', 'avg_word_length']
embedding_features = [col for col in df.columns if col.startswith('embed_')]
text_features = nlp_features + embedding_features

print(f"Structured features: {len(structured_features)}")
print(f"Text features: {len(text_features)} (NLP: {len(nlp_features)}, Embeddings: {len(embedding_features)})")

# Prepare feature sets as per requirements, predict against rating (classification)
X_structured = df[structured_features]
X_text = df[text_features]
X_combined = df[structured_features + text_features]
y = df['Rating']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nTarget classes: {le.classes_}")

# Defining test train splits
X_train_s, X_test_s, y_train, y_test = train_test_split(X_structured, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train_t, X_test_t, _, _ = train_test_split(X_text, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train_c, X_test_c, _, _ = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\nTrain size: {len(X_train_s)}, Test size: {len(X_test_s)}")

# Training the model using trainAndEval
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

results_structured, best_model_s, best_name_s, pred_s = trainAndEval(
    X_train_s, X_test_s, y_train, y_test, "Structured"
)

results_text, best_model_t, best_name_t, pred_t = trainAndEval(
    X_train_t, X_test_t, y_train, y_test, "Text"
)

results_combined, best_model_c, best_name_c, pred_c = trainAndEval(
    X_train_c, X_test_c, y_train, y_test, "Combined"
)

# Comparing the results
print("\n" + "="*50)
print("RESULTS COMPARISON")
print("="*50)

comparison_data = []
for feature_type, results in [('Structured', results_structured), 
                             ('Text', results_text), 
                             ('Combined', results_combined)]:
    for model_name, metrics in results.items():
        comparison_data.append({
            'Feature Type': feature_type,
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1 Score': metrics['f1'],
            'CV Mean': metrics['cv_mean']
        })

comparison_df = pd.DataFrame(comparison_data)

# Best performing models
print("\nBest model for each feature type:")
for feature_type in ['Structured', 'Text', 'Combined']:
    best = comparison_df[comparison_df['Feature Type'] == feature_type].sort_values('Accuracy', ascending=False).iloc[0]
    print(f"{feature_type}: {best['Model']} (Accuracy: {best['Accuracy']})")

# Check and save feature importance if model is Random Forest
if isinstance(best_model_c, RandomForestClassifier):
    feature_importance = best_model_c.feature_importances_
    feature_names = structured_features + text_features
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to 'feature_importance.csv'")
    
    # Print top features
    print("\nTop 5 structured features:")
    struct_importance = importance_df[importance_df['feature'].str.startswith('scaled_')]
    print(struct_importance[['feature', 'importance']].head().to_string(index=False))
    
    print("\nTop 5 text features:")
    text_importance = importance_df[~importance_df['feature'].str.startswith('scaled_')]
    print(text_importance[['feature', 'importance']].head().to_string(index=False))


# Final report
print("\n" + "="*50)
print("FINAL REPORT")
print("="*50)

# Calculate the improvement from using a multimodal model
struct_acc = comparison_df[comparison_df['Feature Type'] == 'Structured']['Accuracy'].max()
text_acc = comparison_df[comparison_df['Feature Type'] == 'Text']['Accuracy'].max()
combined_acc = comparison_df[comparison_df['Feature Type'] == 'Combined']['Accuracy'].max()

improvement = ((combined_acc - struct_acc) / struct_acc) * 100

print(f"\nBest Accuracy Scores:")
print(f"  Structured-only: {struct_acc}")
print(f"  Text-only: {text_acc}")
print(f"  Combined: {combined_acc}")
print(f"\nImprovement of combined over structured: {improvement}%")

# Classification report for best combined model
print(f"\nDetailed Classification Report (Combined - {best_name_c}):")
print(classification_report(y_test, pred_c, target_names=le.classes_))

# Saving model performance 
comparison_df.to_csv('model_results.csv', index=False)
print("\nResults saved to 'model_results.csv'")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)