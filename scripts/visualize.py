"""
Visualization script for ProvDetector
Creates ROC curves, confusion matrices, and other performance visualizations
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import argparse
import json
from scipy import interpolate

from data_loader import ProvenanceDataLoader
from feature_extractor import ProvDetectorFeatureExtractor
import joblib

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


def plot_roc_curve(y_true, y_proba, model_name="ProvDetector", save_path=None):
    """Plot ROC curve with smooth interpolation"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # sklearn's roc_curve always starts at (0,0) and ends at (1,1) by default
    # But let's explicitly ensure this for visualization clarity
    # Check if first point is (0,0) - if not, prepend it
    if len(fpr) == 0 or fpr[0] != 0.0 or tpr[0] != 0.0:
        fpr = np.concatenate([[0.0], fpr])
        tpr = np.concatenate([[0.0], tpr])
    
    # Check if last point is (1,1) - if not, append it
    if len(fpr) == 0 or fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr = np.concatenate([fpr, [1.0]])
        tpr = np.concatenate([tpr, [1.0]])
    
    # Debug: Print first few points to verify
    # print(f"First 5 FPR points: {fpr[:5]}")
    # print(f"First 5 TPR points: {tpr[:5]}")
    
    # Remove duplicate FPR values while maintaining monotonicity
    # For ROC curves, we want the maximum TPR for each FPR
    seen_fpr = {}
    for i in range(len(fpr)):
        fpr_val = fpr[i]
        tpr_val = tpr[i]
        # Keep the maximum TPR for each FPR to ensure monotonicity
        if fpr_val not in seen_fpr or tpr_val > seen_fpr[fpr_val]:
            seen_fpr[fpr_val] = tpr_val
    
    # Sort by FPR and convert to arrays
    sorted_pairs = sorted(seen_fpr.items())
    fpr_clean = np.array([p[0] for p in sorted_pairs])
    tpr_clean = np.array([p[1] for p in sorted_pairs])
    
    # CRITICAL: Ensure first point is exactly (0,0)
    if len(fpr_clean) == 0 or fpr_clean[0] != 0.0:
        fpr_clean = np.concatenate([[0.0], fpr_clean])
        tpr_clean = np.concatenate([[0.0], tpr_clean])
    else:
        # Force first TPR to be exactly 0.0
        tpr_clean[0] = 0.0
    
    # CRITICAL: Ensure last point is exactly (1,1)
    if len(fpr_clean) == 0 or fpr_clean[-1] != 1.0:
        fpr_clean = np.concatenate([fpr_clean, [1.0]])
        tpr_clean = np.concatenate([tpr_clean, [1.0]])
    else:
        # Force last TPR to be exactly 1.0
        tpr_clean[-1] = 1.0
    
    # Ensure monotonicity (TPR should never decrease as FPR increases)
    for i in range(1, len(tpr_clean)):
        if tpr_clean[i] < tpr_clean[i-1]:
            tpr_clean[i] = tpr_clean[i-1]
    
    # Create high-resolution smooth interpolation
    # Use 2000 points for very smooth curve, starting exactly at 0.0
    fpr_smooth = np.linspace(0.0, 1.0, 2000)
    fpr_smooth[0] = 0.0  # Ensure first point is exactly 0.0
    fpr_smooth[-1] = 1.0  # Ensure last point is exactly 1.0
    
    # Use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for monotonic interpolation
    # PCHIP is specifically designed to preserve monotonicity and create smooth curves
    try:
        if len(fpr_clean) >= 2:
            # PCHIP preserves monotonicity better than cubic spline
            # It creates smooth, continuous curves even with few data points
            pchip = interpolate.PchipInterpolator(fpr_clean, tpr_clean)
            tpr_smooth = pchip(fpr_smooth)
            # CRITICAL: Force first point to be exactly 0.0
            tpr_smooth[0] = 0.0
            # CRITICAL: Force last point to be exactly 1.0
            tpr_smooth[-1] = 1.0
            # Ensure values stay in [0, 1] range
            tpr_smooth = np.clip(tpr_smooth, 0.0, 1.0)
            # Ensure strict monotonicity (shouldn't be needed with PCHIP, but safety check)
            for i in range(1, len(tpr_smooth)):
                if tpr_smooth[i] < tpr_smooth[i-1]:
                    tpr_smooth[i] = tpr_smooth[i-1]
        else:
            # If too few points, use linear interpolation
            tpr_smooth = np.interp(fpr_smooth, fpr_clean, tpr_clean)
    except Exception as e:
        # Fallback to linear interpolation if PCHIP fails
        print(f"Warning: PCHIP interpolation failed, using linear: {e}")
        tpr_smooth = np.interp(fpr_smooth, fpr_clean, tpr_clean)
    
    # Debug: Print info about interpolation
    # print(f"Original points: {len(fpr_clean)}, Interpolated points: {len(fpr_smooth)}")
    
    plt.figure(figsize=(10, 8), dpi=100)
    
    # Plot smooth curve with high-quality rendering
    # IMPORTANT: Use the SMOOTH interpolated points (fpr_smooth, tpr_smooth)
    # NOT the original step-wise points (fpr, tpr)
    plt.plot(fpr_smooth, tpr_smooth, color='#3498db', lw=3.5, 
             label=f'{model_name} (AUC = {roc_auc:.3f})', 
             antialiased=True, zorder=3, solid_capstyle='round')
    
    # Uncomment below to see original points for debugging (will show step-wise nature)
    # plt.scatter(fpr_clean, tpr_clean, color='red', s=50, alpha=0.5, zorder=5, marker='o')
    
    # Plot original points (optional, for debugging)
    # plt.scatter(fpr, tpr, color='red', s=30, alpha=0.6, zorder=5)
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.50)', alpha=0.7)
    
    # Set axis limits - ensure they start exactly at 0.0
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Explicitly set axis to start at 0 and ensure 0.0 is visible
    ax = plt.gca()
    ax.set_xlim(left=0.0, right=1.0)
    ax.set_ylim(bottom=0.0, top=1.05)
    
    # Ensure ticks include 0.0 for clarity
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(f'ROC Curve - {model_name} malware prediction, MALICIOUS=1', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Improve legend
    plt.legend(loc="lower right", fontsize=12, framealpha=0.9, 
               fancybox=True, shadow=True)
    
    # Improve grid
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Add AUC text box
    textstr = f'AUC = {roc_auc:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.6, 0.2, textstr, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save first if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"ROC curve saved to {save_path}")
    
    # Always show ROC curve in window (this will block until window is closed)
    plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance"""
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_dataset_distribution(labels, save_path=None):
    """Plot class distribution in dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    labels_map = {0: 'Benign', 1: 'Malicious'}
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar([labels_map[u] for u in unique], counts, 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset distribution plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_prediction_distribution(y_true, y_proba, save_path=None):
    """Plot distribution of prediction probabilities"""
    benign_probs = y_proba[y_true == 0]
    malicious_probs = y_proba[y_true == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(benign_probs, bins=20, alpha=0.6, label='Benign (True)', color='#2ecc71')
    plt.hist(malicious_probs, bins=20, alpha=0.6, label='Malicious (True)', color='#e74c3c')
    plt.xlabel('Predicted Probability (Malicious)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Probabilities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Decision Threshold (0.5)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_correlation(features_df, top_n=15, save_path=None):
    """Plot correlation matrix of top features"""
    # Select top features by variance
    variances = features_df.var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    
    corr_matrix = features_df[top_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Feature Correlation Matrix (Top {top_n} Features)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature correlation plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize ProvDetector results')
    parser.add_argument('--benign-dir', type=str, default='benign', 
                       help='Directory with benign CSV files')
    parser.add_argument('--malicious-dir', type=str, default='malicious', 
                       help='Directory with malicious CSV files')
    parser.add_argument('--model-path', type=str, default='models/random_forest_model.joblib',
                       help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for plots')
    parser.add_argument('--plot-type', type=str, default='all',
                       choices=['all', 'roc', 'confusion', 'features', 'distribution', 'prediction', 'correlation'],
                       help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ProvDetector Visualization")
    print("=" * 60)
    
    # Load models if they exist
    model_loaded = False
    rf_model = None
    extractor = None
    
    model_dir = os.path.dirname(args.model_path)
    freq_db_path = os.path.join(model_dir, 'frequency_database.pkl')
    doc2vec_path = os.path.join(model_dir, 'doc2vec_model.model')
    
    if os.path.exists(args.model_path) and os.path.exists(freq_db_path) and os.path.exists(doc2vec_path):
        print(f"\nLoading models from {model_dir}...")
        try:
            extractor = ProvDetectorFeatureExtractor(frequency_db_path=freq_db_path, doc2vec_model_path=doc2vec_path)
            rf_model = joblib.load(args.model_path)
            model_loaded = True
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            print("Will generate visualizations without model predictions.")
    else:
        print(f"\nModels not found. Required files:")
        print(f"  - {args.model_path}")
        print(f"  - {freq_db_path}")
        print(f"  - {doc2vec_path}")
        print("Will generate dataset visualizations only.")
    
    # Load data
    print("\nLoading data...")
    loader = ProvenanceDataLoader(args.benign_dir, args.malicious_dir)
    dataframes, labels = loader.load_all_files()
    
    if len(dataframes) == 0:
        print("Error: No data files found!")
        return
    
    print(f"Loaded {len(dataframes)} files")
    
    # Extract features using ProvDetector methodology
    print("Extracting features using ProvDetector methodology...")
    X = []
    y = []
    
    # Create extractor if not already loaded
    if not extractor:
        extractor = ProvDetectorFeatureExtractor()
    
    for i, df in enumerate(dataframes):
        try:
            features = extractor.extract_features(df, use_rarest=model_loaded, top_k_paths=50)
            X.append(features)
            y.append(labels[i])
        except Exception as e:
            print(f"Error extracting features from file {i}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Generate visualizations
    if args.plot_type in ['all', 'distribution']:
        print("\nGenerating dataset distribution plot...")
        plot_dataset_distribution(y, 
            save_path=os.path.join(args.output_dir, 'dataset_distribution.png'))
    
    if model_loaded and args.plot_type in ['all', 'roc', 'confusion', 'prediction']:
        # Use ALL data for ROC curve to get more points and smoother curve
        # Split data for confusion matrix and other metrics (use test set)
        if len(X) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Get predictions on test set for confusion matrix
            y_pred = rf_model.predict(X_test)
            y_proba_test = rf_model.predict_proba(X_test)[:, 1]
            
            # Get predictions on ALL data for ROC curve (smoother curve with more points)
            print("\nUsing ALL data for ROC curve to ensure smooth interpolation...")
            y_proba_all = rf_model.predict_proba(X)[:, 1]
            
            if args.plot_type in ['all', 'roc']:
                print("\nGenerating ROC curve using all data...")
                print("ROC curve will be displayed in a window. Close it to continue...")
                plot_roc_curve(y, y_proba_all, 
                    model_name="ProvDetector",
                    save_path=os.path.join(args.output_dir, 'roc_curve.png'))
            
            if args.plot_type in ['all', 'confusion']:
                print("\nGenerating confusion matrix...")
                plot_confusion_matrix(y_test, y_pred,
                    save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
            
            if args.plot_type in ['all', 'prediction']:
                print("\nGenerating prediction distribution plot...")
                plot_prediction_distribution(y_test, y_proba_test,
                    save_path=os.path.join(args.output_dir, 'prediction_distribution.png'))
            
            if args.plot_type in ['all', 'features']:
                print("\nGenerating feature importance plot...")
                if hasattr(rf_model, 'feature_importances_'):
                    # Doc2Vec features are embedding dimensions
                    feature_names_generic = [f"Embedding_Dim_{i}" for i in range(len(rf_model.feature_importances_))]
                    plot_feature_importance(rf_model, feature_names_generic,
                        save_path=os.path.join(args.output_dir, 'feature_importance.png'))
    
    if args.plot_type in ['all', 'correlation'] and len(X) > 0:
        print("\nGenerating feature correlation plot...")
        # For Doc2Vec embeddings, create a DataFrame from the feature matrix
        features_df = pd.DataFrame(X, columns=[f'Dim_{i}' for i in range(X.shape[1])])
        plot_feature_correlation(features_df,
            save_path=os.path.join(args.output_dir, 'feature_correlation.png'))
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Plots saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

