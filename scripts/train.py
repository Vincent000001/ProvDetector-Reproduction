"""
ProvDetector Training Script - Following the paper methodology
Implements: Path-to-Sentence, Doc2Vec, Frequency Database, Rarest Path, Random Forest/LOF
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. SMOTE will be disabled.")
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import argparse
import json
from data_loader import ProvenanceDataLoader
from feature_extractor import ProvDetectorFeatureExtractor
import joblib


def main():
    parser = argparse.ArgumentParser(description='Train ProvDetector model following paper methodology')
    parser.add_argument('--benign-dir', type=str, default='benign', help='Directory with benign CSV files')
    parser.add_argument('--malicious-dir', type=str, default='malicious', help='Directory with malicious CSV files')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--vector-size', type=int, default=128, help='Doc2Vec vector size (increased for better representation)')
    parser.add_argument('--epochs', type=int, default=50, help='Doc2Vec training epochs (increased for better training)')
    parser.add_argument('--top-k-paths', type=int, default=100, help='Number of rarest paths to use (increased for better features)')
    parser.add_argument('--n-estimators', type=int, default=200, help='Random Forest n_estimators (increased for better performance)')
    parser.add_argument('--detector', type=str, default='random_forest', 
                       choices=['random_forest', 'lof'], 
                       help='Detection method: random_forest (supervised) or lof (unsupervised, paper method)')
    parser.add_argument('--lof-contamination', type=float, default=0.04, 
                       help='LOF contamination parameter (expected proportion of outliers)')
    parser.add_argument('--lof-n-neighbors', type=int, default=20, 
                       help='LOF number of neighbors parameter')
    parser.add_argument('--aggregation', type=str, default='enhanced',
                       choices=['mean', 'max', 'concat', 'enhanced'],
                       help='How to aggregate path embeddings (enhanced uses mean+max+std+first+last)')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Use cross-validation instead of single train/test split')
    parser.add_argument('--use-smote', action='store_true',
                       help='Use SMOTE for handling class imbalance')
    parser.add_argument('--use-weighted', action='store_true', default=True,
                       help='Use weighted path aggregation based on rarity')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ProvDetector Training - Following Paper Methodology")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    loader = ProvenanceDataLoader(args.benign_dir, args.malicious_dir)
    dataframes, labels = loader.load_all_files()
    
    if len(dataframes) == 0:
        print("Error: No data files found!")
        return
    
    print(f"Loaded {len(dataframes)} files ({sum(1 for l in labels if l == 0)} benign, {sum(1 for l in labels if l == 1)} malicious)")
    
    # Step 2: Extract paths and build frequency database
    print("\n[2/5] Extracting paths and building frequency database...")
    extractor = ProvDetectorFeatureExtractor()
    
    all_paths_sentences = []
    for i, df in enumerate(dataframes):
        paths = extractor.extract_paths_from_graph(df)
        paths_sentences = [extractor.path_to_sentence(path, df) for path in paths]
        paths_sentences = [p for p in paths_sentences if p]
        all_paths_sentences.extend(paths_sentences)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataframes)} files...")
    
    print(f"Extracted {len(all_paths_sentences)} total path sentences")
    
    # Build frequency database
    extractor.build_frequency_database(all_paths_sentences)
    print(f"Frequency database built with {len(extractor.frequency_db)} unique paths")
    
    # Step 3: Train Doc2Vec model
    print("\n[3/5] Training Doc2Vec model...")
    extractor.train_doc2vec(all_paths_sentences, vector_size=args.vector_size, epochs=args.epochs)
    print("Doc2Vec model trained")
    
    # Step 4: Extract features using ProvDetector methodology
    print("\n[4/5] Extracting features using rarest paths and Doc2Vec...")
    X = []
    y = []
    
    for i, df in enumerate(dataframes):
        try:
            # Extract features using rarest paths and Doc2Vec with enhanced aggregation
            features = extractor.extract_features(df, use_rarest=True, top_k_paths=args.top_k_paths, 
                                                 aggregation=args.aggregation, use_weighted=args.use_weighted)
            X.append(features)
            y.append(labels[i])
        except Exception as e:
            print(f"Error extracting features from file {i}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: Benign={np.sum(y==0)}, Malicious={np.sum(y==1)}")
    
    # Step 5: Train detector (Random Forest or LOF)
    if args.detector == 'lof':
        print("\n[5/5] Training Local Outlier Factor (LOF) anomaly detector (paper method)...")
        print("Note: LOF is unsupervised - trained only on benign data")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # For LOF, train only on benign data (unsupervised anomaly detection)
        X_train_benign = X_train[y_train == 0]
        print(f"Training set: {X_train_benign.shape[0]} benign samples")
        print(f"Test set: {X_test.shape[0]} samples ({np.sum(y_test==0)} benign, {np.sum(y_test==1)} malicious)")
        
        # Train LOF on benign data only
        lof_model = LocalOutlierFactor(
            n_neighbors=args.lof_n_neighbors,
            contamination=args.lof_contamination,
            novelty=True,  # Enable prediction on new data
            n_jobs=-1
        )
        
        lof_model.fit(X_train_benign)
        
        # Predict on test set
        y_pred_lof = lof_model.predict(X_test)
        # LOF returns -1 for outliers (malicious), 1 for inliers (benign)
        y_pred = (y_pred_lof == -1).astype(int)
        # Get outlier scores (negative scores = more anomalous)
        y_scores = -lof_model.score_samples(X_test)
        # Normalize scores to probabilities (higher score = more malicious)
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
        
        detector_model = lof_model
        model_type = 'lof'
        
    else:  # random_forest
        print("\n[5/5] Training Random Forest classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Apply SMOTE if requested to handle class imbalance
        if args.use_smote and SMOTE_AVAILABLE:
            print("Applying SMOTE for class imbalance handling...")
            k_neighbors = min(3, max(1, np.sum(y_train == 0) - 1), max(1, np.sum(y_train == 1) - 1))
            if k_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE - Training set: {X_train.shape[0]} samples ({np.sum(y_train==0)} benign, {np.sum(y_train==1)} malicious)")
            else:
                print("Warning: Not enough samples for SMOTE. Skipping...")
        elif args.use_smote and not SMOTE_AVAILABLE:
            print("Warning: SMOTE requested but imbalanced-learn not available. Skipping...")
        
        # Train Random Forest (supervised) with improved hyperparameters
        # For small datasets, use more aggressive class weighting
        rf_model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=15,  # Reduced for small datasets
            min_samples_split=3,  # Reduced for small datasets
            min_samples_leaf=1,  # Reduced for small datasets
            max_features='sqrt',  # Use sqrt of features (common best practice)
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',  # Better for small datasets
            bootstrap=True,
            oob_score=True,  # Out-of-bag scoring
            max_samples=0.8  # Use 80% of samples per tree for better diversity
        )
        
        if args.cross_validate:
            # Use cross-validation for better evaluation
            print("Using cross-validation for evaluation...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='f1')
            print(f"Cross-validation F1 scores: {cv_scores}")
            print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Still train on full data for final model
            rf_model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = rf_model.predict(X_test)
            y_proba = rf_model.predict_proba(X_test)[:, 1]
        else:
            rf_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = rf_model.predict(X_test)
            y_proba = rf_model.predict_proba(X_test)[:, 1]
        
        detector_model = rf_model
        model_type = 'random_forest'
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")  # Paper reports F1 score of 0.974
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    # Save models
    print("\nSaving models...")
    
    # Save frequency database
    freq_db_path = os.path.join(args.output_dir, 'frequency_database.pkl')
    doc2vec_path = os.path.join(args.output_dir, 'doc2vec_model.model')
    extractor.save_models(freq_db_path, doc2vec_path)
    print(f"Frequency database saved to {freq_db_path}")
    print(f"Doc2Vec model saved to {doc2vec_path}")
    
    # Save detector model
    if model_type == 'lof':
        model_path = os.path.join(args.output_dir, 'lof_model.joblib')
        joblib.dump(detector_model, model_path)
        print(f"LOF model saved to {model_path}")
    else:
        model_path = os.path.join(args.output_dir, 'random_forest_model.joblib')
        joblib.dump(detector_model, model_path)
        print(f"Random Forest model saved to {model_path}")
    
    # Save metrics
    metrics = {
        'detector_type': model_type,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),  # Paper reports F1 score of 0.974
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }
    
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nProvDetector methodology implemented:")
    print("  [OK] Path-to-Sentence conversion")
    print("  [OK] Frequency database construction")
    print("  [OK] Rarest path selection")
    print("  [OK] Doc2Vec embedding (PV-DM model)")
    if model_type == 'lof':
        print("  [OK] Local Outlier Factor anomaly detection (paper method)")
    else:
        print("  [OK] Random Forest classification")
    print(f"\nResults: F1-Score = {f1:.4f} (Paper reports ~0.974)")


if __name__ == '__main__':
    main()

