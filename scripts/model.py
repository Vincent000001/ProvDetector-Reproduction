"""
Machine learning models for malware detection
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import os


class MalwareDetector:
    """Machine learning model for malware detection from provenance graphs"""
    
    def __init__(self, model_type='random_forest', n_estimators=100, random_state=42):
        """
        Initialize the malware detector
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
            n_estimators: Number of trees in the ensemble
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,  # Let trees grow fully for better performance
                min_samples_split=2,  # More sensitive splits
                min_samples_leaf=1,  # More granular leaves
                max_features='sqrt',  # Use sqrt of features per split (common in papers)
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True,
                oob_score=True  # Out-of-bag scoring for better evaluation
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        Train the model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=benign, 1=malicious)
            feature_names: Optional list of feature names
        """
        print(f"Training {self.model_type} model on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Print feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if feature_names:
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                print("\nTop 10 Most Important Features:")
                for name, importance in feature_importance[:10]:
                    print(f"  {name}: {importance:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for samples"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability scores for samples"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model and return metrics
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]  # Probability of being malicious
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(y_pred == y),
            'roc_auc': roc_auc_score(y, y_proba),
        }
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        metrics.update({
            'precision_benign': report.get('0', {}).get('precision', 0),
            'recall_benign': report.get('0', {}).get('recall', 0),
            'f1_benign': report.get('0', {}).get('f1-score', 0),
            'precision_malicious': report.get('1', {}).get('precision', 0),
            'recall_malicious': report.get('1', {}).get('recall', 0),
            'f1_malicious': report.get('1', {}).get('f1-score', 0),
        })
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv=5) -> dict:
        """Perform cross-validation with multiple metrics"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Get multiple metrics from cross-validation
        scoring = {
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'accuracy': 'accuracy'
        }
        
        cv_results = cross_validate(self.model, X_scaled, y, cv=cv, scoring=scoring, return_train_score=False)
        
        return {
            'mean_roc_auc': cv_results['test_roc_auc'].mean(),
            'std_roc_auc': cv_results['test_roc_auc'].std(),
            'mean_f1': cv_results['test_f1'].mean(),
            'std_f1': cv_results['test_f1'].std(),
            'mean_precision': cv_results['test_precision'].mean(),
            'std_precision': cv_results['test_precision'].std(),
            'mean_recall': cv_results['test_recall'].mean(),
            'std_recall': cv_results['test_recall'].std(),
            'mean_accuracy': cv_results['test_accuracy'].mean(),
            'std_accuracy': cv_results['test_accuracy'].std(),
            'cv_scores': cv_results['test_roc_auc'].tolist(),
            'cv_f1_scores': cv_results['test_f1'].tolist()
        }
    
    def save(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data.get('feature_names')
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")

