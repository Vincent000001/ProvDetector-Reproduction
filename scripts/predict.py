"""
Prediction script for ProvDetector
"""
import os
import numpy as np
import pandas as pd
import argparse
from data_loader import ProvenanceDataLoader
from feature_extractor import ProvDetectorFeatureExtractor
import joblib


def main():
    parser = argparse.ArgumentParser(description='Predict malware from provenance graph CSV')
    parser.add_argument('--csv-file', type=str, required=True, help='Path to CSV file to analyze')
    parser.add_argument('--model-path', type=str, default='models/random_forest_model.joblib', 
                       help='Path to trained Random Forest model')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'max', 'concat', 'enhanced'],
                        help='Aggregation method used during training (must match training)')
    parser.add_argument('--top-k-paths', type=int, default=150,
                        help='Top-K rare paths to use (must match training)')
    parser.add_argument('--use-weighted', action='store_true', default=True,
                        help='Use weighted path aggregation (must match training)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Probability threshold for malicious classification')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"ERROR: File not found: {args.csv_file}")
        print("\nPlease provide a valid path to a CSV file.")
        print("Example:")
        print("  python predict.py --csv-file windows/benign/nd-105715-processletevent_0.csv")
        print("  python predict.py --csv-file windows/malicious/nd-1143339833-processletevent_0.csv")
        return
    
    # Load models
    model_dir = os.path.dirname(args.model_path)
    freq_db_path = os.path.join(model_dir, 'frequency_database.pkl')
    doc2vec_path = os.path.join(model_dir, 'doc2vec_model.model')
    
    print(f"Loading models from {model_dir}...")
    
    if not os.path.exists(freq_db_path):
        print(f"ERROR: Frequency database not found: {freq_db_path}")
        return
    
    if not os.path.exists(doc2vec_path):
        print(f"ERROR: Doc2Vec model not found: {doc2vec_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"ERROR: Random Forest model not found: {args.model_path}")
        return
    
    extractor = ProvDetectorFeatureExtractor(frequency_db_path=freq_db_path, doc2vec_model_path=doc2vec_path)
    rf_model = joblib.load(args.model_path)
    print("Models loaded successfully")
    
    # Load and process CSV file
    print(f"Loading CSV file: {args.csv_file}...")
    loader = ProvenanceDataLoader()
    df = loader.load_csv(args.csv_file)
    
    if df.empty:
        print("Error: Could not load CSV file or file is empty")
        return
    
    # Extract features using ProvDetector methodology
    print("Extracting features using ProvDetector methodology...")
    features = extractor.extract_features(
        df,
        use_rarest=True,
        top_k_paths=args.top_k_paths,
        aggregation=args.aggregation,
        use_weighted=args.use_weighted
    )
    
    # Make prediction
    feature_array = features.reshape(1, -1)
    prediction = rf_model.predict(feature_array)[0]
    probability = rf_model.predict_proba(feature_array)[0]
    
    benign_prob = probability[0]
    malicious_prob = probability[1]
    
    # Print results
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"File: {args.csv_file}")
    print(f"\nProbabilities:")
    print(f"  Benign:    {benign_prob:.4f} ({benign_prob*100:.2f}%)")
    print(f"  Malicious: {malicious_prob:.4f} ({malicious_prob*100:.2f}%)")
    print(f"\nPrediction: {'MALICIOUS' if prediction == 1 else 'BENIGN'}")
    
    if malicious_prob >= args.threshold:
        print(f"\nWARNING: File classified as MALICIOUS (threshold: {args.threshold})")
    else:
        print(f"\nFile classified as BENIGN (threshold: {args.threshold})")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
