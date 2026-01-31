#!/usr/bin/env python3
"""
ProvDetector Training Script - Leak Free (Robust)

Goal
----
Reproduce the "paper-like" LOF pipeline while avoiding common evaluation leaks:

1) Split FIRST (stratified or group-aware).
2) Build frequency DB + train Doc2Vec on TRAIN BENIGN ONLY (paper method scope).
3) Extract features for ALL samples using that fixed representation.
4) Train LOF on TRAIN BENIGN ONLY.
5) Evaluate on TEST (with robust metric handling even when a class is missing).

Why this file?
--------------
- Your earlier `train_group_split.py` built the representation using *all* files (incl. test),
  which can leak distribution information. This version prevents that.
- Some splits can yield a single-class test set (e.g., all malicious). This version will NOT crash:
  it prints a report with fixed labels and sets ROC-AUC to NaN when undefined.
- Deterministic-ish runs: optional `--seed` and `--d2v-workers 1` for more stable Doc2Vec.

Notes
-----
- This is still a small dataset setting. Huge F1 might be real separability or could be
  dataset artifacts; use group-aware split (recommended) and multiple seeds to sanity-check.
"""

import os
import re
import json
import argparse
from typing import List, Tuple

import numpy as np

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from imblearn.over_sampling import SMOTE  # noqa: F401
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. SMOTE will be disabled.")

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from data_loader import ProvenanceDataLoader
from feature_extractor import ProvDetectorFeatureExtractor


# -----------------------------
# Utilities
# -----------------------------

def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    # Python's random module isn't heavily used here, but seed it anyway.
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def _infer_file_id(df, fallback: str) -> str:
    try:
        if df is not None and 'file_id' in df.columns and len(df) > 0:
            v = str(df['file_id'].iloc[0])
            return v if v else fallback
    except Exception:
        pass
    return fallback


def _group_id_from_file_id(file_id: str, group_regex: str) -> str:
    """Derive a group id to avoid leakage across split fragments.

    Typical split file name: <base>_p7.csv -> group is <base>
    """
    base = os.path.splitext(os.path.basename(file_id))[0]
    try:
        return re.sub(group_regex, '', base)
    except re.error:
        return base


def _group_train_test_split_indices(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
    tries: int = 200,
    require_both_classes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Group-aware split with best-effort label balance.

    GroupShuffleSplit doesn't support stratification. We try multiple seeds and pick the split
    closest to the overall positive rate, while (optionally) ensuring both classes appear in
    train and test.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)

    overall_pos = float(np.mean(y == 1))
    best = None
    best_gap = 1e9

    # We don't need X for GroupShuffleSplit; provide dummy indices.
    X_dummy = np.zeros((len(y), 1), dtype=np.float32)

    for s in range(seed, seed + max(1, tries)):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
        train_idx, test_idx = next(gss.split(X_dummy, y, groups=groups))
        y_tr, y_te = y[train_idx], y[test_idx]

        if require_both_classes:
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                continue

        test_pos = float(np.mean(y_te == 1)) if len(y_te) else float("nan")
        gap = abs(test_pos - overall_pos) if np.isfinite(test_pos) else 1e9

        if gap < best_gap:
            best_gap = gap
            best = (train_idx, test_idx, s, test_pos)

    if best is None:
        # Last resort: just do one split with the given seed.
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X_dummy, y, groups=groups))
        test_pos = float(np.mean(y[test_idx] == 1)) if len(test_idx) else float("nan")
        return train_idx, test_idx, seed, test_pos

    return best


def _train_doc2vec_sentences(
    sentences: List[str],
    vector_size: int,
    epochs: int,
    seed: int,
    workers: int = 1,
) -> Doc2Vec:
    """Train a Doc2Vec model treating each sentence as a document (paper-like)."""
    documents = [TaggedDocument(words=s.split(), tags=[i]) for i, s in enumerate(sentences)]

    # Determinism: workers=1 recommended; seed helps, but full determinism isn't guaranteed
    # across all BLAS setups. In practice, workers=1 makes it stable enough.
    model = Doc2Vec(
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=workers,
        epochs=epochs,
        dm=1,          # PV-DM
        seed=seed,
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC is undefined if y_true has one class. Return NaN instead of raising."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Always print a 2-class report even if one class is missing."""
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=['Benign', 'Malicious'],
        zero_division=0,
    )


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description='Train ProvDetector LOF (leak-free, robust)')

    parser.add_argument('--benign-dir', type=str, default='benign', help='Directory with benign CSV files')
    parser.add_argument('--malicious-dir', type=str, default='malicious', help='Directory with malicious CSV files')
    parser.add_argument('--output-dir', type=str, default='models_noleak', help='Output directory for models')

    parser.add_argument('--vector-size', type=int, default=100, help='Doc2Vec vector size')
    parser.add_argument('--epochs', type=int, default=50, help='Doc2Vec training epochs')
    parser.add_argument('--top-k-paths', type=int, default=20, help='Number of rarest paths to use')

    parser.add_argument('--detector', type=str, default='lof', choices=['lof'], help='Only LOF in this script')
    parser.add_argument('--lof-contamination', type=float, default=0.04, help='LOF contamination')
    parser.add_argument('--lof-n-neighbors', type=int, default=10, help='LOF n_neighbors')

    parser.add_argument('--aggregation', type=str, default='enhanced',
                        choices=['mean', 'max', 'concat', 'enhanced'],
                        help='How to aggregate path embeddings')

    parser.add_argument('--use-weighted', action='store_true', default=True,
                        help='Use weighted path aggregation based on rarity (recommended)')

    # Split controls
    parser.add_argument('--split-mode', type=str, default='group',
                        choices=['stratified', 'group'],
                        help='stratified: sample-level stratify; group: group-aware split for *_pN.csv fragments.')
    parser.add_argument('--group-regex', type=str, default=r'_p\d+$',
                        help='Regex stripped from file base name to form group id.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test ratio')
    parser.add_argument('--split-seed', type=int, default=42, help='Split seed (start seed for group tries)')

    parser.add_argument('--group-split-tries', type=int, default=200,
                        help='How many seeds to try for group split to get balanced label ratio & both classes.')
    parser.add_argument('--allow-single-class-test', action='store_true',
                        help='If set, do not force both classes in train/test during group split tries.')

    # Repro controls
    parser.add_argument('--seed', type=int, default=42, help='Global seed for numpy/random + Doc2Vec')
    parser.add_argument('--d2v-workers', type=int, default=1, help='Doc2Vec workers (1 for stability)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _set_global_seed(args.seed)

    print("=" * 60)
    print("ProvDetector Training - Leak Free (Robust)")
    print("=" * 60)

    # [1] Load
    print("\n[1/6] Loading data...")
    loader = ProvenanceDataLoader(args.benign_dir, args.malicious_dir)
    dataframes, labels = loader.load_all_files()

    if len(dataframes) == 0:
        print("Error: No data files found!")
        return

    labels = np.asarray(labels, dtype=int)
    n_benign = int(np.sum(labels == 0))
    n_mal = int(np.sum(labels == 1))
    print(f"Loaded {len(dataframes)} files (benign={n_benign}, malicious={n_mal})")

    # Build ids + groups up front
    sample_ids = []
    groups = []
    for i, df in enumerate(dataframes):
        fid = _infer_file_id(df, f"sample_{i}.csv")
        sample_ids.append(fid)
        groups.append(_group_id_from_file_id(fid, args.group_regex))
    groups = np.asarray(groups)

    # [2] Split
    print("\n[2/6] Splitting train/test...")
    if args.split_mode == 'group':
        train_idx, test_idx, used_seed, test_pos = _group_train_test_split_indices(
            y=labels,
            groups=groups,
            test_size=args.test_size,
            seed=args.split_seed,
            tries=args.group_split_tries,
            require_both_classes=not args.allow_single_class_test,
        )
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        print(f"Split mode: group | unique_groups total={len(set(groups))}, train={len(train_groups)}, test={len(test_groups)}")
        print(f"Group split seed used: {used_seed} | test positive rate≈{test_pos:.3f}")
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=args.test_size,
            random_state=args.split_seed,
            stratify=labels,
        )
        used_seed = int(args.split_seed)
        test_pos = float(np.mean(labels[test_idx] == 1))
        print("Split mode: stratified")
        print(f"Stratified seed used: {used_seed} | test positive rate≈{test_pos:.3f}")

    y_train = labels[train_idx]
    y_test = labels[test_idx]
    print(f"Train set: {len(train_idx)} (benign={int(np.sum(y_train==0))}, malicious={int(np.sum(y_train==1))})")
    print(f"Test  set: {len(test_idx)} (benign={int(np.sum(y_test==0))}, malicious={int(np.sum(y_test==1))})")

    # [3] Build representation on TRAIN BENIGN ONLY
    print("\n[3/6] Building frequency DB + training Doc2Vec (train-only)...")
    extractor = ProvDetectorFeatureExtractor()

    train_benign_idx = train_idx[y_train == 0]
    print("Rep learning scope: train benign only (paper method)")

    rep_sentences: List[str] = []
    for j, idx in enumerate(train_benign_idx):
        df = dataframes[int(idx)]
        paths = extractor.extract_paths_from_graph(df)
        sents = [extractor.path_to_sentence(p, df) for p in paths]
        sents = [s for s in sents if s]
        rep_sentences.extend(sents)
        if (j + 1) % 20 == 0:
            print(f"  processed rep samples {j+1}/{len(train_benign_idx)}...")

    print(f"Collected sentences: {len(rep_sentences)}")
    extractor.build_frequency_database(rep_sentences)
    print(f"Frequency DB size: {len(extractor.frequency_db)}")

    extractor.doc2vec_model = _train_doc2vec_sentences(
        rep_sentences,
        vector_size=args.vector_size,
        epochs=args.epochs,
        seed=args.seed,
        workers=args.d2v_workers,
    )
    print("Doc2Vec trained (deterministic-ish settings).")

    # [4] Extract features for ALL samples (but ONLY using the fixed rep)
    print("\n[4/6] Extracting features for train/test...")
    X_all = []
    ok_idx = []
    for i, df in enumerate(dataframes):
        try:
            feat = extractor.extract_features(
                df,
                use_rarest=True,
                top_k_paths=args.top_k_paths,
                aggregation=args.aggregation,
                use_weighted=args.use_weighted,
            )
            X_all.append(feat)
            ok_idx.append(i)
        except Exception as e:
            print(f"  [skip] feature extraction failed for {sample_ids[i]}: {e}")

    X_all = np.asarray(X_all, dtype=np.float32)
    ok_idx = np.asarray(ok_idx, dtype=int)

    # Map original indices -> compact indices after skipping failures
    idx_map = {orig: new for new, orig in enumerate(ok_idx.tolist())}
    train_mask = np.array([idx_map.get(int(i), -1) for i in train_idx], dtype=int)
    test_mask = np.array([idx_map.get(int(i), -1) for i in test_idx], dtype=int)

    # Filter out -1 (skipped)
    train_mask = train_mask[train_mask >= 0]
    test_mask = test_mask[test_mask >= 0]

    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    y_train2 = labels[ok_idx][train_mask]
    y_test2 = labels[ok_idx][test_mask]

    print(f"Feature matrix: {X_all.shape} | used train={X_train.shape[0]} test={X_test.shape[0]}")

    # [5] Train LOF on TRAIN BENIGN ONLY
    print("\n[5/6] Training LOF (benign only)...")
    X_train_benign = X_train[y_train2 == 0]
    print(f"Training LOF on benign: {X_train_benign.shape[0]} samples")

    lof_model = LocalOutlierFactor(
        n_neighbors=args.lof_n_neighbors,
        contamination=args.lof_contamination,
        novelty=True,
        n_jobs=-1,
    )
    lof_model.fit(X_train_benign)

    # Predict on TEST
    y_pred_lof = lof_model.predict(X_test)  # -1 outlier, 1 inlier
    y_pred = (y_pred_lof == -1).astype(int)

    # Scores for ROC-AUC (higher => more malicious)
    y_scores = -lof_model.score_samples(X_test)
    # normalize to [0,1] as a pseudo-proba
    if len(y_scores) > 0:
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
    else:
        y_proba = np.array([], dtype=np.float32)

    # [6] Eval (robust)
    print("\n[6/6] Evaluation")
    print("=" * 60)
    print("Test Set Evaluation (Leak-Free, Robust)")
    print("=" * 60)

    accuracy = float(np.mean(y_pred == y_test2)) if len(y_test2) else float("nan")
    precision = float(precision_score(y_test2, y_pred, zero_division=0)) if len(y_test2) else float("nan")
    recall = float(recall_score(y_test2, y_pred, zero_division=0)) if len(y_test2) else float("nan")
    f1 = float(f1_score(y_test2, y_pred, zero_division=0)) if len(y_test2) else float("nan")
    roc_auc = _safe_roc_auc(y_test2, y_proba) if len(y_test2) else float("nan")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc if np.isfinite(roc_auc) else 'nan'}")

    print("\nClassification Report:")
    print(_safe_classification_report(y_test2, y_pred))

    cm = confusion_matrix(y_test2, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    print("\nConfusion Matrix:")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")

    # Save
    print("\nSaving models...")
    # Save frequency DB + Doc2Vec using extractor helper
    freq_db_path = os.path.join(args.output_dir, 'frequency_database.pkl')
    doc2vec_path = os.path.join(args.output_dir, 'doc2vec_model.model')
    extractor.save_models(freq_db_path, doc2vec_path)

    # Save LOF
    import joblib
    model_path = os.path.join(args.output_dir, 'lof_model.joblib')
    joblib.dump(lof_model, model_path)

    metrics = {
        "mode": "graph_doc2vec_lof_leakfree",
        "detector_type": "lof",
        "split_mode": args.split_mode,
        "test_size": float(args.test_size),
        "split_seed": int(args.split_seed),
        "used_seed": int(used_seed),
        "group_regex": str(args.group_regex),
        "group_split_tries": int(args.group_split_tries),
        "allow_single_class_test": bool(args.allow_single_class_test),
        "seed": int(args.seed),
        "d2v_workers": int(args.d2v_workers),
        "vector_size": int(args.vector_size),
        "epochs": int(args.epochs),
        "top_k_paths": int(args.top_k_paths),
        "aggregation": str(args.aggregation),
        "use_weighted": bool(args.use_weighted),
        "lof_n_neighbors": int(args.lof_n_neighbors),
        "lof_contamination": float(args.lof_contamination),
        "n_total": int(len(labels)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_benign": int(np.sum(y_train == 0)),
        "train_malicious": int(np.sum(y_train == 1)),
        "test_benign": int(np.sum(y_test == 0)),
        "test_malicious": int(np.sum(y_test == 1)),
        "rep_train_benign_used": int(len(train_benign_idx)),
        "rep_sentences": int(len(rep_sentences)),
        "freq_db_size": int(len(extractor.frequency_db)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {args.output_dir}/ (frequency_db, doc2vec, lof_model, test_metrics.json)")
    print("=" * 60)


if __name__ == "__main__":
    main()
