"""
Career Guidance ML Model - Training Script
Model: TF-IDF + Random Forest (with comparison against other classifiers)
Input: combined student text (skills + interests + goals + projects + education)
Output: Career path prediction with confidence probabilities
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("CAREER GUIDANCE ML MODEL — TRAINING")
print("=" * 60)

df = pd.read_csv("data/student_profiles.csv")
print(f"\n[DATA] Loaded {len(df)} records | {df['career_label'].nunique()} career labels")

X = df["combined_text"]
y = df["career_label"]

# Label encode for reference
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"[DATA] Labels: {list(le.classes_)}")

# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. DEFINE PIPELINES TO COMPARE
# ─────────────────────────────────────────────
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=8000,
    sublinear_tf=True,       # apply log normalization
    min_df=2,
    analyzer='word'
)

pipelines = {
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True, min_df=2)),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1))
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs', random_state=42))
    ]),
    "Linear SVC (Calibrated)": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True, min_df=2)),
        ("clf", CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=42)))
    ]),
}

# ─────────────────────────────────────────────
# 4. TRAIN & COMPARE ALL MODELS
# ─────────────────────────────────────────────
print("\n[TRAINING] Comparing classifiers...\n")
results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in pipelines.items():
    print(f"  Training: {name}...")
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1_weighted', n_jobs=-1)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "pipeline": pipe
    }
    print(f"    CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test Accuracy : {test_acc:.4f}")
    print(f"    Test F1       : {test_f1:.4f}\n")

# ─────────────────────────────────────────────
# 5. SELECT BEST MODEL
# ─────────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]["test_f1"])
best_pipeline = results[best_model_name]["pipeline"]

print(f"\n[BEST MODEL] {best_model_name}")
print(f"  Test Accuracy : {results[best_model_name]['test_accuracy']:.4f}")
print(f"  Test F1 Score : {results[best_model_name]['test_f1']:.4f}")

# ─────────────────────────────────────────────
# 6. DETAILED REPORT ON BEST MODEL
# ─────────────────────────────────────────────
y_pred_best = best_pipeline.predict(X_test)
print("\n[CLASSIFICATION REPORT]\n")
print(classification_report(y_test, y_pred_best))

# ─────────────────────────────────────────────
# 7. SAVE MODEL + METADATA
# ─────────────────────────────────────────────
os.makedirs("ml", exist_ok=True)

# Save best pipeline (includes vectorizer + classifier)
pickle.dump(best_pipeline, open("ml/career_classifier.pkl", "wb"))
print(f"\n[SAVED] ml/career_classifier.pkl")

# Save label list for reference
labels = list(best_pipeline.classes_) if hasattr(best_pipeline, 'classes_') else list(le.classes_)
career_labels = list(best_pipeline.named_steps['clf'].classes_) \
    if hasattr(best_pipeline.named_steps['clf'], 'classes_') else list(le.classes_)

model_metadata = {
    "best_model": best_model_name,
    "test_accuracy": round(results[best_model_name]['test_accuracy'], 4),
    "test_f1": round(results[best_model_name]['test_f1'], 4),
    "career_labels": career_labels,
    "total_training_samples": len(X_train),
    "total_test_samples": len(X_test),
    "all_model_results": {
        name: {
            "cv_f1_mean": round(r["cv_f1_mean"], 4),
            "test_accuracy": round(r["test_accuracy"], 4),
            "test_f1": round(r["test_f1"], 4)
        }
        for name, r in results.items()
    }
}

with open("ml/model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)
print(f"[SAVED] ml/model_metadata.json")

# ─────────────────────────────────────────────
# 8. QUICK INFERENCE TEST
# ─────────────────────────────────────────────
print("\n[TEST INFERENCE] Running sample predictions...\n")

test_profiles = [
    {
        "desc": "Data Science student",
        "text": "python machine learning pandas numpy statistics data visualization sql scikit-learn data science want to be data scientist movie recommendation system"
    },
    {
        "desc": "Mobile dev student",
        "text": "flutter dart firebase android mobile ui rest api git mobile app developer food delivery app cross platform"
    },
    {
        "desc": "Frontend student",
        "text": "html css javascript react responsive design tailwind css nextjs frontend developer portfolio website web animations"
    },
    {
        "desc": "Cybersecurity student",
        "text": "networking linux ethical hacking kali linux wireshark security fundamentals ctf challenges penetration tester vulnerability scanner"
    }
]

for profile in test_profiles:
    proba = best_pipeline.predict_proba([profile["text"]])[0]
    top3_idx = proba.argsort()[-3:][::-1]
    top3 = [(career_labels[i], round(proba[i]*100, 1)) for i in top3_idx]
    print(f"  Profile: {profile['desc']}")
    print(f"  Top-3 Predictions:")
    for rank, (label, conf) in enumerate(top3, 1):
        print(f"    {rank}. {label:30s} {conf:.1f}%")
    print()

print("[DONE] Training complete. Model ready to use.")
