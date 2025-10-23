import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

RANDOM_STATE = 42

def toy_dataset():
    texts = [
        "I love this product, it works perfectly",
        "Absolutely terrible, broke on the first day",
        "Decent quality for the price",
        "Customer support was amazing and helpful",
        "Horrible experience, I want a refund",
        "Works fine, does the job",
        "Not bad, but could be better",
        "Fantastic build and easy to use",
        "I hate it, waste of money",
        "Okay overall, shipping was slow",
        "Great value! highly recommend",
        "Awful design, very disappointed",
    ]
    labels = [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
    ]
    return pd.DataFrame({"text": texts, "label": labels})

def main():
    os.makedirs("artifacts", exist_ok=True)
    df = toy_dataset()
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=RANDOM_STATE, stratify=df["label"])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
    ])

    param_grid = {
        "clf__C": [0.5, 1.0, 3.0]
    }
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_macro", cv=5, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Best CV macro-F1:", gs.best_score_)

    y_pred = gs.predict(X_test)
    print("\nTest macro-F1:", f1_score(y_test, y_pred, average="macro"))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(gs.best_estimator_, "artifacts/model.joblib")
    print("Saved model to artifacts/model.joblib")

if __name__ == "__main__":
    main()
