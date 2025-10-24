import os, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42

def main():
    os.makedirs("artifacts", exist_ok=True)
    data = load_digits()
    X, y = data.data, data.target  # 1797 x 64
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=RANDOM_STATE))
    ])

    param_grid = {
        "clf__C": [1.0, 3.0, 10.0],
        "clf__gamma": ["scale", 0.01, 0.001],
        "clf__kernel": ["rbf"]
    }

    gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Best CV accuracy:", gs.best_score_)

    y_pred = gs.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save sample predictions as image grid
    fig, axes = plt.subplots(3, 6, figsize=(9, 5))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=18, replace=False)
    for ax, i in zip(axes.ravel(), idx):
        img = X_test[i].reshape(8, 8)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Pred:{gs.predict([X_test[i]])[0]}")
    plt.tight_layout()
    plt.savefig("artifacts/sample_predictions.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
