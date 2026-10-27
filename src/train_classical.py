
from pathlib import Path
import json
import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORT = ROOT / "report"
CLASSES = ["blackspot", "canker", "fresh", "grenning"]
IMG_SIZE = (128, 128)


def extract_features(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    image = cv2.resize(image, IMG_SIZE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []
    for color_image in (hsv, lab):
        for channel in cv2.split(color_image):
            hist = cv2.calcHist([channel], [0], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(12), range=(0, 11))
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= max(lbp_hist.sum(), 1)
    features.extend(lbp_hist)

    return np.asarray(features, dtype=np.float32)


def load_split(split):
    X, y = [], []
    for cls in CLASSES:
        folder = DATA / split / cls
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in
            {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        for path in files:
            X.append(extract_features(path))
            y.append(cls)

    return np.asarray(X), np.asarray(y)


def evaluate(model, X, y, name, split):
    pred = model.predict(X)
    result = {
        "model": name,
        "split": split,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(
            y, pred, labels=CLASSES, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y, pred, labels=CLASSES, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y, pred, labels=CLASSES, average="macro", zero_division=0
        )),
    }

    print(f"\n{name} — {split.upper()}")
    print(json.dumps(result, indent=2))
    print(classification_report(
        y, pred, labels=CLASSES, zero_division=0
    ))

    if split == "test":
        cm = confusion_matrix(y, pred, labels=CLASSES)
        np.savetxt(
            REPORT / f"{name.lower()}_confusion_matrix.csv",
            cm, delimiter=",", fmt="%d"
        )

    return result


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    print("Loading training features...")
    X_train, y_train = load_split("train")
    print("Loading validation features...")
    X_val, y_val = load_split("val")
    print("Loading official test features...")
    X_test, y_test = load_split("test")

    candidates = {
        "SVM": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", C=3, class_weight="balanced"
            ))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        evaluate(model, X_val, y_val, name, "val")
        results.append(evaluate(model, X_test, y_test, name, "test"))

        joblib.dump(model, MODELS / f"{name.lower()}.joblib")

    with open(REPORT / "classical_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nClassical ML training complete.")
    print(f"Models saved in: {MODELS}")
    print(f"Results saved in: {REPORT}")


if __name__ == "__main__":
    main()
