import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

import sklearn.preprocessing as pp

# =========================
# CONFIG
# =========================
DATA_PATH = "Stars.csv"
MODEL_PATH = "star_classifier_model.h5"
SCALER_PATH = "scaler.pkl"

LABEL_NAMES = [
    "Red Dwarf",
    "Brown Dwarf",
    "White Dwarf",
    "Main Sequence",
    "Super Giant",
    "Hyper Giant"
]

N_CLASSES = 6


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Preprocessing
color_encoder = pp.LabelEncoder()
spectral_class_encoder = pp.LabelEncoder()
color_encoder.fit(df['Color'])
spectral_class_encoder.fit(df['Spectral_Class'])
df['Color_code'] = color_encoder.transform(df['Color'])
df['Spectral_Class_code'] = spectral_class_encoder.transform(df['Spectral_Class'])
df = df.drop(['Color','Spectral_Class'],axis=1)
X = df.drop(columns=["Type"]).values
y = df["Type"].values


# =========================
# LOAD SCALER & MODEL
# =========================
scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH)

X_scaled = scaler.transform(X)


# =========================
# PREDICTION
# =========================
y_prob = model.predict(X_scaled)
y_pred = np.argmax(y_prob, axis=1)


# =========================
# BASIC METRICS
# =========================
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average="macro")
recall = recall_score(y, y_pred, average="macro")
f1 = f1_score(y, y_pred, average="macro")

loss = model.evaluate(
    X_scaled,
    y,
    verbose=1
)[0]

print("===== MODEL EVALUATION =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Loss      : {loss:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABEL_NAMES
)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Star Classification")
plt.tight_layout()
plt.show()


# =========================
# ROC & AUC (One-vs-Rest)
# =========================
y_bin = pp.label_binarize(y, classes=np.arange(N_CLASSES))

plt.figure(figsize=(8, 6))

for i in range(N_CLASSES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{LABEL_NAMES[i]} (AUC = {roc_auc:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Multiclass Star Classification")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
