"""
Step 3: Train a NEURAL NETWORK that predicts pest_risk (Low / Medium / High).

We compare 3 models:
  - Dummy baseline : always guesses the most common label (the "no AI" score)
  - Neural net     : 1 hidden layer of 32 neurons
  - Deep neural net: 3 hidden layers (64 -> 32 -> 16 neurons)
Then the best model is re-tested on 3 sites it never saw (one per continent).
All use scikit-learn's MLPClassifier (MLP = multi-layer perceptron).

Outputs: results/results.csv, results/confusion_<model>.png,
         models/best_model.joblib
Run:     python train.py
"""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
os.makedirs(os.path.join(HERE, "models"), exist_ok=True)

# 1. Load data and pick input columns (X) and the answer column (y)
data = pd.read_csv(os.path.join(HERE, "data", "pest_data.csv"))
TEXT_COLS = ["crop", "crop_stage"]          # location-free, so it works anywhere
NUMBER_COLS = ["temp", "humidity", "rainfall"]
X = data[TEXT_COLS + NUMBER_COLS]
y = data["pest_risk"].to_numpy(dtype=object)  # plain Python strings

# 2. Stratified split: 80% train, 20% test, same Low/Medium/High mix in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Text -> 0/1 columns (one-hot). Numbers -> same scale (neural nets need this)
prepare = ColumnTransformer([
    ("text", OneHotEncoder(handle_unknown="ignore"), TEXT_COLS),
    ("numbers", StandardScaler(), NUMBER_COLS),
])

MODELS = {
    "Dummy baseline": DummyClassifier(strategy="most_frequent"),
    # early_stopping keeps 10% of training data aside and stops when it stops improving
    "Neural net": MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000,
                                early_stopping=True, random_state=42),
    "Deep neural net": MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=2000,
                                     early_stopping=True, random_state=42),
}

results = []
best_f1, best_name, best_model = -1, None, None
for name, model in MODELS.items():
    pipe = Pipeline([("prepare", prepare), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")  # macro = every class counts equally
    print(f"{name:20s} accuracy={acc:.3f}  macro-F1={f1:.3f}")
    results.append({"model": name, "accuracy": round(acc, 3), "macro_f1": round(f1, 3)})

    # Confusion matrix picture: rows = true label, columns = predicted label
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=["Low", "Medium", "High"])
    plt.title(name)
    plt.savefig(os.path.join(HERE, "results", f"confusion_{name.replace(' ', '_')}.png"), dpi=100)
    plt.close()

    if f1 > best_f1:
        best_f1, best_name, best_model = f1, name, pipe

# 4. Does it work in places it has NEVER seen? Train the best model without
#    3 sites (one per continent) and test only on those sites.
UNSEEN = ["Can Tho", "Lubbock", "Soroti"]
is_unseen = data["site"].isin(UNSEEN).to_numpy()
unseen_pipe = Pipeline([("prepare", prepare), ("model", MODELS[best_name])])
unseen_pipe.fit(X[~is_unseen], y[~is_unseen])
pred = unseen_pipe.predict(X[is_unseen])
acc = accuracy_score(y[is_unseen], pred)
f1 = f1_score(y[is_unseen], pred, average="macro")
print(f"{best_name} on unseen sites {UNSEEN}: accuracy={acc:.3f}  macro-F1={f1:.3f}")
results.append({"model": f"{best_name} (unseen sites: {', '.join(UNSEEN)})",
                "accuracy": round(acc, 3), "macro_f1": round(f1, 3)})

pd.DataFrame(results).to_csv(os.path.join(HERE, "results", "results.csv"), index=False)
joblib.dump(best_model, os.path.join(HERE, "models", "best_model.joblib"))
print(f"Best model: {best_name} (macro-F1={best_f1:.3f}) saved to models/best_model.joblib")
