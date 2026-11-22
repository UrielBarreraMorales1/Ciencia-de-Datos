from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import FOLDS, MODELS_DIR, PROCESSED_DIR, SEMILLA


def entrenar_modelos():
  X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
  y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()

  cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEMILLA)
  modelos_guardados = {}

  pipe_lr = Pipeline([
      ("scaler", StandardScaler()),
      ("modelo", LogisticRegression(max_iter=2000, random_state=SEMILLA)),
  ])
  grid_lr = GridSearchCV(
      pipe_lr, {"modelo__C": [0.01, 0.1, 1.0, 10.0]}, cv=cv, scoring="f1"
  )
  grid_lr.fit(X_train, y_train)
  modelos_guardados["baseline_logreg"] = grid_lr.best_estimator_
  print(
      "LogReg -> mejor F1 CV:",
      round(grid_lr.best_score_, 4),
      "| params:",
      grid_lr.best_params_,
  )

  pipe_nb = Pipeline([
      ("scaler", StandardScaler()),
      ("modelo", GaussianNB()),
  ])
  grid_nb = GridSearchCV(
      pipe_nb,
      {"modelo__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
      cv=cv,
      scoring="f1",
  )
  grid_nb.fit(X_train, y_train)
  modelos_guardados["naive_bayes"] = grid_nb.best_estimator_
  print(
      "Naive Bayes -> mejor F1 CV:",
      round(grid_nb.best_score_, 4),
      "| params:",
      grid_nb.best_params_,
  )

  pipe_svm = Pipeline([
      ("scaler", StandardScaler()),
      ("modelo", SVC(probability=True, random_state=SEMILLA)),
  ])
  rejilla_svm = [
      {"modelo__kernel": ["linear"], "modelo__C": [0.1, 1.0, 10.0]},
      {
          "modelo__kernel": ["rbf"],
          "modelo__C": [0.1, 1.0, 10.0, 100.0],
          "modelo__gamma": ["scale", 0.01, 0.1],
      },
  ]
  grid_svm = GridSearchCV(pipe_svm, rejilla_svm, cv=cv, scoring="f1")
  grid_svm.fit(X_train, y_train)
  modelos_guardados["svm_model"] = grid_svm.best_estimator_
  print(
      "SVM -> mejor F1 CV:",
      round(grid_svm.best_score_, 4),
      "| params:",
      grid_svm.best_params_,
  )

  pipe_mlp = Pipeline([
      ("scaler", StandardScaler()),
      (
          "modelo",
          MLPClassifier(
              hidden_layer_sizes=(32, 16),
              activation="relu",
              early_stopping=True,
              validation_fraction=0.15,
              max_iter=1000,
              random_state=SEMILLA,
          ),
      ),
  ])
  rejilla_mlp = {
      "modelo__alpha": [0.0001, 0.001, 0.01, 0.1],
      "modelo__learning_rate_init": [0.001, 0.005],
  }
  grid_mlp = GridSearchCV(pipe_mlp, rejilla_mlp, cv=cv, scoring="f1")
  grid_mlp.fit(X_train, y_train)
  modelos_guardados["mlp_model"] = grid_mlp.best_estimator_
  print(
      "MLP -> mejor F1 CV:",
      round(grid_mlp.best_score_, 4),
      "| params:",
      grid_mlp.best_params_,
  )

  for nombre, est in modelos_guardados.items():
    joblib.dump(est, MODELS_DIR / f"{nombre}.joblib")

  print("\nModelos guardados en:", MODELS_DIR)
  return modelos_guardados


if __name__ == "__main__":
  entrenar_modelos()