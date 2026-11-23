from pathlib import Path
import sys
from sklearn.base import clone
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import FOLDS, MODELS_DIR, PROCESSED_DIR, SEMILLA
from src.visualization.plots import graficas_evaluacion

NOMBRES_MODELOS = ["baseline_logreg", "naive_bayes", "svm_model", "mlp_model"]


def evaluar_modelos():
  X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
  y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
  X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
  y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

  modelos = {}
  tabla = []
  cv_extra = StratifiedKFold(
      n_splits=FOLDS, shuffle=True, random_state=SEMILLA + 10
  )

  for nombre in NOMBRES_MODELOS:
    ruta_mod = MODELS_DIR / f"{nombre}.joblib"
    modelo = joblib.load(ruta_mod)
    modelos[nombre] = modelo

    pred = modelo.predict(X_test)
    prob = modelo.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    scores_cv = cross_val_score(
        clone(modelo), X_train, y_train, cv=cv_extra, scoring="f1"
    )

    tabla.append({
        "modelo": nombre,
        "acc": round(accuracy_score(y_test, pred), 4),
        "prec": round(precision_score(y_test, pred), 4),
        "rec": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "f1_cv_mean": round(float(np.mean(scores_cv)), 4),
        "f1_cv_std": round(float(np.std(scores_cv)), 4),
        "FP": int(fp),
        "FN": int(fn),
    })

  df_res = pd.DataFrame(tabla)
  print("\n--- RESULTADOS EN PRUEBA Y VARIABILIDAD ---")
  print(df_res.to_string(index=False))

  print("\n--- ANALISIS DE ERRORES ---")
  for nombre, modelo in modelos.items():
    pred = modelo.predict(X_test)
    prob = modelo.predict_proba(X_test)[:, 1]
    idx_err = np.where(pred != y_test.values)[0]
    print(f"\n{nombre} -> errores totales: {len(idx_err)}")
    if len(idx_err) > 0:
      df_err = pd.DataFrame({
          "fila_test": idx_err,
          "real": y_test.values[idx_err],
          "pred": pred[idx_err],
          "prob_M": np.round(prob[idx_err], 3),
      })
      print(df_err.to_string(index=False))

  graficas_evaluacion(modelos, X_test, y_test)
  return df_res


if __name__ == "__main__":
  evaluar_modelos()