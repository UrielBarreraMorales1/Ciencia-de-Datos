from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, SEMILLA, TARGET, TAM_PRUEBA
from src.data.make_dataset import cargar_datos


def preparar_datos(df=None):
  archivo_limpio = PROCESSED_DIR / "wdbc_clean.csv"
  if df is None:
    if archivo_limpio.exists():
      df = pd.read_csv(archivo_limpio)
    else:
      df = cargar_datos()

  df_prep = df.copy()

  # Quitamos id porque no sirve para predecir
  if "id" in df_prep.columns:
    df_prep = df_prep.drop(columns=["id"])

  # Limpiamos espacios y pasamos M -> 1, B -> 0
  limpio = df_prep[TARGET].astype(str).str.strip()
  mapeado = limpio.map({"M": 1, "B": 0})
  if mapeado.isna().any():
    print("Valores encontrados en diagnosis:", limpio.unique())
    raise ValueError("Hay valores en diagnosis distintos de 'M' o 'B'.")
  df_prep[TARGET] = mapeado

  X = df_prep.drop(columns=[TARGET])
  y = df_prep[TARGET].astype(int)

  X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size=TAM_PRUEBA,
      random_state=SEMILLA,
      stratify=y,
  )

  X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
  X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
  y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
  y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

  print("X_train:", X_train.shape, "| X_test:", X_test.shape)
  print("Proporción de malignos en train:", round(y_train.mean(), 3))
  print("Proporción de malignos en test:", round(y_test.mean(), 3))
  return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  preparar_datos()