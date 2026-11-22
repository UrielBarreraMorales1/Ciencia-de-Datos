from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ARCHIVO_DATOS, PROCESSED_DIR
from src.data.make_dataset import cargar_datos
from src.data.preprocess import preparar_datos


def probar_todo():
  assert ARCHIVO_DATOS.exists(), "No está wdbc.data en data/raw/"
  df = cargar_datos()
  assert df.shape == (569, 32)
  assert df.isnull().sum().sum() == 0

  X_train, X_test, y_train, y_test = preparar_datos(df)
  assert len(X_train) + len(X_test) == 569
  assert "id" not in X_train.columns
  assert (PROCESSED_DIR / "X_train.csv").exists()
  print("\nPrueba rápida pasada sin broncas.")


if __name__ == "__main__":
  probar_todo()