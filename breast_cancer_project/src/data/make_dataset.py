from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ARCHIVO_DATOS, PROCESSED_DIR, TARGET

MEDIDAS_BASE = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

COLUMNAS = (
    ["id", TARGET]
    + [f"{m}_mean" for m in MEDIDAS_BASE]
    + [f"{m}_se" for m in MEDIDAS_BASE]
    + [f"{m}_worst" for m in MEDIDAS_BASE]
)


def cargar_datos(ruta=ARCHIVO_DATOS):
  print("Leyendo archivo:", ruta)
  df = pd.read_csv(ruta, header=None, names=COLUMNAS)

  print("Filas y columnas:", df.shape)
  print("\nConteo de clases:")
  print(df[TARGET].value_counts())
  print("\nValores nulos:", df.isnull().sum().sum())
  print("Duplicados:", df.duplicated().sum())
  return df


def guardar_limpio(df, nombre="wdbc_clean.csv"):
  salida = PROCESSED_DIR / nombre
  df.to_csv(salida, index=False)
  print("Guardado en:", salida)
  return salida


if __name__ == "__main__":
  datos = cargar_datos()
  guardar_limpio(datos)