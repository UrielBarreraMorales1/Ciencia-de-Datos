from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import FIGURES_DIR, PROCESSED_DIR, TARGET

sns.set_style("whitegrid")


def graficas_eda(df):
  plt.figure(figsize=(6, 4))
  sns.countplot(x=TARGET, data=df)
  plt.title("Cantidad de casos por diagnóstico (B / M)")
  plt.xlabel("Diagnóstico")
  plt.ylabel("Cantidad")
  plt.tight_layout()
  plt.savefig(FIGURES_DIR / "clases.png", dpi=300)
  plt.close()

  cols_mean = [c for c in df.columns if c.endswith("_mean")]
  if cols_mean:
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[cols_mean].corr(), cmap="coolwarm", center=0)
    plt.title("Correlación entre variables _mean")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlacion_mean.png", dpi=300)
    plt.close()

  vars_ejemplo = [
      "radius_mean",
      "texture_mean",
      "area_mean",
      "concavity_mean",
  ]
  vars_existentes = [v for v in vars_ejemplo if v in df.columns]
  if vars_existentes:
    df_largo = df[[TARGET] + vars_existentes].melt(
        id_vars=TARGET, var_name="variable", value_name="valor"
    )
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df_largo, x="variable", y="valor", hue=TARGET)
    plt.title("Comparación de variables por diagnóstico")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplots.png", dpi=300)
    plt.close()

  print("Gráficas de EDA guardadas en:", FIGURES_DIR)


def graficas_evaluacion(modelos, X_test, y_test):
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))
  for ax, (nombre, modelo) in zip(axes.ravel(), modelos.items()):
    ConfusionMatrixDisplay.from_estimator(
        modelo,
        X_test,
        y_test,
        ax=ax,
        colorbar=False,
        display_labels=["Benigno (0)", "Maligno (1)"],
    )
    ax.set_title(nombre)
  plt.tight_layout()
  plt.savefig(FIGURES_DIR / "matrices_confusion.png", dpi=300)
  plt.close()

  plt.figure(figsize=(7, 5))
  ax = plt.gca()
  for nombre, modelo in modelos.items():
    RocCurveDisplay.from_estimator(modelo, X_test, y_test, ax=ax, name=nombre)
  plt.plot([0, 1], [0, 1], "k--", label="Azar")
  plt.title("Curvas ROC en prueba")
  plt.tight_layout()
  plt.savefig(FIGURES_DIR / "curvas_roc.png", dpi=300)
  plt.close()

  print("Gráficas de evaluación guardadas en:", FIGURES_DIR)


if __name__ == "__main__":
  ruta_limpio = PROCESSED_DIR / "wdbc_clean.csv"
  if ruta_limpio.exists():
    graficas_eda(pd.read_csv(ruta_limpio))
  else:
    print("Primero corre make_dataset.py para crear wdbc_clean.csv")