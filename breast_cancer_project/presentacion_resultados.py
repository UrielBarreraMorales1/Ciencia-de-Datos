"""
Script independiente de presentación (EDA + Resultados de Modelos).
Lee los datos procesados y los modelos entrenados (.joblib) y muestra
las tablas y gráficas en pantalla (plt.show) sin guardar ni sobrescribir archivos.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Estilo visual limpio para presentación
sns.set_theme(style="whitegrid", context="talk")

BASE_DIR = Path(__file__).resolve().parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def cargar_todo():
    df_clean = pd.read_csv(DATA_PROCESSED / "wdbc_clean.csv")
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()

    modelos = {
        "Regresión Logística": joblib.load(MODELS_DIR / "baseline_logreg.joblib"),
        "Gaussian Naive Bayes": joblib.load(MODELS_DIR / "naive_bayes.joblib"),
        "SVM (Lineal / RBF)": joblib.load(MODELS_DIR / "svm_model.joblib"),
        "Red Neuronal (MLP)": joblib.load(MODELS_DIR / "mlp_model.joblib"),
    }
    return df_clean, X_train, X_test, y_train, y_test, modelos


def _resumen_hiperparametros(modelo):
    if hasattr(modelo, "best_params_"):
        return modelo.best_params_
    if hasattr(modelo, "named_steps"):
        ultimo_paso = list(modelo.named_steps.values())[-1]
        return ultimo_paso.get_params()
    return modelo.get_params()


def mostrar_eda(df_clean):
    print("\n" + "=" * 70)
    print("1. RESUMEN EXPLORATORIO DEL DATASET (WDBC)")
    print("=" * 70)
    print(f"Total de muestras: {df_clean.shape[0]} | Variables: {df_clean.shape[1]}")
    print(f"Valores nulos: {df_clean.isna().sum().sum()} | Duplicados: {df_clean.duplicated().sum()}")
    conteo = df_clean["diagnosis"].value_counts()
    for clase, n in conteo.items():
        etiqueta = "Benigno (B)" if clase == "B" else "Maligno (M)"
        print(f"  - {etiqueta}: {n} ({n / len(df_clean) * 100:.2f}%)")

    # Figura 1: Distribución de clases + Boxplots clave
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    sns.countplot(
        data=df_clean,
        x="diagnosis",
        order=["B", "M"],
        palette=["#2E86AB", "#D64550"],
        ax=axes[0],
    )
    axes[0].set_title("Distribución de Diagnósticos")
    axes[0].set_xlabel("Diagnóstico (B = Benigno, M = Maligno)")
    axes[0].set_ylabel("Cantidad de muestras")
    for p in axes[0].patches:
        axes[0].annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    sns.boxplot(
        data=df_clean,
        x="diagnosis",
        y="radius_mean",
        order=["B", "M"],
        palette=["#2E86AB", "#D64550"],
        ax=axes[1],
    )
    axes[1].set_title("Radio Promedio por Clase")
    axes[1].set_xlabel("Diagnóstico")
    axes[1].set_ylabel("radius_mean")

    sns.boxplot(
        data=df_clean,
        x="diagnosis",
        y="concave_points_mean",
        order=["B", "M"],
        palette=["#2E86AB", "#D64550"],
        ax=axes[2],
    )
    axes[2].set_title("Puntos Cóncavos Promedio")
    axes[2].set_xlabel("Diagnóstico")
    axes[2].set_ylabel("concave_points_mean")

    plt.tight_layout()
    plt.show()

    # Figura 2: Mapa de calor de correlaciones (_mean)
    cols_mean = [c for c in df_clean.columns if c.endswith("_mean")]
    corr = df_clean[cols_mean].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        annot_kws={"size": 8},
    )
    plt.title("Matriz de Correlación — Características Promedio (_mean)")
    plt.tight_layout()
    plt.show()


def evaluar_y_presentar(X_train, X_test, y_train, y_test, modelos):
    print("\n" + "=" * 70)
    print("2. COMPARACIÓN DE MODELOS EN CONJUNTO DE PRUEBA (N = 114)")
    print("=" * 70)

    cv_alt = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    filas = []
    predicciones = {}
    probabilidades = {}

    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        predicciones[nombre] = y_pred
        probabilidades[nombre] = y_prob

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv_alt, scoring="f1")

        filas.append(
            {
                "Modelo": nombre,
                "Exactitud": accuracy_score(y_test, y_pred),
                "Precisión": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
                "F1 CV (media±std)": f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
                "FP": fp,
                "FN": fn,
            }
        )

    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nParámetros / hiperparámetros de los modelos cargados:")
    for nombre, modelo in modelos.items():
        print(f"  - {nombre}: {_resumen_hiperparametros(modelo)}")

    # Figura 3: Gráfica de barras comparativa de métricas en prueba
    metricas_plot = tabla.melt(
        id_vars="Modelo",
        value_vars=["Exactitud", "Precisión", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Métrica",
        value_name="Valor",
    )
    plt.figure(figsize=(12, 5.5))
    ax = sns.barplot(data=metricas_plot, x="Métrica", y="Valor", hue="Modelo", palette="viridis")
    plt.ylim(0.80, 1.03)
    plt.title("Comparación de Métricas en Conjunto de Prueba")
    plt.legend(loc="lower right", frameon=True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2, rotation=90)
    plt.tight_layout()
    plt.show()

    # Figura 4: Matrices de confusión (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (nombre, y_pred) in zip(axes.ravel(), predicciones.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benigno (0)", "Maligno (1)"],
            yticklabels=["Benigno (0)", "Maligno (1)"],
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(f"Matriz de Confusión — {nombre}")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor Real")

    plt.tight_layout()
    plt.show()

    # Figura 5: Curvas ROC comparativas
    plt.figure(figsize=(8.5, 6.5))
    for nombre, y_prob in probabilidades.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, linewidth=2.2, label=f"{nombre} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Azar (AUC = 0.5000)")
    plt.title("Curvas ROC Comparativas en Conjunto de Prueba")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (Recall / TPR)")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.show()

    # Resumen rápido de errores del mejor modelo (por F1)
    mejor_nombre = tabla.sort_values("F1-Score", ascending=False).iloc[0]["Modelo"]
    y_pred_best = predicciones[mejor_nombre]
    y_prob_best = probabilidades[mejor_nombre]
    errores_idx = np.where(y_pred_best != y_test.values)[0]

    print(f"\n3. ANÁLISIS DE ERRORES DEL MODELO DESTACADO ({mejor_nombre})")
    print("-" * 70)
    if len(errores_idx) == 0:
        print("¡0 errores en el conjunto de prueba!")
    else:
        for idx in errores_idx:
            real = "Maligno (1)" if y_test.iloc[idx] == 1 else "Benigno (0)"
            pred = "Maligno (1)" if y_pred_best[idx] == 1 else "Benigno (0)"
            print(
                f"  - Muestra test #{idx}: Real={real} | Predicho={pred} | "
                f"Prob(Maligno)={y_prob_best[idx]:.4f}"
            )


if __name__ == "__main__":
    df_clean, X_train, X_test, y_train, y_test, modelos = cargar_todo()
    mostrar_eda(df_clean)
    evaluar_y_presentar(X_train, X_test, y_train, y_test, modelos)