# Clasificacion de tumores mamarios (Breast Cancer Wisconsin)

Proyecto de clasificacion binaria (Benigno `B` vs. Maligno `M`) sobre el dataset *Breast Cancer Wisconsin (Diagnostic)* (569 registros y 30 variables numericas).

Se prueba regresion logistica como base y se comparan *Gaussian Naive Bayes*, SVM (lineal y RBF) y un perceptron multicapa con validacion cruzada estratificada y evaluacion en prueba.

## Estructura del proyecto

```text
breast_cancer_project/
├── data/
│   ├── external/
│   │   └── wdbc.names
│   ├── processed/
│   └── raw/
│       └── wdbc.data
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── visualization/
│       └── plots.py
├── test/
│   └── test_pipeline.py
├── config.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Como ejecutarlo en Visual Studio Code

1. Abre la carpeta `breast_cancer_project` en **Visual Studio Code**.
2. Selecciona el interprete de Python de tu entorno virtual (`Ctrl+Shift+P` > `Python: Select Interpreter` > `.\.venv\Scripts\python.exe`).
3. Abre la terminal integrada de Visual Studio Code (`Ctrl + n` o `Terminal` > `New Terminal`) y verifica que aparezca `(.venv)`.
4. Instala las dependencias si aun no lo has hecho:

```text
python -m pip install -r requirements.txt
```

5. Corre los scripts en orden desde la terminal de Visual Studio Code:

```text
python test\test_pipeline.py
python src\data\make_dataset.py
python src\data\preprocess.py
python src\visualization\plots.py
python src\models\train.py
python src\models\evaluate.py
```

## Notas del experimento

- Se usa semilla fija `42` en la division de datos, validacion cruzada y modelos.
- El escalado (`StandardScaler`) va dentro de cada `Pipeline` para ajustarse solo en entrenamiento.
- En `reports/figures/` se guardan las graficas del analisis exploratorio y de evaluacion, y en `models/` los modelos entrenados.
