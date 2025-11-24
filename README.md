
# 🎓 DatasetAcadémico – Regresión Logística de Aprobación

Aplicación web sencilla en **Flask** que genera un **dataset académico sintético**, define una variable binaria **Aprobado / No Aprobado** y entrena una **regresión logística** para clasificar estudiantes según su desempeño.

El objetivo es servir como ejemplo práctico para el **Punto 1 (Regresión Logística)** del taller de Machine Learning.

---

## 🧠 Idea del proyecto

Se simulan N estudiantes (por defecto 10.000).  
Cada fila del dataset representa un estudiante, con variables numéricas que podrían existir en un sistema académico real.

### Variables de entrada (X)

Todas **numéricas**:

- `PromedioAcumulado` – promedio histórico del estudiante (escala 0–5).
- `AsistenciaPct` – porcentaje de asistencia a clase (50–100 %).
- `HorasEstudioSem` – horas de estudio a la semana (0–25).
- `TareasEntregadasPct` – porcentaje de tareas entregadas (30–100 %).
- `Parcial1` – nota del primer parcial (0–5).
- `Parcial2` – nota del segundo parcial (0–5).
- `DificultadMateria` – nivel de dificultad (1–5).
- `IntentosReprobados` – número de veces que ha reprobado la materia (0–2).

### Variables de salida (Y)

- `PromedioFinal` – nota final de la materia (0–5), calculada con una **fórmula fija** que combina parciales, tareas, asistencia y penalizaciones por dificultad e intentos reprobados.
- `Aprobado` – **variable binaria**:
  - `1` si `PromedioFinal ≥ 3.0`
  - `0` si `PromedioFinal < 3.0`

La regresión logística se entrena para predecir `Aprobado` a partir de las variables X.

---

## 🧩 Flujo del modelo

1. **Generación / carga del dataset**
   - Si no existe `dataset_notas.csv`, se genera automáticamente con el tamaño solicitado.
   - Si existe pero está incompleto o con columnas distintas, se vuelve a generar.

2. **Preparación de datos**
   - Se separan:
     - `X` = columnas de entrada (`PromedioAcumulado`, `AsistenciaPct`, …, `IntentosReprobados`)
     - `y` = columna objetivo binaria (`Aprobado`)

3. **Entrenamiento de la regresión logística**
   - Se divide en **train (80%)** y **test (20%)** con `train_test_split`.
   - Se entrena un modelo `LogisticRegression` de `scikit-learn`.

4. **Evaluación**
   - Métricas sobre el conjunto de prueba:
     - **Accuracy** (exactitud),
     - **Error rate** (1 − accuracy),
     - **Precisión (precision)**,
     - **Recall (exhaustividad)**,
     - **F1-score**.
   - Se construye la **matriz de confusión** con:
     - Verdaderos negativos (TN)
     - Falsos positivos (FP)
     - Falsos negativos (FN)
     - Verdaderos positivos (TP)

5. **Interfaz web (Flask)**
   - Permite:
     - Definir el tamaño del dataset.
     - Forzar la recreación del CSV.
     - Ver un resumen del dataset (filas, columnas, X y Y).
     - Visualizar las métricas y la matriz de confusión.
     - Ver una **vista previa** de las tablas X e Y.
     - Descargar:
       - `dataset_notas.csv`
       - `resultados.json` con toda la información del experimento.

---

## 🗂️ Estructura del proyecto

```text
DatasetAcademico/
├─ app.py                  # Backend Flask + generación de dataset + regresión logística
├─ dataset_notas.csv       # Dataset generado (se crea automáticamente si no existe)
├─ requirements.txt        # Dependencias del proyecto
├─ templates/
│   └─ index.html          # Plantilla principal (frontend)
└─ static/
    ├─ style.css           # Estilos (tema oscuro)
    └─ app.js              # Lógica del lado del cliente (llamadas a /start y render de tablas)
```

---

## 💻 Cómo ejecutar el proyecto localmente

Recomendado usar **Python 3.11**.

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/Drownfe/DatasetAcademico.git
cd DatasetAcademico
```

### 2️⃣ Crear y activar entorno virtual (Windows)

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

En Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Este proyecto utiliza principalmente:

- `Flask`
- `pandas`
- `numpy`
- `scikit-learn`

### 4️⃣ Ejecutar la aplicación

```bash
python app.py
```

Deberías ver algo como:

```text
* Running on http://127.0.0.1:5000
```

Abre el navegador en:

```text
http://127.0.0.1:5000
```

---

## 🧪 Uso de la interfaz

1. **Tamaño del dataset**  
   - Campo numérico (ej. 5000, 10000, 20000).  
   - Mientras más grande, más ejemplos para entrenar la regresión logística.

2. **Re-crear dataset**  
   - Si marcas la casilla, se ignora el CSV actual y se genera uno nuevo con el tamaño indicado.

3. **Botón “Empezar”**  
   - Lanza el pipeline:
     1. Generación / carga del dataset.
     2. Separación X / Y.
     3. Entrenamiento de la regresión logística.
     4. Cálculo de métricas y matriz de confusión.
     5. Renderizado del resumen y tablas.

4. **Dataset CSV**  
   - Descarga `dataset_notas.csv` con todas las filas del dataset.

5. **Resultados JSON**  
   - Descarga `resultados.json` con:
     - `dataset_info`
     - `logistic.metrics`
     - `logistic.confusion_matrix`
     - `preview_X` y `preview_Y`

---

## 📈 Cómo interpretar los resultados (para el informe)

- **Accuracy**  
  Proporción de predicciones correctas sobre el conjunto de prueba.  
  Ejemplo: `0.93` → el modelo acierta el 93 % de los casos.

- **Error rate**  
  Complemento del accuracy: `1 − accuracy`.  
  Ejemplo: `0.07` → el modelo se equivoca en el 7 % de los casos.

- **Precisión**  
  De todos los estudiantes que el modelo predijo como “Aprobado”, ¿qué porcentaje realmente aprueba?

- **Recall**  
  De todos los estudiantes que realmente aprueban, ¿qué porcentaje detecta el modelo?

- **F1-score**  
  Media armónica entre precisión y recall.  
  Útil cuando nos interesa equilibrar ambos.

- **Matriz de confusión**  

  |                      | Pred. No aprueba (0) | Pred. Aprueba (1) |
  |----------------------|----------------------|-------------------|
  | **Real: No aprueba** | TN                  | FP                |
  | **Real: Aprueba**    | FN                  | TP                |

  - TN: reprobados clasificados correctamente como reprobados.  
  - FP: reprobados clasificados incorrectamente como aprobados.  
  - FN: aprobados clasificados como reprobados.  
  - TP: aprobados clasificados correctamente como aprobados.

Estos valores son los que se suelen reportar en el documento del taller.

---

## 📚 Posibles extensiones (para el compañero o versiones futuras)

- Entrenar una **Red Neuronal** utilizando el mismo dataset (para el Punto 2 del taller).
- Añadir curvas ROC / AUC u otras métricas.
- Probar con distintos umbrales de aprobación (ej. 2.5, 3.5) y comparar resultados.
- Incluir análisis de importancia de variables a partir de los coeficientes de la regresión logística.

---

## 👨‍🏫 Uso en el taller

Este proyecto está pensado para:

- Mostrar un ejemplo completo de **clasificación binaria con regresión logística**.
- Trabajar con un **dataset grande y coherente**, aunque sea sintético.
- Tener una interfaz clara donde se vean:
  - Entradas X
  - Salidas Y
  - Métricas
  - Matriz de confusión

Se puede usar tanto para la **exposición en clase** como para dejar el repositorio en GitHub como evidencia del trabajo.
