
import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file
from mlp_module import train_mlp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Sirve estáticos desde ./static y plantillas desde ./templates
app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------------- Configuración básica ----------------------
CSV_PATH = "dataset_notas.csv"
X_COLS = [
    "PromedioAcumulado",
    "AsistenciaPct",
    "HorasEstudioSem",
    "TareasEntregadasPct",
    "Parcial1",
    "Parcial2",
    "DificultadMateria",
    "IntentosReprobados",
]
Y_REG_COL = "PromedioFinal"  # para construir Aprobado
Y_BIN_COL = "Aprobado"       # 1 si PromedioFinal >= 3.0, 0 en caso contrario
np.random.seed(42)

# -------------------- Generación del dataset ----------------------
def _calcular_promedio_final(row):
    """Promedio final (0–5) a partir de parciales, tareas y asistencia."""
    p1 = row["Parcial1"]
    p2 = row["Parcial2"]
    tareas = 0.5 * (row["HorasEstudioSem"] / 20) * 5
    asistencia = (row["AsistenciaPct"] / 100) * 5
    base = 0.30 * p1 + 0.40 * p2 + 0.15 * tareas + 0.15 * asistencia
    dificultad = row["DificultadMateria"]
    penal_dif = (dificultad - 3) * 0.15
    penal_reprob = row["IntentosReprobados"] * 0.10
    ruido = np.random.normal(0, 0.15)
    nota = base - penal_dif - penal_reprob + ruido
    return float(np.clip(nota, 1.0, 5.0))

def _crear_dataset(path=CSV_PATH, n=10000):
    """Genera SIEMPRE un dataset sintético de estudiantes con n filas."""
    prom_acum = np.clip(np.random.normal(3.6, 0.5, n), 2.0, 5.0)
    asistencia_pct = np.clip(np.random.normal(85, 10, n), 50, 100)
    horas_estudio = np.clip(np.random.normal(10, 4, n), 0, 25)
    tareas_pct = np.clip(np.random.normal(80, 15, n), 30, 100)
    parcial1 = np.clip(np.random.normal(3.6, 0.7, n), 1.0, 5.0)
    parcial2 = np.clip(np.random.normal(3.7, 0.7, n), 1.0, 5.0)
    dificultad = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    reprob_prev = np.random.choice([0, 1, 2], size=n, p=[0.75, 0.2, 0.05])

    df = pd.DataFrame({
        "PromedioAcumulado": np.round(prom_acum, 2),
        "AsistenciaPct": np.round(asistencia_pct, 1),
        "HorasEstudioSem": np.round(horas_estudio, 1),
        "TareasEntregadasPct": np.round(tareas_pct, 1),
        "Parcial1": np.round(parcial1, 2),
        "Parcial2": np.round(parcial2, 2),
        "DificultadMateria": dificultad.astype(int),
        "IntentosReprobados": reprob_prev.astype(int),
    })
    # Promedio final continuo (0–5)
    df[Y_REG_COL] = df.apply(_calcular_promedio_final, axis=1)
    # Variable binaria: Aprobado (1) / No aprobado (0)
    df[Y_BIN_COL] = (df[Y_REG_COL] >= 3.0).astype(int)
    df.to_csv(path, index=False)
    return df

def _ensure_dataset(path=CSV_PATH, n=10000, force=False):
    """
    Crea el dataset si no existe, si hay error de lectura, si no tiene
    las columnas requeridas o si el usuario fuerza la recreación.
    """
    if force or (not os.path.exists(path)):
        return _crear_dataset(path, n)
    try:
        df = pd.read_csv(path)
    except Exception:
        return _crear_dataset(path, n)

    required = set(X_COLS + [Y_REG_COL, Y_BIN_COL])
    if not required.issubset(df.columns) or len(df) < n:
        return _crear_dataset(path, n)
    return df

# ---------------- Pipeline con Logística + RNA (MLP) ---------------
def pipeline(n=10000, force=False):
    # 1) Cargar / generar dataset
    df = _ensure_dataset(CSV_PATH, n=n, force=force)

    # 2) Separar X (entradas) e Y (salida binaria)
    X = df[X_COLS].copy()
    y = df[Y_BIN_COL].copy()

    # 3) Entrenar / validar (train/test split)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Modelo 1: Regresión Logística ---
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    err = 1.0 - acc
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    cm_log = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_log.ravel()

    # --- Modelo 2: Red Neuronal Artificial (MLP) ---
    mlp_result = train_mlp(X_tr, X_te, y_tr, y_te)

    # --- Vista previa X / Y ---
    preview_X = X.head(10).round(2).to_dict(orient="records")
    preview_Y = df[[Y_BIN_COL, Y_REG_COL]].head(10).round(2).to_dict(orient="records")

    return {
        "dataset_info": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "X_cols": X_COLS,
            "Y_cols": [Y_BIN_COL, Y_REG_COL],
            "note": (
                "PromedioFinal está en escala 0–5. Aprobado=1 si PromedioFinal ≥ 3.0; "
                "Aprobado=0 en caso contrario."
            ),
        },
        "logistic": {
            "metrics": {
                "accuracy": float(acc),
                "error_rate": float(err),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            },
            "confusion_matrix": [
                [int(tn), int(fp)],
                [int(fn), int(tp)],
            ],
            "labels": ["No Aprobado (0)", "Aprobado (1)"],
        },
        "mlp": mlp_result,
        "preview_X": preview_X,
        "preview_Y": preview_Y,
    }

# ---------------------------- Rutas Flask ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start")
def start():
    try:
        n = int(request.args.get("n", 10000))
        force = request.args.get("force", "0") in ("1", "true", "True")
        res = pipeline(n=n, force=force)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/download/dataset")
def download_dataset():
    n = int(request.args.get("n", 10000))
    force = request.args.get("force", "0") in ("1", "true", "True")
    _ensure_dataset(CSV_PATH, n=n, force=force)
    return send_file(CSV_PATH, as_attachment=True)

@app.route("/download/results")
def download_results():
    n = int(request.args.get("n", 10000))
    force = request.args.get("force", "0") in ("1", "true", "True")
    res = pipeline(n=n, force=force)
    buf = io.BytesIO(json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"))
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="resultados.json",
        mimetype="application/json",
    )

@app.route("/start_mlp")
def start_mlp():
    try:
        n = int(request.args.get("n", 10000))
        force = request.args.get("force", "0") in ("1", "true", "True")
        df = _ensure_dataset(CSV_PATH, n=n, force=force)
        X = df[X_COLS].copy()
        y = df[Y_BIN_COL].copy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        mlp_result = train_mlp(X_tr, X_te, y_tr, y_te)
        preview_X = X.head(10).round(2).to_dict(orient="records")
        preview_Y = df[[Y_BIN_COL, Y_REG_COL]].head(10).round(2).to_dict(orient="records")
        res = {
            "dataset_info": {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "X_cols": X_COLS,
                "Y_cols": [Y_BIN_COL, Y_REG_COL],
                "note": (
                    "PromedioFinal está en escala 0–5. Aprobado=1 si PromedioFinal ≥ 3.0; "
                    "Aprobado=0 en caso contrario."
                ),
            },
            "mlp": mlp_result,
            "preview_X": preview_X,
            "preview_Y": preview_Y,
        }
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/download/mlp_results")
def download_mlp_results():
    n = int(request.args.get("n", 10000))
    force = request.args.get("force", "0") in ("1", "true", "True")
    df = _ensure_dataset(CSV_PATH, n=n, force=force)
    X = df[X_COLS].copy()
    y = df[Y_BIN_COL].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mlp_result = train_mlp(X_tr, X_te, y_tr, y_te)
    preview_X = X.head(10).round(2).to_dict(orient="records")
    preview_Y = df[[Y_BIN_COL, Y_REG_COL]].head(10).round(2).to_dict(orient="records")
    res = {
        "dataset_info": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "X_cols": X_COLS,
            "Y_cols": [Y_BIN_COL, Y_REG_COL],
            "note": (
                "PromedioFinal está en escala 0–5. Aprobado=1 si PromedioFinal ≥ 3.0; "
                "Aprobado=0 en caso contrario."
            ),
        },
        "mlp": mlp_result,
        "preview_X": preview_X,
        "preview_Y": preview_Y,
    }
    buf = io.BytesIO(json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"))
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="mlp_resultados.json",
        mimetype="application/json",
    )

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
