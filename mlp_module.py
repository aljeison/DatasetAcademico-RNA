
from typing import Dict, Any
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def train_mlp(X_tr, X_te, y_tr, y_te) -> Dict[str, Any]:
    """
    Entrena una Red Neuronal (MLP) con normalización (StandardScaler)
    y retorna métricas, matriz de confusión y arquitectura detallada.

    Arquitectura: Entrada(n_features) -> Oculta1(16) -> Oculta2(8) -> Salida(1)
    """
    n_features = int(X_tr.shape[1])

    # Pipeline: StandardScaler -> MLPClassifier
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
        )),
    ])

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # Métricas
    acc = accuracy_score(y_te, y_pred)
    err = 1.0 - acc
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Acceso al estimador interno para shapes de pesos y sesgos
    mlp = model.named_steps["mlp"]

    architecture = {
        "input_neurons": n_features,
        "hidden_layers": [16, 8],
        "output_neurons": 1,
        "weights_shapes": [list(w.shape) for w in mlp.coefs_],
        "bias_shapes": [list(b.shape) for b in mlp.intercepts_],
        "activation": mlp.activation,
        "solver": mlp.solver,
        "max_iter": mlp.max_iter,
        "normalized": True,
    }

    return {
        "architecture": architecture,
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
    }
