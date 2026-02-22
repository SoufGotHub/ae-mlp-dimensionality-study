# src/evaluate.py
"""
Évaluation des modèles de classification (MNIST).

Fonctionnalités :
- Accuracy, Precision, Recall, F1-score
- Matrice de confusion
- Affichage clair pour le rapport

Utilisable pour :
- Baseline (images brutes)
- AE → Classifieur (espace latent)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcule les métriques principales.

    Args:
        y_true: labels réels
        y_pred: labels prédits

    Returns:
        dictionnaire de métriques
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    return metrics


def print_metrics(metrics: dict, title: str = "Résultats"):
    """
    Affiche les métriques de manière lisible.
    """
    print(f"\n=== {title} ===")
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Precision (avg) : {metrics['precision_macro']:.4f}")
    print(f"Recall (avg)    : {metrics['recall_macro']:.4f}")
    print(f"F1-score (avg)  : {metrics['f1_macro']:.4f}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names=None,
    normalize: bool = False,
    title: str = "Matrice de confusion"
):
    """
    Affiche la matrice de confusion.

    Args:
        y_true, y_pred: labels
        class_names: liste des noms de classes
        normalize: normaliser ou non
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues", values_format=".2f" if normalize else "d")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names=None,
    show_confusion: bool = True,
    normalize_cm: bool = False,
    title: str = "Évaluation du modèle"
) -> dict:
    """
    Évaluation complète d’un modèle.

    Returns:
        dictionnaire des métriques
    """
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, title=title)

    if show_confusion:
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names=class_names,
            normalize=normalize_cm,
            title=title
        )

    return metrics


if __name__ == "__main__":
    # Test rapide avec données fictives
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 2, 1, 0])

    evaluate_model(
        y_true,
        y_pred,
        class_names=[str(i) for i in range(3)],
        normalize_cm=True,
        title="Test rapide"
    )
