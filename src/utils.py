# src/utils.py
"""
Fonctions utilitaires pour le projet :
- Reproductibilité (seed)
- Mesure du temps
- Affichage des courbes d'entraînement
"""

import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# =========================
# Reproductibilité
# =========================
def set_seed(seed: int = 42):
    """
    Fixe les seeds pour avoir des résultats reproductibles.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# Mesure du temps
# =========================
class Timer:
    """
    Timer simple pour mesurer le temps d'exécution.

    Usage :
        t = Timer()
        t.start()
        ...
        elapsed = t.stop()
    """
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Le timer n'a pas été démarré.")
        elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return elapsed


# =========================
# Visualisation
# =========================
def plot_history(history, title: str = "Entraînement"):
    """
    Affiche les courbes loss / accuracy à partir d'un history Keras.

    Args:
        history: objet retourné par model.fit()
        title: titre du graphique
    """
    if history is None:
        print("Aucun historique à afficher.")
        return

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("loss", []), label="Train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Accuracy (si disponible)
    if "accuracy" in history.history or "val_accuracy" in history.history:
        plt.subplot(1, 2, 2)
        if "accuracy" in history.history:
            plt.plot(history.history["accuracy"], label="Train acc")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="Val acc")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test rapide
    set_seed(123)
    timer = Timer()
    timer.start()
    time.sleep(0.5)
    print(f"Temps mesuré : {timer.stop():.3f} s")
