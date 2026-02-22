# src/train_classifier.py
"""
Entraînement et évaluation du classifieur pour MNIST.

Modes :
- baseline : entraînement sur images brutes (aplaties)
- ae       : entraînement sur espace latent (nécessite encoder.h5)
- both     : exécute baseline puis ae et affiche comparaison

Usage :
    python src/train_classifier.py --mode both

Dépendances :
    tensorflow, numpy, scikit-learn, matplotlib
"""
import os
import time
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf

# Imports locaux (fonctionnent si vous exécutez le script depuis src/ ou en faisant python src/...)
from data_loader import load_mnist
from models.classifier import build_classifier, compile_classifier, train_classifier

# Dossier pour figures / métriques
REPORT_DIR = "report_figs"
os.makedirs(REPORT_DIR, exist_ok=True)


# -------------------------
# Helpers pour sauvegarde / plots
# -------------------------
def save_history_npz(history, filename: str):
    """Sauvegarde history.history dans un .npz"""
    if history is None:
        return
    hist = history.history if hasattr(history, "history") else history
    to_save = {k: np.array(v) for k, v in hist.items()}
    path = os.path.join(REPORT_DIR, filename)
    np.savez(path, **to_save)
    print(f"[saved] {path}")


def plot_history(history, title: str, filename: str):
    """Trace et sauvegarde loss / val_loss et accuracy / val_accuracy si présentes."""
    if history is None:
        print(f"[info] pas d'history pour {title}")
        return
    hist = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    loss = hist.get("loss")
    val_loss = hist.get("val_loss")
    if loss is not None:
        axes[0].plot(loss, label="train loss")
    if val_loss is not None:
        axes[0].plot(val_loss, label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    acc = hist.get("accuracy") or hist.get("acc")
    val_acc = hist.get("val_accuracy") or hist.get("val_acc")
    if acc is not None or val_acc is not None:
        if acc is not None:
            axes[1].plot(acc, label="train acc")
        if val_acc is not None:
            axes[1].plot(val_acc, label="val acc")
        axes[1].set_title("Accuracy")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Pas d'accuracy dans history", ha="center")
        axes[1].axis("off")
    fig.suptitle(title)
    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_confusion_save(y_true, y_pred, classes, filename: str, normalize: bool = True):
    """Trace et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, ylabel="Vrai label", xlabel="Label prédit")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.max() != 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.title("Matrice de confusion" + (" (normalisée)" if normalize else ""))
    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_prf_save(y_true, y_pred, classes, filename: str):
    """Trace et sauvegarde barplots Precision/Recall/F1 par classe."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)))
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, p, width, label="Precision")
    ax.bar(x, r, width, label="Recall")
    ax.bar(x + width, f, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 par classe")
    ax.legend()
    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


# placé près des autres helpers (remplace l'ancienne fonction save_metrics_json)
def _make_serializable(obj):
    """
    Convertit récursivement numpy arrays et numpy scalars en types Python sérialisables par JSON.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalars -> python scalars
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    # python builtins pass-through
    return obj


def save_metrics_json(metrics: dict, filename: str):
    """
    Sauvegarde metrics en JSON en s'assurant que tout est JSON-serializable.
    Remplace l'ancienne implémentation qui crashait pour les ndarray.
    """
    path = os.path.join(REPORT_DIR, filename)
    serializable = _make_serializable(metrics)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")



# -------------------------
# Fonctions existantes (légèrement enrichies)
# -------------------------
def evaluate_preds(y_true: np.ndarray, y_pred: np.ndarray):
    """Calcule métriques classiques et retourne dict (garde confusion matriciale)."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "f1_macro": f1, "precision_macro": prec, "recall_macro": rec, "confusion_matrix": cm}


def prepare_baseline_inputs(x):
    """Aplatis les images si nécessaire (ex: (N,28,28,1) -> (N,784))."""
    if x is None:
        return None
    if len(x.shape) == 4:
        return x.reshape((x.shape[0], -1))
    elif len(x.shape) == 3:
        return x.reshape((x.shape[0], -1))
    else:
        return x  # déjà aplati


def extract_latents(encoder: tf.keras.Model, x: np.ndarray, batch_size: int = 128):
    """Utilise l'encoder pour extraire les représentations latentes."""
    z = encoder.predict(x, batch_size=batch_size, verbose=0)
    return z


def run_baseline(x_train, y_train, x_val, y_val, x_test, y_test, epochs=30, batch_size=128, lr=1e-3, save_path=None):
    print("=== Baseline : entraînement sur images brutes ===")
    X_train = prepare_baseline_inputs(x_train)
    X_val = prepare_baseline_inputs(x_val) if x_val is not None else None
    X_test = prepare_baseline_inputs(x_test)

    input_dim = X_train.shape[1]
    model = build_classifier(input_dim=input_dim)
    compile_classifier(model, lr=lr)
    model.summary()

    start = time.perf_counter()
    history = train_classifier(model, X_train, y_train, x_val=X_val, y_val=y_val, epochs=epochs, batch_size=batch_size,
                               callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    train_time = time.perf_counter() - start

    # sauvegarder history + plot si présent
    save_history_npz(history, "baseline_history.npz")
    plot_history(history, "Baseline - entraînement", "baseline_history.png")

    # Évaluer (temps d'inference)
    t0 = time.perf_counter()
    y_proba = model.predict(X_test, batch_size=batch_size, verbose=0)
    t_inf = time.perf_counter() - t0
    y_pred = np.argmax(y_proba, axis=1)

    # calcul métriques + sauvegarde plots + json
    metrics = evaluate_preds(y_test, y_pred)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = t_inf

    # sauvegarder métriques et plots
    save_metrics_json(metrics, "baseline_metrics.json")
    classes = [str(i) for i in range(10)]
    plot_confusion_save(y_test, y_pred, classes, "baseline_confusion.png", normalize=True)
    plot_prf_save(y_test, y_pred, classes, "baseline_prf.png")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Modèle baseline sauvegardé : {save_path}")
    return metrics, history, model


def run_ae_pipeline(encoder_path, x_train, y_train, x_val, y_val, x_test, y_test, epochs=30, batch_size=128, lr=1e-3, save_path=None):
    print("=== Pipeline AE → Classifieur ===")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder non trouvé : {encoder_path} (exécute d'abord train_ae.py)")

    # Charger l'encoder
    print(f"Chargement de l'encoder depuis {encoder_path} ...")
    encoder = tf.keras.models.load_model(encoder_path)

    # Extraire latents
    print("Extraction des latents (train/val/test)...")
    z_train = extract_latents(encoder, x_train, batch_size=batch_size)
    z_val = extract_latents(encoder, x_val, batch_size=batch_size) if x_val is not None else None
    z_test = extract_latents(encoder, x_test, batch_size=batch_size)

    input_dim = z_train.shape[1]
    model = build_classifier(input_dim=input_dim)
    compile_classifier(model, lr=lr)
    model.summary()

    start = time.perf_counter()
    history = train_classifier(model, z_train, y_train, x_val=z_val, y_val=y_val, epochs=epochs, batch_size=batch_size,
                               callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    train_time = time.perf_counter() - start

    # sauvegarder history + plot si présent
    save_history_npz(history, "ae_history.npz")
    plot_history(history, "AE->Classifier - entraînement", "ae_history.png")

    t0 = time.perf_counter()
    y_proba = model.predict(z_test, batch_size=batch_size, verbose=0)
    t_inf = time.perf_counter() - t0
    y_pred = np.argmax(y_proba, axis=1)

    metrics = evaluate_preds(y_test, y_pred)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = t_inf

    # sauvegarder métriques et plots
    save_metrics_json(metrics, "ae_metrics.json")
    classes = [str(i) for i in range(10)]
    plot_confusion_save(y_test, y_pred, classes, "ae_confusion.png", normalize=True)
    plot_prf_save(y_test, y_pred, classes, "ae_prf.png")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Modèle AE→classifier sauvegardé : {save_path}")
    return metrics, history, model


def pretty_print_metrics(title: str, metrics: dict):
    print(f"\n--- {title} ---")
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"F1 (macro)  : {metrics['f1_macro']:.4f}")
    print(f"Precision   : {metrics['precision_macro']:.4f}")
    print(f"Recall      : {metrics['recall_macro']:.4f}")
    print(f"Train time  : {metrics['train_time_s']:.2f} s")
    print(f"Infer time  : {metrics['inference_time_s']:.2f} s")
    print(f"Confusion matrix:\n{metrics['confusion_matrix']}")


def main(args):
    # Charger les données MNIST
    data = load_mnist(normalize=True, flatten=False, val_split=0.1, as_tf_dataset=False)
    x_train, y_train = data["train"]
    x_val, y_val = data.get("val", (None, None))
    x_test, y_test = data["test"]

    # Si mode baseline, on passera par prepare_baseline_inputs (aplatissement)
    results = {}

    if args.mode in ("baseline", "both"):
        metrics_baseline, history_b, model_b = run_baseline(
            x_train, y_train, x_val, y_val, x_test, y_test,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            save_path=args.save_baseline
        )
        pretty_print_metrics("Baseline (brut)", metrics_baseline)
        results["baseline"] = metrics_baseline

    if args.mode in ("ae", "both"):
        metrics_ae, history_a, model_a = run_ae_pipeline(
            encoder_path=args.encoder_path,
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            save_path=args.save_ae
        )
        pretty_print_metrics("AE → Classifieur", metrics_ae)
        results["ae"] = metrics_ae

    # Comparaison sommaire si both
    if args.mode == "both" and "baseline" in results and "ae" in results:
        print("\n=== Comparaison rapide ===")
        b = results["baseline"]
        a = results["ae"]
        print(f"Accuracy baseline : {b['accuracy']:.4f} | AE : {a['accuracy']:.4f}")
        print(f"Train time baseline : {b['train_time_s']:.2f}s | AE : {a['train_time_s']:.2f}s")
        print(f"Infer time baseline : {b['inference_time_s']:.2f}s | AE : {a['inference_time_s']:.2f}s")

    print("\nTerminé. Figures & métriques dans", REPORT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier (baseline / AE→classifier) on MNIST")
    parser.add_argument("--mode", type=str, default="both", choices=["baseline", "ae", "both"],
                        help="Mode d'exécution")
    parser.add_argument("--encoder_path", type=str, default="encoder.h5", help="Chemin vers encoder sauvegardé (.h5)")
    parser.add_argument("--epochs", type=int, default=30, help="Nombre d'epochs pour le classifieur")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_baseline", type=str, default=None, help="Chemin pour sauvegarder le modèle baseline (optionnel)")
    parser.add_argument("--save_ae", type=str, default=None, help="Chemin pour sauvegarder le modèle AE→classifier (optionnel)")
    args = parser.parse_args()
    main(args)
