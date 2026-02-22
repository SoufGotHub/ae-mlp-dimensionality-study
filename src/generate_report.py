

import os
import argparse
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.decomposition import PCA

OUTDIR = "report_figs"
os.makedirs(OUTDIR, exist_ok=True)


def _save(fig, name: str, dpi=150):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.show()
    print(f"[saved] {path}")


# -------------------------
# Chargement history (npz simple attendu)
# -------------------------
def load_history(path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Charge un fichier history sauvegardé au format numpy .npz ou .npy
    Attendu keys: 'loss', 'val_loss', optionnel 'accuracy', 'val_accuracy'
    """
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"[warn] history introuvable: {path}")
        return None
    try:
        data = np.load(path, allow_pickle=True)
        # .npz donne un objet similaire à un dict
        hist = {}
        for k in data.files:
            hist[k] = data[k].tolist() if data[k].dtype == object else data[k]
        return hist
    except Exception as e:
        print("Erreur chargement history:", e)
        return None


# -------------------------
# Courbes d'entraînement
# -------------------------
def plot_history_from_dict(hist: Dict[str, Any], title: str, filename: str):
    if hist is None:
        print(f"Aucun history pour {title}.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Loss
    loss = hist.get("loss") or hist.get("train_loss")
    val_loss = hist.get("val_loss") or hist.get("validation_loss")
    if loss is not None:
        axes[0].plot(loss, label="train loss")
    if val_loss is not None:
        axes[0].plot(val_loss, label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    # Accuracy si présente
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
    _save(fig, filename)


# -------------------------
# Reconstructions
# -------------------------
def plot_reconstructions(autoencoder: Optional[tf.keras.Model],
                         encoder: Optional[tf.keras.Model],
                         decoder: Optional[tf.keras.Model],
                         x_test: np.ndarray,
                         n: int = 10,
                         filename: str = "reconstructions.png"):
    if x_test is None:
        print("Aucune donnée test fournie pour reconstructions.")
        return
    idx = np.random.choice(len(x_test), size=n, replace=False)
    originals = x_test[idx]
    try:
        if autoencoder is not None:
            recon = autoencoder.predict(originals, batch_size=64, verbose=0)
        elif encoder is not None and decoder is not None:
            z = encoder.predict(originals, batch_size=64, verbose=0)
            recon = decoder.predict(z, batch_size=64, verbose=0)
        else:
            print("Pas d'autoencoder ni encoder+decoder : pas de reconstructions.")
            return
    except Exception as e:
        print("Erreur lors des reconstructions:", e)
        return

    def to_img(x):
        if x.ndim == 2:  # (N,784)
            side = int(np.sqrt(x.shape[1]))
            return x.reshape((-1, side, side))
        if x.ndim == 4 and x.shape[-1] == 1:
            return x.squeeze(-1)
        if x.ndim == 3:
            # déjà (N,H,W) ou (N,H,W,channels) géré ci-dessus
            return x
        return x

    orig_img = to_img(originals)
    recon_img = to_img(recon)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(orig_img[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_img[i], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstruit")
    _save(fig, filename)


# -------------------------
# Matrice de confusion & barplots precision/recall/f1
# -------------------------
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes=None, normalize=False, filename: str = "confusion.png"):
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, ylabel="Vrai label", xlabel="Label prédit")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.title("Matrice de confusion" + (" (normalisée)" if normalize else ""))
    _save(fig, filename)


def plot_prf_bars(y_true: np.ndarray, y_pred: np.ndarray, classes=None, filename: str = "prf_bars.png"):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(classes) if classes is not None else np.unique(y_true).shape[0]))
    labels = classes if classes is not None else [str(i) for i in range(len(p))]
    x = np.arange(len(p))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, p, width, label="Precision")
    ax.bar(x, r, width, label="Recall")
    ax.bar(x + width, f, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 par classe")
    ax.legend()
    _save(fig, filename)


# -------------------------
# Comparatif pipelines
# -------------------------
def plot_pipeline_comparison(results: Dict[str, Dict[str, Any]], filename: str = "pipeline_comparison.png"):
    keys = ["accuracy", "train_time_s", "inference_time_s"]
    labels = ["Accuracy", "Train time (s)", "Inference time (s)"]
    fig, axes = plt.subplots(1, len(keys), figsize=(len(keys) * 4, 4))
    for i, k in enumerate(keys):
        vals = [results.get("baseline", {}).get(k, 0), results.get("ae", {}).get(k, 0)]
        axes[i].bar(["baseline", "ae"], vals)
        axes[i].set_title(labels[i])
    fig.suptitle("Comparaison Baseline vs AE→Classifier")
    _save(fig, filename)


# -------------------------
# Optionnel: PCA latent (désactivé par défaut)
# -------------------------
def plot_latent_pca(encoder: tf.keras.Model, x: np.ndarray, y: np.ndarray, n_samples: int = 2000, filename: str = "latent_pca.png"):
    if encoder is None:
        print("Encoder absent : pas de PCA latent.")
        return
    if n_samples < len(x):
        idx = np.random.choice(len(x), size=n_samples, replace=False)
        x_s, y_s = x[idx], y[idx]
    else:
        x_s, y_s = x, y
    z = encoder.predict(x_s, batch_size=256, verbose=0)
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(z)
    fig = plt.figure(figsize=(6, 6))
    for lbl in np.unique(y_s):
        sel = y_s == lbl
        plt.scatter(z2[sel, 0], z2[sel, 1], label=str(lbl), s=6, alpha=0.7)
    plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Espace latent (PCA 2D)")
    _save(fig, filename)


# -------------------------
# CLI pipeline
# -------------------------
def load_data_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    n_val = int(len(x_train) * 0.1)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train2, y_train2 = x_train[n_val:], y_train[n_val:]
    return x_train2, y_train2, x_val, y_val, x_test, y_test


def main(args):
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_mnist()

    # charger modèles si présents
    baseline_model = None
    ae_model = None
    encoder = None
    decoder = None
    autoencoder = None
    results = {}

    if args.baseline_model and os.path.exists(args.baseline_model):
        try:
            baseline_model = tf.keras.models.load_model(args.baseline_model)
            print("Baseline model chargé.")
        except Exception as e:
            print("Erreur chargement baseline:", e)

    if args.ae_model and os.path.exists(args.ae_model):
        try:
            ae_model = tf.keras.models.load_model(args.ae_model)
            print("AE classifier chargé.")
        except Exception as e:
            print("Erreur chargement ae_model:", e)

    if args.encoder_path and os.path.exists(args.encoder_path):
        try:
            encoder = tf.keras.models.load_model(args.encoder_path)
            print("Encoder chargé.")
        except Exception as e:
            print("Erreur chargement encoder:", e)

    if args.decoder_path and os.path.exists(args.decoder_path):
        try:
            decoder = tf.keras.models.load_model(args.decoder_path)
            print("Decoder chargé.")
        except Exception as e:
            print("Erreur chargement decoder:", e)

    if args.autoencoder_path and os.path.exists(args.autoencoder_path):
        try:
            autoencoder = tf.keras.models.load_model(args.autoencoder_path)
            print("Autoencoder complet chargé.")
        except Exception as e:
            print("Erreur chargement autoencoder:", e)

    # histories optionnels
    baseline_hist = load_history(args.baseline_history) if args.baseline_history else None
    ae_hist = load_history(args.ae_history) if args.ae_history else None

    # tracer courbes si history fournis
    if baseline_hist is not None:
        plot_history_from_dict(baseline_hist, "Baseline - entraînement", "baseline_history.png")
    if ae_hist is not None:
        plot_history_from_dict(ae_hist, "AE→classifier - entraînement", "ae_history.png")

    # évaluer baseline s'il existe
    if baseline_model is not None:
        X_test_flat = x_test.reshape((x_test.shape[0], -1))
        y_proba = baseline_model.predict(X_test_flat, batch_size=256, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        acc = float(accuracy_score(y_test, y_pred))
        results["baseline"] = {
            "accuracy": acc,
            "train_time_s": args.baseline_train_time or 0.0,
            "inference_time_s": args.baseline_inf_time or 0.0
        }
        plot_confusion(y_test, y_pred, classes=[str(i) for i in range(10)], normalize=True, filename="confusion_baseline.png")
        plot_prf_bars(y_test, y_pred, classes=[str(i) for i in range(10)], filename="prf_baseline.png")

    # évaluer AE pipeline s'il existe
    if ae_model is not None:
        if encoder is not None:
            z_test = encoder.predict(x_test, batch_size=256, verbose=0)
            y_proba = ae_model.predict(z_test, batch_size=256, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
            acc = float(accuracy_score(y_test, y_pred))
            results["ae"] = {
                "accuracy": acc,
                "train_time_s": args.ae_train_time or 0.0,
                "inference_time_s": args.ae_inf_time or 0.0
            }
            plot_confusion(y_test, y_pred, classes=[str(i) for i in range(10)], normalize=True, filename="confusion_ae.png")
            plot_prf_bars(y_test, y_pred, classes=[str(i) for i in range(10)], filename="prf_ae.png")
        else:
            print("ae_model fourni mais encoder absent -> impossible d'extraire latents (ou vérifie format).")

    # reconstructions si possible
    if autoencoder is not None or (encoder is not None and decoder is not None):
        plot_reconstructions(autoencoder=autoencoder, encoder=encoder, decoder=decoder, x_test=x_test, n=10, filename="reconstructions.png")
    else:
        print("Aucun autoencoder/decoder disponible : pas de reconstructions.")

    # PCA latent optionnel
    if args.plot_latent and encoder is not None:
        plot_latent_pca(encoder, x_test, y_test, n_samples=args.latent_samples, filename="latent_pca.png")
    elif args.plot_latent and encoder is None:
        print("plot_latent demandé mais encoder absent.")

    # comparatif
    if "baseline" in results or "ae" in results:
        plot_pipeline_comparison(results, filename="pipeline_comparison.png")
    else:
        print("Pas assez de résultats pour comparatif (charger les modèles ou fournir metrics).")

    print("Figures générées dans", OUTDIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère figures/plots pour rapport (MNIST)")
    parser.add_argument("--baseline_model", type=str, default=None, help="Chemin vers modèle baseline (optionnel)")
    parser.add_argument("--ae_model", type=str, default=None, help="Chemin vers modèle AE->classifier (optionnel)")
    parser.add_argument("--encoder_path", type=str, default=None, help="Encoder (optionnel)")
    parser.add_argument("--decoder_path", type=str, default=None, help="Decoder (optionnel)")
    parser.add_argument("--autoencoder_path", type=str, default=None, help="Autoencoder complet (optionnel)")
    parser.add_argument("--baseline_history", type=str, default=None, help="Fichier .npz history baseline (optionnel)")
    parser.add_argument("--ae_history", type=str, default=None, help="Fichier .npz history ae (optionnel)")
    parser.add_argument("--baseline_train_time", type=float, default=None)
    parser.add_argument("--baseline_inf_time", type=float, default=None)
    parser.add_argument("--ae_train_time", type=float, default=None)
    parser.add_argument("--ae_inf_time", type=float, default=None)
    parser.add_argument("--plot_latent", action="store_true", help="Tracer PCA 2D de l'espace latent (optionnel)")
    parser.add_argument("--latent_samples", type=int, default=2000, help="Nb échantillons pour PCA latent")

    args = parser.parse_args()
    main(args)
