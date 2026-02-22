# src/train_ae.py
"""
Entraînement de l'autoencodeur sur MNIST
et sauvegarde de l'encoder (réduction de dimension)

Modifications ajoutées :
- sauvegarde autoencoder.h5, encoder.h5, decoder.h5
- sauvegarde history (.npz) et plot loss / val_loss
- reconstructions (images) sauvegardées
- calcul et sauvegarde MSE reconstruction sur test (metrics JSON)
- sorties dans dossier report_figs/
"""
import os
import json
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import load_mnist
from models.autoencoder import (
    build_autoencoder,
    compile_autoencoder,
    train_autoencoder,
    save_encoder
)

# =====================
# Hyperparamètres
# =====================
LATENT_DIM = 64        # dimension de l'espace latent
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
CONV = True            # True = autoencodeur convolutionnel
ENCODER_SAVE_PATH = "encoder.h5"
DECODER_SAVE_PATH = "decoder.h5"
AUTOENCODER_SAVE_PATH = "autoencoder.h5"

REPORT_DIR = "report_figs"
os.makedirs(REPORT_DIR, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
def _make_serializable(obj):
    """Convert numpy types to Python builtins for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def save_history_npz(history: Optional[tf.keras.callbacks.History], filename: str):
    if history is None:
        return
    hist = history.history if hasattr(history, "history") else history
    to_save = {k: np.array(v) for k, v in hist.items()}
    path = os.path.join(REPORT_DIR, filename)
    np.savez(path, **to_save)
    print(f"[saved] {path}")


def plot_history(history: Optional[tf.keras.callbacks.History], title: str, filename: str):
    if history is None:
        print("[info] pas d'history à tracer.")
        return
    hist = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    loss = hist.get("loss")
    val_loss = hist.get("val_loss")
    if loss is not None:
        ax.plot(loss, label="train loss")
    if val_loss is not None:
        ax.plot(val_loss, label="val loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_reconstructions(autoencoder: Optional[tf.keras.Model],
                         encoder: Optional[tf.keras.Model],
                         decoder: Optional[tf.keras.Model],
                         x_test: np.ndarray,
                         n: int = 10,
                         filename: str = "reconstructions.png"):
    """Affiche n images originales et reconstruites et sauvegarde."""
    if x_test is None:
        print("[info] pas de x_test pour reconstructions.")
        return
    idx = np.random.choice(len(x_test), size=min(n, len(x_test)), replace=False)
    originals = x_test[idx]

    if autoencoder is not None:
        recon = autoencoder.predict(originals, batch_size=64, verbose=0)
    elif encoder is not None and decoder is not None:
        z = encoder.predict(originals, batch_size=64, verbose=0)
        recon = decoder.predict(z, batch_size=64, verbose=0)
    else:
        print("[info] pas de modèle pour reconstructions.")
        return

    # convert to (N,H,W)
    def to_img(x):
        if x.ndim == 2:
            side = int(np.sqrt(x.shape[1]))
            return x.reshape((-1, side, side))
        if x.ndim == 4 and x.shape[-1] == 1:
            return x.squeeze(-1)
        if x.ndim == 3:
            return x
        return x

    orig_img = to_img(originals)
    recon_img = to_img(recon)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3))
    for i in range(n):
        axes[0, i].imshow(orig_img[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_img[i], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstruit")
    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


def compute_reconstruction_mse(autoencoder: Optional[tf.keras.Model],
                               encoder: Optional[tf.keras.Model],
                               decoder: Optional[tf.keras.Model],
                               x_test: np.ndarray,
                               batch_size: int = 256) -> float:
    """Calcule l'MSE moyen de reconstruction sur x_test."""
    if x_test is None:
        return float("nan")
    if autoencoder is not None:
        rec = autoencoder.predict(x_test, batch_size=batch_size, verbose=0)
    elif encoder is not None and decoder is not None:
        z = encoder.predict(x_test, batch_size=batch_size, verbose=0)
        rec = decoder.predict(z, batch_size=batch_size, verbose=0)
    else:
        raise RuntimeError("Aucun modèle disponible pour calculer la reconstruction.")
    # mse per sample
    mse_per_sample = np.mean((x_test - rec) ** 2, axis=tuple(range(1, x_test.ndim)))
    return float(np.mean(mse_per_sample))


# =====================
# Main
# =====================
def main():
    print("📥 Chargement des données MNIST...")
    data = load_mnist(
        normalize=True,
        flatten=not CONV,   # si conv=True -> images (28,28,1)
        val_split=0.1,
        as_tf_dataset=False
    )

    x_train, _ = data["train"]
    x_val, _ = data["val"]
    x_test, _ = data["test"]

    input_shape = x_train.shape[1:]
    print(f"Input shape : {input_shape}")

    print("🧠 Construction de l'autoencodeur...")
    autoencoder, encoder, decoder = build_autoencoder(
        input_shape=input_shape,
        latent_dim=LATENT_DIM,
        conv=CONV
    )

    compile_autoencoder(
        autoencoder,
        lr=LEARNING_RATE,
        loss="mse"
    )

    autoencoder.summary()

    print("🚀 Entraînement de l'autoencodeur...")
    history = train_autoencoder(
        autoencoder=autoencoder,
        x_train=x_train,
        x_val=x_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    # sauvegarder history + plot
    save_history_npz(history, "autoencoder_history.npz")
    plot_history(history, "Autoencoder - reconstruction loss", "autoencoder_history.png")

    # sauvegarder autoencoder complet + encoder + decoder
    print(f"💾 Sauvegarde de l'encoder → {ENCODER_SAVE_PATH}")
    save_encoder(encoder, ENCODER_SAVE_PATH)
    # save decoder and full autoencoder
    try:
        decoder.save(DECODER_SAVE_PATH)
        print(f"💾 Sauvegarde du decoder → {DECODER_SAVE_PATH}")
    except Exception as e:
        print(f"[warn] impossible de sauvegarder decoder: {e}")
    try:
        autoencoder.save(AUTOENCODER_SAVE_PATH)
        print(f"💾 Sauvegarde de l'autoencoder complet → {AUTOENCODER_SAVE_PATH}")
    except Exception as e:
        print(f"[warn] impossible de sauvegarder autoencoder complet: {e}")

    # reconstructions visuelles
    plot_reconstructions(autoencoder=autoencoder, encoder=encoder, decoder=decoder, x_test=x_test, n=10, filename="reconstructions.png")

    # calculer MSE de reconstruction sur test
    try:
        mse = compute_reconstruction_mse(autoencoder=autoencoder, encoder=encoder, decoder=decoder, x_test=x_test, batch_size=BATCH_SIZE)
        metrics = {"reconstruction_mse": mse}
    except Exception as e:
        metrics = {"reconstruction_mse": None, "error": str(e)}
    # sauvegarder metrics json
    metrics_serial = _make_serializable(metrics)
    metrics_path = os.path.join(REPORT_DIR, "autoencoder_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_serial, f, indent=2, ensure_ascii=False)
    print(f"[saved] {metrics_path}")

    print("✅ Entraînement terminé avec succès.")


if __name__ == "__main__":
    main()
