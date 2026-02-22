# src/data_loader.py
"""
Chargement et prétraitement minimal pour MNIST.

Fonctions principales :
- set_seed(seed)
- load_mnist(normalize=True, flatten=False, val_split=0.1,
             batch_size=None, as_tf_dataset=False, seed=42)

Retour :
- dict contenant 'train', 'val' (si val_split>0), 'test'
  chaque entrée est un tuple (X, y) en numpy arrays ou tf.data.Dataset
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import random
import tensorflow as tf


def set_seed(seed: int = 42):
    """Fixe les seeds pour numpy, random et tensorflow (reproductibilité)."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def _maybe_normalize(x: np.ndarray, normalize: bool) -> np.ndarray:
    x = x.astype("float32")
    return x / 255.0 if normalize else x


def _to_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: Optional[int] = None, shuffle: bool = False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))
    if batch_size is not None:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load_mnist(
    normalize: bool = True,
    flatten: bool = False,
    val_split: float = 0.1,
    batch_size: Optional[int] = None,
    as_tf_dataset: bool = False,
    seed: int = 42
) -> Dict[str, Tuple[Any, Any]]:
    """
    Charge MNIST depuis Keras, prétraite et retourne train/val/test.

    Args:
        normalize: met à l'échelle [0,1] si True.
        flatten: aplati les images en vecteurs (shape = (N, 784)) si True.
        val_split: fraction (0.0-0.5) de train réservée pour validation. 0 -> pas de val.
        batch_size: si as_tf_dataset True et batch_size donné, renvoie des datasets batchés.
        as_tf_dataset: si True, renvoie des tf.data.Dataset, sinon numpy arrays.
        seed: seed pour le split.

    Returns:
        dict avec clés 'train', 'val' (si val_split>0), 'test'.
        chaque valeur est (X, y) soit numpy arrays soit tf.data.Dataset.
    """
    assert 0.0 <= val_split < 0.5, "val_split doit être dans [0.0, 0.5)."
    set_seed(seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisation
    x_train = _maybe_normalize(x_train, normalize)
    x_test = _maybe_normalize(x_test, normalize)

    # Reshape selon flatten ou non (ajoute canal si nécessaire pour CNN)
    if flatten:
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
    else:
        # shape -> (N, 28, 28, 1)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    # Création du split validation à partir du training
    result: Dict[str, Tuple[Any, Any]] = {}
    if val_split > 0.0:
        # shuffle avant de splitter
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]
        n_val = int(len(x_train) * val_split)
        x_val, y_val = x_train[:n_val], y_train[:n_val]
        x_train, y_train = x_train[n_val:], y_train[n_val:]
        result["val"] = (x_val, y_val)

    result["train"] = (x_train, y_train)
    result["test"] = (x_test, y_test)

    # Convertir en tf.data.Dataset si demandé
    if as_tf_dataset:
        result_tf: Dict[str, Any] = {}
        for k, (X, Y) in result.items():
            shuffle_flag = (k == "train")
            ds = _to_tf_dataset(X, Y, batch_size=batch_size, shuffle=shuffle_flag)
            result_tf[k] = ds
        return result_tf

    return result


if __name__ == "__main__":
    # Exemple d'utilisation simple
    ds = load_mnist(normalize=True, flatten=False, val_split=0.1, as_tf_dataset=False, seed=123)
    x_train, y_train = ds["train"]
    x_val, y_val = ds["val"]
    x_test, y_test = ds["test"]
    print("Shapes :",
          "train:", x_train.shape, y_train.shape,
          "val:", x_val.shape, y_val.shape,
          "test:", x_test.shape, y_test.shape)
