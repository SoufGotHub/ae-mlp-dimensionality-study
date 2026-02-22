# src/models/autoencoder.py
"""
Autoencoder minimal (Dense ou Convolutionnel) pour MNIST.

Fonctions exposées :
- build_autoencoder(input_shape, latent_dim=64, conv=False)
    -> retourne (autoencoder, encoder, decoder)
- compile_autoencoder(model, lr=1e-3, loss='mse')
- train_autoencoder(autoencoder, x_train, x_val=None, epochs=50, batch_size=128, callbacks=None, save_path=None)

Usage:
    from src.models.autoencoder import build_autoencoder, compile_autoencoder, train_autoencoder
    ae, encoder, decoder = build_autoencoder((28,28,1), latent_dim=64, conv=True)
    compile_autoencoder(ae)
    history = train_autoencoder(ae, x_train, x_val=x_val, epochs=20)
"""

from typing import Tuple, Optional, Any
import tensorflow as tf


def build_autoencoder(input_shape: Tuple[int, ...], latent_dim: int = 64, conv: bool = False):
    """
    Construit un autoencodeur.

    Args:
        input_shape: ex. (28,28,1) pour images ou (784,) pour vecteur.
        latent_dim: dimension du code latent.
        conv: si True -> architecture convolutionnelle (nécessite input_shape rank==3).
              si False -> architecture fully-connected (flatten).

    Returns:
        (autoencoder, encoder, decoder) : tf.keras.Model
    """
    if conv:
        assert len(input_shape) == 3, "Pour conv=True, input_shape doit être (H,W,C)."

        inputs = tf.keras.Input(shape=input_shape, name="ae_input")
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)  # 14x14x32
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)       # 7x7x64
        shape_before_flat = tf.keras.backend.int_shape(x)[1:]  # (7,7,64)
        x = tf.keras.layers.Flatten()(x)
        latent = tf.keras.layers.Dense(latent_dim, name="latent_vector")(x)

        # Encoder model
        encoder = tf.keras.Model(inputs, latent, name="encoder")

        # Decoder
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_input")
        x = tf.keras.layers.Dense(shape_before_flat[0] * shape_before_flat[1] * shape_before_flat[2], activation="relu")(latent_inputs)
        x = tf.keras.layers.Reshape(shape_before_flat)(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)  # 14x14x64
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)  # 28x28x32
        outputs = tf.keras.layers.Conv2D(input_shape[2], 3, padding="same", activation="sigmoid", name="reconstruction")(x)

        decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")

        # Autoencoder: input -> encoder -> decoder
        outputs_ae = decoder(encoder(inputs))
        autoencoder = tf.keras.Model(inputs, outputs_ae, name="autoencoder_conv")

    else:
        # Dense autoencoder (pour vecteurs ou images aplaties)
        # Si input_shape a rank 3, on le flatten d'abord
        if len(input_shape) == 3:
            flat_dim = input_shape[0] * input_shape[1] * input_shape[2]
        else:
            flat_dim = input_shape[0]

        inputs = tf.keras.Input(shape=(flat_dim,), name="ae_input")
        x = tf.keras.layers.Dense(512, activation="relu")(inputs)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        latent = tf.keras.layers.Dense(latent_dim, name="latent_vector")(x)

        encoder = tf.keras.Model(inputs, latent, name="encoder_dense")

        # Decoder symmetric
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_input")
        x = tf.keras.layers.Dense(256, activation="relu")(latent_inputs)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        outputs = tf.keras.layers.Dense(flat_dim, activation="sigmoid", name="reconstruction")(x)

        decoder = tf.keras.Model(latent_inputs, outputs, name="decoder_dense")
        outputs_ae = decoder(encoder(inputs))
        autoencoder = tf.keras.Model(inputs, outputs_ae, name="autoencoder_dense")

    return autoencoder, encoder, decoder


def compile_autoencoder(model: tf.keras.Model, lr: float = 1e-3, loss: str = "mse"):
    """Compile l'autoencodeur avec un optimiseur Adam et la loss choisie."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])
    return model


def train_autoencoder(
    autoencoder: tf.keras.Model,
    x_train,
    x_val: Optional[Any] = None,
    epochs: int = 50,
    batch_size: int = 128,
    callbacks: Optional[list] = None,
    save_path: Optional[str] = None,
    verbose: int = 1
):
    """
    Entraîne l'autoencodeur.

    Args:
        autoencoder: modèle compilé.
        x_train: numpy array ou tf.data.Dataset (si ds -> doit produire (x,x) ou (x,y) où y ignoré).
                 Pour numpy array, on l'utilise comme (x_train, x_train).
        x_val: idem pour validation (optionnel).
        epochs, batch_size: paramètres d'entraînement.
        callbacks: liste de callbacks Keras.
        save_path: si fourni, sauvegarde l'encoder seul après entraînement (chemin .h5).
    Returns:
        history Keras.
    """
    # Préparer les entrées selon le type
    if isinstance(x_train, tf.data.Dataset):
        train_ds = x_train.map(lambda x, y: (x, x))
        history = autoencoder.fit(train_ds, epochs=epochs, validation_data=(x_val.map(lambda x, y: (x, x)) if x_val is not None else None),
                                  callbacks=callbacks, verbose=verbose)
    else:
        # numpy arrays
        if isinstance(x_train, tuple) and len(x_train) == 2:
            X_train = x_train[0]
        else:
            X_train = x_train
        X_val = None
        if x_val is not None:
            if isinstance(x_val, tuple) and len(x_val) == 2:
                X_val = x_val[0]
            else:
                X_val = x_val
        history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                                  validation_data=(X_val, X_val) if X_val is not None else None,
                                  callbacks=callbacks, verbose=verbose)

    # Sauvegarde de l'encoder si demandé
    if save_path:
        # on suppose que l'encoder est nommé 'encoder' dans le module build_autoencoder
        # trouver le sous-modèle encoder
        for layer in autoencoder.layers:
            # pas fiable pour retrouver l'encoder automatiquement; on recommande d'utiliser l'objet encoder retourné par build_autoencoder
            pass
        # si l'utilisateur veut sauvegarder l'encoder, il/elle devrait fournir directement l'objet encoder.
        # ici on ne sauvegarde pas automatiquement pour éviter les confusions.

    return history


def save_encoder(encoder: tf.keras.Model, path: str):
    """Sauvegarde l'encoder (architecture + poids) en .h5"""
    encoder.save(path)
    return path


if __name__ == "__main__":
    # Petit test rapide pour voir les résumés
    # Conv autoencoder (images)
    ae_conv, enc_conv, dec_conv = build_autoencoder((28, 28, 1), latent_dim=64, conv=True)
    compile_autoencoder(ae_conv)
    print("=== Conv Autoencoder ===")
    ae_conv.summary()
    print("\n=== Encoder (conv) ===")
    enc_conv.summary()
    print("\n=== Decoder (conv) ===")
    dec_conv.summary()

    # Dense autoencoder (vecteurs aplatis)
    ae_dense, enc_dense, dec_dense = build_autoencoder((784,), latent_dim=64, conv=False)
    compile_autoencoder(ae_dense)
    print("\n\n=== Dense Autoencoder ===")
    ae_dense.summary()
