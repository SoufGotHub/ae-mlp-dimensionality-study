# src/models/classifier.py
"""
Classifieur supervisé simple (MLP) pour MNIST.

Utilisation :
- Baseline : entrée = images aplaties (784)
- AE → Classifier : entrée = vecteur latent (latent_dim)

Fonctions :
- build_classifier(input_dim, num_classes=10)
- compile_classifier(model, lr=1e-3)
- train_classifier(model, x_train, y_train, x_val=None, y_val=None)
"""

from typing import Optional, Any
import tensorflow as tf


def build_classifier(
    input_dim: int,
    num_classes: int = 10
) -> tf.keras.Model:
    """
    Construit un classifieur MLP.

    Args:
        input_dim: dimension d'entrée (784 ou latent_dim).
        num_classes: nombre de classes (MNIST = 10).

    Returns:
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="classifier_input")

    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="mlp_classifier")
    return model


def compile_classifier(
    model: tf.keras.Model,
    lr: float = 1e-3
):
    """
    Compile le classifieur.

    Args:
        model: modèle Keras.
        lr: learning rate.

    Returns:
        modèle compilé.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_classifier(
    model: tf.keras.Model,
    x_train,
    y_train,
    x_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    epochs: int = 30,
    batch_size: int = 128,
    callbacks: Optional[list] = None,
    verbose: int = 1
):
    """
    Entraîne le classifieur.

    Args:
        model: modèle compilé.
        x_train, y_train: données d'entraînement.
        x_val, y_val: données de validation (optionnel).
        epochs, batch_size: paramètres d'entraînement.
        callbacks: callbacks Keras (EarlyStopping, etc.).

    Returns:
        history Keras.
    """
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val) if x_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    return history


if __name__ == "__main__":
    # Test rapide
    model = build_classifier(input_dim=64)
    compile_classifier(model)
    model.summary()
