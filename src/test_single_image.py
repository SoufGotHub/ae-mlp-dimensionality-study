import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# =============================
# CONFIG
# =============================
MODEL_PATH = "classifier.h5"      # ou ae_classifier.h5
ENCODER_PATH = None               # mettre "encoder.h5" si AE
IMAGE_INDEX = 7                   # index de l'image à tester
SAVE_FIG = True

# =============================
# LOAD DATA
# =============================
(_, _), (x_test, y_test) = mnist.load_data()

x = x_test[IMAGE_INDEX]
y_true = y_test[IMAGE_INDEX]

# normalisation
x_norm = x.astype("float32") / 255.0
x_flat = x_norm.reshape(1, -1)

# =============================
# LOAD MODELS
# =============================
classifier = load_model(MODEL_PATH)

if ENCODER_PATH:
    encoder = load_model(ENCODER_PATH)
    x_input = encoder.predict(x_flat)
else:
    x_input = x_flat

# =============================
# PREDICTION
# =============================
proba = classifier.predict(x_input)[0]
y_pred = np.argmax(proba)

# =============================
# VISUALISATION
# =============================
plt.figure(figsize=(6, 4))
plt.imshow(x, cmap="gray")
plt.axis("off")

plt.title(
    f"Vrai label : {y_true} | Prédit : {y_pred}",
    fontsize=12
)

if SAVE_FIG:
    plt.savefig(f"prediction_image_{IMAGE_INDEX}.png", dpi=300)

plt.show()

# =============================
# PRINT DETAILS
# =============================
print("Label réel :", y_true)
print("Label prédit :", y_pred)
print("Probabilités :", np.round(proba, 3))
