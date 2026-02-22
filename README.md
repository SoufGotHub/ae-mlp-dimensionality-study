# Projet 13 — Classification MNIST (baseline & autoencodeur)

Projet de classification des chiffres MNIST avec TensorFlow/Keras, proposant deux approches : un classifieur sur images brutes (baseline) et un classifieur sur l’espace latent d’un autoencodeur.

## Structure du projet

```
Projet_13/
├── src/
│   ├── data_loader.py      # Chargement et prétraitement MNIST
│   ├── train_ae.py         # Entraînement de l’autoencodeur
│   ├── train_classifier.py # Entraînement du classifieur (baseline / AE)
│   ├── evaluate.py         # Métriques (accuracy, F1, matrice de confusion)
│   ├── generate_report.py  # Génération des figures pour le rapport
│   ├── test_single_image.py # Test sur une image
│   ├── utils.py
│   └── models/
│       ├── autoencoder.py  # Définition de l’autoencodeur
│       └── classifier.py   # Définition du classifieur
├── report_figs/            # Figures et métriques (créé à l’exécution)
└── README.md
```

## Dépendances

- Python 3.x
- TensorFlow
- NumPy
- scikit-learn
- Matplotlib

Installation suggérée :

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

## Utilisation

### 1. Entraîner l’autoencodeur

À exécuter depuis la racine du projet (ou en adaptant les chemins) :

```bash
python src/train_ae.py
```

Produit notamment : `encoder.h5`, `decoder.h5`, `autoencoder.h5`, ainsi que des courbes et métriques dans `report_figs/`.

### 2. Entraîner le classifieur

Trois modes possibles :

- **baseline** : classifieur sur images brutes (aplaties)
- **ae** : classifieur sur l’espace latent (nécessite `encoder.h5`)
- **both** : lance baseline puis ae et compare les résultats

```bash
python src/train_classifier.py --mode both
```

Options utiles : `--encoder_path`, `--epochs`, `--batch_size`, `--lr`, `--save_baseline`, `--save_ae`.

### 3. Générer le rapport (figures)

```bash
python src/generate_report.py [--baseline_model ...] [--ae_model ...] [--encoder_path ...] [--baseline_history ...] [--ae_history ...] [--plot_latent]
```

Les chemins des modèles et des fichiers history (`.npz`) sont optionnels ; le script peut être utilisé pour regénérer les figures à partir des fichiers déjà présents dans `report_figs/`.

### 4. Tester une image

Dans `src/test_single_image.py`, configurer :

- `MODEL_PATH` : `classifier.h5` ou `ae_classifier.h5`
- `ENCODER_PATH` : `None` ou `"encoder.h5"` si classifieur AE
- `IMAGE_INDEX` : index de l’image dans le jeu de test

Puis exécuter :

```bash
python src/test_single_image.py
```

## Fichiers générés

| Fichier        | Description                    |
|----------------|--------------------------------|
| `encoder.h5`   | Encodeur (réduction de dimension) |
| `decoder.h5`   | Décodeur                       |
| `autoencoder.h5` | Autoencodeur complet        |
| `classifier.h5`  | Classifieur baseline        |
| `ae_classifier.h5` | Classifieur sur espace latent |
| `report_figs/`   | Courbes, métriques, figures pour le rapport |

## Licence

Usage interne / projet pédagogique.
