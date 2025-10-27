# Projet de Reconnaissance de Visages - Groupe 5
## MDSMS1 - Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

### Membres du groupe
- **NGOUANA FAUMETE Etienne**
- **BEINDI FÉLIX HOUMBI**
- **SCHOUAME Jean Pierre**
- **DEDIM GUELBE APPOLINAIRE**

---

## 📋 Description du projet

Ce projet implémente un système de **reconnaissance de visages** pour identifier automatiquement les étudiants de la promotion MDSMS1. Le modèle utilise le **deep learning** avec une approche de **transfer learning** basée sur MobileNetV2 pour atteindre une haute précision sur le jeu de test.

### Objectifs
1. Collecter et préparer un dataset de visages d'étudiants
2. Pré-traiter les images pour la reconnaissance faciale
3. Entraîner un modèle de deep learning performant
4. Déployer le modèle pour des prédictions en temps réel

---

## 📚 Table des matières
1. [Collecte des données](#collecte-des-données)
2. [Pré-traitement](#pré-traitement)
3. [Architecture du modèle](#architecture-du-modèle)
4. [Entraînement](#entraînement)
5. [Résultats](#résultats)
6. [Installation](#installation)
7. [Utilisation](#utilisation)
8. [Structure du projet](#structure-du-projet)
9. [Améliorations futures](#améliorations-futures)

---

## 📊 Collecte des données

### Méthodologie
- **Nombre d'étudiants:** ~20 étudiants de MDSMS1
- **Images par étudiant:** 10-20 images variées
- **Total d'images:** ~200+ images
- **Conditions variées:**
  - Différents angles (face, 3/4, profil)
  - Différents éclairages (naturel, artificiel)
  - Différentes expressions (neutre, sourire)
  - Avec/sans lunettes (si applicable)

### Organisation des données
Les données ont été collectées de manière collaborative avec l'ensemble de la promotion pour garantir la cohérence et la qualité du dataset.

```
Data/
├── NGOUANA/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── BEINDI/
│   ├── image_001.jpg
│   └── ...
└── ...
```

---

## 🔧 Pré-traitement

### Étapes appliquées

1. **Détection des visages**
   - Algorithme: **MTCNN** (Multi-task Cascaded Convolutional Networks)
   - Extraction et recadrage automatique des visages
   - Détection robuste même avec variations d'angle et d'éclairage

2. **Redimensionnement**
   - Taille cible: **160x160 pixels**
   - Maintien du ratio d'aspect avec marge de 20%

3. **Normalisation**
   - Pixels normalisés entre 0 et 1
   - Formule: `pixel_value / 255.0`

4. **Augmentation des données** (Data Augmentation)
   - Rotation: ±20°
   - Décalage horizontal/vertical: ±20%
   - Flip horizontal
   - Zoom: ±20%
   - Ajustement de luminosité: 80-120%

---

## 🏗️ Architecture du modèle

### Approche: Transfer Learning

Nous avons utilisé **MobileNetV2** pré-entraîné sur ImageNet comme modèle de base, avec des couches personnalisées pour la classification des étudiants.

### Structure du modèle

```
Input (160x160x3)
    ↓
MobileNetV2 (pré-entraîné sur ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(num_classes, activation='softmax')
```

### Justification du choix

- **MobileNetV2:** Léger et efficace, idéal pour le déploiement
- **Transfer Learning:** Performances élevées même avec un petit dataset
- **Architecture simple:** Évite l'overfitting grâce aux couches Dropout

---

## 🎯 Entraînement

### Hyperparamètres

- **Optimizer:** Adam
- **Learning rate initial:** 0.001
- **Learning rate fine-tuning:** 0.0001
- **Batch size:** 32
- **Loss function:** Categorical Crossentropy
- **Epochs phase 1:** 20 (couches de base gelées)
- **Epochs phase 2:** 30 (fine-tuning complet)

### Callbacks utilisés

1. **EarlyStopping**
   - Monitor: `val_loss`
   - Patience: 10 epochs
   - Restaure les meilleurs poids

2. **ModelCheckpoint**
   - Sauvegarde du meilleur modèle basé sur `val_accuracy`

3. **ReduceLROnPlateau**
   - Réduction du learning rate si plateau détecté
   - Factor: 0.5

### Stratégie d'entraînement en 2 phases

1. **Phase 1:** Entraînement des couches personnalisées uniquement (couches de base gelées)
   - Permet d'apprendre les caractéristiques spécifiques aux visages des étudiants

2. **Phase 2:** Fine-tuning de toutes les couches avec un learning rate réduit
   - Affine le modèle pour améliorer les performances

### Séparation des données

- **Training set:** 70% des données
- **Validation set:** 15% des données
- **Test set:** 15% des données

---

## 📈 Résultats

### Performance du modèle

- **Accuracy (Train):** 71.37%
- **Accuracy (Validation):** 71.15%
- **Accuracy (Test):** 57.69%
- **Loss (Test):** 1.2307

### Statistiques du dataset

- **Nombre de classes:** 19 étudiants
- **Total d'images:** 345 images
- **Training set:** 241 images (70%)
- **Validation set:** 52 images (15%)
- **Test set:** 52 images (15%)

### Graphiques

Les graphiques d'entraînement et la matrice de confusion sont générés automatiquement dans le notebook et sauvegardés dans le dossier `models/`.

- `training_history.png` - Évolution de l'accuracy et de la loss
- `confusion_matrix.png` - Matrice de confusion sur le jeu de test

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip
- (Optionnel) GPU avec CUDA pour accélérer l'entraînement

### Installation des dépendances

```bash
# Cloner ou télécharger le projet
cd projet-reconnaissance-visages

# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement virtuel
# Sur Linux/Mac:
source venv/bin/activate
# Sur Windows:
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 💻 Utilisation

### 1. Entraîner le modèle

Ouvrez le notebook Jupyter et exécutez toutes les cellules:

```bash
jupyter notebook Face_Recognition_Project.ipynb
```

Le notebook vous guidera à travers toutes les étapes:
- Exploration des données
- Pré-traitement
- Entraînement
- Évaluation
- Sauvegarde du modèle

### 2. Tester le modèle sur une image

```bash
python inference.py path/to/image.jpg
```

**Sortie attendue:**
```
Résultats de la prédiction:
1. NGOUANA FAUMETE Etienne: 95.67%
2. SCHOUAME Jean Pierre: 3.21%
3. BEINDI MEMANG HOUMBI: 0.89%
```

### 3. Tester avec la webcam

```bash
python inference.py --webcam
```

Appuyez sur 'q' pour quitter le mode webcam.

### 4. Traiter un lot d'images

```bash
python inference.py --batch ./test_images
```

### 5. Options avancées

```bash
# Afficher les 5 meilleures prédictions
python inference.py image.jpg --top-k 5

# Utiliser un modèle personnalisé
python inference.py image.jpg --model custom_model.h5

# Afficher l'aide
python inference.py --help
```

### 6. Charger le modèle dans votre code

```python
from tensorflow.keras.models import load_model
import pickle
import json
import numpy as np
import cv2

# Charger le modèle
model = load_model('models/face_recognition_model.h5')

# Charger le label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Charger une image et prédire
img = cv2.imread('test_image.jpg')
img_resized = cv2.resize(img, (160, 160))
img_normalized = img_resized / 255.0
img_batch = np.expand_dims(img_normalized, axis=0)

# Prédiction
predictions = model.predict(img_batch)[0]
best_idx = np.argmax(predictions)
student_name = label_encoder.inverse_transform([best_idx])[0]
confidence = predictions[best_idx] * 100

print(f"Prédiction: {student_name} ({confidence:.2f}%)")
```

---

## 📁 Structure du projet

```
projet-reconnaissance-visages/
├── Data/                                # Images brutes collectées
│   ├── NGOUANA/
│   ├── BEINDI/
│   ├── SCHOUAME/
│   ├── DEDIM/
│   └── ...
│
├── processed_data/                      # Images pré-traitées (générées)
│   ├── NGOUANA/
│   └── ...
│
├── models/                              # Modèles et artefacts (générés)
│   ├── face_recognition_model.h5        # Modèle entraîné final
│   ├── best_model_phase1.h5             # Meilleur modèle phase 1
│   ├── best_model_phase2.h5             # Meilleur modèle phase 2
│   ├── label_encoder.pkl                # Encodeur des labels
│   ├── model_config.json                # Configuration du modèle
│   ├── confusion_matrix.png             # Matrice de confusion
│   └── training_history.png             # Graphiques d'entraînement
│
├── Face_Recognition_Project.ipynb       # Notebook principal
├── inference.py                         # Script d'inférence
├── requirements.txt                     # Dépendances Python
├── README.md                            # Ce fichier
├── train.csv                            # Dataset d'entraînement (généré)
├── val.csv                              # Dataset de validation (généré)
└── test.csv                             # Dataset de test (généré)
```

---

## 🔬 Détails techniques

### Technologies utilisées

- **TensorFlow/Keras:** Framework de deep learning
- **OpenCV:** Traitement d'images
- **MTCNN:** Détection de visages
- **NumPy/Pandas:** Manipulation de données
- **Matplotlib/Seaborn:** Visualisations
- **Scikit-learn:** Prétraitement et métriques

### Configuration matérielle recommandée

- **CPU:** Intel Core i5 ou équivalent
- **RAM:** 8 GB minimum
- **GPU:** NVIDIA GPU avec CUDA (optionnel mais recommandé)
- **Espace disque:** 2 GB minimum

---

## ⚠️ Limitations et considérations

### Limitations actuelles

1. **Taille du dataset:** Performance limitée par le nombre d'images par personne
2. **Conditions d'éclairage:** Peut être moins précis dans des conditions extrêmes
3. **Angles de vue:** Meilleure performance avec des visages de face
4. **Occlusions:** Difficulté avec masques, lunettes de soleil, etc.

### Considérations éthiques

- Ce modèle est développé à des fins **éducatives uniquement**
- Respect de la vie privée et consentement des participants
- Ne pas utiliser à des fins de surveillance non autorisée

---

## 🚧 Améliorations futures

### Court terme

- [ ] Augmenter le dataset avec plus d'images variées
- [ ] Implémenter la reconnaissance de visages multiples
- [ ] Optimiser les performances en temps réel

### Moyen terme

- [ ] Tester d'autres architectures (VGGFace, FaceNet, ArcFace)
- [ ] Implémenter un système de seuil de confiance
- [ ] Ajouter la détection d'émotions

### Long terme

- [ ] Créer une interface web avec Flask/Streamlit
- [ ] Déployer sur mobile (TensorFlow Lite)
- [ ] Implémenter un système d'apprentissage continu

---

## 🐛 Dépannage

### Problèmes courants

**1. Erreur "No module named 'tensorflow'"**
```bash
pip install tensorflow
```

**2. Erreur avec MTCNN**
```bash
pip install mtcnn
```

**3. Problèmes de webcam**
- Vérifiez que votre webcam est connectée
- Essayez de changer l'indice: `cv2.VideoCapture(1)` au lieu de `0`

**4. Erreur de mémoire GPU**
- Réduisez le batch size dans le notebook
- Utilisez le CPU en ajoutant:
  ```python
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  ```

---

## 📖 Références

### Articles scientifiques

- **MobileNetV2:** [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- **MTCNN:** [Zhang et al., 2016](https://arxiv.org/abs/1604.02878)
- **Transfer Learning:** [Yosinski et al., 2014](https://arxiv.org/abs/1411.1792)

### Documentation

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [MTCNN GitHub](https://github.com/ipazc/mtcnn)

### Ressources d'apprentissage

- [Deep Learning Specialization - Andrew Ng](https://www.coursera.org/specializations/deep-learning)
- [Face Recognition Guide](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras/)

---

## 📝 Historique des versions

### Version 1.0 (Octobre 2025)
- Implémentation initiale du projet
- MobileNetV2 avec transfer learning
- Script d'inférence complet
- Documentation complète

---

## 👥 Contribution

Ce projet a été réalisé dans le cadre du cours de Deep Learning - MDSMS1.

### Répartition des tâches

- **NGOUANA FAUMETE Etienne:** Configuration, entraînement, documentation technique
- **BEINDI FÉLIX HOUMBI:** Pré-traitement, détection de visages
- **SCHOUAME Jean Pierre:** Organisation des données, visualisations
- **DEDIM GUELBE APPOLINAIRE:** Script d'inférence, tests, documentation utilisateur

---

## 📧 Contact

Pour toute question concernant ce projet, contactez les membres du Groupe 5.

---

## 📄 Licence

Ce projet est réalisé dans le cadre du cours de Deep Learning - MDSMS1.
**Usage académique uniquement.**

---

## 🙏 Remerciements

- Professeur et équipe pédagogique de MDSMS1
- Tous les étudiants ayant participé à la collecte de données
- Communauté TensorFlow et Keras

---

**Développé avec ❤️ par le Groupe 5 - MDSMS1**

*Dernière mise à jour: Octobre 2025*
