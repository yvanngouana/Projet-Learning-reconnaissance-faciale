"""
Script d'inférence pour le modèle de reconnaissance de visages
Groupe 5 - MDSMS1

Usage:
    python inference.py <image_path>           # Prédire sur une image
    python inference.py --webcam               # Test avec webcam
    python inference.py --batch <folder_path>  # Prédire sur un dossier
"""

import os
import sys
import cv2
import numpy as np
import pickle
import json
import argparse
from tensorflow.keras.models import load_model
from mtcnn import MTCNN


class FaceRecognizer:
    """
    Classe pour la reconnaissance de visages
    """

    def __init__(self,
                 model_path='models/face_recognition_model.h5',
                 config_path='models/model_config.json',
                 encoder_path='models/label_encoder.pkl'):
        """
        Initialise le reconnaisseur de visages

        Args:
            model_path: Chemin vers le modèle .h5
            config_path: Chemin vers la configuration
            encoder_path: Chemin vers le label encoder
        """
        print("Initialisation du reconnaisseur de visages...")

        # Vérifier que les fichiers existent
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration introuvable: {config_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder introuvable: {encoder_path}")

        # Charger le modèle
        print(f"Chargement du modèle depuis {model_path}...")
        self.model = load_model(model_path)
        print("✓ Modèle chargé avec succès!")

        # Charger la configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        print("✓ Configuration chargée!")

        # Charger le label encoder
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("✓ Label encoder chargé!")

        # Initialiser le détecteur de visages
        print("Initialisation du détecteur de visages MTCNN...")
        self.face_detector = MTCNN()
        print("✓ Détecteur initialisé!")

        self.input_shape = tuple(self.config['input_shape'][:2])

        print("\n" + "="*50)
        print("Reconnaisseur initialisé avec succès!")
        print(f"Nombre de classes: {self.config['num_classes']}")
        print(f"Accuracy du modèle (test): {self.config['test_accuracy']*100:.2f}%")
        print("="*50 + "\n")

    def preprocess_image(self, image):
        """
        Pré-traite une image pour la prédiction

        Args:
            image: Image numpy array (BGR)

        Returns:
            face: Visage pré-traité ou None
            box: Coordonnées du visage détecté (x, y, w, h) ou None
        """
        # Convertir en RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Détecter le visage
        faces = self.face_detector.detect_faces(image_rgb)

        if len(faces) == 0:
            return None, None

        # Prendre le visage avec la meilleure confiance
        best_face = max(faces, key=lambda x: x['confidence'])
        x, y, w, h = best_face['box']

        # S'assurer que les coordonnées sont valides
        x = max(0, x)
        y = max(0, y)

        # Extraire le visage avec une petite marge
        margin = int(0.2 * min(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image_rgb.shape[1], x + w + margin)
        y2 = min(image_rgb.shape[0], y + h + margin)

        face = image_rgb[y1:y2, x1:x2]

        # Redimensionner
        face_resized = cv2.resize(face, self.input_shape)

        # Normaliser
        face_normalized = face_resized / 255.0

        return face_normalized, (x, y, w, h)

    def predict(self, image_path, top_k=3, display=True):
        """
        Prédit l'identité de la personne sur l'image

        Args:
            image_path: Chemin vers l'image
            top_k: Nombre de prédictions top à retourner
            display: Afficher l'image avec les prédictions

        Returns:
            results: Liste des top-k prédictions avec probabilités
            error: Message d'erreur ou None
        """
        # Charger l'image
        image = cv2.imread(image_path)

        if image is None:
            return None, f"Impossible de charger l'image: {image_path}"

        # Pré-traiter
        face, box = self.preprocess_image(image)

        if face is None:
            return None, "Aucun visage détecté dans l'image"

        # Prédire
        face_batch = np.expand_dims(face, axis=0)
        predictions = self.model.predict(face_batch, verbose=0)[0]

        # Obtenir les top-k prédictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            student_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = predictions[idx] * 100
            results.append({
                'name': student_name,
                'confidence': confidence
            })

        # Afficher l'image avec les prédictions
        if display:
            self.display_prediction(image, results, box)

        return results, None

    def display_prediction(self, image, results, box=None):
        """
        Affiche l'image avec les prédictions

        Args:
            image: Image originale
            results: Résultats de la prédiction
            box: Coordonnées du visage (x, y, w, h)
        """
        img_display = image.copy()

        # Dessiner le rectangle autour du visage
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Ajouter le texte avec les prédictions
        y_offset = 30
        for i, result in enumerate(results):
            text = f"{i+1}. {result['name']}: {result['confidence']:.1f}%"
            color = (0, 255, 0) if i == 0 else (255, 255, 0)
            cv2.putText(img_display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        # Afficher
        cv2.imshow('Reconnaissance de Visages', img_display)
        print("\nAppuyez sur une touche pour continuer...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict_webcam(self):
        """
        Démonstration en temps réel avec webcam
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la webcam")
            return

        print("\n" + "="*50)
        print("Mode Webcam - Reconnaissance en temps réel")
        print("Appuyez sur 'q' pour quitter")
        print("="*50 + "\n")

        frame_count = 0
        process_every_n_frames = 5  # Traiter 1 frame sur 5 pour améliorer les performances

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur: Impossible de lire la frame")
                break

            frame_count += 1

            # Traiter seulement certaines frames
            if frame_count % process_every_n_frames == 0:
                # Pré-traiter
                face, box = self.preprocess_image(frame)

                if face is not None:
                    # Prédire
                    face_batch = np.expand_dims(face, axis=0)
                    predictions = self.model.predict(face_batch, verbose=0)[0]

                    # Meilleure prédiction
                    best_idx = np.argmax(predictions)
                    best_name = self.label_encoder.inverse_transform([best_idx])[0]
                    confidence = predictions[best_idx] * 100

                    # Dessiner le rectangle
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Afficher le nom et la confiance
                    text = f"{best_name}: {confidence:.1f}%"
                    cv2.putText(frame, text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Aucun visage detecte", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Reconnaissance de Visages - Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\nMode webcam terminé.")

    def predict_batch(self, folder_path, top_k=3):
        """
        Prédit sur toutes les images d'un dossier

        Args:
            folder_path: Chemin vers le dossier contenant les images
            top_k: Nombre de prédictions top à retourner
        """
        if not os.path.exists(folder_path):
            print(f"Erreur: Le dossier {folder_path} n'existe pas")
            return

        # Lister toutes les images
        image_files = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"Aucune image trouvée dans {folder_path}")
            return

        print(f"\nTraitement de {len(image_files)} images...\n")
        print("="*70)

        results_all = []

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)

            print(f"\nImage: {img_file}")
            results, error = self.predict(img_path, top_k=top_k, display=False)

            if error:
                print(f"  ✗ {error}")
            else:
                print(f"  ✓ Prédictions:")
                for i, result in enumerate(results, 1):
                    print(f"    {i}. {result['name']}: {result['confidence']:.2f}%")

                results_all.append({
                    'image': img_file,
                    'predictions': results
                })

        print("\n" + "="*70)
        print(f"Traitement terminé: {len(results_all)}/{len(image_files)} images traitées avec succès")
        print("="*70)


def main():
    """
    Fonction principale
    """
    parser = argparse.ArgumentParser(
        description='Script d\'inférence pour la reconnaissance de visages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python inference.py image.jpg                    # Prédire sur une image
  python inference.py --webcam                     # Test avec webcam
  python inference.py --batch ./test_images        # Prédire sur un dossier
  python inference.py image.jpg --top-k 5          # Afficher top 5 prédictions
        """
    )

    parser.add_argument('image_path', nargs='?', help='Chemin vers l\'image à analyser')
    parser.add_argument('--webcam', action='store_true', help='Mode webcam')
    parser.add_argument('--batch', type=str, help='Traiter toutes les images d\'un dossier')
    parser.add_argument('--top-k', type=int, default=3, help='Nombre de prédictions à afficher (défaut: 3)')
    parser.add_argument('--model', type=str, default='models/face_recognition_model.h5',
                       help='Chemin vers le modèle .h5')
    parser.add_argument('--config', type=str, default='models/model_config.json',
                       help='Chemin vers la configuration')
    parser.add_argument('--encoder', type=str, default='models/label_encoder.pkl',
                       help='Chemin vers le label encoder')

    args = parser.parse_args()

    try:
        # Créer le reconnaisseur
        recognizer = FaceRecognizer(
            model_path=args.model,
            config_path=args.config,
            encoder_path=args.encoder
        )

        # Mode webcam
        if args.webcam:
            recognizer.predict_webcam()

        # Mode batch
        elif args.batch:
            recognizer.predict_batch(args.batch, top_k=args.top_k)

        # Mode image unique
        elif args.image_path:
            print(f"\nAnalyse de l'image: {args.image_path}\n")
            results, error = recognizer.predict(args.image_path, top_k=args.top_k, display=True)

            if error:
                print(f"Erreur: {error}")
                sys.exit(1)
            else:
                print("\n" + "="*50)
                print("RÉSULTATS DE LA PRÉDICTION")
                print("="*50)
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['name']}: {result['confidence']:.2f}%")
                print("="*50)

        else:
            parser.print_help()
            print("\nErreur: Vous devez spécifier une image, --webcam ou --batch")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"\nErreur: {e}")
        print("\nAssurez-vous que les fichiers suivants existent:")
        print("  - models/face_recognition_model.h5")
        print("  - models/model_config.json")
        print("  - models/label_encoder.pkl")
        sys.exit(1)

    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
