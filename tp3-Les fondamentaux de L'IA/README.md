# 🎓 Mes Projets d'Intelligence Artificielle (B3)
> **Auteur :** Ilyes Alouata | **Matière :** Les fondamentaux de l'IA | **Projet :** TP3 - Deep Learning Fashion

Bienvenue sur mon répertoire pour le troisième travail pratique portant sur le Deep Learning ! 🚀

## 🌿 Architecture du Projet
Ce projet est structuré pour une analyse approfondie du dataset Fashion-MNIST :
- **`src/`** : Code source Python (`main.py`) utilisant **TensorFlow / Keras**.
- **`outputs/`** : Dossier regroupant les visuels (Échantillons, Courbes d'apprentissage, Matrice CNN, Filtres).
- **`docs/`** : Réponses détaillées sur les architectures neuronales et le diagnostic.
- **`Livrable_Final_TP3.html`** : Le rapport interactif complet regroupant toutes les analyses.

---

## 🔹 [TP3] - Deep Learning : Classification Zalando
Ce projet compare le ML classique (Random Forest) au Deep Learning (MLP et CNN) pour la classification d'images de vêtements.

### ⚙️ Installation et Usage
1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Lancez le pipeline :
   ```bash
   python src/main.py
   ```

### 🔍 Synthèse : 3 Insights Clés (Debrief TP3)

#### 1. ML Classique vs Deep Learning
*   **Observation** : Le CNN surpasse largement le Random Forest (gain de ~5% d'accuracy).
*   **Explication** : Le CNN capture la structure spatiale (contours, textures) alors que le RF traite chaque pixel comme une donnée isolée.
*   **Implication pratique** : Pour le traitement d'images, le surcoût de calcul du Deep Learning est indispensable pour atteindre les objectifs industriels.

#### 2. Interprétabilité du modèle
*   **Observation** : La visualisation des filtres montre que le réseau apprend des détecteurs de bords et de textures.
*   **Explication** : Cela confirme que le modèle ne "devine" pas mais s'appuie sur des critères visuels cohérents avec l'œil humain.
*   **Implication pratique** : Indispensable pour valider la robustesse du modèle face à des artefacts de fond ou de couleur.

#### 3. Diagnostic de production
*   **Observation** : Les erreurs de classification sont concentrées sur les articles de formes proches (Pull vs Chemise).
*   **Explication** : La basse résolution (28x28) limite la distinction des détails fins.
*   **Implication pratique** : Prévoir une validation humaine pour les prédictions à faible score de confiance et passer à de la haute résolution en production réelle.

---
*Note : Ce dépôt est optimisé pour démontrer une maîtrise des réseaux de neurones convolutifs (CNN) et de leur interprétation.*
