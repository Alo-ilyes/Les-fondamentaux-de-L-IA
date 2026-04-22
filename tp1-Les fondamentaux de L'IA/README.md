# 🎓 Mes Projets d'Intelligence Artificielle (B3)
> **Auteur :** Ilyes Alouata | **Matière :** Les fondamentaux de l'IA | **Projet :** TP1 - Analyse d'Algorithme

Bienvenue sur mon répertoire de travaux pratiques en Machine Learning ! 🚀

## 🌿 Architecture du Projet
Pour garantir un espace de travail clair et professionnel, ce projet est organisé comme suit :
- **`src/`** : Contient le code source Python (`main.py`).
- **`outputs/`** : Dossier regroupant tous les graphiques générés (Overfitting, Confusion Matrix, SHAP).
- **`docs/`** : Documentation complémentaire et réponses aux questions théoriques.
- **`Livrable_Final_TP1.html`** : Le rapport interactif complet regroupant toutes les analyses.

---

## 🔹 [TP1] - Analyse d'un algorithme de ML (Iris)
Ce projet implémente un pipeline complet de bout en bout sur le dataset Iris.

### ⚙️ Installation et Usage
1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Lancez le pipeline :
   ```bash
   python src/main.py
   ```

### 🔍 Synthèse : 3 Insights Clés (Debrief TP)

#### 1. Performance
*   **Observation** : La baseline (Decision Tree) atteint une accuracy de ~96.7%, prouvant la simplicité du dataset Iris.
*   **Explication** : Le dataset Iris est linéairement séparable pour la plupart des classes (surtout Setosa).
*   **Implication pratique** : Toujours commencer par une baseline simple avant de passer à des modèles complexes.

#### 2. Overfitting / Généralisation
*   **Observation** : On observe un début d'overfitting à partir d'une **profondeur de 5**.
*   **Explication** : Le modèle mémorise le bruit des données d'entraînement au lieu d'apprendre les tendances générales.
*   **Implication pratique** : Il faut limiter la complexité du modèle (via `max_depth`) pour garantir une meilleure fiabilité.

#### 3. Explicabilité (SHAP)
*   **Observation** : La variable **petal_width** est la plus déterminante selon SHAP.
*   **Explication** : Les espèces d'Iris se distinguent principalement par la morphologie de leurs pétales.
*   **Implication pratique** : L'explicabilité est cruciale pour garantir que le modèle prend des décisions basées sur des variables pertinentes.

---
*Note : Ce dépôt est structuré pour faciliter la lecture et l'évaluation par le professeur.*
