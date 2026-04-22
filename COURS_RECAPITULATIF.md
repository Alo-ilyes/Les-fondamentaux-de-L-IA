# 🧠 Cours Récapitulatif : Les Fondamentaux de l'IA (B3)

Ce document explique les concepts clés mis en œuvre dans vos trois travaux pratiques (TP). L'objectif est de vous donner les clés pour comprendre et expliquer vos projets.

---

## 🏗️ 1. Le Pipeline de Machine Learning
Dans chaque projet, nous avons suivi une structure standardisée appelée **Pipeline**. C'est le flux de travail d'un Data Scientist :
1.  **Chargement & Exploration (EDA)** : Comprendre les données avant d'agir (ex: Pairplot pour Iris).
2.  **Prétraitement** : Nettoyer et transformer (ex: **Normalisation** entre 0 et 1 pour que le réseau de neurones ne soit pas "ébloui" par de grandes valeurs).
3.  **Baseline** : Entraîner un modèle simple pour avoir une référence (ex: Decision Tree).
4.  **Optimisation** : Tester des modèles plus complexes (ex: Random Forest, XGBoost, CNN).
5.  **Évaluation** : Utiliser des métriques (Accuracy, F1-Score, Recall) pour mesurer le succès.
6.  **Explicabilité** : Comprendre "pourquoi" l'IA a pris cette décision (SHAP).

---

## 🌳 2. Du Simple Arbre à la Forêt (TP1)
### L'Arbre de Décision (Baseline)
C'est comme un jeu de "Qui est-ce ?". L'algorithme pose des questions successives (ex: "La largeur du pétale est-elle > 0.8 ?").
*   **Avantage** : Très facile à lire.
*   **Inconvénient** : Très instable (un petit changement de donnée peut changer tout l'arbre).

### Le Random Forest (Modèle d'Ensemble)
Au lieu d'un seul arbre, on en crée 100. On fait voter ces arbres.
*   **Concept clé :** La sagesse de la foule. Cela réduit la variance et évite que l'IA ne mémorise par cœur les données.

---

## 🛡️ 3. Le Boosting et la Cybersécurité (TP2)
### XGBoost (Extreme Gradient Boosting)
Contrairement au Random Forest où les arbres sont indépendants, dans **XGBoost**, les arbres sont créés les uns après les autres. Chaque nouvel arbre essaie de corriger les erreurs du précédent.
*   **Pourquoi en Cyber ?** C'est l'un des algorithmes les plus puissants pour les données tabulaires (flux réseau). Il est extrêmement précis pour détecter des signaux faibles (attaques furtives).

### Métrique Cruciale : Le RECALL
En cybersécurité, on préfère un système qui fait une "fausse alerte" (Faux Positif) plutôt qu'un système qui laisse passer une attaque (Faux Négatif). Le **Recall** mesure notre capacité à ne rien laisser passer.

---

## 👁️ 4. Le Deep Learning et la Vision (TP3)
### MLP (Réseau Dense)
C'est un réseau où chaque "neurone" est connecté à tous les pixels de l'image.
*   **Problème :** Il traite un pixel en haut à gauche de la même manière qu'un pixel au milieu. Il perd la notion de "forme".

### CNN (Réseau de Neurones Convolutifs)
C'est l'IA qui imite l'œil humain.
1.  **Convolution** : Des petits filtres (3x3) glissent sur l'image pour détecter des bords, des coins, puis des textures.
2.  **MaxPooling** : On réduit la taille de l'image pour ne garder que l'information importante (résumé spatial).
*   **Résultat :** C'est pour cela que le CNN est bien meilleur pour reconnaître un T-shirt d'une chaussure.

---

## ⚖️ 5. Le Diagnostic : Overfitting vs Underfitting
C'est la partie la plus importante pour un ingénieur IA.
*   **Underfitting** : L'IA est trop simple (ex: une ligne droite pour des points en courbe). Elle est mauvaise partout.
*   **Overfitting (Surapprentissage)** : L'IA est trop complexe. Elle apprend par cœur les données d'entraînement.
    *   **Signe** : 100% d'accuracy sur le Train, mais seulement 70% sur le Test.
    *   **Solution** : Utiliser le **Dropout** (TP3) ou limiter la **Profondeur** (TP1).

---

## 🔍 6. L'IA Explicable (XAI)
### SHAP (SHapley Additive exPlanations)
C'est une méthode mathématique qui attribue à chaque caractéristique (ex: longueur du pétale, volume de données source) une valeur de contribution à la décision finale.
*   **Utilité** : Prouver au client (ou à la loi AI Act) que l'IA ne décide pas au hasard.

---

### 📝 Résumé pour votre oral :
*   **TP1** : J'ai appris à gérer l'overfitting en limitant la profondeur des arbres.
*   **TP2** : J'ai utilisé des modèles d'ensemble puissants pour sécuriser un réseau, en priorisant le Recall.
*   **TP3** : J'ai construit un CNN capable de "voir" les caractéristiques visuelles d'un produit e-commerce.

*Document rédigé par votre assistant IA pour Ilyes Alouata - 2026*
