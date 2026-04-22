# Réponses rédigées — TP3 (Deep Learning & Fashion-MNIST)

## Étape 1 — Exploration
- **Le catalogue est-il équilibré ?** Oui, Fashion-MNIST est parfaitement équilibré avec 6000 images par classe dans l'ensemble d'entraînement. Cela facilite l'apprentissage car aucune classe n'est sous-représentée.
- **Risque de confusion ?** Oui, les pulls et les chemises (ou manteaux) ont des formes globales similaires. Cela peut induire le modèle en erreur si les détails fins (boutons, textures) ne sont pas bien capturés.
- **Résolution 28x28 ?** Pour Zalando en production, cette résolution est trop basse. Elle est utilisée ici pour la rapidité. En production, on utiliserait des images haute résolution (ex: 224x224 ou plus) avec des réseaux plus profonds (ResNet, EfficientNet).

## Étape 2 — Prétraitement
- **Pourquoi normaliser [0, 1] ?** Les réseaux de neurones convergent plus vite et de manière plus stable. Sans cela, les grands écarts de valeurs (0-255) peuvent provoquer des gradients explosifs ou instables.

## Étape 3 — Baseline (Random Forest)
- **Accuracy suffisante ?** Généralement, le Random Forest atteint ~88%. Ce n'est pas assez pour l'objectif métier (< 5% d'erreur).
- **Voisins ?** Non, le RF traite chaque pixel comme une variable indépendante. Il n'a aucune notion de "voisinage" ou de structure spatiale (texture, bords).

## Étape 4 — MLP (Réseau Dense)
- **Nombre de paramètres ?** (784*128 + 128) + (128*64 + 64) + (64*10 + 10) = 100 480 + 8 256 + 650 = 109 386 paramètres. C'est beaucoup plus que le nombre de pixels (784).
- **Softmax ?** Elle transforme les sorties en probabilités dont la somme vaut 1. C'est idéal pour choisir la classe la plus probable.
- **Sparse Categorical Crossentropy ?** Elle est utilisée quand les labels sont des entiers (0, 1, 2...). La version "binary" est réservée à seulement 2 classes.

## Étape 5 — CNN (Réseau Convolutif)
- **26x26 ?** Un filtre 3x3 perd 1 pixel de chaque côté quand il n'y a pas de padding. Donc 28 - (3-1) = 26.
- **MaxPooling2D ?** Il réduit la taille spatiale (sous-échantillonnage), ce qui rend le modèle invariant aux petites translations et réduit le coût de calcul.
- **Dropout ?** Il désactive des neurones au hasard, forçant le réseau à ne pas trop compter sur un seul neurone (réduction de la co-dépendance), ce qui limite le surapprentissage.

## Étape 10 — 3 Insights clés

### 1. ML Classique vs Deep Learning
**Observation :** Le CNN surpasse largement le Random Forest et le MLP (gain de ~3-5% d'accuracy).
**Explication :** Contrairement aux modèles denses, le CNN "voit" les formes et les textures grâce aux filtres de convolution.
**Recommandation :** Le surcoût GPU est justifié car c'est le seul moyen d'atteindre l'objectif métier de < 5% d'erreur.

### 2. Diagnostic de production (Overfitting)
**Observation :** Sans Dropout, le réseau dense surapprend plus vite que le CNN régularisé.
**Explication :** Les convolutions partagent les poids, ce qui réduit naturellement le nombre de paramètres par rapport à un réseau dense géant.
**Recommandation :** Utiliser systématiquement le Dropout et l'augmentation de données pour garantir la généralisation.

### 3. Fiabilité et Confiance
**Observation :** Les erreurs les plus fréquentes (Pull vs Manteau) sont visuellement logiques.
**Explication :** La résolution 28x28 masque les détails discriminants (matière, petits boutons).
**Recommandation :** Pour descendre sous les 5% d'erreur, une validation humaine reste nécessaire pour les cas où le modèle a une confiance < 90%.
