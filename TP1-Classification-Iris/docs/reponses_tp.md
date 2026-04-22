# Réponses rédigées — TP1

## Étape 1 — Exploration des données
- Le dataset Iris contient 150 échantillons répartis sur 3 classes.
- Les classes sont équilibrées avec 50 échantillons par espèce.
- Il n'y a pas de valeurs manquantes.
- Les variables liées aux pétales semblent les plus discriminantes, surtout `petal_length` et `petal_width`.

## Étape 2 — Baseline et modèle principal
Commencer par une baseline simple permet d'avoir un point de référence clair. Cela permet de vérifier si le Random Forest apporte réellement un gain de performance par rapport à un modèle plus simple et plus facile à interpréter.

## Étape 3 — Métriques
Le Random Forest doit normalement surperformer ou au moins égaler le Decision Tree sur Iris. L'accuracy mesure la proportion totale de bonnes prédictions, tandis que le F1-score est plus pertinent quand les classes sont déséquilibrées, car il combine précision et rappel.

## Étape 4 — Overfitting
L'overfitting apparaît quand le score d'entraînement continue de monter alors que le score de test stagne ou baisse. La validation croisée permet de mieux estimer la généralisation du modèle en multipliant les splits au lieu de dépendre d'un seul découpage.

## Étape 5 — SHAP
Les variables de pétale ressortent généralement comme les plus importantes selon SHAP. La différence entre l'importance sklearn (MDI) et SHAP est que SHAP explique la contribution réelle des variables aux prédictions, alors que MDI mesure une importance interne au modèle basée sur la réduction d'impureté.

## Étape 6 — 3 insights clés
### Insight 1 — Performance
**Observation :** Le Random Forest obtient les meilleures performances sur le jeu de test.

**Explication :** Il agrège plusieurs arbres, ce qui réduit la variance et améliore la robustesse par rapport à un seul arbre de décision.

**Implication pratique :** Il est pertinent de comparer systématiquement un modèle avancé à une baseline simple avant de choisir le modèle final.

### Insight 2 — Overfitting / Généralisation
**Observation :** L'écart entre le score d'entraînement et le score de test augmente quand la profondeur devient trop importante.

**Explication :** Le modèle apprend trop précisément les données d'entraînement et perd en capacité de généralisation.

**Implication pratique :** Il faut ajuster la complexité avec `max_depth` et confirmer le bon compromis par validation croisée.

### Insight 3 — Explicabilité
**Observation :** Les variables de pétale dominent l'explication du modèle selon SHAP.

**Explication :** Cela est cohérent avec la bonne séparabilité visuelle des espèces observée dans le pairplot.

**Implication pratique :** L'explicabilité est cruciale dans les secteurs réglementés comme le médical ou le crédit bancaire, où chaque décision doit pouvoir être justifiée.
