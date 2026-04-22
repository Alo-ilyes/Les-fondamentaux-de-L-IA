# Réponses rédigées — TP2 (Mini-projet Cybersécurité)

## Étape 1 — Cadrage du projet
- **Problème métier** : Détecter automatiquement les intrusions et cyber-attaques dans les flux réseau pour protéger l'infrastructure.
- **KPI métier** : Taux de détection des attaques (Recall), Temps de réponse aux incidents, Taux de faux positifs (pour ne pas bloquer le trafic légitime).
- **Variable cible (y)** : `label_binary` (0: Normal, 1: Attack).
- **Type de tâche ML** : Classification binaire.
- **Métrique ML principale** : Recall (il est plus critique de rater une attaque que de bloquer un utilisateur légitime par erreur).
- **Risques identifiés** : Faux négatifs (attaque non détectée), Faux positifs (interruption de service légitime).
- **Niveau AI Act** : Haut risque (Utilisé dans la gestion et l'exploitation des infrastructures critiques).

## Étape 2 — Préparation des données
- Chargement du dataset NSL-KDD (sous-ensemble 20%).
- Encodage des variables catégorielles (`protocol_type`, `service`, `flag`) via `LabelEncoder`.
- Création d'une cible binaire (Normal vs Attack).
- Standardisation des variables numériques.

## Étape 3 — Modélisation
- Comparaison de modèles (Régression Logistique, Random Forest, XGBoost).
- Le Random Forest et XGBoost sont particulièrement efficaces pour détecter les patterns d'attaques complexes.

## Étape 4 — Évaluation
- Analyse de la matrice de confusion.
- Focus sur le Recall pour minimiser les "Faux Négatifs".

## Étape 5 — SHAP
- Identification des caractéristiques réseau les plus suspectes (ex: `src_bytes`, `count`, `dst_host_srv_count`).

## Étape 6 — Conformité AI Act
- **Niveau de risque** : Haut risque.
- **Justification** : Sécurité des infrastructures numériques et des données.
- **Base légale RGPD** : Intérêt légitime (sécurité du réseau et des données).
- **DPIA requis** : Oui, car traitement de données de trafic pouvant inclure des données personnelles et profilage de sécurité.
- **Supervision humaine** : Nécessaire (SOC Analyst) pour valider les alertes critiques.
