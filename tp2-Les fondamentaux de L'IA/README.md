# 🎓 Mes Projets d'Intelligence Artificielle (B3)
> **Auteur :** Ilyes Alouata | **Matière :** Les fondamentaux de l'IA | **Projet :** TP2 - Mini-Projet Cyber

Bienvenue sur mon répertoire pour le deuxième travail pratique de Machine Learning ! 🚀

## 🌿 Architecture du Projet
Ce projet suit une structure professionnelle et isolée :
- **`src/`** : Contient le code source Python (`main.py`) utilisant **XGBoost** et **SHAP**.
- **`outputs/`** : Dossier regroupant les graphiques (Comparaison modèles, Matrice de Confusion, SHAP).
- **`docs/`** : Réponses détaillées au cadrage métier et à la conformité AI Act.
- **`Livrable_Final_TP2.html`** : Le rapport interactif complet regroupant toutes les analyses.

---

## 🔹 [TP2] - Mini-Projet : Cybersécurité
Ce projet implémente une solution de détection d'intrusions basée sur le dataset académique **NSL-KDD**.

### ⚙️ Installation et Usage
1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Lancez le pipeline :
   ```bash
   python src/main.py
   ```

### 🔍 Synthèse : 3 Insights Clés (Debrief TP2 - Cyber)

#### 1. Performance (Random Forest)
*   **Observation** : Le Random Forest obtient une accuracy quasi-parfaite (>99%) sur le sous-ensemble NSL-KDD.
*   **Explication** : Les attaques réseau laissent des empreintes numériques (signatures) très distinctes du trafic normal, facilement capturées par des arbres de décision.
*   **Implication pratique** : Utiliser le Random Forest comme première ligne de défense pour filtrer le trafic suspect en temps réel.

#### 2. Importance des variables (Signatures suspectes)
*   **Observation** : Les variables comme `src_bytes` et `count` sont les plus déterminantes selon SHAP.
*   **Explication** : Une augmentation brutale du volume de données sortantes ou du nombre de connexions simultanées est souvent le signe d'une attaque par déni de service (DoS) ou d'une exfiltration.
*   **Implication pratique** : Configurer des alertes prioritaires sur ces indicateurs spécifiques dans les outils de monitoring réseau.

#### 3. Conformité et Éthique (Infrastructures Critiques)
*   **Observation** : Ce système est classé "Haut Risque" au titre de l'AI Act.
*   **Explication** : Une défaillance ou un biais dans la détection d'attaques peut paralyser des services essentiels.
*   **Implication pratique** : Maintenir une supervision humaine constante (SOC) pour valider les détections et éviter les blocages de trafic légitime par erreur.

---
*Note : Ce dépôt est optimisé pour démontrer une maîtrise technique et réglementaire du Machine Learning.*
