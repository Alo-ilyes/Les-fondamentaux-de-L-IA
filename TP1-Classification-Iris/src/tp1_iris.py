import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

os.makedirs("outputs", exist_ok=True)

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("=== EXPLORATION DES DONNÉES ===")
print(f"Forme du dataset : {df.shape}")
print("\nDistribution des classes :")
print(df["species"].value_counts())
print(f"\nValeurs manquantes : {df.isnull().sum().sum()}")
print("\nAperçu :")
print(df.head(3))
print("\nStatistiques descriptives :")
print(df.describe())

sns.pairplot(df, hue="species", markers=["o", "s", "D"], plot_kws=dict(alpha=0.7), diag_kind="hist")
plt.suptitle("Dataset Iris — Séparabilité des classes", y=1.02)
plt.savefig("outputs/iris_pairplot.png", bbox_inches="tight")
plt.close()

le = LabelEncoder()
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
y = le.fit_transform(df["species"])
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nEntraînement : {len(X_train)} échantillons | Test : {len(X_test)} échantillons")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_sc, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_sc, y_train)

print("Modèles entraînés.")

y_pred_dt = dt.predict(X_test_sc)
y_pred_rf = rf.predict(X_test_sc)

results = []
for nom, pred in [("Decision Tree", y_pred_dt), ("Random Forest", y_pred_rf)]:
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    results.append({"modele": nom, "accuracy": round(acc, 4), "f1_weighted": round(f1, 4)})

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/metrics_comparison.csv", index=False)

print("\n=== COMPARAISON BASELINE vs RANDOM FOREST ===")
print(results_df)
print("\n=== RAPPORT DÉTAILLÉ — RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matrice de confusion — Random Forest")
plt.ylabel("Réalité")
plt.xlabel("Prédiction")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

n_estimators_range = [10, 25, 50, 100, 200, 500]
scores_cv = []
print("\n=== VALIDATION CROISÉE ===")
for n in n_estimators_range:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    cv = cross_val_score(m, X_train_sc, y_train, cv=5, scoring="accuracy")
    scores_cv.append({"n_estimators": n, "cv_accuracy_mean": round(cv.mean(), 4), "cv_accuracy_std": round(cv.std(), 4)})
    print(f"n_estimators={n:4d} → CV accuracy : {cv.mean()*100:.1f}% (±{cv.std()*100:.1f}%)")

pd.DataFrame(scores_cv).to_csv("outputs/cv_results.csv", index=False)

profondeurs = range(1, 20)
scores_train = []
scores_test = []
for d in profondeurs:
    m = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42)
    m.fit(X_train_sc, y_train)
    scores_train.append(m.score(X_train_sc, y_train))
    scores_test.append(m.score(X_test_sc, y_test))

pd.DataFrame({"max_depth": list(profondeurs), "train_accuracy": scores_train, "test_accuracy": scores_test}).to_csv("outputs/overfitting_scores.csv", index=False)

plt.figure(figsize=(9, 4))
plt.plot(profondeurs, scores_train, "b-o", label="Score entraînement")
plt.plot(profondeurs, scores_test, "r-o", label="Score test")
plt.xlabel("Profondeur maximale (max_depth)")
plt.ylabel("Accuracy")
plt.title("Underfitting vs Overfitting — Random Forest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/overfitting.png")
plt.close()

rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_final.fit(X_train_sc, y_train)
explainer = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_test_sc)

plt.figure()
shap.summary_plot(shap_values, X_test_sc, feature_names=feature_names, class_names=list(le.classes_), plot_type='bar', show=False)
plt.title("SHAP — Importance globale des variables (3 classes)")
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", bbox_inches="tight")
plt.close()

importances_sklearn = rf_final.feature_importances_
if isinstance(shap_values, list):
    importances_shap = np.abs(shap_values[2]).mean(axis=0)
else:
    importances_shap = np.abs(shap_values[:, :, 2]).mean(axis=0)

pd.DataFrame({
    "feature": feature_names,
    "importance_sklearn": importances_sklearn,
    "importance_shap_virginica": importances_shap,
}).sort_values("importance_shap_virginica", ascending=False).to_csv("outputs/feature_importance.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.barh(feature_names, importances_sklearn, color="steelblue")
ax1.set_title("Feature Importance — sklearn (MDI)")
ax1.set_xlabel("Importance")
ax2.barh(feature_names, importances_shap, color="darkorange")
ax2.set_title("Feature Importance — SHAP (classe virginica)")
ax2.set_xlabel("|SHAP value| moyen")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

with open("outputs/synthese.txt", "w", encoding="utf-8") as f:
    f.write("""Insight 1 — Performance\nObservation : Le Random Forest obtient de meilleures performances globales que le Decision Tree.\nExplication : L'agrégation de plusieurs arbres réduit la variance et améliore la robustesse des prédictions.\nImplication pratique : Il faut toujours comparer un modèle avancé à une baseline simple avant validation finale.\n\nInsight 2 — Overfitting / Généralisation\nObservation : Lorsque la profondeur augmente, le score d'entraînement devient plus élevé que le score de test.\nExplication : Le modèle mémorise davantage les données d'entraînement et généralise moins bien.\nImplication pratique : Il faut régler max_depth et confirmer les choix par validation croisée.\n\nInsight 3 — Explicabilité\nObservation : Les variables de pétale dominent généralement l'importance SHAP.\nExplication : Cela est cohérent avec la bonne séparabilité visuelle des espèces sur ces mesures.\nImplication pratique : L'explicabilité est indispensable dans les domaines réglementés comme la santé ou le crédit.\n""")

print("Pack TP généré.")
