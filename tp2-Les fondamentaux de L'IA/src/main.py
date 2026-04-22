import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap

# Configuration
sns.set_theme(style="whitegrid")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data():
    print("[1/6] Chargement et préparation des données (Cas C)...")
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+_20Percent.txt"
    df = pd.read_csv(url, header=None, names=columns)
    
    # Cible binaire
    df['label_binary'] = (df['label'] != 'normal').astype(int)
    
    # Encodage
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    X = df.drop(['label', 'difficulty', 'label_binary'], axis=1)
    y = df['label_binary']
    
    feature_names = list(X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    X_train_sc = pd.DataFrame(X_train_sc, columns=feature_names)
    X_test_sc = pd.DataFrame(X_test_sc, columns=feature_names)
    
    return X_train_sc, X_test_sc, y_train, y_test, feature_names

def compare_models(X_train, X_test, y_train, y_test, results_dict):
    print("[2/6] Comparaison des modèles...")
    modeles = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
    }
    
    comparaison = {}
    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        pred = modele.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1_mac = f1_score(y_test, pred, average='macro')
        comparaison[nom] = {'accuracy': acc, 'f1_macro': f1_mac}
        print(f"  {nom:20} -> F1-macro : {f1_mac:.3f}")
    
    # Visualisation comparaison
    df_res = pd.DataFrame(comparaison).T
    df_res[['accuracy', 'f1_macro']].plot(kind='bar', figsize=(10, 5))
    plt.title('Comparaison des modèles (Cyber-détection)')
    plt.ylabel('Score')
    plt.ylim(0.8, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cyber_comparaison_modeles.png"))
    plt.close()
    
    results_dict['model_comparison'] = comparaison
    return modeles["Random Forest"] # RF est souvent excellent sur KDD

def detailed_evaluation(model, X_test, y_test, results_dict):
    print("[3/6] Évaluation approfondie...")
    y_pred = model.predict(X_test)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Matrice de Confusion - Cyber Détection')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cyber_confusion_matrix.png"))
    plt.close()
    
    results_dict['best_model_metrics'] = classification_report(y_test, y_pred, output_dict=True)

def analyze_shap(model, X_test, feature_names, results_dict):
    print("[4/6] Analyse SHAP...")
    X_sample = X_test[:100] # Plus petit pour la rapidité sur KDD (41 features)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure()
    # Gestion des versions de SHAP
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    if len(sv.shape) == 3: sv = sv[:, :, 1]
    
    shap.summary_plot(sv, X_sample, feature_names=feature_names, max_display=10, show=False)
    plt.title("Importance SHAP - Facteurs de détection d'attaque")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cyber_shap_summary.png"), bbox_inches='tight')
    plt.close()

def generate_html_report(results_dict):
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Livrable Final TP2 - Cybersécurité (Cas C)</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        body {{ font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 40px 20px; line-height: 1.6; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 40px; padding: 50px 30px; background: linear-gradient(135deg, #b91c1c, #7f1d1d); border-radius: 16px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.3); border: 1px solid #ef444450; }}
        .header h1 {{ margin: 0; font-size: 2.8em; font-weight: 700; color: white; }}
        .card {{ background: #1e293b; padding: 35px; border-radius: 16px; margin-bottom: 30px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2); border: 1px solid #334155; }}
        h2 {{ color: #ef4444; border-bottom: 2px solid #334155; padding-bottom: 12px; margin-top: 0; font-size: 1.8em; }}
        h3 {{ color: #93c5fd; margin-top: 25px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #0f172a; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #334155; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #ef4444; }}
        .metric-label {{ font-size: 0.9em; color: #94a3b8; }}
        .img-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
        .img-card {{ background: #0f172a; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #334155; }}
        img {{ max-width: 100%; border-radius: 8px; background-color: white; }}
        .insight-box {{ border-left: 4px solid #ef4444; background: linear-gradient(90deg, #ef444415, transparent); padding: 20px; margin: 20px 0; border-radius: 0 12px 12px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #ef4444; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Livrable Final TP2 - Cybersécurité</h1>
            <p><strong>Détection d'intrusions sur flux réseau (NSL-KDD)</strong></p>
            <p style="font-size: 0.9em; margin-top: 5px; color: #fecaca;">Réalisé par Ilyes Alouata / B3</p>
        </div>

        <div class="card">
            <h2>🧩 1. Cadrage du Projet (Etape 1)</h2>
            <table>
                <tr><th>Dimension</th><th>Réponse</th></tr>
                <tr><td>Problème métier</td><td>Détection proactive d'attaques réseau.</td></tr>
                <tr><td>Variable cible (y)</td><td>label_binary (Normal vs Attaque)</td></tr>
                <tr><td>Métrique principale</td><td>Recall (Crucial pour ne rater aucune attaque)</td></tr>
                <tr><td>Risques identifiés</td><td>Faux positifs bloquant le trafic légitime.</td></tr>
                <tr><td>Niveau AI Act</td><td>Haut risque (Infrastructures critiques)</td></tr>
            </table>
        </div>

        <div class="card">
            <h2>📊 2. Résultats de la Modélisation</h2>
            <div class="metric-grid">
                <div class="metric-card"><div class="metric-value">{results_dict['model_comparison']['Random Forest']['accuracy']*100:.1f}%</div><div class="metric-label">Accuracy (RF)</div></div>
                <div class="metric-card"><div class="metric-value">{results_dict['model_comparison']['Random Forest']['f1_macro']:.3f}</div><div class="metric-label">F1-macro</div></div>
                <div class="metric-card"><div class="metric-value">NSL-KDD</div><div class="metric-label">Dataset</div></div>
            </div>
            <div class="img-grid">
                <div class="img-card"><img src="outputs/cyber_comparaison_modeles.png"><div class="img-title">Comparatif Logistic vs RF vs XGB</div></div>
                <div class="img-card"><img src="outputs/cyber_confusion_matrix.png"><div class="img-title">Matrice de Confusion (Cyber)</div></div>
            </div>
        </div>

        <div class="card">
            <h2>🧠 3. Analyse SHAP & Explicabilité</h2>
            <p>Quelles sont les signatures réseau qui trahissent une attaque ?</p>
            <div class="img-card" style="margin-top: 15px;">
                <img src="outputs/cyber_shap_summary.png">
                <div class="img-title">Top variables prédictives d'attaques (SHAP)</div>
            </div>
        </div>

        <div class="card">
            <h2>⚖️ 4. Conformité & AI Act (Etape 6)</h2>
            <div class="insight-box">
                <h4>Justification AI Act</h4>
                <p>Le système est classé <strong>Haut Risque</strong> car il assure la sécurité d'infrastructures numériques essentielles et manipule des données de trafic sensibles.</p>
            </div>
            <table>
                <tr><th>Critère</th><th>Analyse</th></tr>
                <tr><td>Base légale RGPD</td><td>Intérêt légitime (Sécurité)</td></tr>
                <tr><td>DPIA</td><td>Requis (Surveillance réseau systématique)</td></tr>
                <tr><td>Supervision humaine</td><td>Indispensable via une équipe SOC dédiée.</td></tr>
            </table>
        </div>
    </div>
</body>
</html>"""
    
    with open("Livrable_Final_TP2.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("\n[+] Rapport Premium généré : Livrable_Final_TP2.html")

def main():
    print("=== Début du Pipeline TP2 (Cyber) ===")
    results = {}
    
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    best_model = compare_models(X_train, X_test, y_train, y_test, results)
    
    detailed_evaluation(best_model, X_test, y_test, results)
    
    analyze_shap(best_model, X_test, feature_names, results)
    
    generate_html_report(results)
    print("\n=== Pipeline TP2 (Cyber) terminé avec succès ! ===")

if __name__ == "__main__":
    main()
