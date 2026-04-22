import os
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
sns.set_theme(style="whitegrid")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_explore():
    print("[1/10] Chargement et exploration de Fashion-MNIST...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
                   'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine']
    
    # Visualisation exemples
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        idx = np.where(y_train == i)[0][0]
        ax.imshow(X_train[idx], cmap='gray')
        ax.set_title(class_names[i], fontsize=10)
        ax.axis('off')
    plt.suptitle('Catalogue Zalando - Exemples par catégorie', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fashion_samples.png"))
    plt.close()
    
    return (X_train, y_train), (X_test, y_test), class_names

def preprocess(X_train, X_test):
    print("[2/10] Prétraitement des images...")
    # Normalisation
    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0
    
    # Pour ML classique
    X_train_flat = X_train_norm.reshape(-1, 784)
    X_test_flat = X_test_norm.reshape(-1, 784)
    
    # Pour CNN
    X_train_cnn = X_train_norm.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test_norm.reshape(-1, 28, 28, 1)
    
    return X_train_norm, X_test_norm, X_train_flat, X_test_flat, X_train_cnn, X_test_cnn

def train_baseline(X_train_flat, y_train, X_test_flat, y_test):
    print("[3/10] Entraînement Baseline (Random Forest)...")
    # Utilisation d'un sous-ensemble pour la baseline si nécessaire, mais Fashion-MNIST est gérable
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train)
    y_pred = rf.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Random Forest Accuracy : {acc*100:.1f}%")
    return acc

def train_mlp(X_train_norm, y_train, X_test_norm, y_test):
    print("[4/10] Entraînement MLP (Réseau Dense)...")
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_norm, y_train, epochs=10, batch_size=64, validation_split=0.15, verbose=0)
    acc = model.evaluate(X_test_norm, y_test, verbose=0)[1]
    print(f"  MLP Accuracy : {acc*100:.1f}%")
    return model, history, acc

def train_cnn(X_train_cnn, y_train, X_test_cnn, y_test):
    print("[5/10] Entraînement CNN (Convolutionnel)...")
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_cnn, y_train, epochs=10, batch_size=64, validation_split=0.15, verbose=0)
    acc = model.evaluate(X_test_cnn, y_test, verbose=0)[1]
    print(f"  CNN Accuracy : {acc*100:.1f}%")
    return model, history, acc

def plot_learning_curves(h_dense, h_cnn):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(h_dense.history['accuracy'], label='Train')
    axes[0].plot(h_dense.history['val_accuracy'], label='Val')
    axes[0].set_title('Apprentissage Réseau Dense')
    axes[0].legend()
    
    axes[1].plot(h_cnn.history['accuracy'], label='Train')
    axes[1].plot(h_cnn.history['val_accuracy'], label='Val')
    axes[1].set_title('Apprentissage CNN')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fashion_learning_curves.png"))
    plt.close()

def error_analysis(model, X_test_cnn, y_test, class_names):
    y_pred = model.predict(X_test_cnn)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion - CNN')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fashion_confusion_matrix.png"))
    plt.close()
    
    # Analyse des erreurs
    errors = np.where(y_pred_classes != y_test)[0]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flat):
        idx = errors[i]
        ax.imshow(X_test_cnn[idx].reshape(28,28), cmap='gray')
        conf = y_pred[idx, y_pred_classes[idx]] * 100
        ax.set_title(f"P: {class_names[y_pred_classes[idx]]} ({conf:.0f}%)\nR: {class_names[y_test[idx]]}", 
                     color='red', fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fashion_errors.png"))
    plt.close()

def visualize_activations(model, X_test_cnn, y_test, class_names):
    # Filtres 1re couche
    filters, biases = model.layers[0].get_weights()
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[:, :, 0, i], cmap='gray')
        ax.axis('off')
    plt.suptitle('Filtres appris par la 1re couche du CNN', fontsize=13)
    plt.savefig(os.path.join(OUTPUT_DIR, "fashion_filters.png"))
    plt.close()
    
    # Activations
    activation_model = keras.Model(inputs=model.inputs, outputs=model.layers[0].output)
    sample_idx = np.where(y_test == 7)[0][0] # Sneaker
    sample = X_test_cnn[sample_idx:sample_idx+1]
    activations = activation_model.predict(sample, verbose=0)
    
    fig, axes = plt.subplots(1, 9, figsize=(16, 2.5))
    axes[0].imshow(X_test_cnn[sample_idx].reshape(28,28), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    for i in range(8):
        axes[i+1].imshow(activations[0, :, :, i], cmap='viridis')
        axes[i+1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fashion_activations.png"))
    plt.close()

def get_base64_img(filename):
    """Encode une image en base64 pour l'intégrer au HTML."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_report(results):
    """
    Génère un rapport HTML Ultra-Premium (Standalone) pour le TP3.
    """
    print("[10/10] Génération du rapport Ultra-Premium...")
    
    img_samples = get_base64_img("fashion_samples.png")
    img_curves = get_base64_img("fashion_learning_curves.png")
    img_cm = get_base64_img("fashion_confusion_matrix.png")
    img_errors = get_base64_img("fashion_errors.png")
    img_filters = get_base64_img("fashion_filters.png")
    img_activations = get_base64_img("fashion_activations.png")

    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Livrable Final TP3 - Ilyes Alouata</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        :root {{
            --bg-color: #050a18;
            --card-bg: rgba(15, 23, 42, 0.85);
            --accent: #f59e0b;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --cnn-color: #38bdf8;
        }}

        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(245, 158, 11, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(56, 189, 248, 0.1) 0px, transparent 50%);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
        }}

        .container {{ max-width: 1100px; margin: 0 auto; }}

        .header {{
            padding: 60px 40px;
            background: var(--card-bg);
            border-radius: 24px;
            border: 1px solid rgba(245, 158, 11, 0.2);
            text-align: center;
            margin-bottom: 50px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        }}

        .header h1 {{
            margin: 0;
            font-size: 3.2em;
            background: linear-gradient(to right, #fbbf24, #f59e0b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .section {{
            background: var(--card-bg);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 40px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}

        h2 {{ color: var(--accent); display: flex; align-items: center; gap: 15px; }}

        .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #0c1222; padding: 20px; border-radius: 16px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.05); }}
        .metric-val {{ font-size: 2.2em; font-weight: 700; color: var(--cnn-color); }}
        
        .img-wrapper {{ background: #000; border-radius: 16px; padding: 10px; border: 1px solid rgba(245, 158, 11, 0.1); margin-top: 15px; }}
        .img-wrapper img {{ width: 100%; border-radius: 10px; }}
        .img-caption {{ text-align: center; color: var(--text-dim); font-size: 0.85em; margin-top: 10px; }}

        .insight-card {{
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.05), transparent);
            border-left: 4px solid var(--accent);
            padding: 20px;
            margin-top: 20px;
        }}

        .footer {{ text-align: center; color: var(--text-dim); font-size: 0.85em; margin-top: 60px; }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div style="text-transform: uppercase; letter-spacing: 2px; font-size: 0.8em; color: var(--accent); margin-bottom: 10px;">Deep Learning & Computer Vision</div>
            <h1>Classification E-Commerce</h1>
            <p>Réseaux de Neurones Convolutifs sur Fashion-MNIST</p>
            <div style="margin-top:30px; color: var(--text-dim);">Auteur : <strong>Ilyes Alouata</strong> | Zalando Catalog Intelligence</div>
        </header>

        <section class="section">
            <h2>🧩 1. Contexte Métier</h2>
            <p>L'objectif est de corriger les erreurs de catégorisation (12% initialement) pour descendre sous les 5%. Nous utilisons le dataset <strong>Fashion-MNIST</strong> (images 28x28 pixels).</p>
            <div class="img-wrapper">
                <img src="data:image/png;base64,{img_samples}">
                <div class="img-caption">Échantillons du catalogue (Sneakers, Pulls, Robes...)</div>
            </div>
        </section>

        <section class="section">
            <h2>📊 2. Benchmarking Algorithmique</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-val">{results['rf']*100:.1f}%</div>
                    <div style="color: var(--text-dim)">Random Forest (Baseline)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{results['mlp']*100:.1f}%</div>
                    <div style="color: var(--text-dim)">Réseau Dense (MLP)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{results['cnn']*100:.1f}%</div>
                    <div style="color: var(--text-dim)">CNN (Production)</div>
                </div>
            </div>
            <div class="img-wrapper">
                <img src="data:image/png;base64,{img_curves}">
                <div class="img-caption">Évolution de l'apprentissage (Accuracy Train/Val)</div>
            </div>
        </section>

        <section class="section">
            <h2>🔍 3. Analyse des Erreurs</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                <div class="img-wrapper">
                    <img src="data:image/png;base64,{img_cm}">
                </div>
                <div class="img-wrapper">
                    <img src="data:image/png;base64,{img_errors}">
                </div>
            </div>
            <p style="margin-top:20px;">Le CNN atteint ~91% d'accuracy. Les erreurs persistantes concernent les "Pulls" confondus avec les "Chemises", ce qui est cohérent vu la résolution 28x28.</p>
        </section>

        <section class="section">
            <h2>👁️ 4. Interprétabilité : L'œil du CNN</h2>
            <p>Visualisation des filtres et des activations de la première couche.</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                <div class="img-wrapper">
                    <img src="data:image/png;base64,{img_filters}">
                    <div class="img-caption">Filtres de convolution (Bords/Textures)</div>
                </div>
                <div class="img-wrapper">
                    <img src="data:image/png;base64,{img_activations}">
                    <div class="img-caption">Activations (Ce que le réseau "voit")</div>
                </div>
            </div>
        </section>

        <section class="section">
            <h2>💡 5. Recommandations Comité Technique</h2>
            <div class="insight-card">
                <h4>🚀 Avantage CNN</h4>
                <p>Le mécanisme de convolution permet de capturer la structure spatiale, surpassant les modèles denses. Le surcoût calcul est justifié par le gain de performance.</p>
            </div>
            <div class="insight-card">
                <h4>🧠 Optimisation de Production</h4>
                <p>Pour passer sous les 5% d'erreur, il est recommandé de passer à des images HD (224x224) et d'utiliser l'augmentation de données.</p>
            </div>
        </section>

        <div class="footer">
            Rapport Deep Learning généré par Ilyes Alouata — 2026
        </div>
    </div>
</body>
</html>"""
    
    with open("Livrable_Final_TP3.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("\n[+] Rapport Ultra-Premium généré : Livrable_Final_TP3.html")

def main():
    print("=== Début du Pipeline TP3 ===")
    (X_train, y_train), (X_test, y_test), class_names = load_and_explore()
    X_train_norm, X_test_norm, X_train_flat, X_test_flat, X_train_cnn, X_test_cnn = preprocess(X_train, X_test)
    
    acc_rf = train_baseline(X_train_flat, y_train, X_test_flat, y_test)
    m_dense, h_dense, acc_dense = train_mlp(X_train_norm, y_train, X_test_norm, y_test)
    m_cnn, h_cnn, acc_cnn = train_cnn(X_train_cnn, y_train, X_test_cnn, y_test)
    
    plot_learning_curves(h_dense, h_cnn)
    error_analysis(m_cnn, X_test_cnn, y_test, class_names)
    visualize_activations(m_cnn, X_test_cnn, y_test, class_names)
    
    generate_report({'rf': acc_rf, 'mlp': acc_dense, 'cnn': acc_cnn})
    print("\n=== Pipeline TP3 terminé avec succès ! ===")

if __name__ == "__main__":
    main()
