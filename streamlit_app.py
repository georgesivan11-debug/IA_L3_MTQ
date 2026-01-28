import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Configuration de la page
st.set_page_config(
    page_title="Classification des Iris 🌸",
    page_icon="🌸",
    layout="wide"
)

# Titre principal
st.title("🌸 Classification des Fleurs Iris")
st.markdown("---")

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    """Charge le modèle et le scaler s'ils existent"""
    try:
        if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
            with open('best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        else:
            return None, None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None

# Fonction pour charger les données
@st.cache_data
def load_data():
    """Charge le dataset Iris"""
    try:
        # Essayer différents chemins possibles
        possible_paths = ['Iris.csv', 'iris.csv', 'IRIS.csv']
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path, sep=';')
                # Normaliser les noms de colonnes
                df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
                return df
        
        # Si aucun fichier trouvé, utiliser le dataset de sklearn
        st.warning("Fichier Iris.csv non trouvé. Utilisation du dataset sklearn à la place.")
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
        df['species'] = [iris.target_names[i] for i in iris.target]
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

# Charger le modèle et les données
model, scaler = load_model()
df = load_data()

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisissez une page:",
    ["🏠 Accueil", "🔮 Prédiction", "📊 Analyse Complète", "📈 Visualisations Avancées", "ℹ️ À propos"]
)

# ========== PAGE ACCUEIL ==========
if page == "🏠 Accueil":
    st.header("Bienvenue sur l'application de classification des Iris !")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 À propos du projet")
        st.write("""
        Cette application utilise le machine learning pour classifier les fleurs iris 
        en trois espèces différentes :
        - **Iris Setosa**
        - **Iris Versicolor**
        - **Iris Virginica**
        
        Les prédictions sont basées sur quatre caractéristiques :
        - Longueur du sépale
        - Largeur du sépale
        - Longueur du pétale
        - Largeur du pétale
        """)
    
    with col2:
        st.subheader("🎯 Fonctionnalités")
        st.write("""
        - ✅ Prédiction interactive en temps réel
        - 📊 Visualisations complètes (tous les exercices)
        - 📈 Analyse statistique approfondie
        - 🤖 6 modèles de ML comparés
        - 🎨 Interface intuitive et interactive
        """)
    
    # Statut du modèle et données
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if model is not None:
            st.success("✅ Modèle chargé avec succès !")
        else:
            st.warning("⚠️ Modèle non disponible. Exécutez `tp_iris_complet.py` d'abord.")
    
    with col2:
        if df is not None:
            st.success(f"✅ Dataset chargé : {len(df)} échantillons")
        else:
            st.error("❌ Dataset non disponible")
    
    st.markdown("---")
    st.info("👈 Utilisez le menu à gauche pour naviguer entre les différentes pages")

# ========== PAGE PRÉDICTION ==========
elif page == "🔮 Prédiction":
    st.header("Prédiction d'espèce d'Iris")
    
    if model is None or scaler is None:
        st.error("❌ Modèle non disponible. Veuillez d'abord entraîner le modèle.")
        st.info("Exécutez le fichier `tp_iris_complet.py` pour créer les fichiers nécessaires.")
        st.code("python tp_iris_complet.py", language="bash")
        st.stop()
    
    st.write("Entrez les mesures de la fleur pour prédire son espèce :")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Caractéristiques du Sépale")
        sepal_length = st.slider(
            "Longueur du sépale (cm)", 
            min_value=4.0, 
            max_value=8.0, 
            value=5.1, 
            step=0.1
        )
        sepal_width = st.slider(
            "Largeur du sépale (cm)", 
            min_value=2.0, 
            max_value=4.5, 
            value=3.5, 
            step=0.1
        )
    
    with col2:
        st.subheader("Caractéristiques du Pétale")
        petal_length = st.slider(
            "Longueur du pétale (cm)", 
            min_value=1.0, 
            max_value=7.0, 
            value=1.4, 
            step=0.1
        )
        petal_width = st.slider(
            "Largeur du pétale (cm)", 
            min_value=0.1, 
            max_value=2.5, 
            value=0.2, 
            step=0.1
        )
    
    st.markdown("---")
    
    if st.button("🔮 Prédire l'espèce", type="primary"):
        try:
            # Préparer les données
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            features_scaled = scaler.transform(features)
            
            # Faire la prédiction
            prediction = model.predict(features_scaled)[0]
            
            st.success(f"🎯 Espèce prédite : **{prediction.upper()}**")
            
            # Afficher les probabilités si disponible
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                
                st.subheader("📊 Probabilités :")
                
                # Créer un dataframe pour les probabilités
                if df is not None and 'species' in df.columns:
                    species_names = sorted(df['species'].unique())
                else:
                    species_names = ['setosa', 'versicolor', 'virginica']
                
                prob_df = pd.DataFrame({
                    'Espèce': species_names,
                    'Probabilité': probabilities
                })
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Graphique des probabilités
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                    bars = ax.barh(prob_df['Espèce'], prob_df['Probabilité'], color=colors)
                    ax.set_xlabel('Probabilité', fontsize=12)
                    ax.set_xlim([0, 1])
                    ax.set_title('Probabilités par espèce', fontsize=14, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Ajouter les valeurs sur les barres
                    for i, (bar, v) in enumerate(zip(bars, prob_df['Probabilité'])):
                        ax.text(v + 0.02, i, f'{v:.2%}', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Afficher le tableau
                    st.dataframe(
                        prob_df.style.format({'Probabilité': '{:.2%}'})
                        .background_gradient(cmap='RdYlGn', subset=['Probabilité']),
                        use_container_width=True
                    )
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.info("Vérifiez que le modèle a été correctement entraîné.")

# ========== PAGE ANALYSE COMPLÈTE ==========
elif page == "📊 Analyse Complète":
    st.header("Analyse Complète des Données Iris")
    
    if df is None:
        st.error("❌ Dataset non disponible")
        st.stop()
    
    # Aperçu des données
    st.subheader("📋 Aperçu des données")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.metric("Nombre total d'échantillons", len(df))
        st.metric("Nombre de variables", len(df.columns) - 1)
        if 'species' in df.columns:
            st.metric("Nombre d'espèces", df['species'].nunique())
    
    st.markdown("---")
    
    # Statistiques descriptives
    st.subheader("📊 Statistiques descriptives")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # EXERCICE 1 : Visualisations des effectifs
    st.subheader("📊 Exercice 1 : Effectifs par espèce")
    
    if 'species' in df.columns:
        effectifs = df['species'].value_counts()
        
        # Afficher les effectifs
        st.write("**Effectifs :**")
        st.dataframe(effectifs, use_container_width=True)
        
        # Créer 4 graphiques différents
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramme
            fig, ax = plt.subplots(figsize=(8, 5))
            effectifs.plot(kind='bar', ax=ax, color=['green', 'orange', 'blue'])
            ax.set_title("Histogramme des espèces", fontsize=14, fontweight='bold')
            ax.set_xlabel("Espèces")
            ax.set_ylabel("Effectif")
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Barres groupées
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(effectifs.index, effectifs.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title("Barres groupées des espèces", fontsize=14, fontweight='bold')
            ax.set_xlabel("Espèces")
            ax.set_ylabel("Effectif")
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Diagramme circulaire
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax.pie(effectifs.values, labels=effectifs.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            ax.set_title("Répartition des espèces (diagramme circulaire)", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Diagramme en cascade
            fig, ax = plt.subplots(figsize=(8, 5))
            values = effectifs.values
            cum = np.cumsum(values)
            ax.bar(effectifs.index, values, color=['green', 'orange', 'blue'], alpha=0.7)
            ax.plot(effectifs.index, cum, marker='o', color='red', linewidth=2, 
                   markersize=8, label='Cumulé')
            ax.set_title("Diagramme en cascade", fontsize=14, fontweight='bold')
            ax.set_ylabel("Effectif")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # EXERCICE 2 : Variables quantitatives
    st.subheader("📈 Exercice 2 : Variables quantitatives")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Créer des onglets pour chaque variable
        tabs = st.tabs([col.upper() for col in numeric_cols])
        
        for i, col_name in enumerate(numeric_cols):
            with tabs[i]:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Statistiques :**")
                    stats = df[col_name].describe()
                    st.dataframe(stats, use_container_width=True)
                
                with col2:
                    # Histogramme
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(df[col_name], bins=15, edgecolor='black', color='skyblue', alpha=0.7)
                    ax.set_title(f"Distribution de {col_name}", fontsize=14, fontweight='bold')
                    ax.set_xlabel(f"{col_name} (cm)")
                    ax.set_ylabel("Fréquence")
                    ax.grid(axis='y', alpha=0.3)
                    ax.axvline(df[col_name].mean(), color='red', linestyle='--', 
                              linewidth=2, label=f'Moyenne: {df[col_name].mean():.2f}')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

# ========== PAGE VISUALISATIONS AVANCÉES ==========
elif page == "📈 Visualisations Avancées":
    st.header("Visualisations Avancées")
    
    if df is None:
        st.error("❌ Dataset non disponible")
        st.stop()
    
    # EXERCICE 3 : Pairplot
    st.subheader("🔗 Exercice 3 : Nuages de points (Pairplot)")
    st.write("Relations entre toutes les paires de variables, colorées par espèce")
    
    if st.checkbox("Afficher le Pairplot (peut être lent)", value=False):
        with st.spinner("Génération du pairplot..."):
            fig = sns.pairplot(df, hue='species', palette=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                             diag_kind='hist', height=2.5)
            fig.fig.suptitle("Matrice de nuages de points par espèce", y=1.02, fontsize=16, fontweight='bold')
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # EXERCICE 4 : Boxplots
    st.subheader("📦 Exercice 4 : Boxplots par espèce")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols and 'species' in df.columns:
        # Sélecteur de variable
        selected_var = st.selectbox(
            "Choisir une variable à analyser:",
            numeric_cols,
            format_func=lambda x: x.upper()
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot simple
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column=selected_var, by='species', ax=ax)
            plt.suptitle('')
            ax.set_title(f'{selected_var} par espèce', fontsize=14, fontweight='bold')
            ax.set_xlabel('Espèce')
            ax.set_ylabel(f'{selected_var} (cm)')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Boxplot avec seaborn (plus esthétique)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='species', y=selected_var, ax=ax,
                       palette=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title(f'{selected_var} par espèce (Seaborn)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Espèce')
            ax.set_ylabel(f'{selected_var} (cm)')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Afficher tous les boxplots
        if st.checkbox("Afficher tous les boxplots ensemble", value=True):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols):
                sns.boxplot(data=df, x='species', y=col, ax=axes[i],
                           palette=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Espèce')
                axes[i].set_ylabel(f'{col} (cm)')
                axes[i].grid(axis='y', alpha=0.3)
            
            plt.suptitle('Comparaison de toutes les variables par espèce', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # EXERCICE 5 : Corrélations
    st.subheader("🔗 Exercice 5 : Corrélations et visualisations avancées")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Matrice de corrélation
            st.write("**Matrice de corrélation :**")
            correlation = numeric_df.corr()
            st.dataframe(correlation.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1),
                        use_container_width=True)
        
        with col2:
            # Heatmap de corrélation
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                       fmt='.2f')
            ax.set_title("Heatmap de corrélation", fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Nuage de points pétales avec distinction par espèce
        st.write("**Nuage de points : Longueur vs Largeur du pétale**")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'species' in df.columns:
            colors_map = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
            
            for esp in df['species'].unique():
                sous_df = df[df['species'] == esp]
                color = colors_map.get(esp, 'gray')
                ax.scatter(
                    sous_df['petallength'],
                    sous_df['petalwidth'],
                    label=esp.capitalize(),
                    s=100,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.5,
                    color=color
                )
        
        ax.set_title("Relation Longueur/Largeur des pétales par espèce", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Longueur du pétale (cm)", fontsize=12)
        ax.set_ylabel("Largeur du pétale (cm)", fontsize=12)
        ax.legend(title='Espèce', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Scatter matrix interactif
        st.write("**Choix personnalisé de variables à comparer :**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            var_x = st.selectbox("Variable X:", numeric_cols, index=0)
        
        with col2:
            var_y = st.selectbox("Variable Y:", numeric_cols, index=1)
        
        if var_x and var_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if 'species' in df.columns:
                for esp in df['species'].unique():
                    sous_df = df[df['species'] == esp]
                    color = colors_map.get(esp, 'gray')
                    ax.scatter(
                        sous_df[var_x],
                        sous_df[var_y],
                        label=esp.capitalize(),
                        s=100,
                        alpha=0.6,
                        edgecolors='black',
                        linewidths=0.5,
                        color=color
                    )
            
            ax.set_title(f"Relation {var_x} vs {var_y}", fontsize=14, fontweight='bold')
            ax.set_xlabel(f"{var_x} (cm)", fontsize=12)
            ax.set_ylabel(f"{var_y} (cm)", fontsize=12)
            ax.legend(title='Espèce')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ========== PAGE À PROPOS ==========
elif page == "ℹ️ À propos":
    st.header("À propos de ce projet")
    
    st.markdown("""
    ### 🎓 Projet TP - Classification des Iris
    
    Ce projet a été développé dans le cadre d'un travail pratique sur le Machine Learning.
    
    #### 🛠️ Technologies utilisées :
    - **Python** : Langage de programmation
    - **Scikit-learn** : Bibliothèque de Machine Learning
    - **Pandas & NumPy** : Manipulation de données
    - **Matplotlib & Seaborn** : Visualisation
    - **Streamlit** : Interface web interactive
    - **Flask** : API REST (optionnel)
    
    #### 📚 Dataset :
    Le dataset Iris est un classique en Machine Learning, créé par Edgar Anderson 
    et popularisé par R.A. Fisher en 1936.
    
    - **150 échantillons** (50 par espèce)
    - **4 caractéristiques** numériques
    - **3 classes** équilibrées
    
    #### 🤖 Modèles testés :
    - K-Nearest Neighbors (KNN)
    - Régression Logistique
    - Arbre de Décision
    - Naive Bayes
    - SVM (Support Vector Machine)
    - Réseau de Neurones (MLP)
    
    #### 📊 Exercices inclus :
    - ✅ **Exercice 1** : Visualisation des effectifs (histogramme, pie, barres, cascade)
    - ✅ **Exercice 2** : Analyse des variables quantitatives
    - ✅ **Exercice 3** : Nuages de points et pairplot
    - ✅ **Exercice 4** : Boxplots par espèce
    - ✅ **Exercice 5** : Corrélations et visualisations avancées
    
    ---
    
    ### 📝 Instructions de déploiement :
    
    **1. Entraîner le modèle :**
    ```bash
    python tp_iris_complet.py
    ```
    
    **2. Lancer l'application Streamlit :**
    ```bash
    streamlit run streamlit_app.py
    ```
    
    **3. GitHub et Streamlit Cloud :**
    - Créer un repo GitHub
    - Ajouter tous les fichiers + Iris.csv
    - Déployer sur Streamlit Cloud
    
    ---
    
    ### 🎯 Résultats typiques :
    
    Les modèles atteignent généralement une exactitude de **95-100%** sur ce dataset,
    démontrant l'efficacité du Machine Learning pour la classification de données bien structurées.
    """)
    
    st.success("✅ Application développée avec ❤️ pour l'apprentissage du ML")
    
    st.markdown("---")
    
    # Informations sur le dataset
    if df is not None:
        st.subheader("📊 Informations sur le dataset actuel")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📝 Total échantillons", len(df))
        
        with col2:
            if 'species' in df.columns:
                st.metric("🌸 Nombre d'espèces", df['species'].nunique())
        
        with col3:
            st.metric("📊 Nombre de variables", len(df.columns) - 1)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>"
    "🌸 Iris Classifier - ML Project 2025 | "
    "Développé avec Streamlit & Scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
