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
    ["🏠 Accueil", "🔮 Prédiction", "📊 Analyse des Données", "ℹ️ À propos"]
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
        - 📊 Visualisation des données
        - 📈 Analyse statistique
        - 🤖 Modèles de ML entraînés
        - 🎨 Interface intuitive
        """)
    
    # Statut du modèle
    st.markdown("---")
    if model is not None:
        st.success("✅ Modèle chargé avec succès ! Vous pouvez faire des prédictions.")
    else:
        st.warning("⚠️ Modèle non disponible. Veuillez d'abord exécuter `tp_iris_complet.py` pour entraîner le modèle.")
        st.code("python tp_iris_complet.py", language="bash")
    
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
                
                # Graphique des probabilités
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                bars = ax.barh(prob_df['Espèce'], prob_df['Probabilité'], color=colors)
                ax.set_xlabel('Probabilité')
                ax.set_xlim([0, 1])
                ax.set_title('Probabilités par espèce')
                
                # Ajouter les valeurs sur les barres
                for i, (bar, v) in enumerate(zip(bars, prob_df['Probabilité'])):
                    ax.text(v + 0.02, i, f'{v:.2%}', va='center')
                
                st.pyplot(fig)
                plt.close()
                
                # Afficher le tableau
                st.dataframe(prob_df.style.format({'Probabilité': '{:.2%}'}))
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.info("Vérifiez que le modèle a été correctement entraîné.")

# ========== PAGE ANALYSE ==========
elif page == "📊 Analyse des Données":
    st.header("Analyse des Données Iris")
    
    if df is None:
        st.error("❌ Dataset non disponible")
        st.stop()
    
    st.subheader("📋 Aperçu des données")
    st.dataframe(df.head(10))
    
    st.subheader("📊 Statistiques descriptives")
    st.dataframe(df.describe())
    
    st.markdown("---")
    
    # Visualisations
    tab1, tab2, tab3 = st.tabs(["Distribution", "Corrélations", "Boxplots"])
    
    with tab1:
        st.subheader("Distribution des espèces")
        if 'species' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            counts = df['species'].value_counts()
            ax.bar(counts.index, counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_ylabel('Nombre')
            ax.set_xlabel('Espèce')
            ax.set_title('Répartition des espèces')
            plt.xticks(rotation=0)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Colonne 'species' non trouvée dans le dataset")
    
    with tab2:
        st.subheader("Matrice de corrélation")
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax, center=0)
                ax.set_title('Corrélations entre variables')
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Aucune variable numérique trouvée")
        except Exception as e:
            st.error(f"Erreur lors de la création de la matrice de corrélation : {e}")
    
    with tab3:
        st.subheader("Boxplots par espèce")
        
        # Trouver les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            variable = st.selectbox("Choisir une variable:", numeric_cols)
            
            if 'species' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                df.boxplot(column=variable, by='species', ax=ax)
                plt.suptitle('')
                ax.set_title(f'{variable} par espèce')
                ax.set_xlabel('Espèce')
                ax.set_ylabel(variable)
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Colonne 'species' non trouvée")
        else:
            st.warning("Aucune variable numérique disponible")

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
    - **Streamlit** : Interface web
    - **Flask** : API REST (optionnel)
    
    #### 📚 Dataset :
    Le dataset Iris est un classique en Machine Learning, créé par Edgar Anderson 
    et popularisé par R.A. Fisher en 1936.
    
    #### 🤖 Modèles testés :
    - K-Nearest Neighbors (KNN)
    - Régression Logistique
    - Arbre de Décision
    - Naive Bayes
    - SVM
    - Réseau de Neurones
    
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
    """)
    
    st.success("✅ Application développée avec ❤️ pour l'apprentissage du ML")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>🌸 Iris Classifier - ML Project 2025</div>",
    unsafe_allow_html=True
)
