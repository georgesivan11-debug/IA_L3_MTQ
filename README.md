# 🌸 Classification des Fleurs Iris - Projet ML

Application complète de Machine Learning pour la classification des fleurs Iris avec interface Streamlit et API Flask.

## 📋 Description

Ce projet implémente un système de classification des fleurs Iris basé sur leurs caractéristiques morphologiques (longueur/largeur des sépales et pétales). Il permet de prédire l'espèce parmi : Setosa, Versicolor et Virginica.

## 🎯 Fonctionnalités

- ✅ **Analyse exploratoire** complète des données
- 🤖 **6 modèles de ML** testés et comparés (KNN, LR, DT, NB, SVM, ANN)
- 🔧 **Optimisation des hyperparamètres** avec GridSearchCV
- 🌐 **API REST** avec Flask
- 🎨 **Dashboard interactif** avec Streamlit
- 📊 **Visualisations** avancées des données et résultats

## 🛠️ Technologies utilisées

- **Python 3.8+**
- **Scikit-learn** - Machine Learning
- **Pandas & NumPy** - Manipulation de données
- **Matplotlib & Seaborn** - Visualisation
- **Flask** - API REST
- **Streamlit** - Interface web interactive

## 📦 Installation

1. **Cloner le repository :**
```bash
git clone https://github.com/votre-username/iris-classification.git
cd iris-classification
```

2. **Créer un environnement virtuel (recommandé) :**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### 1️⃣ Entraîner les modèles

Exécutez d'abord le script principal pour entraîner les modèles :

```bash
python iris_classification_complete.py
```

Cela va :
- Charger et analyser les données
- Entraîner 6 modèles différents
- Optimiser les hyperparamètres
- Sauvegarder le meilleur modèle (`best_model.pkl`)

### 2️⃣ Lancer l'API Flask

Dans un terminal :

```bash
python app.py
```

L'API sera accessible sur `http://localhost:5000`

**Exemple de requête :**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### 3️⃣ Lancer le Dashboard Streamlit

Dans un autre terminal :

```bash
streamlit run streamlit_app.py
```

L'application sera accessible sur `http://localhost:8501`

## 📁 Structure du projet

```
iris-classification/
│
├── iris_classification_complete.py  # Script principal d'entraînement
├── app.py                           # API Flask
├── streamlit_app.py                 # Dashboard Streamlit
├── pp.py                            # Analyses exploratoires
│
├── Iris.csv                         # Dataset
├── best_model.pkl                   # Modèle entraîné (généré)
├── scaler.pkl                       # Scaler pour normalisation (généré)
│
├── requirements.txt                 # Dépendances Python
├── README.md                        # Ce fichier
└── .gitignore                       # Fichiers à ignorer
```

## 📊 Résultats

Les modèles ont été évalués sur un ensemble de test (20% des données). Voici les performances typiques :

| Modèle | Exactitude |
|--------|-----------|
| KNN (optimisé) | ~97% |
| SVM | ~97% |
| Logistic Regression | ~95% |
| Decision Tree | ~93% |
| Neural Network | ~95% |
| Naive Bayes | ~95% |

## 🎓 Contexte académique

Ce projet a été réalisé dans le cadre d'un TP sur l'apprentissage automatique. Il couvre :

1. L'analyse exploratoire de données
2. La préparation et le preprocessing
3. L'entraînement de modèles de classification
4. L'évaluation et l'optimisation
5. Le déploiement avec Flask et Streamlit

## 📚 Dataset

Le **Iris Dataset** est un classique du Machine Learning :
- Créé par Edgar Anderson (1935)
- Popularisé par R.A. Fisher (1936)
- 150 échantillons (50 par espèce)
- 4 caractéristiques numériques
- 3 classes équilibrées

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👤 Auteur

Votre Nom - Projet TP Machine Learning 2025

## 🙏 Remerciements

- Dataset Iris : R.A. Fisher & Edgar Anderson
- Communauté Scikit-learn
- Documentation Streamlit et Flask

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !**
