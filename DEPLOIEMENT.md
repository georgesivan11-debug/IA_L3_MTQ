# 🚀 Guide de Déploiement sur GitHub et Streamlit Cloud

## 📋 Étape 1 : Préparation des fichiers

Avant de déployer, assurez-vous d'avoir ces fichiers dans votre dossier :

```
votre-projet/
├── iris_classification_complete.py
├── app.py
├── streamlit_app.py
├── pp.py
├── Iris.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## 🐙 Étape 2 : Déploiement sur GitHub

### 2.1 Créer un compte GitHub (si vous n'en avez pas)
1. Allez sur https://github.com
2. Cliquez sur "Sign up"
3. Suivez les instructions

### 2.2 Créer un nouveau repository

1. Sur GitHub, cliquez sur le bouton vert **"New"** ou **"+"** → **"New repository"**
2. Remplissez les informations :
   - **Repository name** : `iris-classification` (ou autre nom)
   - **Description** : "Classification des fleurs Iris avec ML"
   - **Public** ou **Private** : à votre choix
   - **Ne cochez PAS** "Add a README" (on a déjà le nôtre)
3. Cliquez sur **"Create repository"**

### 2.3 Initialiser Git localement

Ouvrez un terminal dans votre dossier projet et exécutez :

```bash
# Initialiser Git
git init

# Ajouter tous les fichiers
git add .

# Créer le premier commit
git commit -m "Premier commit - Projet classification Iris"

# Renommer la branche en 'main' (si nécessaire)
git branch -M main

# Lier au repository GitHub (remplacez YOUR-USERNAME et YOUR-REPO)
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git

# Pousser le code vers GitHub
git push -u origin main
```

### 2.4 Vérifier sur GitHub

Retournez sur votre page GitHub et rafraîchissez. Vous devriez voir tous vos fichiers !

## ☁️ Étape 3 : Déploiement sur Streamlit Cloud

### 3.1 Créer un compte Streamlit Cloud

1. Allez sur https://streamlit.io/cloud
2. Cliquez sur **"Sign up"**
3. Connectez-vous avec votre compte GitHub

### 3.2 Déployer l'application

1. Une fois connecté, cliquez sur **"New app"**
2. Remplissez les informations :
   - **Repository** : Sélectionnez `YOUR-USERNAME/iris-classification`
   - **Branch** : `main`
   - **Main file path** : `streamlit_app.py`
   - **App URL** : Choisissez un nom (ex: `iris-classifier`)

3. Cliquez sur **"Deploy!"**

### 3.3 Ajouter les fichiers nécessaires

Pour que Streamlit Cloud trouve le fichier `Iris.csv` et les modèles, vous avez deux options :

**Option A : Uploader le CSV dans le repo**
- Le fichier `Iris.csv` doit être dans votre repo GitHub
- Streamlit le trouvera automatiquement

**Option B : Générer les modèles au démarrage**
- Modifiez `streamlit_app.py` pour entraîner les modèles s'ils n'existent pas
- Ajoutez ce code au début :

```python
import os
if not os.path.exists('best_model.pkl'):
    os.system('python iris_classification_complete.py')
```

### 3.4 Configuration avancée (optionnel)

Si vous avez des secrets (API keys, etc.), utilisez Streamlit Secrets :

1. Dans les paramètres de l'app sur Streamlit Cloud
2. Allez dans **"Secrets"**
3. Ajoutez vos variables secrètes au format TOML

## 🔄 Étape 4 : Mises à jour futures

Pour mettre à jour votre code :

```bash
# Après avoir modifié vos fichiers
git add .
git commit -m "Description des changements"
git push
```

Streamlit Cloud redéploiera automatiquement votre app !

## 🐛 Dépannage

### Problème : "ModuleNotFoundError"
**Solution** : Vérifiez que toutes les dépendances sont dans `requirements.txt`

### Problème : "FileNotFoundError: Iris.csv"
**Solution** : Assurez-vous que `Iris.csv` est bien dans le repo GitHub

### Problème : L'app ne démarre pas
**Solution** : Vérifiez les logs dans Streamlit Cloud pour identifier l'erreur

### Problème : Git demande un mot de passe
**Solution** : Utilisez un Personal Access Token :
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token
3. Utilisez le token comme mot de passe

## 📱 Étape 5 : Partager votre application

Une fois déployée, vous obtiendrez une URL comme :
```
https://YOUR-APP-NAME.streamlit.app
```

Partagez cette URL avec qui vous voulez ! 🎉

## 🎯 Checklist finale

- [ ] Code poussé sur GitHub
- [ ] README.md bien formaté
- [ ] requirements.txt complet
- [ ] Iris.csv présent dans le repo
- [ ] App déployée sur Streamlit Cloud
- [ ] App testée et fonctionnelle
- [ ] URL partageable obtenue

## 💡 Conseils supplémentaires

1. **Badge GitHub** : Ajoutez un badge dans votre README :
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP.streamlit.app)
```

2. **Demo GIF** : Créez un GIF de votre app avec https://www.screentogif.com/ et ajoutez-le au README

3. **Documentation** : Mettez à jour le README avec des screenshots de votre app

Bon déploiement ! 🚀
