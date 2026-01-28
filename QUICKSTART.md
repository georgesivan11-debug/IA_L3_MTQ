# 🚀 Guide de Démarrage Rapide

## ⚡ Démarrage en 3 étapes

### Étape 1 : Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 2 : Entraîner le modèle
```bash
python iris_classification_complete.py
```
Cela va créer les fichiers `best_model.pkl` et `scaler.pkl`

### Étape 3 : Lancer l'application
```bash
streamlit run streamlit_app.py
```

Votre app sera disponible sur http://localhost:8501 🎉

---

## 🐙 Déploiement GitHub Express

### Méthode automatique (recommandée)

**Linux/Mac :**
```bash
chmod +x deploy.sh
./deploy.sh "Premier commit"
```

**Windows :**
```bash
deploy.bat "Premier commit"
```

### Méthode manuelle

```bash
git init
git add .
git commit -m "Premier commit"
git branch -M main
git remote add origin https://github.com/VOTRE-USERNAME/VOTRE-REPO.git
git push -u origin main
```

---

## ☁️ Déploiement Streamlit Cloud

1. Allez sur https://streamlit.io/cloud
2. Connectez-vous avec GitHub
3. Cliquez sur "New app"
4. Sélectionnez votre repo
5. Fichier principal : `streamlit_app.py`
6. Cliquez sur "Deploy"

Attendez 2-3 minutes et votre app sera en ligne ! 🚀

---

## 🧪 Test de l'API Flask (optionnel)

### Lancer l'API
```bash
python app.py
```

### Tester avec curl
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## 📂 Fichiers importants

| Fichier | Description |
|---------|-------------|
| `iris_classification_complete.py` | Script d'entraînement complet |
| `streamlit_app.py` | Application Streamlit |
| `app.py` | API Flask |
| `pp.py` | Analyses exploratoires |
| `Iris.csv` | Dataset |
| `requirements.txt` | Dépendances Python |

---

## ❓ Problèmes courants

### "No module named 'xxx'"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: Iris.csv"
Assurez-vous que `Iris.csv` est dans le même dossier que vos scripts.

### L'app Streamlit ne démarre pas
Vérifiez que vous avez bien exécuté `iris_classification_complete.py` d'abord.

---

## 🎯 Checklist

- [ ] Dépendances installées
- [ ] Modèle entraîné (`best_model.pkl` créé)
- [ ] App Streamlit testée en local
- [ ] Code poussé sur GitHub
- [ ] App déployée sur Streamlit Cloud

---

Bon développement ! 💪
